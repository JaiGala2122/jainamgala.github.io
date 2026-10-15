"""Seven-day warehouse plan that prices freight after orders are grouped.

The slow model gives the solver a yes/no choice for every order, carrier,
weight band, and day. Those last two choices are lookups once a lane's
orders are known, and most orders have only one legal warehouse. This
script keeps the same eligibility and freight rules, and solves only the
warehouse assignment.
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

HORIZON_DAYS = 7


class Lane:
    def __init__(self, mode, carrier, port, transit, min_w, max_w, rate, min_cost):
        self.mode = mode
        self.carrier = carrier
        self.port = port
        self.transit = transit
        self.min_w = min_w
        self.max_w = max_w
        self.rate = rate
        self.min_cost = min_cost


def strip_text(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.columns = table.columns.str.strip()
    for column in table.select_dtypes(include=["object", "str"]).columns:
        table[column] = table[column].str.strip()
    return table


def load_tables(path: Path) -> dict[str, pd.DataFrame]:
    tables = pd.read_excel(path, sheet_name=None)
    return {name: strip_text(table) for name, table in tables.items()}


def drop_dominated(min_w, max_w, rate, min_cost, mode):
    """Remove a band when another band covers it at a lower charge."""
    keep = []
    count = len(min_w)
    for i in range(count):
        dominated = False
        for j in range(count):
            if i == j:
                continue
            covers = min_w[j] <= min_w[i] and max_w[j] >= max_w[i]
            if not covers:
                continue
            if mode == "GROUND":
                cheaper = rate[j] < rate[i] or (
                    rate[j] == rate[i]
                    and (min_w[j] < min_w[i] or max_w[j] > max_w[i])
                )
            else:
                cheaper = (
                    rate[j] <= rate[i]
                    and min_cost[j] <= min_cost[i]
                    and (
                        rate[j] < rate[i]
                        or min_cost[j] < min_cost[i]
                        or min_w[j] < min_w[i]
                        or max_w[j] > max_w[i]
                    )
                )
            if cheaper:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def build_lanes(freight: pd.DataFrame) -> dict[tuple[str, str], list[Lane]]:
    lanes: dict[tuple[str, str], list[Lane]] = defaultdict(list)
    keys = [
        "orig_port_cd",
        "dest_port_cd",
        "Carrier",
        "svc_cd",
        "mode_dsc",
        "tpt_day_cnt",
    ]
    for key, group in freight.groupby(keys, sort=False):
        port, _dest, carrier, service, mode, transit = key
        deduped = group.drop_duplicates(
            ["minm_wgh_qty", "max_wgh_qty", "rate", "minimum cost"]
        )
        min_w = deduped["minm_wgh_qty"].to_numpy(dtype=float)
        max_w = deduped["max_wgh_qty"].to_numpy(dtype=float)
        rate = deduped["rate"].to_numpy(dtype=float)
        min_cost = deduped["minimum cost"].to_numpy(dtype=float)
        chosen = drop_dominated(min_w, max_w, rate, min_cost, mode)
        lanes[(port, service)].append(
            Lane(
                mode,
                carrier,
                port,
                int(transit),
                min_w[chosen],
                max_w[chosen],
                rate[chosen],
                min_cost[chosen],
            )
        )
    return lanes


def cost_on_lane(weights: np.ndarray, lane: Lane) -> float | None:
    """Price one lane from its combined weight. None if no band contains it."""
    if len(weights) == 0:
        return 0.0
    total = float(weights.sum())
    fits = (lane.min_w <= total + 1e-8) & (total <= lane.max_w + 1e-8)
    if not np.any(fits):
        return None
    if lane.mode == "GROUND":
        # Flat truck charge. Orders on the lane split this charge by weight,
        # so the group pays the selected rate once.
        return float(lane.rate[fits].min())
    rates = lane.rate[fits]
    minimums = lane.min_cost[fits]
    charges = np.maximum(weights[:, None] * rates[None, :], minimums[None, :])
    return float(charges.sum(axis=0).min())


def price_air(weights: np.ndarray, air_lanes: list[Lane]) -> float | None:
    best = None
    for lane in air_lanes:
        cost = cost_on_lane(weights, lane)
        if cost is not None and (best is None or cost < best):
            best = cost
    if best is not None:
        return best
    return split_across_lanes(weights, air_lanes)


def split_across_lanes(weights: np.ndarray, lanes: list[Lane]) -> float | None:
    """Place orders onto lanes when one lane cannot hold the combined weight."""
    if len(weights) == 0:
        return 0.0
    if not lanes:
        return None
    ordered = sorted(lanes, key=lambda lane: float(lane.max_w.max()), reverse=True)
    bags: list[list[float]] = [[] for _ in ordered]
    totals = [0.0] * len(ordered)
    for weight in sorted((float(value) for value in weights), reverse=True):
        placed = False
        for index, lane in enumerate(ordered):
            if totals[index] + weight <= float(lane.max_w.max()) + 1e-6:
                bags[index].append(weight)
                totals[index] += weight
                placed = True
                break
        if not placed:
            return None
    total_cost = 0.0
    for bag in bags:
        if not bag:
            continue
        bundle = np.asarray(bag, dtype=float)
        best = None
        for lane in lanes:
            cost = cost_on_lane(bundle, lane)
            if cost is not None and (best is None or cost < best):
                best = cost
        if best is None:
            return None
        total_cost += best
    return total_cost


def ground_then_air(weights: np.ndarray, lanes: list[Lane]) -> float | None:
    """Fill the cheapest ground trucks, then send the rest by air.

    A ground lane charges its rate once for the whole truck. Extra orders
    that fit are free, so a truck is filled with the smallest orders before
    the dearer air minimums are paid.
    """
    ground = [lane for lane in lanes if lane.mode == "GROUND"]
    air = [lane for lane in lanes if lane.mode == "AIR"]
    if not ground:
        return price_air(weights, air)

    remaining = sorted(float(value) for value in weights)
    truck_cost = 0.0
    for lane in sorted(ground, key=lambda item: float(item.rate.min())):
        cap = float(lane.max_w.max())
        taken: list[float] = []
        total = 0.0
        still: list[float] = []
        for weight in remaining:
            if total + weight <= cap + 1e-6:
                taken.append(weight)
                total += weight
            else:
                still.append(weight)
        if not taken:
            continue
        ground_cost = cost_on_lane(np.asarray(taken, dtype=float), lane)
        if ground_cost is None:
            continue
        rest = np.asarray(still, dtype=float)
        rest_cost = 0.0 if len(rest) == 0 else price_air(rest, air)
        before = price_air(np.asarray(remaining, dtype=float), air)
        if rest_cost is None:
            continue
        if before is not None and ground_cost + rest_cost >= before - 1e-6:
            continue
        truck_cost += ground_cost
        remaining = still
    rest = np.asarray(remaining, dtype=float)
    rest_cost = 0.0 if len(rest) == 0 else price_air(rest, air)
    if rest_cost is None:
        return None
    return truck_cost + rest_cost


def price_weights(weights: np.ndarray, lanes: list[Lane]) -> float | None:
    if len(weights) == 0:
        return 0.0
    weights = np.asarray(weights, dtype=float)
    options = []
    for lane in lanes:
        cost = cost_on_lane(weights, lane)
        if cost is not None:
            options.append(cost)
    mixed = ground_then_air(weights, lanes)
    if mixed is not None:
        options.append(mixed)
    if not options:
        split = split_across_lanes(weights, lanes)
        if split is not None:
            options.append(split)
    if not options:
        return None
    return min(options)


def eligible_assignments(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = tables["OrderList"].rename(columns={"Plant Code": "Historical Plant"})
    products = (
        tables["ProductsPerPlant"]
        .rename(columns={"Plant Code": "Eligible Warehouse"})
        .drop_duplicates()
    )
    options = orders.merge(products, on="Product ID", how="left")
    vmi = (
        tables["VmiCustomers"]
        .rename(columns={"Plant Code": "Eligible Warehouse", "Customers": "Customer"})
        .drop_duplicates()
    )
    restricted = set(vmi["Eligible Warehouse"])
    vmi = vmi.copy()
    vmi["Customer Match"] = True
    options = options.merge(
        vmi,
        on=["Eligible Warehouse", "Customer"],
        how="left",
        validate="many_to_one",
    )
    allowed = options["Eligible Warehouse"].notna() & (
        ~options["Eligible Warehouse"].isin(restricted)
        | options["Customer Match"].eq(True)
    )
    eligible = options.loc[allowed].drop(columns="Customer Match")
    return eligible.drop_duplicates(["Order ID", "Eligible Warehouse"]).reset_index(
        drop=True
    )


def solve_warehouses(pair_order, pair_plant, objective, order_count, plant_count, upper):
    """Assign flexible orders. `upper` is remaining horizon capacity by plant."""
    variable_count = len(pair_order)
    rows = np.concatenate([pair_order, order_count + pair_plant])
    columns = np.concatenate([np.arange(variable_count), np.arange(variable_count)])
    matrix = coo_matrix(
        (np.ones(2 * variable_count), (rows, columns)),
        shape=(order_count + plant_count, variable_count),
    ).tocsc()
    limits_low = np.concatenate([np.ones(order_count), np.zeros(plant_count)])
    limits_high = np.concatenate([np.ones(order_count), upper.astype(float)])
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.uint8),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, limits_low, limits_high),
        options={"time_limit": 30, "mip_rel_gap": 0},
    )
    return result


def pack_days(order_indexes, service, weight, daily_capacity):
    """Fill earlier days first, keeping each service together."""
    rank = {"DTP": 0, "DTD": 1, "CRF": 2}
    ordered = sorted(
        order_indexes,
        key=lambda index: (rank[service[index]], -weight[index], index),
    )
    loads = np.zeros(HORIZON_DAYS, dtype=int)
    assigned_day = {}
    for index in ordered:
        placed = False
        for day in range(HORIZON_DAYS):
            if loads[day] < daily_capacity:
                assigned_day[index] = day
                loads[day] += 1
                placed = True
                break
        if not placed:
            raise RuntimeError("Daily packing failed despite horizon capacity.")
    return assigned_day, loads


def main() -> None:
    started = time.perf_counter()
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("Supply chain logisitcs problem.xlsx")
    )
    tables = load_tables(path)
    orders = tables["OrderList"]
    print(f"Orders: {len(orders):,}")

    eligible = eligible_assignments(tables)
    lanes = build_lanes(tables["FreightRates"])
    band_count = sum(len(lane.min_w) for group in lanes.values() for lane in group)
    raw_bands = len(tables["FreightRates"])
    print(f"Freight bands kept after dropping dominated bands: {band_count:,} of {raw_bands:,}")

    plants = tables["WhCapacities"]["Plant ID"].tolist()
    plant_index = {plant: index for index, plant in enumerate(plants)}
    daily_capacity = tables["WhCapacities"]["Daily Capacity"].to_numpy(dtype=int)
    unit_cost = tables["WhCosts"].set_index("WH")["Cost/unit"].to_dict()
    plant_ports = (
        tables["PlantPorts"]
        .groupby("Plant Code")["Port"]
        .apply(lambda values: list(dict.fromkeys(values)))
        .to_dict()
    )
    plant_lanes = {}
    for plant, ports in plant_ports.items():
        for service in ("DTD", "DTP"):
            collected = []
            for port in ports:
                collected.extend(lanes.get((port, service), []))
            plant_lanes[(plant, service)] = collected

    order_ids = orders["Order ID"].to_numpy()
    order_index = {order_id: index for index, order_id in enumerate(order_ids)}
    weight = orders["Weight"].to_numpy(dtype=float)
    quantity = orders["Unit quantity"].to_numpy(dtype=float)
    service = orders["Service Level"].to_numpy()

    # Worked example from the Excel file: 387 units of product 1689547.
    example = eligible[
        (eligible["Product ID"] == 1689547)
        & (eligible["Unit quantity"] == 387)
        & (eligible["Historical Plant"] == "PLANT03")
    ].copy()
    example_orders = example["Order ID"].unique()
    if len(example_orders) != 1:
        raise ValueError("Expected one 387-unit example order at PLANT03.")
    example = example[example["Order ID"] == example_orders[0]]
    example["Warehouse Cost"] = example["Unit quantity"] * example[
        "Eligible Warehouse"
    ].map(unit_cost)
    expected_cost = {"PLANT02": 184.7938618, "PLANT03": 200.2732321}
    for plant, amount in expected_cost.items():
        actual = example.loc[
            example["Eligible Warehouse"] == plant, "Warehouse Cost"
        ].iloc[0]
        if abs(actual - amount) >= 0.01:
            raise AssertionError(f"Warehouse cost does not match Excel for {plant}.")
    print("Warehouse costs for the worked example match Excel.")

    kept_rows = []
    route_options = 0
    air_route_options = 0
    eligible_orders = eligible["Order ID"].to_numpy()
    eligible_plants = eligible["Eligible Warehouse"].to_numpy()
    for order_id, plant in zip(eligible_orders, eligible_plants):
        index = order_index[order_id]
        if service[index] == "CRF":
            freight = 0.0
            route_options += 1
        else:
            available = plant_lanes.get((plant, service[index]), [])
            freight = price_weights(weight[index : index + 1], available)
            if freight is not None:
                for lane in available:
                    if weight[index] <= float(lane.max_w.max()) + 1e-8:
                        route_options += 1
                        if lane.mode == "AIR":
                            air_route_options += 1
        if freight is None:
            continue
        kept_rows.append((index, plant_index[plant], freight))

    pairs = pd.DataFrame(kept_rows, columns=["Order", "Plant", "Solo Freight"])
    missing = set(range(len(orders))) - set(pairs["Order"])
    if missing:
        raise ValueError(f"{len(missing)} orders have no priced warehouse.")
    print(
        f"Priced order-warehouse pairs: {len(pairs):,} "
        f"({time.perf_counter() - started:.1f}s)"
    )

    choice_counts = pairs.groupby("Order").size()
    flexible_orders = choice_counts[choice_counts > 1].index.to_numpy()
    forced = pairs[~pairs["Order"].isin(flexible_orders)].copy()
    flex = pairs[pairs["Order"].isin(flexible_orders)].copy()
    print(
        f"Orders with one legal warehouse: {len(forced):,}. "
        f"Orders the solver must place: {len(flexible_orders):,} "
        f"({len(flex):,} choices)."
    )

    forced_load = (
        forced.groupby("Plant").size().reindex(range(len(plants)), fill_value=0).to_numpy()
    )
    horizon = daily_capacity * HORIZON_DAYS
    remaining = horizon - forced_load
    overloaded = [
        f"{plants[i]} needs {forced_load[i]:,} and can take {horizon[i]:,}"
        for i in range(len(plants))
        if remaining[i] < 0
    ]
    if overloaded:
        print("No seven-day assignment can hold the orders with only one warehouse:")
        print("\n".join(overloaded))
        return

    # Remap flexible orders to constraint rows.
    flex = flex.reset_index(drop=True)
    flex_order_codes, _ = pd.factorize(flex["Order"], sort=False)
    plant_codes = flex["Plant"].to_numpy(dtype=int)

    def objective_for(kind: str) -> np.ndarray:
        if kind == "feasible":
            return np.zeros(len(flex))
        if kind == "plant03":
            return (plant_codes == plant_index["PLANT03"]).astype(float)
        warehouse_cost = quantity[flex["Order"].to_numpy()] * np.array(
            [unit_cost[plants[plant]] for plant in plant_codes]
        )
        return warehouse_cost + flex["Solo Freight"].to_numpy()

    print("\nSEVEN-DAY WAREHOUSE CHECK")
    feasibility = solve_warehouses(
        flex_order_codes,
        plant_codes,
        objective_for("feasible"),
        len(flexible_orders),
        len(plants),
        remaining,
    )
    if not feasibility.success:
        print("No assignment satisfies seven-day warehouse capacity.")
        print(feasibility.message)
        return
    print(f"Feasible assignment found in {time.perf_counter() - started:.1f}s.")

    dependence = solve_warehouses(
        flex_order_codes,
        plant_codes,
        objective_for("plant03"),
        len(flexible_orders),
        len(plants),
        remaining,
    )
    if dependence.success:
        chosen = flex.loc[dependence.x > 0.5, "Plant"].to_numpy()
        plant03_orders = int(forced_load[plant_index["PLANT03"]] + np.count_nonzero(
            chosen == plant_index["PLANT03"]
        ))
        print("\nMINIMUM PLANT03 DEPENDENCE — WAREHOUSE CONSTRAINTS")
        print(f"Orders assigned to PLANT03: {plant03_orders:,}")
        print(f"Share of all orders: {plant03_orders / len(orders):.1%}")
    else:
        print("Minimum PLANT03 check did not finish:", dependence.message)

    cost_result = solve_warehouses(
        flex_order_codes,
        plant_codes,
        objective_for("cost"),
        len(flexible_orders),
        len(plants),
        remaining,
    )
    if not cost_result.success:
        print("Cost assignment did not finish:", cost_result.message)
        return

    chosen_flex = flex.loc[cost_result.x > 0.5, ["Order", "Plant"]]
    assignment = pd.concat(
        [forced[["Order", "Plant"]], chosen_flex],
        ignore_index=True,
    )
    if len(assignment) != len(orders) or assignment["Order"].nunique() != len(orders):
        raise RuntimeError("The warehouse assignment does not cover every order once.")

    warehouse_of = np.empty(len(orders), dtype=int)
    warehouse_of[assignment["Order"].to_numpy()] = assignment["Plant"].to_numpy()
    day_of = np.full(len(orders), -1, dtype=int)
    for plant in range(len(plants)):
        indexes = np.flatnonzero(warehouse_of == plant)
        if len(indexes) == 0:
            continue
        placed, loads = pack_days(indexes, service, weight, int(daily_capacity[plant]))
        if loads.max() > daily_capacity[plant]:
            raise RuntimeError("A day exceeds warehouse capacity.")
        for index, day in placed.items():
            day_of[index] = day
    if np.any(day_of < 0):
        raise RuntimeError("An order was not given a day.")

    members: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for index in range(len(orders)):
        if service[index] == "CRF":
            continue
        members[(warehouse_of[index], day_of[index], service[index])].append(index)

    cost_cache: dict[tuple[int, int, str], float] = {}

    def group_cost(plant: int, day: int, svc: str) -> float:
        key = (plant, day, svc)
        if key not in cost_cache:
            indexes = members.get(key, [])
            if not indexes:
                cost_cache[key] = 0.0
            else:
                priced = price_weights(
                    weight[np.asarray(indexes, dtype=int)],
                    plant_lanes[(plants[plant], svc)],
                )
                if priced is None:
                    raise RuntimeError(
                        f"No freight band fits {plants[plant]} day {day + 1} {svc}."
                    )
                cost_cache[key] = priced
        return cost_cache[key]

    def freight_total() -> float:
        return sum(group_cost(*key) for key in list(members))

    warehouse_total = float(
        sum(quantity[i] * unit_cost[plants[warehouse_of[i]]] for i in range(len(orders)))
    )
    freight_before = freight_total()
    search_started = time.perf_counter()
    moves = 0
    flexible_sorted = sorted(
        (int(order) for order in flexible_orders),
        key=lambda index: -quantity[index],
    )
    alternatives = defaultdict(list)
    for order, plant in zip(flex["Order"].to_numpy(), flex["Plant"].to_numpy()):
        alternatives[int(order)].append(int(plant))

    loads = np.zeros((len(plants), HORIZON_DAYS), dtype=int)
    for index in range(len(orders)):
        loads[warehouse_of[index], day_of[index]] += 1

    improved = True
    while improved and time.perf_counter() - search_started < 60:
        improved = False
        for index in flexible_sorted:
            origin = int(warehouse_of[index])
            origin_day = int(day_of[index])
            svc = service[index]
            best_saving = 0.01
            best_move = None
            for plant in alternatives[index]:
                day_choices = [
                    day
                    for day in range(HORIZON_DAYS)
                    if not (plant == origin and day == origin_day)
                    and loads[plant, day] < daily_capacity[plant]
                ]
                # Prefer a day that already carries this service.
                day_choices.sort(
                    key=lambda day: (
                        0 if members.get((plant, day, svc)) else 1,
                        loads[plant, day],
                    )
                )
                for day in day_choices[:2]:
                    warehouse_delta = quantity[index] * (
                        unit_cost[plants[plant]] - unit_cost[plants[origin]]
                    )
                    freight_delta = 0.0
                    if svc != "CRF":
                        old_members = members[(origin, origin_day, svc)]
                        reduced = [item for item in old_members if item != index]
                        old_now = group_cost(origin, origin_day, svc)
                        if reduced:
                            reduced_cost = price_weights(
                                weight[np.asarray(reduced, dtype=int)],
                                plant_lanes[(plants[origin], svc)],
                            )
                        else:
                            reduced_cost = 0.0
                        added = members.get((plant, day, svc), [])
                        added_before = group_cost(plant, day, svc) if added else 0.0
                        grown = added + [index]
                        grown_cost = price_weights(
                            weight[np.asarray(grown, dtype=int)],
                            plant_lanes[(plants[plant], svc)],
                        )
                        if reduced_cost is None or grown_cost is None:
                            continue
                        freight_delta = (reduced_cost - old_now) + (
                            grown_cost - added_before
                        )
                    saving = -(warehouse_delta + freight_delta)
                    if saving > best_saving:
                        best_saving = saving
                        best_move = (plant, day, reduced_cost if svc != "CRF" else 0.0, grown_cost if svc != "CRF" else 0.0)
            if best_move is None:
                continue
            plant, day, reduced_cost, grown_cost = best_move
            if svc != "CRF":
                members[(origin, origin_day, svc)] = [
                    item
                    for item in members[(origin, origin_day, svc)]
                    if item != index
                ]
                members[(plant, day, svc)].append(index)
                cost_cache[(origin, origin_day, svc)] = reduced_cost
                cost_cache[(plant, day, svc)] = grown_cost
            loads[origin, origin_day] -= 1
            loads[plant, day] += 1
            warehouse_of[index] = plant
            day_of[index] = day
            moves += 1
            improved = True
            warehouse_total += quantity[index] * (
                unit_cost[plants[plant]] - unit_cost[plants[origin]]
            )

    cost_cache.clear()
    freight_after = freight_total()
    if np.any(loads > daily_capacity[:, None]):
        raise RuntimeError("Local search exceeded a daily capacity.")
    if int(loads.sum()) != len(orders):
        raise RuntimeError("Local search lost or duplicated an order.")

    plant03 = int(np.count_nonzero(warehouse_of == plant_index["PLANT03"]))
    elapsed = time.perf_counter() - started
    print("\nWAREHOUSE AND SHIPPING PLAN")
    print(f"Orders assigned: {len(orders):,}")
    print(f"Warehouse cost: ${warehouse_total:,.2f}")
    print(f"Freight cost: ${freight_after:,.2f}")
    print(f"Total scenario cost: ${warehouse_total + freight_after:,.2f}")
    print(f"Orders assigned to PLANT03: {plant03:,}")
    print(f"Flexible orders moved after the first assignment: {moves:,}")
    print(f"Freight before those moves: ${freight_before:,.2f}")
    recalculated_warehouse = float(
        sum(quantity[i] * unit_cost[plants[warehouse_of[i]]] for i in range(len(orders)))
    )
    if abs(recalculated_warehouse - warehouse_total) > 0.01:
        raise RuntimeError("Warehouse cost did not reconcile after the search.")
    print(f"Finished in {elapsed:.1f}s")
    print(
        "Feasible scenario under these freight rules. "
        "Carrier and day are priced after the warehouse assignment, "
        "so the dollar total is not a proved minimum."
    )
    print(
        "Integrated model size avoided: "
        f"{route_options * HORIZON_DAYS:,} route-day choices and "
        f"{air_route_options * HORIZON_DAYS:,} air-charge variables."
    )

    print("\nDaily loads by warehouse (orders):")
    header = "Warehouse  " + " ".join(f"D{day+1:>5}" for day in range(HORIZON_DAYS))
    print(header)
    for plant, name in enumerate(plants):
        if loads[plant].sum() == 0:
            continue
        cells = " ".join(f"{int(loads[plant, day]):5d}" for day in range(HORIZON_DAYS))
        print(f"{name:10} {cells}")

    minimum_days = max(
        math.ceil(len(orders) / daily_capacity.sum()),
        math.ceil((forced_load / daily_capacity).max()),
    )
    print(f"\nMinimum days implied by one-warehouse orders: {minimum_days}")


if __name__ == "__main__":
    main()
