const toggle = document.querySelector(".nav-toggle");
const nav = document.querySelector("#site-nav");

if (toggle && nav) {
  toggle.addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", String(open));
    toggle.textContent = open ? "Close" : "Menu";
  });

  nav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      nav.classList.remove("is-open");
      toggle.setAttribute("aria-expanded", "false");
      toggle.textContent = "Menu";
    });
  });
}

const resume = document.querySelector("#resume");

if (resume) {
  document.querySelectorAll("[data-open-resume]").forEach((button) => {
    button.addEventListener("click", () => resume.showModal());
  });

  const close = resume.querySelector("[data-close-resume]");
  if (close) close.addEventListener("click", () => resume.close());

  resume.addEventListener("click", (event) => {
    if (event.target === resume) resume.close();
  });
}

const links = [...document.querySelectorAll('#site-nav a[href^="#"]')];
const sections = links
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

if (sections.length && "IntersectionObserver" in window) {
  const spy = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = entry.target.id;
        links.forEach((link) => {
          link.classList.toggle("is-active", link.getAttribute("href") === `#${id}`);
        });
      });
    },
    { rootMargin: "-45% 0px -45% 0px" }
  );

  sections.forEach((section) => spy.observe(section));
}
