/* Ke Xu - shared UI: theme, scroll progress, nav spy, reveal, back-to-top. */

(function () {
  "use strict";

  function onReady(fn) {
    if (document.readyState !== "loading") { fn(); }
    else { document.addEventListener("DOMContentLoaded", fn); }
  }

  function lang() { return document.documentElement.getAttribute("lang") === "zh" ? "zh" : "en"; }
  function reduced() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function setTheme(theme) {
    theme = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("theme", theme); } catch (e) {}
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) { meta.setAttribute("content", theme === "dark" ? "#0d0c0b" : "#f4f1ea"); }
    document.dispatchEvent(new CustomEvent("themechange", { detail: { theme: theme } }));
    updateThemeButton();
  }

  function updateThemeButton() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) { return; }
    var dark = document.documentElement.getAttribute("data-theme") === "dark";
    var label = lang() === "zh"
      ? (dark ? "切换到浅色模式" : "切换到深色模式")
      : (dark ? "Switch to light mode" : "Switch to dark mode");
    btn.setAttribute("aria-label", label);
    btn.setAttribute("title", label);
    btn.setAttribute("aria-pressed", dark ? "true" : "false");
  }

  function bindTheme() {
    var btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.addEventListener("click", function () {
        setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
      });
    }
    updateThemeButton();
    document.addEventListener("langchange", updateThemeButton);
  }

  function bindProgress() {
    var bar = document.getElementById("scroll-progress");
    if (!bar) { return; }
    var ticking = false;
    function update() {
      var doc = document.documentElement;
      var max = doc.scrollHeight - window.innerHeight;
      var pct = max > 0 ? (window.scrollY / max) * 100 : 0;
      bar.style.width = Math.max(0, Math.min(100, pct)).toFixed(2) + "%";
      ticking = false;
    }
    window.addEventListener("scroll", function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
    }, { passive: true });
    window.addEventListener("resize", update, { passive: true });
    update();
  }

  function bindNav() {
    var links = document.querySelectorAll(".nav-bar a[href^='#']");
    var linkById = {};
    var sections = [];
    Array.prototype.forEach.call(links, function (a) {
      var id = a.getAttribute("href").slice(1);
      var sec = document.getElementById(id);
      if (sec) {
        linkById[id] = a;
        sections.push(sec);
      }
    });
    if (!("IntersectionObserver" in window) || !sections.length) { return; }
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting || !linkById[en.target.id]) { return; }
        Object.keys(linkById).forEach(function (k) {
          linkById[k].classList.toggle("active", k === en.target.id);
        });
      });
    }, { rootMargin: "-28% 0px -62% 0px" });
    sections.forEach(function (s) { spy.observe(s); });
  }

  function bindBackToTop() {
    var top = document.getElementById("back-to-top");
    if (!top) { return; }
    function onScroll() { top.classList.toggle("show", window.scrollY > 700); }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    top.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: reduced() ? "auto" : "smooth" });
    });
  }

  function bindReveal() {
    if (reduced() || !("IntersectionObserver" in window)) { return; }
    var candidates = document.querySelectorAll(".section, .job, .research-item, .news-item, .gallery-item, .travel-stat");
    if (!candidates.length) { return; }
    Array.prototype.forEach.call(candidates, function (el, i) {
      el.classList.add("reveal");
      el.style.setProperty("--reveal-delay", Math.min(80, i * 18) + "ms");
    });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          en.target.classList.add("is-visible");
          io.unobserve(en.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
    Array.prototype.forEach.call(candidates, function (el) { io.observe(el); });
  }

  onReady(function () {
    bindTheme();
    bindProgress();
    bindNav();
    bindBackToTop();
    bindReveal();
  });
})();
