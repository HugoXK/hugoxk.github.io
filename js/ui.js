/* Ke Xu - UI enhancements: nav scroll-spy, back-to-top, gallery lightbox */

(function () {
  "use strict";

  function onReady(fn) {
    if (document.readyState !== "loading") { fn(); }
    else { document.addEventListener("DOMContentLoaded", fn); }
  }

  onReady(function () {

    /* ---- nav scroll-spy ---- */
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
    if ("IntersectionObserver" in window && sections.length) {
      var spy = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting && linkById[en.target.id]) {
            Object.keys(linkById).forEach(function (k) {
              linkById[k].classList.toggle("active", k === en.target.id);
            });
          }
        });
      }, { rootMargin: "-30% 0px -60% 0px" });
      sections.forEach(function (s) { spy.observe(s); });
    }

    /* ---- back to top ---- */
    var top = document.getElementById("back-to-top");
    if (top) {
      var onScroll = function () {
        top.classList.toggle("show", window.scrollY > 600);
      };
      window.addEventListener("scroll", onScroll, { passive: true });
      onScroll();
      top.addEventListener("click", function () {
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    }

    /* ---- gallery lightbox ---- */
    var grid = document.querySelector(".gallery-grid");
    if (grid) {
      grid.addEventListener("click", function (e) {
        var img = e.target;
        if (!img || img.tagName !== "IMG") { return; }
        var fig = img.closest("figure");
        var cap = fig ? fig.querySelector("figcaption") : null;

        var overlay = document.createElement("div");
        overlay.className = "lightbox";
        var big = document.createElement("img");
        big.src = img.src;
        big.alt = img.alt || "";
        overlay.appendChild(big);
        if (cap) {
          var fc = document.createElement("figcaption");
          fc.textContent = cap.textContent;
          overlay.appendChild(fc);
        }
        var close = function () {
          if (overlay.parentNode) { overlay.parentNode.removeChild(overlay); }
          document.removeEventListener("keydown", onKey);
        };
        var onKey = function (ev) { if (ev.key === "Escape") { close(); } };
        overlay.addEventListener("click", close);
        document.addEventListener("keydown", onKey);
        document.body.appendChild(overlay);
      });
    }
  });
})();
