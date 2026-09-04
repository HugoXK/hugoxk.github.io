/* Ke Xu - i18n (EN / ZH)
   Any element carrying a data-zh attribute is bilingual:
   the original (English) markup is cached into data-en on
   first run, and the element swaps between the two on toggle.
   The choice persists in localStorage. */

(function () {
  "use strict";

  var STORAGE_KEY = "lang";

  function currentLang() {
    try { return localStorage.getItem(STORAGE_KEY) || "en"; }
    catch (e) { return "en"; }
  }

  function applyLang(lang) {
    lang = (lang === "zh") ? "zh" : "en";
    document.documentElement.setAttribute("lang", lang);

    var nodes = document.querySelectorAll("[data-zh]");
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (!el.hasAttribute("data-en")) {
        el.setAttribute("data-en", el.innerHTML);
      }
      el.innerHTML = (lang === "zh") ? el.getAttribute("data-zh") : el.getAttribute("data-en");
    }

    var buttons = document.querySelectorAll(".lang-switch button");
    for (var j = 0; j < buttons.length; j++) {
      buttons[j].className = (buttons[j].getAttribute("data-lang") === lang) ? "active" : "";
    }

    try { localStorage.setItem(STORAGE_KEY, lang); } catch (e) {}

    document.dispatchEvent(new CustomEvent("langchange", { detail: { lang: lang } }));
  }

  function onReady(fn) {
    if (document.readyState !== "loading") { fn(); }
    else { document.addEventListener("DOMContentLoaded", fn); }
  }

  onReady(function () {
    var toggle = document.querySelector(".lang-switch");
    if (toggle) {
      toggle.addEventListener("click", function (e) {
        var btn = e.target;
        while (btn && btn !== toggle && btn.tagName !== "BUTTON") { btn = btn.parentNode; }
        if (btn && btn.getAttribute && btn.getAttribute("data-lang")) {
          applyLang(btn.getAttribute("data-lang"));
        }
      });
    }
    applyLang(currentLang());
  });
})();
