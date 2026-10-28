/* Ke Xu - i18n (EN / ZH)
   Any element carrying data-zh is bilingual: the original English markup
   is cached in data-en, then swapped on toggle. Attribute translations use
   data-zh-placeholder, data-zh-title, data-zh-aria-label and data-zh-alt. */

(function () {
  "use strict";

  var STORAGE_KEY = "lang";
  var ATTRS = ["placeholder", "title", "aria-label", "alt"];

  function currentLang() {
    try { return localStorage.getItem(STORAGE_KEY) || "en"; }
    catch (e) { return "en"; }
  }

  function applyAttrs(el, lang) {
    ATTRS.forEach(function (attr) {
      var key = "data-zh-" + attr;
      if (!el.hasAttribute(key)) { return; }
      var enKey = "data-en-" + attr;
      if (!el.hasAttribute(enKey)) { el.setAttribute(enKey, el.getAttribute(attr) || ""); }
      el.setAttribute(attr, lang === "zh" ? el.getAttribute(key) : el.getAttribute(enKey));
    });
  }

  function applyLang(lang) {
    lang = (lang === "zh") ? "zh" : "en";
    document.documentElement.setAttribute("lang", lang);

    var nodes = document.querySelectorAll("[data-zh]");
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (!el.hasAttribute("data-en")) { el.setAttribute("data-en", el.innerHTML); }
      el.innerHTML = (lang === "zh") ? el.getAttribute("data-zh") : el.getAttribute("data-en");
    }
    var attrNodes = document.querySelectorAll("[data-zh-placeholder], [data-zh-title], [data-zh-aria-label], [data-zh-alt]");
    for (var j = 0; j < attrNodes.length; j++) { applyAttrs(attrNodes[j], lang); }

    var buttons = document.querySelectorAll(".lang-switch button");
    for (var k = 0; k < buttons.length; k++) {
      buttons[k].className = (buttons[k].getAttribute("data-lang") === lang) ? "active" : "";
      buttons[k].setAttribute("aria-pressed", buttons[k].getAttribute("data-lang") === lang ? "true" : "false");
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
