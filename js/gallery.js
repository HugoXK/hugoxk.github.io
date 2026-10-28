/* Ke Xu - Portfolio gallery.
   Reads the public manifest written by the admin console at
   https://hugoxk.online/data/portfolio.json and renders a filterable
   masonry grid with a keyboard-accessible lightbox. */

(function () {
  "use strict";

  var state = {
    items: [],
    filtered: [],
    year: "all",
    lightboxIndex: -1,
    updatedAt: null,
    loaded: false
  };

  function onReady(fn) {
    if (document.readyState !== "loading") { fn(); }
    else { document.addEventListener("DOMContentLoaded", fn); }
  }

  function cfg() { return window.HUGOXK_SITE || {}; }
  function currentLang() { return document.documentElement.getAttribute("lang") === "zh" ? "zh" : "en"; }

  function resolveAsset(path) {
    if (!path) { return ""; }
    if (/^https?:\/\//i.test(path)) { return path; }
    var base = (cfg().assetBase || "").replace(/\/$/, "");
    return base + (path.charAt(0) === "/" ? path : "/" + path);
  }

  function pick(item, key) {
    if (currentLang() === "zh") {
      var zh = item[key + "Zh"];
      if (zh && String(zh).trim()) { return String(zh).trim(); }
    }
    return item[key] ? String(item[key]).trim() : "";
  }

  function itemTitle(item) {
    return pick(item, "title") || pick(item, "location") || (currentLang() === "zh" ? "未命名照片" : "Untitled photo");
  }

  function itemMeta(item) {
    var bits = [];
    var location = pick(item, "location");
    if (location) { bits.push(location); }
    if (item.date) { bits.push(String(item.date).replace("-", ".")); }
    return bits.join(" \u00b7 ");
  }

  function normalize(items) {
    return (items || []).filter(function (item) {
      return item && item.id && (item.thumb || item.file);
    });
  }

  function fetchRemote() {
    var url = cfg().portfolioUrl;
    if (!url || typeof fetch !== "function") { return Promise.reject(new Error("no endpoint")); }
    var controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    var timer = controller ? setTimeout(function () { controller.abort(); }, 7000) : null;
    return fetch(url, {
      cache: "no-store",
      credentials: "omit",
      headers: { Accept: "application/json" },
      signal: controller ? controller.signal : undefined
    }).then(function (res) {
      if (!res.ok) { throw new Error("HTTP " + res.status); }
      return res.json();
    }).then(function (data) {
      var list = normalize(data && (data.items || (Array.isArray(data) ? data : null)));
      if (!list.length) { throw new Error("empty portfolio"); }
      return { items: list, updatedAt: data.updatedAt || null };
    }).finally(function () { if (timer) { clearTimeout(timer); } });
  }

  function availableYears() {
    var years = {};
    state.items.forEach(function (item) {
      var y = String(item.date || "").slice(0, 4);
      if (/^\d{4}$/.test(y)) { years[y] = true; }
    });
    return Object.keys(years).sort(function (a, b) { return b.localeCompare(a); });
  }

  function applyFilter() {
    state.filtered = state.items.filter(function (item) {
      return state.year === "all" || String(item.date || "").slice(0, 4) === state.year;
    });
  }

  function renderFilters() {
    var box = document.getElementById("gallery-filters");
    if (!box) { return; }
    var years = availableYears();
    if (years.length < 2) { box.innerHTML = ""; box.hidden = true; return; }
    box.hidden = false;
    var all = [{ key: "all", label: currentLang() === "zh" ? "全部" : "All" }].concat(years.map(function (y) {
      return { key: y, label: y };
    }));
    box.innerHTML = "";
    all.forEach(function (entry) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "filter-chip" + (state.year === entry.key ? " active" : "");
      btn.setAttribute("data-gallery-year", entry.key);
      btn.setAttribute("aria-pressed", state.year === entry.key ? "true" : "false");
      btn.textContent = entry.label;
      box.appendChild(btn);
    });
  }

  function renderGrid() {
    var grid = document.getElementById("gallery-grid");
    if (!grid) { return; }
    grid.innerHTML = "";
    if (!state.filtered.length) {
      var empty = document.createElement("p");
      empty.className = "gallery-empty";
      empty.textContent = currentLang() === "zh" ? "照片正在整理中，稍后再来看看。" : "Photos are being curated. Check back soon.";
      grid.appendChild(empty);
      return;
    }
    state.filtered.forEach(function (item, index) {
      var fig = document.createElement("figure");
      fig.className = "gallery-item";
      fig.setAttribute("data-index", String(index));

      var button = document.createElement("button");
      button.type = "button";
      button.className = "gallery-open";
      button.setAttribute("aria-label", (currentLang() === "zh" ? "查看大图：" : "Open photo: ") + itemTitle(item));

      var img = document.createElement("img");
      img.src = resolveAsset(item.thumb || item.file);
      img.alt = itemTitle(item);
      img.loading = "lazy";
      img.decoding = "async";
      if (item.width) { img.width = item.width; }
      if (item.height) { img.height = item.height; }
      button.appendChild(img);

      var zoom = document.createElement("span");
      zoom.className = "gallery-zoom";
      zoom.setAttribute("aria-hidden", "true");
      zoom.innerHTML = "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round'><circle cx='11' cy='11' r='7'/><path d='M21 21l-4.3-4.3M11 8v6M8 11h6'/></svg>";
      button.appendChild(zoom);
      fig.appendChild(button);

      var cap = document.createElement("figcaption");
      var title = document.createElement("b");
      title.textContent = itemTitle(item);
      cap.appendChild(title);
      var meta = itemMeta(item);
      if (meta) {
        var span = document.createElement("span");
        span.textContent = meta;
        cap.appendChild(span);
      }
      fig.appendChild(cap);
      grid.appendChild(fig);
    });
  }

  function renderAll() {
    applyFilter();
    renderFilters();
    renderGrid();
    var count = document.getElementById("gallery-count");
    if (count) { count.textContent = state.items.length; }
    var updated = document.getElementById("gallery-updated");
    if (updated) {
      if (state.updatedAt) {
        var d = new Date(state.updatedAt);
        updated.textContent = isNaN(d.getTime()) ? state.updatedAt : d.toLocaleDateString(currentLang() === "zh" ? "zh-CN" : "en-GB", { year: "numeric", month: "short", day: "numeric" });
      } else {
        updated.textContent = currentLang() === "zh" ? "本地" : "local";
      }
    }
  }

  function openLightbox(index) {
    if (index < 0 || index >= state.filtered.length) { return; }
    state.lightboxIndex = index;
    var item = state.filtered[index];

    var overlay = document.createElement("div");
    overlay.className = "lightbox";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", itemTitle(item));

    var stage = document.createElement("div");
    stage.className = "lightbox-stage";

    var img = document.createElement("img");
    img.src = resolveAsset(item.file || item.thumb);
    img.alt = itemTitle(item);
    stage.appendChild(img);

    var close = document.createElement("button");
    close.type = "button";
    close.className = "lightbox-close";
    close.setAttribute("aria-label", currentLang() === "zh" ? "关闭" : "Close");
    close.innerHTML = "\u00d7";

    var prev = document.createElement("button");
    prev.type = "button";
    prev.className = "lightbox-nav prev";
    prev.setAttribute("aria-label", currentLang() === "zh" ? "上一张" : "Previous photo");
    prev.innerHTML = "\u2039";

    var next = document.createElement("button");
    next.type = "button";
    next.className = "lightbox-nav next";
    next.setAttribute("aria-label", currentLang() === "zh" ? "下一张" : "Next photo");
    next.innerHTML = "\u203a";

    var footer = document.createElement("div");
    footer.className = "lightbox-caption";
    var title = document.createElement("b");
    title.textContent = itemTitle(item);
    footer.appendChild(title);
    var meta = itemMeta(item);
    if (meta) {
      var span = document.createElement("span");
      span.textContent = meta;
      footer.appendChild(span);
    }
    var counter = document.createElement("small");
    counter.textContent = (index + 1) + " / " + state.filtered.length;
    footer.appendChild(counter);

    overlay.appendChild(close);
    overlay.appendChild(prev);
    overlay.appendChild(stage);
    overlay.appendChild(next);
    overlay.appendChild(footer);
    document.body.appendChild(overlay);
    document.body.classList.add("lightbox-open");
    close.focus();

    function closeBox() {
      document.removeEventListener("keydown", onKey);
      if (overlay.parentNode) { overlay.parentNode.removeChild(overlay); }
      document.body.classList.remove("lightbox-open");
      var trigger = document.querySelector('.gallery-item[data-index="' + index + '"] .gallery-open');
      if (trigger) { trigger.focus(); }
      state.lightboxIndex = -1;
    }
    function show(delta) {
      var target = state.lightboxIndex + delta;
      if (target < 0) { target = state.filtered.length - 1; }
      if (target >= state.filtered.length) { target = 0; }
      closeBox();
      openLightbox(target);
    }
    function onKey(ev) {
      if (ev.key === "Escape") { closeBox(); }
      else if (ev.key === "ArrowLeft") { show(-1); }
      else if (ev.key === "ArrowRight") { show(1); }
    }
    close.addEventListener("click", closeBox);
    prev.addEventListener("click", function () { show(-1); });
    next.addEventListener("click", function () { show(1); });
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay || e.target === stage) { closeBox(); }
    });
    document.addEventListener("keydown", onKey);

    if (state.filtered.length < 2) {
      prev.hidden = true;
      next.hidden = true;
    }
  }

  function bindUi() {
    var grid = document.getElementById("gallery-grid");
    if (grid) {
      grid.addEventListener("click", function (e) {
        var btn = e.target.closest(".gallery-open");
        if (!btn) { return; }
        var fig = btn.closest(".gallery-item");
        if (!fig) { return; }
        openLightbox(parseInt(fig.getAttribute("data-index"), 10));
      });
    }
    var filters = document.getElementById("gallery-filters");
    if (filters) {
      filters.addEventListener("click", function (e) {
        var btn = e.target.closest("[data-gallery-year]");
        if (!btn) { return; }
        state.year = btn.getAttribute("data-gallery-year");
        renderAll();
      });
    }
    document.addEventListener("langchange", renderAll);
  }

  onReady(function () {
    bindUi();
    var grid = document.getElementById("gallery-grid");
    fetchRemote().then(function (data) {
      state.items = data.items;
      state.updatedAt = data.updatedAt;
      state.loaded = true;
      renderAll();
    }).catch(function () {
      state.loaded = true;
      renderAll();
      if (grid && !state.items.length) {
        grid.innerHTML = "";
        var p = document.createElement("p");
        p.className = "gallery-empty";
        p.textContent = currentLang() === "zh" ? "照片正在整理中，稍后再来看看。" : "Photos are being curated. Check back soon.";
        grid.appendChild(p);
      }
    });
  });
})();
