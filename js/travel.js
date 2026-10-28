/* Ke Xu - Travel map and checklist.
   Data source: https://hugoxk.online/data/travel.json (written by the
   authenticated admin console). Falls back to js/countries-list.js when
   the network or the remote file is unavailable. */

(function () {
  "use strict";

  var CHECK_RE = /^\[([ xXpP])([^\]]*)\]\s*(.*)$/;
  var MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var STATUS = { visited: "visited", planned: "planned", unvisited: "unvisited" };
  var FALLBACK = {
    en: "Map could not be loaded. The checklist below still works.",
    zh: "地图加载失败，下方清单仍可使用。"
  };

  var state = {
    entries: [],
    byName: {},
    layersByName: {},
    markers: {},
    filter: "all",
    query: "",
    updatedAt: null,
    source: "local",
    map: null,
    geoLayer: null,
    markerLayer: null,
    pendingMap: false,
    total: 0
  };

  function onReady(fn) {
    if (document.readyState !== "loading") { fn(); }
    else { document.addEventListener("DOMContentLoaded", fn); }
  }

  function cfg() { return window.HUGOXK_SITE || {}; }

  function currentLang() {
    return (document.documentElement.getAttribute("lang") === "zh") ? "zh" : "en";
  }

  function countryName(name) {
    if (currentLang() === "zh" && typeof COUNTRY_ZH !== "undefined") {
      return COUNTRY_ZH[name] || name;
    }
    return name;
  }

  function monthLabel(date, kind) {
    var m = /^(\d{4})-(\d{1,2})$/.exec(date || "");
    if (!m) { return date || ""; }
    var year = m[1];
    var month = parseInt(m[2], 10);
    if (!month || month > 12) { return date; }
    if (currentLang() === "zh") {
      return (kind === "planned" ? "计划 " : "首次到访 ") + year + "年" + month + "月";
    }
    return (kind === "planned" ? "planned " : "first visited ") + MONTHS_EN[month - 1] + " " + year;
  }

  function shortMonth(date) {
    var m = /^(\d{4})-(\d{1,2})$/.exec(date || "");
    if (!m) { return date || ""; }
    return currentLang() === "zh" ? (m[1] + "." + m[2]) : (MONTHS_EN[parseInt(m[2], 10) - 1] + " " + m[1]);
  }

  function parseChecklist() {
    var out = [];
    if (typeof COUNTRY_CHECKLIST === "undefined") { return out; }
    COUNTRY_CHECKLIST.forEach(function (line) {
      var m = CHECK_RE.exec(line);
      if (!m) { return; }
      var marker = m[1];
      var d = /(\d{4})-(\d{1,2})/.exec(m[2]);
      var date = d ? (d[1] + "-" + (d[2].length < 2 ? "0" + d[2] : d[2])) : "";
      var status = /[xX]/.test(marker) ? STATUS.visited : (/[pP]/.test(marker) ? STATUS.planned : STATUS.unvisited);
      out.push({ name: m[3].trim(), status: status, date: status === STATUS.unvisited ? "" : date });
    });
    return out;
  }

  function normalize(entries) {
    var out = [];
    var seen = {};
    (entries || []).forEach(function (item) {
      var name = item && (item.name || item.country);
      if (!name || typeof name !== "string") { return; }
      name = name.trim();
      if (!name || seen[name]) { return; }
      var status = item.status;
      if (status !== STATUS.visited && status !== STATUS.planned) { status = STATUS.unvisited; }
      var date = /^\d{4}-(0[1-9]|1[0-2])$/.test(item.date || "") ? item.date : "";
      out.push({ name: name, status: status, date: status === STATUS.unvisited ? "" : date });
      seen[name] = true;
    });
    return out;
  }

  function fetchRemote() {
    var url = cfg().travelUrl;
    if (!url || typeof fetch !== "function") { return Promise.reject(new Error("no endpoint")); }
    var controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    var timer = controller ? setTimeout(function () { controller.abort(); }, 6500) : null;
    return fetch(url, {
      cache: "no-store",
      credentials: "omit",
      headers: { Accept: "application/json" },
      signal: controller ? controller.signal : undefined
    }).then(function (res) {
      if (!res.ok) { throw new Error("HTTP " + res.status); }
      return res.json();
    }).then(function (data) {
      var list = data && (data.countries || (Array.isArray(data) ? data : null));
      var entries = normalize(list);
      if (!entries.length) { throw new Error("empty travel data"); }
      return { entries: entries, updatedAt: data.updatedAt || data.updated || null };
    }).finally(function () {
      if (timer) { clearTimeout(timer); }
    });
  }

  function statusOf(name) {
    var item = state.byName[name];
    return item ? item.status : STATUS.unvisited;
  }

  function tooltipFor(name) {
    var item = state.byName[name];
    var text = countryName(name);
    if (!item) { return text; }
    if (item.status === STATUS.visited) {
      text += " \u2713";
      if (item.date) { text += " \u00b7 " + monthLabel(item.date, STATUS.visited); }
    } else if (item.status === STATUS.planned) {
      text += " \u00b7 " + (item.date ? monthLabel(item.date, STATUS.planned) : (currentLang() === "zh" ? "计划中" : "planned"));
    }
    return text;
  }

  function themeIsDark() {
    return document.documentElement.getAttribute("data-theme") === "dark";
  }

  function mapColors() {
    if (themeIsDark()) {
      return {
        ocean: "#0d0c0b",
        visited: { color: "#ff9b91", weight: 1.1, fillColor: "#ff6b5e", fillOpacity: 0.78 },
        planned: { color: "#ff8b80", weight: 1, dashArray: "5 4", fillColor: "#5b2d2a", fillOpacity: 0.75 },
        unvisited: { color: "#3a352e", weight: 0.7, fillColor: "#1a1815", fillOpacity: 0.95 },
        markerVisited: { radius: 5, color: "#ff9b91", weight: 1, fillColor: "#ff6b5e", fillOpacity: 0.9 },
        markerPlanned: { radius: 4, color: "#ff8b80", weight: 1, dashArray: "3 2", fillColor: "#5b2d2a", fillOpacity: 0.9 }
      };
    }
    return {
      ocean: "#eef1f4",
      visited: { color: "#7d1712", weight: 1.2, fillColor: "#b3261e", fillOpacity: 0.84 },
      planned: { color: "#b3261e", weight: 1, dashArray: "5 4", fillColor: "#e8b4ae", fillOpacity: 0.72 },
      unvisited: { color: "#cfc6b7", weight: 0.7, fillColor: "#e7e2d8", fillOpacity: 0.92 },
      markerVisited: { radius: 5, color: "#7d1712", weight: 1, fillColor: "#b3261e", fillOpacity: 0.9 },
      markerPlanned: { radius: 4, color: "#b3261e", weight: 1, dashArray: "3 2", fillColor: "#e8b4ae", fillOpacity: 0.9 }
    };
  }

  function styleFor(feature) {
    var c = mapColors();
    var status = statusOf(feature.properties.name);
    if (status === STATUS.visited) { return c.visited; }
    if (status === STATUS.planned) { return c.planned; }
    return c.unvisited;
  }

  function setText(id, value) {
    var el = document.getElementById(id);
    if (el) { el.textContent = value; }
  }

  function applyData(entries, updatedAt, source) {
    state.entries = entries.slice();
    state.byName = {};
    state.total = entries.length;
    entries.forEach(function (item) { state.byName[item.name] = item; });
    state.updatedAt = updatedAt || null;
    state.source = source;
    renderAll();
  }

  function visibleEntries() {
    var q = state.query.trim().toLowerCase();
    return state.entries.filter(function (item) {
      if (state.filter !== "all" && item.status !== state.filter) { return false; }
      if (!q) { return true; }
      return (item.name.toLowerCase().indexOf(q) !== -1) ||
             (countryName(item.name).toLowerCase().indexOf(q) !== -1);
    });
  }

  function sortGroup(list, status) {
    return list.sort(function (a, b) {
      if (status === STATUS.visited) {
        if (a.date && b.date && a.date !== b.date) { return b.date.localeCompare(a.date); }
        if (a.date && !b.date) { return -1; }
        if (!a.date && b.date) { return 1; }
      } else if (status === STATUS.planned) {
        if (a.date && b.date && a.date !== b.date) { return a.date.localeCompare(b.date); }
        if (a.date && !b.date) { return -1; }
        if (!a.date && b.date) { return 1; }
      }
      return countryName(a.name).localeCompare(countryName(b.name));
    });
  }

  function listRow(item) {
    var li = document.createElement("li");
    li.className = item.status;
    li.setAttribute("data-name", item.name);
    li.setAttribute("title", currentLang() === "zh" ? "在地图上查看" : "Show on map");
    var chk = document.createElement("span");
    chk.className = "chk";
    chk.setAttribute("aria-hidden", "true");
    var label = document.createElement("span");
    label.className = "cname";
    label.textContent = countryName(item.name);
    li.appendChild(chk);
    li.appendChild(label);
    if (item.date) {
      var d = document.createElement("span");
      d.className = "vdate";
      d.textContent = shortMonth(item.date);
      li.appendChild(d);
    }
    return li;
  }

  function renderList() {
    var listEl = document.getElementById("travel-list");
    if (!listEl) { return; }
    var list = visibleEntries();
    listEl.innerHTML = "";
    if (!list.length) {
      var empty = document.createElement("li");
      empty.className = "travel-empty";
      empty.textContent = currentLang() === "zh" ? "没有匹配的国家。" : "No matching countries.";
      listEl.appendChild(empty);
      return;
    }
    var groups = [
      { status: STATUS.visited, label: currentLang() === "zh" ? "已到访" : "Visited" },
      { status: STATUS.planned, label: currentLang() === "zh" ? "计划中" : "Planned" },
      { status: STATUS.unvisited, label: currentLang() === "zh" ? "未到访" : "Not yet" }
    ];
    groups.forEach(function (group) {
      var items = list.filter(function (item) { return item.status === group.status; });
      if (!items.length) { return; }
      sortGroup(items, group.status);
      var head = document.createElement("li");
      head.className = "travel-group";
      head.textContent = group.label + " \u00b7 " + items.length;
      listEl.appendChild(head);
      items.forEach(function (item) { listEl.appendChild(listRow(item)); });
    });
  }

  function renderFilters() {
    var counts = { all: state.entries.length, visited: 0, planned: 0, unvisited: 0 };
    state.entries.forEach(function (item) { counts[item.status] = (counts[item.status] || 0) + 1; });
    var nodes = document.querySelectorAll("[data-travel-filter]");
    Array.prototype.forEach.call(nodes, function (btn) {
      var key = btn.getAttribute("data-travel-filter");
      btn.classList.toggle("active", key === state.filter);
      btn.setAttribute("aria-pressed", key === state.filter ? "true" : "false");
      var count = btn.querySelector(".count");
      if (count) { count.textContent = counts[key] || 0; }
    });
  }

  function renderStats() {
    var visited = 0;
    var planned = 0;
    state.entries.forEach(function (item) {
      if (item.status === STATUS.visited) { visited++; }
      else if (item.status === STATUS.planned) { planned++; }
    });
    var total = state.total || 1;
    var pct = (visited / total * 100);
    setText("visited-count", visited);
    setText("planned-count", planned);
    setText("visited-summary", visited);
    setText("total-count-summary", state.total);
    setText("total-count", state.total);
    setText("travel-progress-pct", pct.toFixed(1) + "%");
    var fill = document.getElementById("travel-progress-fill");
    if (fill) { fill.style.width = pct.toFixed(1) + "%"; }
    var label = document.getElementById("travel-progress-label");
    if (label) { label.textContent = visited + " / " + state.total + " \u00b7 " + pct.toFixed(1) + "%"; }
    var updated = document.getElementById("travel-updated");
    if (updated) {
      if (state.updatedAt) {
        var d = new Date(state.updatedAt);
        updated.textContent = isNaN(d.getTime()) ? state.updatedAt : d.toLocaleDateString(currentLang() === "zh" ? "zh-CN" : "en-GB", { year: "numeric", month: "short", day: "numeric" });
      } else {
        updated.textContent = currentLang() === "zh" ? "本地" : "local";
      }
    }
    var source = document.getElementById("travel-source");
    if (source) {
      source.textContent = state.source === "remote"
        ? (currentLang() === "zh" ? "实时数据" : "live data")
        : (currentLang() === "zh" ? "本地备用数据" : "local fallback");
      source.setAttribute("data-source", state.source);
    }
  }

  function renderAll() {
    renderStats();
    renderFilters();
    renderList();
    updateMap();
  }

  function loadScript(src, cb) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = cb;
    s.onerror = cb;
    document.head.appendChild(s);
  }

  function initMap() {
    if (typeof L === "undefined" || typeof COUNTRIES_GEOJSON === "undefined") {
      var box = document.getElementById("travel-map");
      if (box) {
        box.innerHTML = '<p class="travel-map-fallback">' + (FALLBACK[currentLang()] || FALLBACK.en) + "</p>";
      }
      return;
    }
    var el = document.getElementById("travel-map");
    if (!el || state.map) { return; }

    var map = L.map(el, {
      scrollWheelZoom: false,
      minZoom: 1,
      maxZoom: 8,
      maxBounds: [[-85, -180], [85, 180]],
      maxBoundsViscosity: 0.8,
      worldCopyJump: true
    }).setView([20, 10], 2);
    map.attributionControl.addAttribution("Country shapes: Natural Earth");
    map.getContainer().style.background = mapColors().ocean;

    var HomeControl = L.Control.extend({
      options: { position: "topleft" },
      onAdd: function (m) {
        var container = L.DomUtil.create("div", "leaflet-bar leaflet-control leaflet-control-home");
        var link = L.DomUtil.create("a", "", container);
        link.href = "#";
        link.title = currentLang() === "zh" ? "重置视图" : "Reset view";
        link.setAttribute("aria-label", link.title);
        link.innerHTML = "\u2302";
        L.DomEvent.on(link, "click", function (e) {
          L.DomEvent.preventDefault(e);
          m.setView([20, 10], 2);
        });
        return container;
      }
    });
    map.addControl(new HomeControl());

    state.layersByName = {};
    state.geoLayer = L.geoJSON(COUNTRIES_GEOJSON, {
      style: styleFor,
      onEachFeature: function (feature, flayer) {
        var name = feature.properties.name;
        state.layersByName[name] = flayer;
        flayer.bindTooltip(function () { return tooltipFor(name); }, { sticky: true });
        flayer.on("click", function () {
          try { map.fitBounds(flayer.getBounds(), { padding: [26, 26], maxZoom: 4 }); } catch (err) {}
        });
      }
    }).addTo(map);

    state.markerLayer = L.layerGroup().addTo(map);
    renderMarkers();
    state.map = map;
    state.pendingMap = false;
  }

  function renderMarkers() {
    if (!state.markerLayer || typeof COUNTRY_COORDS === "undefined") { return; }
    var c = mapColors();
    state.markerLayer.clearLayers();
    state.markers = {};
    state.entries.forEach(function (item) {
      if (item.status === STATUS.unvisited) { return; }
      if (state.layersByName[item.name]) { return; }
      var coord = COUNTRY_COORDS[item.name];
      if (!coord) { return; }
      var opts = item.status === STATUS.visited ? c.markerVisited : c.markerPlanned;
      var marker = L.circleMarker([coord[0], coord[1]], opts).addTo(state.markerLayer);
      marker.bindTooltip(function () { return tooltipFor(item.name); });
      marker.on("click", function () { state.map.setView([coord[0], coord[1]], 4); });
      state.markers[item.name] = marker;
    });
  }

  function updateMap() {
    if (!state.map) { return; }
    state.map.getContainer().style.background = mapColors().ocean;
    state.geoLayer.setStyle(styleFor);
    state.geoLayer.eachLayer(function (layer) {
      var name = layer.feature && layer.feature.properties && layer.feature.properties.name;
      if (name) { layer.setTooltipContent(tooltipFor(name)); }
    });
    renderMarkers();
  }

  function focusCountry(name) {
    if (!state.map) { return; }
    var layer = state.layersByName[name];
    if (layer) {
      try {
        state.map.fitBounds(layer.getBounds(), { padding: [26, 26], maxZoom: 4 });
        layer.openTooltip();
        return;
      } catch (err) {}
    }
    var marker = state.markers[name];
    if (marker) {
      state.map.setView(marker.getLatLng(), 4);
      marker.openTooltip();
    }
  }

  function startMap() {
    if (state.map || state.pendingMap) { return; }
    state.pendingMap = true;
    loadScript("js/leaflet.min.js", function () {
      loadScript("js/countries-geo.js", initMap);
    });
  }

  function bindUi() {
    var search = document.getElementById("travel-search");
    if (search) {
      search.addEventListener("input", function () {
        state.query = search.value || "";
        renderList();
      });
    }
    var filters = document.querySelector("[data-travel-filters]");
    if (filters) {
      filters.addEventListener("click", function (e) {
        var btn = e.target.closest("[data-travel-filter]");
        if (!btn) { return; }
        state.filter = btn.getAttribute("data-travel-filter");
        renderFilters();
        renderList();
      });
    }
    var list = document.getElementById("travel-list");
    if (list) {
      list.addEventListener("click", function (e) {
        var row = e.target.closest("li[data-name]");
        if (row) { focusCountry(row.getAttribute("data-name")); }
      });
    }
    var refresh = document.getElementById("travel-refresh");
    if (refresh) {
      refresh.addEventListener("click", function () {
        refresh.disabled = true;
        loadRemote(true).finally(function () { refresh.disabled = false; });
      });
    }
    document.addEventListener("langchange", function () {
      state.query = search ? search.value : state.query;
      renderAll();
    });
    document.addEventListener("themechange", function () {
      updateMap();
    });
  }

  var loadedRemote = false;
  function loadRemote(force) {
    if (loadedRemote && !force) { return Promise.resolve(); }
    return fetchRemote().then(function (data) {
      loadedRemote = true;
      applyData(data.entries, data.updatedAt, "remote");
    }).catch(function () {
      loadedRemote = true;
      applyData(parseChecklist(), null, "local");
    });
  }

  onReady(function () {
    bindUi();
    applyData(parseChecklist(), null, "local");

    var target = document.getElementById("travel");
    function start() {
      loadRemote(false);
      startMap();
    }
    if (!target) { start(); return; }
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        for (var i = 0; i < entries.length; i++) {
          if (entries[i].isIntersecting) { io.disconnect(); start(); return; }
        }
      }, { rootMargin: "500px 0px" });
      io.observe(target);
    } else {
      start();
    }
  });
})();
