/* Ke Xu - Travel map.
   Data source: data/travel.json (synced from the authenticated console).
   Falls back to js/countries-list.js when the live file is unavailable. */

(function () {
  "use strict";

  var CHECK_RE = /^\[([ xXpP])([^\]]*)\]\s*(.*)$/;
  var MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var STATUS = { visited: "visited", planned: "planned", unvisited: "unvisited" };
  var FALLBACK = {
    en: "Map could not be loaded.",
    zh: "地图加载失败。"
  };

  var state = {
    entries: [],
    byName: {},
    layersByName: {},
    markers: {},
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
  function currentLang() { return document.documentElement.getAttribute("lang") === "zh" ? "zh" : "en"; }

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
    var timer = controller ? setTimeout(function () { controller.abort(); }, 8000) : null;
    var bust = url + (url.indexOf("?") === -1 ? "?" : "&") + "t=" + Date.now();
    return fetch(bust, {
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

  function themeIsDark() { return document.documentElement.getAttribute("data-theme") === "dark"; }

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
    renderStats();
    updateMap();
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
    setText("travel-progress-pct", pct.toFixed(1) + "%");
    setText("total-count", state.total);
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

  function startMap() {
    if (state.map || state.pendingMap) { return; }
    state.pendingMap = true;
    loadScript("js/leaflet.min.js", function () {
      loadScript("js/countries-geo.js", initMap);
    });
  }

  function bindUi() {
    var refresh = document.getElementById("travel-refresh");
    if (refresh) {
      refresh.addEventListener("click", function () {
        refresh.disabled = true;
        loadRemote(true).finally(function () { refresh.disabled = false; });
      });
    }
    document.addEventListener("langchange", function () {
      renderStats();
      updateMap();
    });
    document.addEventListener("themechange", updateMap);
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
