/* Ke Xu - Travel map (choropleth, bilingual, lazy-loaded)
   Data files (lazy-loaded only when the travel section nears the viewport):
     - js/leaflet.min.js  (self-hosted Leaflet 1.9.4)
     - js/countries-geo.js
   Checklist (loaded eagerly, always available):
     - js/countries-list.js
         "[ ] Name"          not visited yet
         "[X] Name"          visited
         "[X 2019-09] Name"  visited, first visited 2019-09 (YYYY-MM)
         "[P] Name"          planned
         "[P 2026-10] Name"  planned for 2026-10 (YYYY-MM) */

(function () {
  "use strict";

  var ENTRY = /^\[([ xXpP])([^\]]*)\]\s*(.*)$/;
  var MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var FALLBACK = {
    en: "Map could not be loaded. The checklist below still works.",
    zh: "地图加载失败，下方清单仍可使用。"
  };

  function onReady(fn) {
    if (document.readyState !== "loading") { fn(); }
    else { document.addEventListener("DOMContentLoaded", fn); }
  }

  function currentLang() {
    return (document.documentElement.getAttribute("lang") === "zh") ? "zh" : "en";
  }

  function countryName(enName) {
    if (currentLang() === "zh" && typeof COUNTRY_ZH !== "undefined") {
      return COUNTRY_ZH[enName] || enName;
    }
    return enName;
  }

  function dateLabel(date, kind) {
    var m = /^(\d{4})-(\d{1,2})$/.exec(date);
    if (!m) { return date; }
    var year = m[1];
    var month = parseInt(m[2], 10);
    if (currentLang() === "zh") {
      return (kind === "planned")
        ? "计划 " + year + "年" + month + "月到访"
        : "首次到访 " + year + "年" + month + "月";
    }
    return (kind === "planned")
      ? "planned for " + MONTHS_EN[month - 1] + " " + year
      : "first visited " + MONTHS_EN[month - 1] + " " + year;
  }

  var state = {
    visitedSet: {},
    plannedSet: {},
    visited: [],
    firstVisited: {},
    plannedFor: {},
    total: 0
  };

  function parseChecklist() {
    if (typeof COUNTRY_CHECKLIST === "undefined") { return; }
    COUNTRY_CHECKLIST.forEach(function (line) {
      var m = ENTRY.exec(line);
      if (!m) { return; }
      state.total++;
      var name = m[3].trim();
      var marker = m[1];
      var d = /(\d{4})-(\d{1,2})/.exec(m[2]);
      var date = d ? (d[1] + "-" + (d[2].length < 2 ? "0" + d[2] : d[2])) : null;
      if (/[xX]/.test(marker)) {
        state.visitedSet[name] = true;
        state.visited.push(name);
        if (date) { state.firstVisited[name] = date; }
      } else if (/[pP]/.test(marker)) {
        state.plannedSet[name] = true;
        if (date) { state.plannedFor[name] = date; }
      }
    });
  }

  function tooltipFor(name) {
    var s = countryName(name);
    if (state.visitedSet[name]) {
      s += " \u2713";
      if (state.firstVisited[name]) {
        s += " · " + dateLabel(state.firstVisited[name], "visited");
      }
    } else if (state.plannedSet[name]) {
      if (state.plannedFor[name]) {
        s += " · " + dateLabel(state.plannedFor[name], "planned");
      } else {
        s += " · " + ((currentLang() === "zh") ? "计划中" : "planned");
      }
    }
    return s;
  }

  function renderList() {
    var listEl = document.getElementById("travel-list");
    if (!listEl || typeof COUNTRY_CHECKLIST === "undefined") { return; }
    var zh = (currentLang() === "zh");
    listEl.innerHTML = "";
    COUNTRY_CHECKLIST.forEach(function (line) {
      var m = ENTRY.exec(line);
      if (!m) { return; }
      var name = m[3].trim();
      var cls = null;
      var stateWord = null;
      if (state.visitedSet[name]) {
        cls = "visited";
        stateWord = zh ? "已到访" : "visited";
      } else if (state.plannedSet[name]) {
        cls = "planned";
        stateWord = zh ? "计划中" : "planned";
      }
      var li = document.createElement("li");
      if (cls) {
        li.className = cls;
        li.setAttribute("aria-label", countryName(name) + " \u2014 " + stateWord);
      }
      var chk = document.createElement("span");
      chk.className = "chk";
      chk.setAttribute("aria-hidden", "true");
      var label = document.createElement("span");
      label.className = "cname";
      label.textContent = countryName(name);
      li.appendChild(chk);
      li.appendChild(label);
      var date = state.firstVisited[name] || state.plannedFor[name];
      if (cls && date) {
        var d = document.createElement("span");
        d.className = "vdate";
        d.textContent = date;
        li.appendChild(d);
      }
      listEl.appendChild(li);
    });
  }

  function setText(id, v) {
    var el = document.getElementById(id);
    if (el) { el.textContent = v; }
  }

  function updateCounts() {
    setText("visited-count", state.visited.length);
    setText("total-count", state.total);
    setText("visited-summary", state.visited.length);
  }

  function updateProgress() {
    var fill = document.getElementById("travel-progress-fill");
    var label = document.getElementById("travel-progress-label");
    if (!fill || !label) { return; }
    var pct = state.total ? (state.visited.length / state.total * 100) : 0;
    fill.style.width = pct.toFixed(1) + "%";
    label.textContent = state.visited.length + " / " + state.total + " \u00b7 " + pct.toFixed(1) + "%";
  }

  function loadScript(src, cb) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = cb;
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

    var map = L.map("travel-map", {
      scrollWheelZoom: false,
      minZoom: 1,
      maxZoom: 8,
      maxBounds: [[-85, -180], [85, 180]],
      maxBoundsViscosity: 0.8
    }).setView([20, 0], 2);
    map.attributionControl.addAttribution("Country shapes: Natural Earth");

    var HomeControl = L.Control.extend({
      options: { position: "topleft" },
      onAdd: function (m) {
        var container = L.DomUtil.create("div", "leaflet-bar leaflet-control leaflet-control-home");
        var link = L.DomUtil.create("a", "", container);
        link.href = "#";
        link.title = "Reset view / 重置视图";
        link.innerHTML = "\u2302";
        L.DomEvent.on(link, "click", function (e) {
          L.DomEvent.preventDefault(e);
          m.setView([20, 0], 2);
        });
        return container;
      }
    });
    map.addControl(new HomeControl());

    var hasPolygon = {};
    L.geoJSON(COUNTRIES_GEOJSON, {
      style: function (feature) {
        var name = feature.properties.name;
        if (state.visitedSet[name]) {
          return { color: "#7a0000", weight: 1.2, fillColor: "#a00", fillOpacity: 0.7 };
        }
        if (state.plannedSet[name]) {
          return { color: "#c98a8a", weight: 1, dashArray: "4 3", fillColor: "#eec3c3", fillOpacity: 0.55 };
        }
        return { color: "#b7c0cc", weight: 0.7, fillColor: "#e8ebef", fillOpacity: 0.9 };
      },
      onEachFeature: function (feature, flayer) {
        var name = feature.properties.name;
        if (state.visitedSet[name] || state.plannedSet[name]) {
          hasPolygon[name] = true;
        }
        flayer.bindTooltip(function () { return tooltipFor(name); }, { sticky: true });
        flayer.on("click", function () {
          try { map.fitBounds(flayer.getBounds(), { padding: [25, 25] }); } catch (err) {}
        });
      }
    }).addTo(map);

    if (typeof COUNTRY_COORDS !== "undefined") {
      var dotted = [];
      state.visited.forEach(function (name) {
        if (!hasPolygon[name]) { dotted.push(name); }
      });
      Object.keys(state.plannedSet).forEach(function (name) {
        if (!hasPolygon[name]) { dotted.push(name); }
      });
      dotted.forEach(function (name) {
        var c = COUNTRY_COORDS[name];
        if (!c) { return; }
        var planned = !state.visitedSet[name];
        var opts = planned
          ? { radius: 4, color: "#c98a8a", weight: 1, dashArray: "3 2", fillColor: "#eec3c3", fillOpacity: 0.8 }
          : { radius: 5, color: "#7a0000", weight: 1, fillColor: "#a00", fillOpacity: 0.85 };
        L.circleMarker([c[0], c[1]], opts).addTo(map)
          .bindTooltip(function () { return tooltipFor(name); });
      });
    }
  }

  onReady(function () {
    parseChecklist();
    renderList();
    updateCounts();
    updateProgress();

    document.addEventListener("langchange", function () {
      renderList();
      updateCounts();
      updateProgress();
    });

    var target = document.getElementById("travel");
    function start() {
      loadScript("js/leaflet.min.js", function () {
        loadScript("js/countries-geo.js", initMap);
      });
    }
    if (!target) { start(); return; }
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        for (var i = 0; i < entries.length; i++) {
          if (entries[i].isIntersecting) { io.disconnect(); start(); return; }
        }
      }, { rootMargin: "400px 0px" });
      io.observe(target);
    } else {
      start();
    }
  });
})();
