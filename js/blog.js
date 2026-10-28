/* Ke Xu - Blog
   Two views in one page:
     - list view  (default): renders BLOG_POSTS from js/blog-posts.js
     - post view  (blog.html?slug=xxx): fetches blog/posts/<slug>.md,
       renders it with the lazily-loaded marked.js, and mounts
       giscus comments (if configured in BLOG_GISCUS). */

(function () {
  "use strict";

  var SLUG_RE = /^[a-z0-9-]+$/;

  function onReady(fn) {
    if (document.readyState !== "loading") { fn(); }
    else { document.addEventListener("DOMContentLoaded", fn); }
  }

  function currentLang() {
    return (document.documentElement.getAttribute("lang") === "zh") ? "zh" : "en";
  }

  function loadScript(src, cb) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = cb;
    document.head.appendChild(s);
  }

  function currentSlug() {
    var m = /[?&]slug=([a-zA-Z0-9-]+)/.exec(window.location.search);
    return (m && SLUG_RE.test(m[1])) ? m[1] : null;
  }

  /* ---------- list view ---------- */

  function renderList() {
    var list = document.getElementById("post-list");
    var empty = document.getElementById("blog-empty");
    if (!list) { return; }
    var posts = (typeof BLOG_POSTS !== "undefined") ? BLOG_POSTS.slice() : [];
    posts.sort(function (a, b) { return String(b.date || "").localeCompare(String(a.date || "")); });

    list.innerHTML = "";
    if (empty) { empty.style.display = posts.length ? "none" : "block"; }
    if (!posts.length) { return; }

    posts.forEach(function (p) {
      var a = document.createElement("a");
      a.className = "post-card";
      a.href = "blog.html?slug=" + encodeURIComponent(p.slug);

      var head = document.createElement("div");
      head.className = "post-card-head";
      var t = document.createElement("span");
      t.className = "post-card-title";
      t.textContent = p.title || p.slug;
      var d = document.createElement("span");
      d.className = "post-card-date";
      d.textContent = p.date || "";
      head.appendChild(t);
      head.appendChild(d);
      a.appendChild(head);

      if (p.summary) {
        var s = document.createElement("p");
        s.className = "post-card-summary";
        s.textContent = p.summary;
        a.appendChild(s);
      }

      if (p.tags && p.tags.length) {
        var tg = document.createElement("div");
        tg.className = "post-card-tags";
        p.tags.forEach(function (x) {
          var sp = document.createElement("span");
          sp.textContent = x;
          tg.appendChild(sp);
        });
        a.appendChild(tg);
      }

      list.appendChild(a);
    });
  }

  /* ---------- post view ---------- */

  function renderPost(slug) {
    var view = document.getElementById("view-post");
    var listView = document.getElementById("view-list");
    if (!view) { return; }
    view.style.display = "";
    if (listView) { listView.style.display = "none"; }

    var post = null;
    if (typeof BLOG_POSTS !== "undefined") {
      BLOG_POSTS.forEach(function (p) { if (p.slug === slug) { post = p; } });
    }

    document.title = ((post && post.title) ? post.title + " — " : "") + "Ke Xu";
    document.getElementById("post-title").textContent = post ? (post.title || slug) : slug;

    var meta = [];
    if (post && post.date) { meta.push(post.date); }
    if (post && post.tags && post.tags.length) { meta.push(post.tags.join(" \u00b7 ")); }
    document.getElementById("post-meta").textContent = meta.join(" \u00b7 ");

    var body = document.getElementById("post-body");
    body.innerHTML = "<p class='giscus-note'>Loading\u2026</p>";

    fetch("blog/posts/" + slug + ".md")
      .then(function (r) {
        if (!r.ok) { throw new Error("HTTP " + r.status); }
        return r.text();
      })
      .then(function (md) {
        loadScript("js/marked.min.js", function () {
          if (window.marked && typeof window.marked.parse === "function") {
            body.innerHTML = window.marked.parse(md);
          } else {
            body.innerHTML = "";
            var pre = document.createElement("pre");
            pre.textContent = md;
            body.appendChild(pre);
          }
        });
      })
      .catch(function () {
        body.innerHTML = "";
        var p = document.createElement("p");
        p.className = "giscus-note";
        p.textContent = (currentLang() === "zh")
          ? "文章加载失败。本地预览请先运行 python3 -m http.server 再访问 http://localhost:8000/blog.html。"
          : "Failed to load the post. For local preview, run python3 -m http.server and open http://localhost:8000/blog.html.";
        body.appendChild(p);
      });

    initComments(slug);
  }

  /* ---------- comments (giscus) ---------- */

  function initComments(slug) {
    var container = document.getElementById("giscus-container");
    var note = document.getElementById("giscus-note");
    if (!container) { return; }
    container.innerHTML = "";

    var cfg = (typeof BLOG_GISCUS !== "undefined") ? BLOG_GISCUS : null;
    if (!cfg || !cfg.repoId || !cfg.categoryId) {
      if (note) { note.style.display = "block"; }
      return;
    }
    if (note) { note.style.display = "none"; }

    var s = document.createElement("script");
    s.src = "https://giscus.app/client.js";
    s.crossOrigin = "anonymous";
    s.async = true;
    s.setAttribute("data-repo", cfg.repo);
    s.setAttribute("data-repo-id", cfg.repoId);
    s.setAttribute("data-category", cfg.category);
    s.setAttribute("data-category-id", cfg.categoryId);
    s.setAttribute("data-mapping", "specific");
    s.setAttribute("data-term", "blog-" + slug);
    s.setAttribute("data-strict", "0");
    s.setAttribute("data-reactions-enabled", "1");
    s.setAttribute("data-emit-metadata", "0");
    s.setAttribute("data-input-position", "top");
    s.setAttribute("data-theme", "light");
    s.setAttribute("data-lang", (currentLang() === "zh") ? "zh-CN" : "en");
    container.appendChild(s);
  }

  /* ---------- boot ---------- */

  onReady(function () {
    renderList();
    var slug = currentSlug();
    if (slug) { renderPost(slug); }

    document.addEventListener("langchange", function () {
      var s = currentSlug();
      if (s) { initComments(s); }
    });
  });
})();
