/* ============================================================
   Ke Xu's blog — POST MANIFEST + comment config
   ------------------------------------------------------------
   To publish a post:
     1. Write it in Markdown:  blog/posts/<slug>.md
     2. Add an entry to BLOG_POSTS below.
        (slug: lowercase letters, digits and dashes only;
         the list is sorted by date automatically)

   Entry format:
     {
       slug: "my-post",
       title: "My Post",
       date: "2026-09-10",
       summary: "One-line summary shown in the list.",
       tags: ["RSI", "agents"]
     }

   Comments (giscus, backed by GitHub Discussions):
     1. Enable Discussions on HugoXK/hugoxk.github.io and create
        a category for the blog (e.g. "Announcements").
     2. Install the giscus app on the repo:
        https://github.com/apps/giscus
     3. Visit https://giscus.app, fill in the repo, then copy
        data-repo-id and data-category-id into BLOG_GISCUS.
   ============================================================ */

var BLOG_POSTS = [
  // Add your posts here, for example:
  // {
  //   slug: "hello-world",
  //   title: "Hello, World",
  //   date: "2026-09-10",
  //   summary: "Why I am starting this blog.",
  //   tags: ["meta"]
  // },
];

var BLOG_GISCUS = {
  repo: "HugoXK/hugoxk.github.io",
  repoId: "",            // [TODO] fill from https://giscus.app
  category: "Announcements",
  categoryId: ""         // [TODO] fill from https://giscus.app
};
