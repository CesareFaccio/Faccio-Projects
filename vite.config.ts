import { defineConfig } from "vite";

export default defineConfig({
  // Relative base so the build works whether GitHub Pages serves it at
  // the domain root (username.github.io) or under a repo subpath
  // (username.github.io/repo-name/).
  base: "./",
});
