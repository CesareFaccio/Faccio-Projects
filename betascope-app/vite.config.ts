import { defineConfig } from "vite";

export default defineConfig({
  // Relative base so this still resolves correctly once it's built into
  // dist/betascope/ under the main site (which may itself be served from
  // a GitHub Pages project subpath rather than the domain root).
  base: "./",
});
