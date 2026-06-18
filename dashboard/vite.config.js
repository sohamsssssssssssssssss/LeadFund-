import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Single-page app. `base: "./"` keeps asset paths relative so the same build
// deploys cleanly on Vercel or a Hugging Face Space subpath without config.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
