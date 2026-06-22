/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        // finance/quant palette
        ink: "#070a10",
        panel: "#0d1320",
        edge: "#1c2433",
        leadfund: "#34d399", // emerald — the winner
        naive: "#94a3b8", // slate — "what everyone builds"
      },
      keyframes: {
        pulseGap: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.65" },
        },
        glowPulse: {
          "0%, 100%": { opacity: "0.35", transform: "scale(1)" },
          "50%": { opacity: "0.75", transform: "scale(1.12)" },
        },
      },
      animation: {
        pulseGap: "pulseGap 1.8s ease-in-out infinite",
        glowPulse: "glowPulse 2.5s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
