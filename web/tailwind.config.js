/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        // Orikai memo palette — Turf Green primary, mirrored from 001
        turf: {
          50:  "#e8f3ee",
          100: "#c7e1d3",
          200: "#94c4ac",
          300: "#5fa483",
          400: "#2c845c",
          500: "#04724D",
          600: "#03603f",
          700: "#024c33",
          800: "#013a26",
          900: "#01281b"
        },
        ink:    "#101418",
        graphite:"#2b3138",
        rule:   "#d8dde2",
        soft:   "#f3f1ec",
        paper:  "#fbfaf6",
        flag: {
          pos: "#04724D",
          neg: "#8a1c1c",
          warn: "#a06b00"
        }
      },
      fontFamily: {
        serif: ["'Source Serif 4'", "'Source Serif Pro'", "Georgia", "serif"],
        sans:  ["'Inter'", "system-ui", "sans-serif"],
        mono:  ["'IBM Plex Mono'", "ui-monospace", "monospace"]
      },
      letterSpacing: {
        memo: "0.14em"
      }
    }
  },
  plugins: []
};
