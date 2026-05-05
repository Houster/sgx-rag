/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,

  // Force Vercel to bundle web/data/{chunks.json,embeddings.bin} into the
  // serverless function for every API route that imports lib/retriever.ts.
  //
  // Two non-obvious things had to be right at the same time:
  //
  //   1. Placement: in Next 14.x the option lives under `experimental.*`. It
  //      was only promoted to a top-level key in Next 15. At the top level on
  //      14.x, Next emits `Unrecognized key(s) in object` and silently drops
  //      the whole include block.
  //
  //   2. Key format: Next runs picomatch against the *normalized route path*
  //      (e.g. "/api/chat"), NOT against the source-file form
  //      ("pages/api/chat"). The latter looks plausible but matches nothing,
  //      so the route gets traced without the data files.
  //
  // Verified against node_modules/next/dist/build/collect-build-traces.js
  // (lines 510–540) for next@14.2.5: keys are picomatch globs against
  // `route = normalizePagePath(entryName)`, which strips the `pages/` prefix.
  experimental: {
    outputFileTracingIncludes: {
      "/api/chat":  ["./data/chunks.json", "./data/embeddings.bin"],
      "/api/stats": ["./data/chunks.json", "./data/embeddings.bin"],
    },
  },
};
module.exports = nextConfig;
