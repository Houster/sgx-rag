/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,

  // Force Vercel to bundle web/data/{chunks.json,embeddings.bin} into every
  // serverless function that imports lib/retriever.ts. Each route is traced
  // independently — an entry for /api/chat does NOT cascade to /api/stats —
  // so list both.
  //
  // Version note: in Next 14.x this option is `experimental.outputFileTracing
  // Includes`. It was promoted to a top-level config key only in Next 15.
  // Putting it at the top level on 14.x triggers `Unrecognized key(s) in
  // object` and the whole include block is silently ignored.
  //
  // Keys are page identifiers (the source file path is the most reliable form
  // across versions). Values are globs relative to the project root.
  experimental: {
    outputFileTracingIncludes: {
      "pages/api/chat":  ["./data/chunks.json", "./data/embeddings.bin"],
      "pages/api/stats": ["./data/chunks.json", "./data/embeddings.bin"],
    },
  },
};
module.exports = nextConfig;
