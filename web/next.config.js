/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,

  // Force Vercel to bundle web/data/{chunks.json,embeddings.bin} into the
  // serverless function for every API route that touches the retriever.
  //
  // We list the files explicitly (instead of `./data/**`) because:
  //   - `./data/**` is interpreted differently across Next versions and has
  //     historically failed to pick up binary files;
  //   - we know exactly what we need, so explicitness avoids the tracing
  //     guesswork entirely.
  //
  // Both keys are required: Next traces each route separately, so an entry
  // for /api/chat does NOT cascade to /api/stats.
  outputFileTracingIncludes: {
    "/api/chat":  ["./data/chunks.json", "./data/embeddings.bin"],
    "/api/stats": ["./data/chunks.json", "./data/embeddings.bin"],
  },
};
module.exports = nextConfig;
