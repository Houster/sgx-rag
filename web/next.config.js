/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Force Vercel to bundle web/data/* (chunks.json + embeddings.bin) into the
  // /api/chat serverless function. Without this the function reads the files
  // from disk locally but they get tree-shaken out of the production bundle.
  outputFileTracingIncludes: {
    "/api/chat": ["./data/**"],
  },
};
module.exports = nextConfig;
