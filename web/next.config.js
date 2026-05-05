/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,

  // Data file strategy (after three failed deploys with outputFileTracingIncludes
  // and one stalled build with inline 24 MB asset/inline):
  //
  //   - web/data/chunks.json   → imported via ES module in lib/retriever.ts.
  //                              Webpack inlines JSON natively; the chunk text
  //                              ships inside the function bundle (NOT publicly
  //                              downloadable), and Next's tracer never has to
  //                              know about it.
  //
  //   - web/public/embeddings.bin → served as a static asset on Vercel's CDN.
  //                                 The retriever fetches it once on cold start
  //                                 (~24 MB, sub-second on the same region) and
  //                                 caches the Float32Array view in module
  //                                 memory for the rest of the function's life.
  //                                 Embeddings are derivative of the public
  //                                 chunks, so the public URL isn't a leak.
  //
  // No webpack rules, no experimental tracer config, nothing exotic. The two
  // mechanisms in use here (JSON import, public/ static assets) are the
  // foundational primitives every Next.js project relies on.
};
module.exports = nextConfig;
