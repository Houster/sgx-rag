// (intentionally empty)
//
// Earlier this declared `*.bin` imports for an asset/inline approach we tried
// in next.config.js. We dropped that approach in favour of serving
// embeddings.bin from web/public/ and fetching it on cold start, so no module
// declaration is needed any more. Left as a stub so existing tsconfig globs
// don't fail if anyone references "types/assets.d.ts".
export {};
