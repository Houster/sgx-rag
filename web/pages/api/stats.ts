// GET /api/stats — corpus stats for the sidebar (chunk count, doc-type breakdown).
// Included as its own route so the front-end doesn't have to wait for /api/chat.
import type { NextApiRequest, NextApiResponse } from "next";
import { getCorpusStats } from "../../lib/retriever";

export default async function handler(
  _req: NextApiRequest,
  res: NextApiResponse
) {
  try {
    const stats = await getCorpusStats();
    return res.status(200).json(stats);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return res.status(500).json({ error: msg });
  }
}

export const config = { maxDuration: 30 };
