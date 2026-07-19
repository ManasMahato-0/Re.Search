// Vercel serverless proxy: forwards search queries to the HF Space
// with a server-side token so anonymous visitors use our ZeroGPU quota.

const API = "https://thearchimedes-re-search.hf.space";

export default async function handler(req, res) {
  const q = req.query.q;
  if (!q || !q.trim()) {
    return res.status(400).json({ status: "error", query: q, results: [] });
  }

  const headers = { "Content-Type": "application/json" };
  if (process.env.HF_TOKEN) {
    headers["Authorization"] = `Bearer ${process.env.HF_TOKEN}`;
  }

  try {
    const call = await fetch(`${API}/gradio_api/call/search`, {
      method: "POST",
      headers,
      body: JSON.stringify({ data: [q] }),
    });
    const { event_id } = await call.json();

    const stream = await fetch(`${API}/gradio_api/call/search/${event_id}`, {
      headers,
    });
    const text = await stream.text();

    // SSE frames: "event: <type>" then "data: <json>"
    if (/^event:\s*error/m.test(text)) {
      const errLine = text.split("\n").filter((l) => l.startsWith("data:")).pop();
      const err = errLine ? JSON.parse(errLine.slice(5)) : {};
      return res.status(503).json({
        status: "error",
        query: q,
        results: [],
        message: err.error || "backend error",
      });
    }

    const dataLine = text.split("\n").filter((l) => l.startsWith("data:")).pop();
    if (!dataLine) throw new Error("empty response from backend");
    const payload = JSON.parse(dataLine.slice(5))[0];
    return res.status(200).json(payload);
  } catch (e) {
    return res.status(502).json({
      status: "error",
      query: q,
      results: [],
      message: String(e.message || e),
    });
  }
}
