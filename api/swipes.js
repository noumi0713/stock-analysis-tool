const REPO = process.env.GITHUB_REPO || "noumi0713/stock-analysis-tool";
const BRANCH = process.env.SWIPE_DATA_BRANCH || "swipe-decisions";
const TOKEN = process.env.GITHUB_TOKEN || "";

function headers() {
  const h = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "stock-swipe-review"
  };
  if (TOKEN) h.Authorization = `Bearer ${TOKEN}`;
  return h;
}

function validDate(v) {
  return /^\d{4}-\d{2}-\d{2}$/.test(String(v || ""));
}

function sameOrigin(req) {
  const origin = req.headers.origin;
  const host = req.headers["x-forwarded-host"] || req.headers.host;
  if (!origin || !host) return true;
  try { return new URL(origin).host === host; } catch { return false; }
}

function pathFor(date) {
  return `swipe_review/data/${date}.json`;
}

async function readFile(date) {
  const path = pathFor(date);
  const url = `https://api.github.com/repos/${REPO}/contents/${path}?ref=${encodeURIComponent(BRANCH)}`;
  const r = await fetch(url, { headers: headers(), cache: "no-store" });
  if (r.status === 404) return { sha: null, body: { date, decisions: [] } };
  if (!r.ok) throw new Error(`GitHub read failed: ${r.status}`);
  const j = await r.json();
  const text = Buffer.from(String(j.content || "").replace(/\n/g, ""), "base64").toString("utf8");
  let body;
  try { body = JSON.parse(text); } catch { body = { date, decisions: [] }; }
  if (!Array.isArray(body.decisions)) body.decisions = [];
  return { sha: j.sha, body };
}

async function writeFile(date, body, sha) {
  const path = pathFor(date);
  const url = `https://api.github.com/repos/${REPO}/contents/${path}`;
  const payload = {
    message: `Update swipe decisions ${date}`,
    content: Buffer.from(JSON.stringify(body, null, 2) + "\n", "utf8").toString("base64"),
    branch: BRANCH
  };
  if (sha) payload.sha = sha;
  return fetch(url, {
    method: "PUT",
    headers: { ...headers(), "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

function normalize(items, date) {
  const out = [];
  for (const d of Array.isArray(items) ? items : []) {
    const code = String(d.stock_code || "").trim();
    if (!/^[0-9A-Z]{4,5}$/.test(code)) continue;
    if (d.decision !== "interested" && d.decision !== "rejected") continue;
    out.push({
      date,
      stock_code: code,
      stock_name: String(d.stock_name || "").slice(0, 120),
      bbs_rank: Number.isFinite(Number(d.bbs_rank)) ? Number(d.bbs_rank) : null,
      five_day_return_pct: Number.isFinite(Number(d.five_day_return_pct)) ? Number(d.five_day_return_pct) : null,
      decision: d.decision,
      recommended: Boolean(d.recommended),
      recorded_at: String(d.recorded_at || new Date().toISOString())
    });
  }
  return out;
}

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");

  if (req.method === "GET") {
    const date = String(req.query?.date || "");
    if (!validDate(date)) return res.status(400).json({ error: "invalid date" });
    try {
      const { body } = await readFile(date);
      return res.status(200).json(body);
    } catch (e) {
      return res.status(502).json({ error: e.message });
    }
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "method not allowed" });
  }

  if (!sameOrigin(req)) return res.status(403).json({ error: "origin rejected" });
  if (!TOKEN) return res.status(503).json({ error: "server persistence is not configured" });

  const date = String(req.body?.date || "");
  if (!validDate(date)) return res.status(400).json({ error: "invalid date" });

  const incoming = normalize(req.body?.decisions, date);
  if (!incoming.length && Array.isArray(req.body?.decisions) && req.body.decisions.length) {
    return res.status(400).json({ error: "no valid decisions" });
  }
  if (incoming.length > 100) return res.status(413).json({ error: "too many decisions" });

  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      const current = await readFile(date);
      const byCode = new Map((current.body.decisions || []).map(d => [String(d.stock_code), d]));
      for (const d of incoming) byCode.set(d.stock_code, d);
      const body = {
        date,
        updated_at: new Date().toISOString(),
        count: byCode.size,
        decisions: [...byCode.values()].sort((a,b)=>(a.bbs_rank ?? 999)-(b.bbs_rank ?? 999))
      };
      const wr = await writeFile(date, body, current.sha);
      if (wr.ok) return res.status(200).json({ ok: true, count: body.count });
      if (wr.status !== 409 && wr.status !== 422) {
        const detail = await wr.text();
        return res.status(502).json({ error: `GitHub write failed: ${wr.status}`, detail: detail.slice(0,300) });
      }
    }
    return res.status(409).json({ error: "concurrent update, retry" });
  } catch (e) {
    return res.status(502).json({ error: e.message });
  }
}