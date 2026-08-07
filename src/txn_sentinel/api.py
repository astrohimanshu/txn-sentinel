"""HTTP service exposing the fraud-scoring model.

The /score contract is frozen. It carries account history alongside the transaction
so that the behaviour-sequence transformer can be swapped in later without any
change visible to a caller: LightGBM ignores all but the recent window, the
transformer will consume the last 64 entries, and the request and response shapes
stay identical either way.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from txn_sentinel import __version__
from txn_sentinel.scoring import AccountStats, Scorer, Txn

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAY_SAMPLE = REPO_ROOT / "models" / "replay_sample.json"

app = FastAPI(
    title="txn-sentinel",
    version=__version__,
    description="Fraud scoring for card transactions on a synthetic dataset.",
)


class TransactionIn(BaseModel):
    timestamp: datetime
    amount: float
    mcc: int
    use_chip: str
    merchant_id: int
    merchant_state: str | None = None
    errors: str | None = None

    def to_txn(self) -> Txn:
        return Txn(
            timestamp=self.timestamp,
            amount=self.amount,
            mcc=self.mcc,
            use_chip=self.use_chip,
            merchant_id=self.merchant_id,
            merchant_state=self.merchant_state,
            errors=self.errors,
        )


class AccountStatsIn(BaseModel):
    prior_count: int = Field(description="Transactions on this account before this one.")
    prior_amount_mean: float | None = Field(
        default=None, description="Mean amount over all prior transactions."
    )


class ScoreRequest(BaseModel):
    user: int
    card_index: int = Field(
        description="Index of the card within the user. (user, card_index) is the account."
    )
    transaction: TransactionIn
    history: list[TransactionIn] = Field(
        default_factory=list,
        description="Prior transactions on this account. Entries at or after the "
        "transaction's timestamp are ignored.",
    )
    account_stats: AccountStatsIn | None = Field(
        default=None,
        description="Lifetime aggregates from a feature store. Without these, "
        "cumulative features are capped at the length of `history`, which does not "
        "match what the model saw in training.",
    )


class ModelInfo(BaseModel):
    name: str
    version: str


class ScoreResponse(BaseModel):
    score: float = Field(description="Fraud probability in [0, 1].")
    decision: str = Field(description="'review' if score >= threshold, else 'approve'.")
    threshold: float
    model: ModelInfo
    features: dict[str, float | None]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


@lru_cache(maxsize=1)
def get_scorer() -> Scorer:
    return Scorer()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe for the container platform and uptime monitoring."""
    try:
        get_scorer()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(status="ok", version=__version__, model_loaded=loaded)


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """Score one transaction against its account history."""
    try:
        scorer = get_scorer()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    value, features = scorer.score(
        request.user,
        request.card_index,
        request.transaction.to_txn(),
        [h.to_txn() for h in request.history],
        AccountStats(
            prior_count=request.account_stats.prior_count,
            prior_amount_mean=request.account_stats.prior_amount_mean,
        )
        if request.account_stats
        else None,
    )
    return ScoreResponse(
        score=value,
        decision="review" if value >= scorer.threshold else "approve",
        threshold=scorer.threshold,
        model=ModelInfo(name=scorer.name, version=__version__),
        features=features,
    )


@app.get("/replay/sample")
def replay_sample() -> dict:
    """Held-out transactions used by the dashboard. Test split only, never training."""
    if not REPLAY_SAMPLE.exists():
        raise HTTPException(status_code=404, detail="No replay sample bundled")
    return json.loads(REPLAY_SAMPLE.read_text())


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>txn-sentinel &middot; replay</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1115; --fg:#e6e8eb; --dim:#8b93a1;
          --line:#232733; --ok:#3ddc97; --warn:#ffb454; --bad:#ff5c5c; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#fbfbfd; --fg:#1a1c20; --dim:#666e7d; --line:#e3e6ec; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 ui-sans-serif,
         system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:1100px; margin:0 auto; padding:32px 20px 64px; }
  h1 { font-size:20px; margin:0 0 4px; letter-spacing:-.01em; }
  .sub { color:var(--dim); margin:0 0 24px; font-size:13px; }
  .bar { display:flex; gap:10px; align-items:center; margin-bottom:20px; flex-wrap:wrap; }
  button { background:var(--fg); color:var(--bg); border:0; border-radius:7px;
           padding:8px 16px; font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.45; cursor:default; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
           gap:10px; margin-bottom:22px; }
  .card { border:1px solid var(--line); border-radius:9px; padding:11px 13px; }
  .card .k { color:var(--dim); font-size:11px; text-transform:uppercase;
             letter-spacing:.05em; }
  .card .v { font-size:21px; font-weight:650; font-variant-numeric:tabular-nums; }
  .scroll { overflow-x:auto; border:1px solid var(--line); border-radius:9px; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th,td { padding:8px 11px; text-align:left; border-bottom:1px solid var(--line);
          white-space:nowrap; }
  th { color:var(--dim); font-weight:600; font-size:11px; text-transform:uppercase;
       letter-spacing:.05em; position:sticky; top:0; background:var(--bg); }
  tbody tr:last-child td { border-bottom:0; }
  .pill { padding:2px 8px; border-radius:99px; font-size:11px; font-weight:650; }
  .review { background:color-mix(in srgb,var(--bad) 18%,transparent); color:var(--bad); }
  .approve{ background:color-mix(in srgb,var(--ok) 15%,transparent); color:var(--ok); }
  .hit { color:var(--ok); } .miss { color:var(--bad); } .dim { color:var(--dim); }
  .note { color:var(--dim); font-size:12px; margin-top:18px; }
</style>
</head>
<body><div class="wrap">
  <h1>txn-sentinel</h1>
  <p class="sub">Replaying held-out 2019 transactions through <code>POST /score</code>.
     Synthetic data (IBM/Altman) &mdash; not real cardholder activity.</p>

  <div class="bar">
    <button id="go">Start replay</button>
    <span id="status" class="dim">idle</span>
  </div>

  <div class="stats">
    <div class="card"><div class="k">Scored</div><div class="v" id="n">0</div></div>
    <div class="card"><div class="k">Flagged</div><div class="v" id="flag">0</div></div>
    <div class="card"><div class="k">Actual fraud</div><div class="v" id="fraud">0</div></div>
    <div class="card"><div class="k">Caught</div><div class="v" id="caught">0</div></div>
    <div class="card"><div class="k">Median latency</div><div class="v" id="lat">&ndash;</div></div>
  </div>

  <div class="scroll"><table>
    <thead><tr><th>Time</th><th>Amount</th><th>Channel</th><th>State</th>
      <th>Score</th><th>Decision</th><th>Truth</th></tr></thead>
    <tbody id="rows"></tbody>
  </table></div>

  <p class="note">Each row is one live call. Features are rebuilt from the account's
     own prior transactions using the same code path as training.</p>
</div>
<script>
const $ = id => document.getElementById(id);
let running = false;

function fmtAmount(a) {
  return (a < 0 ? "-$" : "$") + Math.abs(a).toFixed(2);
}

async function replay() {
  running = true; $("go").disabled = true; $("status").textContent = "loading sample...";
  let sample;
  try {
    sample = await (await fetch("replay/sample")).json();
  } catch (e) {
    $("status").textContent = "could not load sample"; $("go").disabled = false; return;
  }
  const cases = sample.cases || [];
  $("status").textContent = `replaying ${cases.length} transactions`;

  let n = 0, flagged = 0, fraud = 0, caught = 0;
  const lats = [];
  $("rows").innerHTML = "";

  for (const c of cases) {
    if (!running) break;
    const t0 = performance.now();
    let res;
    try {
      res = await (await fetch("score", {
        method: "POST", headers: {"content-type": "application/json"},
        body: JSON.stringify({user: c.user, card_index: c.card_index,
                              transaction: c.transaction, history: c.history || [],
                              account_stats: c.account_stats || null})
      })).json();
    } catch (e) { continue; }
    lats.push(performance.now() - t0);

    n++;
    const isFlag = res.decision === "review";
    if (isFlag) flagged++;
    if (c.is_fraud) { fraud++; if (isFlag) caught++; }

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="dim">${c.transaction.timestamp.replace("T", " ").slice(0, 16)}</td>
      <td>${fmtAmount(c.transaction.amount)}</td>
      <td class="dim">${(c.transaction.use_chip || "").replace(" Transaction", "")}</td>
      <td class="dim">${c.transaction.merchant_state || "&ndash;"}</td>
      <td>${res.score.toFixed(4)}</td>
      <td><span class="pill ${isFlag ? "review" : "approve"}">${res.decision}</span></td>
      <td class="${c.is_fraud ? (isFlag ? "hit" : "miss") : "dim"}">${
        c.is_fraud ? (isFlag ? "fraud - caught" : "fraud - missed") : "legit"}</td>`;
    $("rows").prepend(tr);

    $("n").textContent = n; $("flag").textContent = flagged;
    $("fraud").textContent = fraud; $("caught").textContent = caught;
    const sorted = [...lats].sort((a, b) => a - b);
    $("lat").textContent = Math.round(sorted[Math.floor(sorted.length / 2)]) + "ms";
  }
  $("status").textContent = "done"; $("go").disabled = false; running = false;
}
$("go").onclick = replay;
</script>
</body></html>
"""
