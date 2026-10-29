"""Self-contained HTML tearsheet (charts embedded as PNG). Requires matplotlib (`pip install alphascout[dev]`)."""
from __future__ import annotations

import base64
import html
import io

import numpy as np


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def _table(rows: list[list], head: list[str]) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def make_tearsheet(fin: dict, dates: np.ndarray, net: np.ndarray, vt: np.ndarray, n_res: int,
                   replay: dict | None, gate: dict, ppy: int = 252) -> str:
    from matplotlib.figure import Figure   # object API: no pyplot, so no global backend side effects

    d = dates.astype("datetime64[D]")
    fig = Figure(figsize=(9, 8.5)); ax = fig.subplots(3, 1, sharex=True)
    for a_, (x, lbl) in zip(ax[:2], [(net, "net of costs, unscaled"), (vt, f"vol-targeted")]):
        eq = np.cumprod(1 + np.nan_to_num(x))
        a_.plot(d, eq, lw=1.1)
        a_.axvline(d[n_res], color="crimson", ls="--", lw=1)
        a_.set_ylabel(f"equity ({lbl})"); a_.grid(alpha=.3)
    ax[0].text(d[n_res], ax[0].get_ylim()[1], " hold-out starts", color="crimson", va="top", fontsize=8)
    eq = np.cumprod(1 + np.nan_to_num(vt)); dd = eq / np.maximum.accumulate(eq) - 1
    ax[2].fill_between(d, dd, 0, color="tab:red", alpha=.4); ax[2].set_ylabel("drawdown (vol-targeted)"); ax[2].grid(alpha=.3)
    img1 = _png(fig)

    fig2 = Figure(figsize=(9, 2.6)); ax2 = fig2.subplots()
    w = 126
    x = np.nan_to_num(vt)
    rs = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        s = x[t - w:t].std(ddof=1)
        rs[t] = x[t - w:t].mean() / s * np.sqrt(ppy) if s > 0 else np.nan
    ax2.plot(d, rs, lw=1); ax2.axhline(0, color="k", lw=.6); ax2.axvline(d[n_res], color="crimson", ls="--", lw=1)
    ax2.set_ylabel("rolling 6m Sharpe"); ax2.grid(alpha=.3)
    img2 = _png(fig2)

    banner = ("<p class='bad'>DIAGNOSTIC ONLY: this strategy did NOT pass the multiple-testing gate. Its raw numbers "
              "are what cherry-picking would have shown you.</p>" if not fin["survivor"] else
              "<p class='ok'>Passed the corrected-significance gate (DSR and PBO) on the research window.</p>")
    stress = _table([[k, _f(v["research_sharpe"]), _f(v["holdout_sharpe"]), _f(v["research_ann_return"] * 100, 1) + "%"]
                     for k, v in fin["cost_stress"].items()], ["cost multiplier", "research Sharpe", "hold-out Sharpe", "research ann. return"])
    reg = _table([[k, v["n"], _f(v["sharpe"])] for k, v in fin["regimes_research"].items()], ["regime", "days", "Sharpe"])
    rk = fin["risk_research_vol_targeted"]; rh = fin["risk_holdout_vol_targeted"]
    risk = _table([[m, _f(rk[m], 3), _f(rh[m], 3)] for m in ("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "cvar_5pct_daily", "calmar")],
                  ["metric", "research", "hold-out"])
    gt = _table([["trials N", gate["n_trials"]], ["PBO (whole candidate set)", _f(gate["pbo"], 3)],
                 ["SR0 = expected best Sharpe from luck (ann.)", _f(gate["sr0_annualised"])],
                 ["this strategy: raw Sharpe (ann.)", _f(fin["research_sharpe"])],
                 ["this strategy: naive p-value", _f(fin["p_naive"], 4)], ["this strategy: DSR", _f(fin["dsr"], 3)],
                 ["BH-FDR q-value", _f(fin["q_bh"], 4)]], ["gate quantity", "value"])
    rp = ""
    if replay:
        rows = [[k, _f(v["sharpe"]), _f(v["total_return"] * 100, 1) + "%", _f(v["fill_ratio"], 2), v["n_events"]] for k, v in replay.items()]
        rp = "<h3>Paper-trading replay on the untouched hold-out (event-driven fills)</h3>" + _table(rows, ["scenario", "Sharpe", "total return", "fill ratio", "control events"])
    return f"""<!doctype html><meta charset=utf-8><title>{html.escape(fin['name'])}</title>
<style>body{{font:14px system-ui,sans-serif;max-width:960px;margin:2em auto;padding:0 1em;color:#222}}
table{{border-collapse:collapse;margin:.5em 0 1.2em}}td,th{{border:1px solid #ccc;padding:3px 9px;text-align:right}}th{{background:#f3f3f3}}
.bad{{background:#fdecea;border-left:4px solid #c62828;padding:8px 12px}}.ok{{background:#e8f5e9;border-left:4px solid #2e7d32;padding:8px 12px}}
img{{max-width:100%}}h1{{font-size:20px}}h3{{margin-bottom:.2em}}</style>
<h1>{html.escape(fin['name'])} <small>({fin['family']})</small></h1>{banner}
<img src="data:image/png;base64,{img1}"><img src="data:image/png;base64,{img2}">
<h3>Multiple-testing gate</h3>{gt}<h3>Cost stress (2x, 3x transaction costs)</h3>{stress}
<h3>Regime-conditional Sharpe (research window; regime known before each bar)</h3>{reg}
<h3>Risk (vol-targeted)</h3>{risk}{rp}
<p style="color:#666;font-size:12px">Costs are assumptions, not measurements. Daily bars only. Synthetic or real data as stated in summary.json.</p>"""
