"""Backtest: how much revenue does each recovery strategy win back on unseen failed payments?

Run from the repository root:   python -m backend.backtest
Writes reports/backtest.json and three PNG charts. Everything here is SIMULATED: outcomes
come from the world in simulation.py, so the result shows how the method behaves in a
plausible world, not how it would perform on real payments.

Strategies (all see the same payments and the same random draws, so differences are paired):
  naive      retry every payment 1h, 24h and 72h after the failure (typical gateway dunning)
  rules      retry timeouts/declines on that schedule; ask the customer to update the card or
             re-authenticate when the reason says retrying cannot work (two requests at most)
  recoverai  pick the action and timing with the best expected value under the trained model,
             and only act when the model's probability clears the policy threshold
All strategies get up to 3 attempts and skip payments at or above the high-value approval limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml_engine
from . import simulation as sim
from .policy import DEFAULT_POLICY

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

RETRY_SCHEDULE_H = [1, 23, 48]      # incremental waits: attempts land at +1h, +24h, +72h
CONTACT_SCHEDULE_H = [0, 24, 48]


# ---------------------------------------------------------------------------------------
# Strategies: each returns (action array, delay array, act mask) for the active payments
# ---------------------------------------------------------------------------------------
def naive_strategy(sub: pd.DataFrame, k: int, when_base):
    n = len(sub)
    return (np.full(n, "retry", dtype=object), np.full(n, float(RETRY_SCHEDULE_H[k - 1])),
            np.ones(n, bool))


def rules_strategy(sub: pd.DataFrame, k: int, when_base):
    reason = sub["failure_reason"].to_numpy()
    action = np.where(reason == "Card expired", "card_update",
                      np.where(reason == "Authentication failed", "reauth", "retry")).astype(object)
    delay = np.where(action == "retry", RETRY_SCHEDULE_H[k - 1], CONTACT_SCHEDULE_H[k - 1]).astype(float)
    act = ~((action != "retry") & (k >= 3))          # at most two customer requests
    return action, delay, act


def make_recoverai_strategy(threshold: float):
    def strategy(sub: pd.DataFrame, k: int, when_base):
        c = ml_engine.choose_actions(sub, k, when_base, threshold=threshold)
        return c["action"].to_numpy(object), c["delay_h"].to_numpy(float), c["act"].to_numpy(bool)
    return strategy


# ---------------------------------------------------------------------------------------
# Simulation of one strategy against the world
# ---------------------------------------------------------------------------------------
def run_strategy(payments: pd.DataFrame, strategy, U: np.ndarray) -> dict:
    n = len(payments)
    active = np.ones(n, bool)
    elapsed = np.zeros(n)
    base = payments["timestamp"].to_numpy()
    out = {"recovered": np.zeros(n, bool), "attempts": np.zeros(n, int),
           "failed_retries": np.zeros(n, int), "contacts": np.zeros(n, int),
           "hours_to_recover": np.full(n, np.nan)}
    for k in range(1, sim.MAX_ATTEMPTS + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        sub = payments.iloc[idx]
        when_base = pd.to_datetime(base[idx]) + pd.to_timedelta(elapsed[idx], unit="h")
        action, delay, act = strategy(sub, k, when_base)
        active[idx[~act]] = False                       # strategy chose to stop
        go = idx[act]
        if go.size == 0:
            break
        frame = sim.attempt_frame(sub[act], action[act], delay[act], k, when_base[act])
        success = U[go, k - 1] < sim.true_success_prob(frame)
        is_retry = action[act] == "retry"
        out["attempts"][go] += 1
        out["contacts"][go] += (~is_retry).astype(int)
        out["failed_retries"][go] += (is_retry & ~success).astype(int)
        elapsed[go] += delay[act]
        won = go[success]
        out["recovered"][won] = True
        out["hours_to_recover"][won] = elapsed[won]
        active[won] = False
    return out


def summarise(payments: pd.DataFrame, r: dict) -> dict:
    amount = payments["amount"].to_numpy(float)
    value = float((amount * r["recovered"]).sum())
    return {
        "payments": int(len(payments)),
        "recovered_payments": int(r["recovered"].sum()),
        "recovered_value": round(value),
        "recovered_value_pct": round(100 * value / amount.sum(), 2),
        "attempts": int(r["attempts"].sum()),
        "attempts_per_100_payments": round(100 * r["attempts"].mean(), 1),
        "failed_retries": int(r["failed_retries"].sum()),
        "customer_contacts": int(r["contacts"].sum()),
        "customer_contacts_per_100_payments": round(100 * r["contacts"].mean(), 1),
        "value_recovered_per_attempt": round(value / max(int(r["attempts"].sum()), 1)),
        "median_hours_to_recover": float(np.nanmedian(r["hours_to_recover"])) if r["recovered"].any() else None,
    }


def by_reason(payments: pd.DataFrame, r: dict) -> dict:
    amount = payments["amount"].to_numpy(float)
    out = {}
    for reason in sim.REASONS:
        m = (payments["failure_reason"] == reason).to_numpy()
        if m.any():
            out[reason] = round(100 * float((amount[m] * r["recovered"][m]).sum() / amount[m].sum()), 1)
    return out


def paired_bootstrap(payments, a: dict, b: dict, reps: int = 1000, seed: int = 0) -> dict:
    """95% interval for (strategy a - strategy b), resampling payments."""
    rng = np.random.default_rng(seed)
    amount = payments["amount"].to_numpy(float)
    va, vb = amount * a["recovered"], amount * b["recovered"]
    aa, ab = a["attempts"].astype(float), b["attempts"].astype(float)
    n = len(amount)
    d_val, d_att = [], []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        d_val.append(100 * (va[i].sum() - vb[i].sum()) / amount[i].sum())
        d_att.append(100 * (aa[i].mean() - ab[i].mean()))
    q = lambda x: [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]
    return {"recovered_value_pct_points": {"mean": round(float(np.mean(d_val)), 2), "ci95": q(d_val)},
            "attempts_per_100_payments": {"mean": round(float(np.mean(d_att)), 1), "ci95": q(d_att)}}


# ---------------------------------------------------------------------------------------
# Choosing the confidence threshold on a separate validation set
# ---------------------------------------------------------------------------------------
THRESHOLD_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def select_threshold(seed: int) -> dict:
    """Pick the confidence threshold using validation payments the test set never touches.

    Rule (fixed before looking at test results): among thresholds whose recovered-value % is at
    least that of the rules baseline on the validation payments, take the one using the fewest
    attempts. Falls back to 0.0 (no gate) if none qualifies."""
    val = sim.generate_failures(8000, seed=seed + 100, start="2026-05-01", days=14)
    val = val[val["amount"] < DEFAULT_POLICY["high_value_threshold"]].reset_index(drop=True)
    U = np.random.default_rng(seed + 101).random((len(val), sim.MAX_ATTEMPTS))
    rules = summarise(val, run_strategy(val, rules_strategy, U))
    rows = []
    for t in THRESHOLD_GRID:
        s = summarise(val, run_strategy(val, make_recoverai_strategy(t), U))
        rows.append({"threshold": t, "recovered_value_pct": s["recovered_value_pct"],
                     "attempts_per_100_payments": s["attempts_per_100_payments"]})
    ok = [r for r in rows if r["recovered_value_pct"] >= rules["recovered_value_pct"]]
    chosen = min(ok, key=lambda r: r["attempts_per_100_payments"])["threshold"] if ok else 0.0
    return {"validation_payments": int(len(val)), "rules_recovered_value_pct": rules["recovered_value_pct"],
            "rules_attempts_per_100_payments": rules["attempts_per_100_payments"],
            "grid": rows, "selected_threshold": chosen}


# ---------------------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------------------
def run(n: int = 20000, seed: int = 99, sweep: bool = True) -> dict:
    ml_engine.ensure_model()
    pay = sim.generate_failures(n, seed=seed, start="2026-05-15", days=60)
    threshold_hv = DEFAULT_POLICY["high_value_threshold"]
    high_value = pay[pay["amount"] >= threshold_hv]
    pay = pay[pay["amount"] < threshold_hv].reset_index(drop=True)
    U = np.random.default_rng(seed + 1).random((len(pay), sim.MAX_ATTEMPTS))

    selection = select_threshold(seed)
    threshold = selection["selected_threshold"]
    runs = {
        "naive": run_strategy(pay, naive_strategy, U),
        "rules": run_strategy(pay, rules_strategy, U),
        "recoverai_strict_0.60": run_strategy(pay, make_recoverai_strategy(0.60), U),
        "recoverai": run_strategy(pay, make_recoverai_strategy(threshold), U),
    }
    result = {
        "note": "Simulated payments and simulated outcomes. Shows the method on a plausible world, "
                "not production performance.",
        "setup": {
            "test_payments": int(len(pay)), "seed": seed, "max_attempts": sim.MAX_ATTEMPTS,
            "selected_recovery_threshold": threshold,
            "default_policy_recovery_threshold": DEFAULT_POLICY["recovery_threshold"],
            "excluded_high_value_payments": int(len(high_value)),
            "excluded_high_value_amount": round(float(high_value["amount"].sum())),
            "failed_value_in_scope": round(float(pay["amount"].sum())),
        },
        "threshold_selection": selection,
        "model": {k: v for k, v in ml_engine.MODEL_INFO.items() if k != "calibration"},
        "strategies": {k: summarise(pay, r) for k, r in runs.items()},
        "by_reason_recovered_value_pct": {k: by_reason(pay, r) for k, r in runs.items()},
        "recoverai_vs_naive": paired_bootstrap(pay, runs["recoverai"], runs["naive"]),
        "recoverai_vs_rules": paired_bootstrap(pay, runs["recoverai"], runs["rules"]),
    }
    if sweep:
        points = []
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            s = summarise(pay, run_strategy(pay, make_recoverai_strategy(t), U))
            points.append({"threshold": t, "attempts_per_100_payments": s["attempts_per_100_payments"],
                           "recovered_value_pct": s["recovered_value_pct"]})
        result["threshold_sweep"] = points
    result["_calibration"] = ml_engine.MODEL_INFO["calibration"]
    return result


# ---------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------
def make_charts(res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, GREY, LIGHT, RED = "#2563eb", "#64748b", "#cbd5e1", "#dc2626"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    names = ["naive", "rules", "recoverai_strict_0.60", "recoverai"]
    t_sel = res["setup"]["selected_recovery_threshold"]
    labels = ["Naive\nretries", "Rules", "RecoverAI\nstrict 0.60 gate",
              "RecoverAI\nEV gate" if t_sel == 0 else f"RecoverAI\ngate {t_sel:.1f}"]
    colors = [LIGHT, GREY, "#93c5fd", BLUE]
    S = res["strategies"]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    v = [S[k]["recovered_value_pct"] for k in names]
    b = ax[0].bar(labels, v, color=colors)
    ax[0].bar_label(b, fmt="%.1f%%", padding=3)
    ax[0].set_ylabel("% of failed revenue recovered")
    ax[0].set_ylim(0, max(v) * 1.2)
    ax[0].tick_params(axis="x", labelsize=10)
    ax[1].tick_params(axis="x", labelsize=10)
    ax[0].set_title("Revenue recovered")
    retries = [S[k]["attempts_per_100_payments"] - S[k]["customer_contacts_per_100_payments"] for k in names]
    contacts = [S[k]["customer_contacts_per_100_payments"] for k in names]
    ax[1].bar(labels, retries, color=colors, label="Retries")
    ax[1].bar(labels, contacts, bottom=retries, color=colors, hatch="//", edgecolor="white", label="Customer requests")
    for i, k in enumerate(names):
        ax[1].text(i, retries[i] + contacts[i] + 3, f"{S[k]['attempts_per_100_payments']:.0f}", ha="center")
    ax[1].set_ylabel("Attempts per 100 failed payments")
    ax[1].set_ylim(0, max(S[k]["attempts_per_100_payments"] for k in names) * 1.18)
    ax[1].set_title("Effort (solid: retries, hatched: customer requests)")
    fig.suptitle("Simulated backtest on unseen failed payments", x=0.01, ha="left", fontsize=10, color=GREY)
    fig.tight_layout()
    fig.savefig(out / "backtest_summary.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    sw = res["threshold_sweep"]
    ax.plot([p["attempts_per_100_payments"] for p in sw], [p["recovered_value_pct"] for p in sw],
            "-o", color=BLUE, label="RecoverAI at different confidence thresholds", ms=4)
    for p in sw:
        if p["threshold"] == 0.1:       # identical to 0.0, avoid overlapping labels
            continue
        ax.annotate(f"{p['threshold']:.1f}", (p["attempts_per_100_payments"], p["recovered_value_pct"]),
                    textcoords="offset points", xytext=(5, -10), fontsize=8, color=GREY)
    ax.scatter([S["naive"]["attempts_per_100_payments"]], [S["naive"]["recovered_value_pct"]], color=RED, zorder=3, label="Naive retries")
    ax.scatter([S["rules"]["attempts_per_100_payments"]], [S["rules"]["recovered_value_pct"]], color=GREY, zorder=3, label="Rules")
    ax.scatter([S["recoverai"]["attempts_per_100_payments"]], [S["recoverai"]["recovered_value_pct"]],
               color=BLUE, marker="*", s=180, zorder=4, label=("RecoverAI, expected-value gate" if t_sel == 0 else f"RecoverAI, gate {t_sel:.1f}"))
    ax.set_xlabel("Attempts per 100 failed payments (effort)")
    ax.set_ylabel("% of failed revenue recovered")
    ax.set_title("Recovery vs effort: the confidence threshold trades one for the other")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "backtest_frontier.png", dpi=150)
    plt.close(fig)

    cal = res["_calibration"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 0.8], [0, 0.8], "--", color=LIGHT, label="Perfect calibration")
    ax.plot([c["predicted"] for c in cal], [c["observed"] for c in cal], "-o", color=BLUE, label="Model (held-out attempts)")
    ax.set_xlabel("Predicted success probability")
    ax.set_ylabel("Observed success rate")
    m = res["model"]
    ax.set_title(f"Calibration  (PR-AUC {m['pr_auc']} vs {m['pr_auc_base_rate']} base rate)", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "model_calibration.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()
    res = run(args.n, args.seed)
    REPORTS.mkdir(exist_ok=True)
    make_charts(res, REPORTS)
    public = {k: v for k, v in res.items() if not k.startswith("_")}
    (REPORTS / "backtest.json").write_text(json.dumps(public, indent=2))
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
