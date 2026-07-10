"""
Task 5: Dynamic Quoting Under Inventory Pressure

We design a quoting function

    Q(I, sigma, alpha, eta) -> (delta_bid, delta_ask)

that sets bid/ask half-spreads around the mid using only the observable state, so
it adapts to unknown, regime-shifting parameters (lambda, gamma, phi) through the
signals alone. Four design principles:

1. Work in units of sigma (scale / regime invariance). Every half-spread is
   delta = k*sigma with k dimensionless. The fill model (eq. 15) depends on
   delta/sigma = k, the floor (eq. 14) is k >= c_min = 0.5, and delta_max = 50 bps
   of mid is ~17 sigma in the data. So k rescales automatically with volatility
   (no re-tuning after a regime shift), is invariant to return- vs price-units,
   and keeping k in [c_min, k_max=8] respects both bounds.

2. Adversity-aware base spread (toxicity protection). The symmetric base spread
   is a steep ramp in the model adversity score: k_base = base0 + base1*alpha,
   clipped to [c_min, k_max]. Closed-form justification: for per-unit drift
   g = side*(M_bar - M0), the expected PnL of quoting delta = k*sigma under the
   fill law exp(-gamma*k) is exp(-gamma*k)*(g + k*sigma), maximised at
   k* = 1/gamma - g/sigma. The optimal spread moves one-for-one with -g/sigma and
   the gamma-term is just a constant shift, so the shape is independent of the
   hidden gamma. Empirically g/sigma spans ~+42 (benign A) to -115 (toxic F), so
   the optimum is almost a step: benign clients -> tight floor (capture positive
   drift at a high fill rate); toxic clients -> wide cap (so they seldom fill).
   alpha is a monotone proxy for -g/sigma, hence base1 is large. This is the
   per-trade version of Task 2's delta* / Task 4's externalization - a quote so
   wide a toxic client rarely fills is "soft" externalization.

3. Inventory skew (Avellaneda-Stoikov style, bounded). We skew quotes to
   mean-revert inventory toward zero: when long we tighten the ask and widen the
   bid, when short the opposite. skew = k_skew * tanh(I / I_ref) * urgency(eta);
   delta_bid = (k_base + skew)*sigma, delta_ask = (k_base - skew)*sigma. tanh
   saturates, so even if a regime shift blows up trade sizes the skew can't
   explode - it keeps end-of-day inventory small and tames the phi*I^2*sigma_D
   penalty (eq. 16) without knowing phi.

4. End-of-day urgency (penalty avoidance). The penalty bites at the close, so the
   skew grows through the day: urgency(eta) = 1 + eta_gain * eta**2. Near the
   close we skew hard to flatten inventory before the penalty applies.

All half-spreads are clipped to [c_min, k_max] * sigma.

Required interface:
    quote(inventory, sigma, alpha, eta) -> (delta_bid, delta_ask)
    validate_quote(...) -> None      (backtests across a grid of hidden params)
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

import task3 as T3  # model M: predict_adversity(client, tau)


# Constraints from the problem statement.
C_MIN = 0.5          # eq. 14 lower-bound coefficient: delta >= c_min * sigma
K_MAX = 8.0          # cap in sigma-units; stays below delta_max (~17 sigma) safely
N_VOL = 20           # eq. 10 window for realized volatility

# Quoting-function hyperparameters. The functional form is fixed; these only
# shape it. Tuned on the provided data as a proxy and chosen for robustness
# across a grid of hidden (lambda, gamma, phi) - see validate_quote.
PARAMS = {
    "base0": -22.0,     # intercept of the adversity ramp (vol units)
    "base1": 55.0,      # slope of half-spread in adversity alpha (steep on purpose)
    "k_skew": 3.0,      # max inventory skew strength (vol units)
    "i_ref": 800.0,     # inventory scale at which skew reaches ~tanh(1)=0.76
    "eta_gain": 3.0,    # end-of-day urgency growth
}
# base1 is large on purpose: the optimal spread k* = 1/gamma - g/sigma moves
# one-for-one with -g/sigma, and g/sigma swings hugely across clients (~+42 for
# benign A to -115 for toxic F), so the optimum is almost a step - tight floor
# for benign flow, wide cap for toxic. See module docstring, principle 2.


# Required: quote
def quote(inventory: float, sigma: float, alpha: float, eta: float,
          params: dict = None) -> Tuple[float, float]:
    """
    Dynamic quoting function Q(I, sigma, alpha, eta) -> (delta_bid, delta_ask).

    Parameters
    ----------
    inventory : float
        Current signed inventory I (positive = long, negative = short).
    sigma : float
        Current realized volatility sigma_t (> 0); same units used by the
        evaluator.  All spreads are returned in these units.
    alpha : float
        Adversity score in [0, 1] from model M for the incoming client.
    eta : float
        Elapsed fraction of the trading day in [0, 1].
    params : dict, optional
        Override the default hyperparameters (used by validate_quote tuning).

    Returns
    -------
    (delta_bid, delta_ask) : tuple of float
        Bid/ask half-spreads (distance below/above mid), each >= 0 and within
        [c_min * sigma, k_max * sigma].
    """
    p = PARAMS if params is None else {**PARAMS, **params}

    # Robust input handling
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-9                      # avoid div-by-zero; keep spreads finite
    a = float(alpha)
    a = 0.0 if not np.isfinite(a) else min(1.0, max(0.0, a))
    e = float(eta)
    e = 0.0 if not np.isfinite(e) else min(1.0, max(0.0, e))
    I = float(inventory)
    if not np.isfinite(I):
        I = 0.0

    # Base symmetric half-spread (vol units), widening with adversity
    k_base = p["base0"] + p["base1"] * a

    # Bounded inventory skew with end-of-day urgency (vol units)
    urgency = 1.0 + p["eta_gain"] * e * e
    skew = p["k_skew"] * np.tanh(I / p["i_ref"]) * urgency

    # Long (I>0): widen bid (buy less), tighten ask (sell more); short: opposite
    k_bid = k_base + skew
    k_ask = k_base - skew

    # Clip to the feasible band [c_min, k_max] (in vol units), then -> price units
    k_bid = min(K_MAX, max(C_MIN, k_bid))
    k_ask = min(K_MAX, max(C_MIN, k_ask))

    return float(k_bid * sigma), float(k_ask * sigma)


# Adversity score alpha for a client: a single robust scalar from model M.
def client_alpha(client: str) -> float:
    """alpha = mean over the six horizons of M's predicted adverse probability."""
    ps = [T3.predict_adversity(client=client, tau=t) for t in T3.ALL_TAU]
    return float(np.mean(ps))


# Backtest engine
def _prepare_stream(df: pd.DataFrame) -> dict:
    """Pre-compute per-trade arrays: realized vol (eq.10, N=20), alpha, eta, PnL parts."""
    df = df.reset_index(drop=True)
    m0 = df["M0"].to_numpy(dtype=float)
    side = df["Side"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    mbar = df[[T3.MID_COL[t] for t in T3.ALL_TAU]].to_numpy(dtype=float).mean(axis=1)

    # realized volatility, eq.10: rms of last N mid-returns (causal, shifted by 1)
    r = np.empty_like(m0)
    r[0] = 0.0
    r[1:] = (m0[1:] - m0[:-1]) / m0[:-1]
    r2 = r * r
    sig = np.sqrt(pd.Series(r2).rolling(N_VOL, min_periods=1).mean().to_numpy())
    med = np.nanmedian(sig[sig > 0]) if np.any(sig > 0) else 1e-4
    sig = np.where((sig > 0) & np.isfinite(sig), sig, med)

    alpha = df["Name"].map({c: client_alpha(c) for c in T3.CLIENTS}).to_numpy(dtype=float)

    # eta: elapsed fraction of the day, per date, by trade order
    sec = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour * 3600 \
        + pd.to_datetime(df["time"], format="%H:%M:%S").dt.minute * 60 \
        + pd.to_datetime(df["time"], format="%H:%M:%S").dt.second
    sec = sec.to_numpy(dtype=float)
    eta = np.zeros_like(sec)
    date = df["Date"].to_numpy()
    for d in np.unique(date):
        m = date == d
        s = sec[m]
        lo, hi = s.min(), s.max()
        eta[m] = (s - lo) / (hi - lo) if hi > lo else 0.0

    sigma_D = float(np.mean(sig))  # average daily volatility proxy for the penalty
    return {
        "m0": m0, "side": side, "vol": vol, "mbar": mbar,
        "sig": sig, "alpha": alpha, "eta": eta, "date": date, "sigma_D": sigma_D,
    }


def backtest(stream: dict, lam: float, gamma: float, phi: float,
             params: dict = None, seed: int = 0) -> dict:
    """
    Replay the trade stream under the quoting strategy and a hidden-parameter
    setting (lam, gamma, phi).  Fills are stochastic per eq. 15.

    Returns total net PnL, the per-day PnL series, end-of-day inventories and
    the inventory path.
    """
    rng = np.random.default_rng(seed)
    m0 = stream["m0"]; side = stream["side"]; vol = stream["vol"]; mbar = stream["mbar"]
    sig = stream["sig"]; alpha = stream["alpha"]; eta = stream["eta"]; date = stream["date"]
    sigma_D = stream["sigma_D"]
    n = len(m0)

    inv = 0.0
    cur_date = date[0]
    day_pnl = 0.0
    day_pnls = []
    eod_inv = []
    inv_path = np.empty(n)

    for i in range(n):
        if date[i] != cur_date:
            # close-of-day inventory penalty (eq. 16) then reset
            day_pnl -= phi * inv * inv * sigma_D
            day_pnls.append(day_pnl)
            eod_inv.append(inv)
            day_pnl = 0.0
            inv = 0.0                       # flat at start of each new day
            cur_date = date[i]

        s = sig[i]
        db, da = quote(inv, s, alpha[i], eta[i], params=params)
        # which half-spread applies depends on the side we trade (Side is LP-side)
        if side[i] > 0:                     # we buy at the bid -> delta_bid
            delta = db
        else:                               # we sell at the ask -> delta_ask
            delta = da

        # fill probability (eq. 15), clipped to [0,1]
        pf = lam * np.exp(-gamma * (delta / s))
        pf = 0.0 if pf < 0 else (1.0 if pf > 1 else pf)

        if rng.random() < pf:               # F = 1 (filled)
            # trade PnL (Def 2.3, uniform weights): T_P = M0 - side*delta
            tp = m0[i] - side[i] * delta
            pnl = side[i] * vol[i] * (mbar[i] - tp)
            day_pnl += pnl
            inv += side[i] * vol[i]         # running inventory (eq. 11)
        inv_path[i] = inv

    # final day
    day_pnl -= phi * inv * inv * sigma_D
    day_pnls.append(day_pnl)
    eod_inv.append(inv)

    day_pnls = np.asarray(day_pnls, dtype=float)
    return {
        "total_pnl": float(day_pnls.sum()),
        "day_pnls": day_pnls,
        "eod_inv": np.asarray(eod_inv, dtype=float),
        "inv_path": inv_path,
    }


def _score(day_pnls: np.ndarray, sigma_floor: float = 1.0) -> float:
    """Sharpe-like score (eq. 18): total PnL / max(std(daily PnL), sigma_floor)."""
    sd = float(np.std(day_pnls))
    return float(day_pnls.sum() / max(sd, sigma_floor))


def _max_drawdown(day_pnls: np.ndarray) -> float:
    """Max drawdown of the cumulative daily-PnL curve (lower is better)."""
    cum = np.cumsum(day_pnls)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum)) if len(cum) else 0.0


# A simple fixed-spread baseline (no inventory control) for comparison.
def _baseline_quote(inventory, sigma, alpha, eta, k=2.0):
    s = sigma if sigma > 0 else 1e-9
    return k * s, k * s


def _backtest_baseline(stream, lam, gamma, phi, k=2.0, seed=0):
    rng = np.random.default_rng(seed)
    m0 = stream["m0"]; side = stream["side"]; vol = stream["vol"]; mbar = stream["mbar"]
    sig = stream["sig"]; date = stream["date"]; sigma_D = stream["sigma_D"]
    inv = 0.0; cur = date[0]; dp = 0.0; days = []
    for i in range(len(m0)):
        if date[i] != cur:
            dp -= phi * inv * inv * sigma_D; days.append(dp); dp = 0.0; inv = 0.0; cur = date[i]
        s = sig[i]; delta = k * s
        pf = lam * np.exp(-gamma * (delta / s)); pf = min(1.0, max(0.0, pf))
        if rng.random() < pf:
            tp = m0[i] - side[i] * delta
            dp += side[i] * vol[i] * (mbar[i] - tp); inv += side[i] * vol[i]
    dp -= phi * inv * inv * sigma_D; days.append(dp)
    return np.asarray(days, float)


# Required: validate_quote
def validate_quote(params: dict = None, seeds=(0, 1, 2), save_fig: bool = True,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Backtest the quoting strategy across a grid of hidden parameters
    (lambda, gamma, phi) - which are unknown and regime-shifting - and report
    the Sharpe-like score (eq. 18), total PnL, end-of-day inventory control and
    max drawdown, averaged over random fill seeds.  Compares against a fixed
    symmetric-spread baseline with no inventory control.

    Prints a summary table and saves 'task5_validation.png'.

    Returns the per-(lambda,gamma,phi) results as a DataFrame.
    """
    df = T3._load_raw()
    stream = _prepare_stream(df)

    lam_grid = [0.3, 0.6, 0.9]
    gamma_grid = [0.5, 1.0, 2.0]
    phi_grid = [1e-7, 1e-6, 1e-5]

    rows = []
    for lam in lam_grid:
        for gamma in gamma_grid:
            for phi in phi_grid:
                sc, pnl, eod, dd = [], [], [], []
                bsc, bpnl = [], []
                for sd in seeds:
                    res = backtest(stream, lam, gamma, phi, params=params, seed=sd)
                    sc.append(_score(res["day_pnls"]))
                    pnl.append(res["total_pnl"])
                    eod.append(np.mean(np.abs(res["eod_inv"])))
                    dd.append(_max_drawdown(res["day_pnls"]))
                    bdays = _backtest_baseline(stream, lam, gamma, phi, seed=sd)
                    bsc.append(_score(bdays)); bpnl.append(float(bdays.sum()))
                rows.append({
                    "lambda": lam, "gamma": gamma, "phi": phi,
                    "score": np.mean(sc), "total_pnl": np.mean(pnl),
                    "mean_abs_eod_inv": np.mean(eod), "max_drawdown": np.mean(dd),
                    "baseline_score": np.mean(bsc), "baseline_pnl": np.mean(bpnl),
                })
    out = pd.DataFrame(rows)

    if verbose:
        pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
        print("Validation across hidden (lambda, gamma, phi) grid "
              f"[{len(lam_grid)}x{len(gamma_grid)}x{len(phi_grid)} = {len(out)} regimes, "
              f"{len(seeds)} seeds each]:\n")
        print(out.to_string(index=False))
        print("\nSummary:")
        print(f"  strategy  mean score = {out['score'].mean():,.2f}  "
              f"(median {out['score'].median():,.2f}, min {out['score'].min():,.2f})")
        print(f"  baseline  mean score = {out['baseline_score'].mean():,.2f}")
        win = (out['score'] > out['baseline_score']).mean() * 100
        print(f"  strategy beats baseline in {win:.0f}% of regimes")
        print(f"  mean total PnL = {out['total_pnl'].mean():,.0f}  "
              f"(baseline {out['baseline_pnl'].mean():,.0f})")
        print(f"  mean |EOD inventory| = {out['mean_abs_eod_inv'].mean():,.0f} units")

    if save_fig:
        _make_validation_fig(stream, out)
    return out


def _make_validation_fig(stream, out, path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task5_validation.png")

    # representative mid-regime run for the PnL / inventory illustration
    res = backtest(stream, lam=0.6, gamma=1.0, phi=1e-6, seed=0)
    base = _backtest_baseline(stream, 0.6, 1.0, 1e-6, seed=0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    ax.plot(np.cumsum(res["day_pnls"]), marker="o", ms=3, label="strategy")
    ax.plot(np.cumsum(base), marker="s", ms=3, alpha=0.7, label="fixed-spread baseline")
    ax.set_title("Cumulative daily PnL (lam=0.6, gamma=1.0, phi=1e-6)")
    ax.set_xlabel("trading day"); ax.set_ylabel("cumulative PnL"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.plot(res["inv_path"][:8000])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Inventory path (first 8k trades) - bounded by skew")
    ax.set_xlabel("trade"); ax.set_ylabel("inventory"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    piv = out.pivot_table(index="gamma", columns="lambda", values="score", aggfunc="mean")
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("lambda"); ax.set_ylabel("gamma")
    ax.set_title("Sharpe-like score across regimes (avg over phi)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    x = np.arange(len(out))
    ax.plot(x, out["score"].values, marker="o", ms=3, label="strategy")
    ax.plot(x, out["baseline_score"].values, marker="s", ms=3, alpha=0.7, label="baseline")
    ax.set_title("Score: strategy vs baseline, every regime")
    ax.set_xlabel("regime index (lambda,gamma,phi)"); ax.set_ylabel("score")
    ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    # sanity demo of the quoting function
    print("quote() demo (sigma=0.03):")
    for I in (-2000, 0, 2000):
        for a in (0.1, 0.6):
            for e in (0.1, 0.9):
                db, da = quote(I, 0.03, a, e)
                print(f"  I={I:6d} alpha={a:.1f} eta={e:.1f} -> "
                      f"delta_bid={db:.4f} delta_ask={da:.4f}")
    print()
    validate_quote()
