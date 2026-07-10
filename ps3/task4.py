"""
Task 4: Optimal Externalization Threshold

For a horizon tau, every trade is assumed closed at t = tau, so its internalized
PnL (LP perspective) is

    pnl_i(tau) = side_i * V_i * (M_tau,i - T_P,i).

Externalization (Section 3.1): from the Task-3 model M we get a per-trade adverse
probability p_i = predict_adversity(client_i, tau). We externalize iff p_i > theta;
an externalized trade nets to zero for us, so the strategy PnL sums over the
internalized trades only (eq. 8):

    PnL(theta) = sum_{i : p_i <= theta} pnl_i(tau).

I sweep theta in [0,1], pick theta* maximizing PnL on the validation set, and
report PnL on the held-out test set at theta*.

Structural note: the Task-3 model uses only client identity and tau, so p_i is
constant within a (client, tau) cell. "Externalize iff p > theta" then reduces to
choosing which CLIENTS to internalize, and since PnL falls monotonically as p
rises, a single global theta selects a prefix of clients ordered by p. This global
optimum coincides exactly with the client-specific optimum here, so a global theta
is sufficient; client-specific thresholds are the general answer.

Required interface:
    optimal_threshold(tau, mode='global'|'client', ...) -> dict
    plot_pnl_vs_theta(tau=ALL_TAU, ...) -> None     (saves pnl_vs_theta.png)
"""

import os
from typing import List, Union

import numpy as np
import pandas as pd

import task3 as T  # reuse the trained model, split, labels and predict_adversity


ALL_TAU = T.ALL_TAU
CLIENTS = T.CLIENTS
MID_COL = T.MID_COL

# theta grid for the sweep; fine enough to resolve the (well-separated) client p's
THETA_GRID = np.linspace(0.0, 1.0, 2001)


# Helpers
def _splits():
    T._ensure_trained()
    return T._STATE["splits"]


def _trade_pnl(df: pd.DataFrame, tau: int) -> np.ndarray:
    """Internalized PnL per trade at horizon tau:  side * V * (M_tau - T_P)."""
    side = df["Side"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    m_tau = df[MID_COL[tau]].to_numpy(dtype=float)
    tp = df["Trade Price"].to_numpy(dtype=float)
    return side * vol * (m_tau - tp)


def _trade_p(df: pd.DataFrame, tau: int) -> np.ndarray:
    """Model adverse probability per trade (constant within a client at given tau)."""
    p_by_client = {c: T.predict_adversity(client=c, tau=tau) for c in CLIENTS}
    return df["Name"].map(p_by_client).to_numpy(dtype=float)


def _pnl_at_theta(df: pd.DataFrame, tau: int, theta: float) -> float:
    """Strategy PnL on df at horizon tau and threshold theta (eq. 8)."""
    p = _trade_p(df, tau)
    pnl = _trade_pnl(df, tau)
    internalize = p <= theta          # externalize iff p > theta
    return float(pnl[internalize].sum())


def _pnl_curve(df: pd.DataFrame, tau: int, grid: np.ndarray = THETA_GRID) -> np.ndarray:
    """Vectorised PnL(theta) over the grid."""
    p = _trade_p(df, tau)
    pnl = _trade_pnl(df, tau)
    order = np.argsort(p)
    p_sorted = p[order]
    pnl_sorted = pnl[order]
    csum = np.concatenate([[0.0], np.cumsum(pnl_sorted)])
    # for each theta: number of trades with p <= theta = searchsorted(p_sorted, theta, 'right')
    k = np.searchsorted(p_sorted, grid, side="right")
    return csum[k]


def _argmax_theta(curve: np.ndarray, grid: np.ndarray = THETA_GRID) -> float:
    """theta maximizing the curve; ties -> midpoint of the optimal plateau (robust)."""
    best = curve.max()
    idx = np.flatnonzero(np.isclose(curve, best, rtol=0, atol=1e-9))
    # midpoint of the longest contiguous run of optimal indices
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    longest = max(runs, key=len)
    mid = longest[len(longest) // 2]
    return float(grid[mid])


# Client-specific optimum
def _client_threshold(tau: int):
    """
    Per-client optimal threshold on the validation set.

    For client c at tau, p_c is constant, so PnL_c(theta) = val_pnl_c if
    theta >= p_c (internalize) else 0 (externalize).  Maximising:
      * profitable client (val_pnl_c > 0): keep -> smallest theta >= p_c (~= p_c);
      * costly client      (val_pnl_c <= 0): externalize -> theta = 0.

    Returns {client: theta*}, and the keep/externalize decision per client.
    """
    val = _splits()["validation"]
    thetas, decision = {}, {}
    for c in CLIENTS:
        g = val[val["Name"] == c]
        p_c = T.predict_adversity(client=c, tau=tau)
        val_pnl_c = float(_trade_pnl(g, tau).sum())
        if val_pnl_c > 0:
            # smallest grid theta >= p_c
            cand = THETA_GRID[THETA_GRID >= p_c]
            thetas[c] = float(cand[0]) if len(cand) else 1.0
            decision[c] = "internalize"
        else:
            thetas[c] = 0.0
            decision[c] = "externalize"
    return thetas, decision


def _pnl_under_client_thresholds(df: pd.DataFrame, tau: int, thetas: dict) -> float:
    """Total PnL on df where each trade uses its client's threshold."""
    total = 0.0
    for c in CLIENTS:
        g = df[df["Name"] == c]
        if len(g) == 0:
            continue
        p_c = T.predict_adversity(client=c, tau=tau)
        if p_c <= thetas[c]:                      # internalize this client
            total += float(_trade_pnl(g, tau).sum())
    return total


# Required: optimal_threshold
def optimal_threshold(tau: int, mode: str = "global", **kwargs) -> dict:
    """
    Optimal externalization threshold for a given horizon tau.

    Parameters
    ----------
    tau : int
        Horizon in {5, 10, 15, 20, 25, 30}.
    mode : {'global', 'client'}
        'global'  -> a single theta shared by all clients (default);
        'client'  -> a client-specific dict {client: theta}.

    Returns
    -------
    dict with keys:
        'theta'          : float (global)  or  Dict[str, float] (client-specific)
        'validation_pnl' : float  PnL at theta* on the validation set
        'test_pnl'       : float  PnL at theta* on the held-out test set
    """
    tau = int(tau)
    if tau not in MID_COL:
        raise ValueError(f"Unsupported horizon tau={tau}; expected one of {ALL_TAU}.")
    sp = _splits()
    val, test = sp["validation"], sp["test"]

    if mode == "global":
        curve = _pnl_curve(val, tau)
        theta_star = _argmax_theta(curve)
        return {
            "theta": theta_star,
            "validation_pnl": _pnl_at_theta(val, tau, theta_star),
            "test_pnl": _pnl_at_theta(test, tau, theta_star),
        }
    elif mode == "client":
        thetas, _ = _client_threshold(tau)
        return {
            "theta": thetas,
            "validation_pnl": _pnl_under_client_thresholds(val, tau, thetas),
            "test_pnl": _pnl_under_client_thresholds(test, tau, thetas),
        }
    else:
        raise ValueError("mode must be 'global' or 'client'.")


# Required: plot_pnl_vs_theta
def plot_pnl_vs_theta(tau: Union[int, List[int]] = None,
                      path: str = None, **kwargs) -> None:
    """
    Plot PnL_validation(theta) for theta in [0, 1] and save to 'pnl_vs_theta.png'.

    Parameters
    ----------
    tau : int or list of int, optional
        Horizon(s) to plot.  Defaults to all six horizons on one figure.
    path : str, optional
        Output path; defaults to 'pnl_vs_theta.png' next to this file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if tau is None:
        taus = list(ALL_TAU)
    elif isinstance(tau, (list, tuple, np.ndarray)):
        taus = [int(t) for t in tau]
    else:
        taus = [int(tau)]

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pnl_vs_theta.png")

    val = _splits()["validation"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in taus:
        curve = _pnl_curve(val, t)
        ax.plot(THETA_GRID, curve, label=f"tau={t}")
        ts = _argmax_theta(curve)
        ax.scatter([ts], [_pnl_at_theta(val, t, ts)], s=30, zorder=5)
    ax.set_xlabel("Threshold theta")
    ax.set_ylabel("Validation PnL(theta)")
    ax.set_title("Validation PnL vs externalization threshold theta\n(markers = optimal theta*)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


# Results CSV: per client, per tau, client-specific theta* and test PnL.
def _build_results_csv(path: str = "task4_results.csv") -> pd.DataFrame:
    sp = _splits()
    test = sp["test"]
    rows = []
    for tau in ALL_TAU:
        thetas, decision = _client_threshold(tau)
        for c in CLIENTS:
            g = test[test["Name"] == c]
            p_c = T.predict_adversity(client=c, tau=tau)
            keep = p_c <= thetas[c]
            final_pnl = float(_trade_pnl(g, tau).sum()) if keep else 0.0
            rows.append([c, tau, thetas[c], final_pnl])
    res = pd.DataFrame(rows, columns=["client", "tau", "theta_star", "final_pnl"])
    res = res.sort_values(["client", "tau"]).reset_index(drop=True)
    res.to_csv(path, index=False)
    return res


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Global optimal thresholds per horizon:")
    for tau in ALL_TAU:
        g = optimal_threshold(tau, mode="global")
        c = optimal_threshold(tau, mode="client")
        print(f"  tau={tau:2d}: global theta*={g['theta']:.4f}  "
              f"val={g['validation_pnl']:10.1f}  test={g['test_pnl']:10.1f}  "
              f"|| client-mode val={c['validation_pnl']:10.1f}  test={c['test_pnl']:10.1f}")

    res = _build_results_csv(os.path.join(out_dir, "task4_results.csv"))
    plot_pnl_vs_theta(path=os.path.join(out_dir, "pnl_vs_theta.png"))
    print("\ntask4_results.csv:")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(res.to_string(index=False))
