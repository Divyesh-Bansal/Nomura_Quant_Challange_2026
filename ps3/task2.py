"""
Task 2: Client Profitability and Spread Recommendation

Everything is from the LP's perspective: side = +1 if we buy, -1 if we sell.

Per-trade PnL closed at horizon tau (Corollary 1, eq. 5):
    PnL(tau) = side * V * (M_tau - T_P)

Per-trade aggregate PnL, uniform closing weights (Corollary 2, eq. 6):
    AggPnL = side * (1/6) * sum_{tau in 5..30} V * (M_tau - T_P)

"Expected" means the sample average over a client's trades.

Minimum half-spread delta*:
If instead of T_P we had quoted M0 +/- delta, we'd fill at the bid M0 - delta
when buying and the ask M0 + delta when selling. Both give an effective price
T_P^eff = M0 - side*delta. Plugging into eq. 6 (side^2 = 1):

    AggPnL(delta) = side * V * (Mbar - M0) + V * delta,   Mbar = mean(M5..M30)

Averaging and requiring it to be non-negative:
    E[AggPnL(delta)] = A + delta*E[V] >= 0,   A = E[side*V*(Mbar - M0)]
    => delta* = max(0, -A / E[V])

A is the PnL we'd earn quoting at the mid; it goes negative exactly when the
client adversely selects us, so delta* is the spread that offsets that cost.

Submission functions:
    expected_pnl(client, tau) -> dict
    classify_client(client) -> str
    min_half_spread(client) -> float
"""

import os
from typing import List

import numpy as np
import pandas as pd


# Data loading, cached.
_DATA_CACHE = {}

_ALL_TAU = [5, 10, 15, 20, 25, 30]
_MID_COL = {5: "M5", 10: "M10", 15: "M15", 20: "M20", 25: "M25", 30: "M30"}


def _data_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "trade_data.csv"),
        os.path.join(os.path.dirname(here), "trade_data.csv"),
        "trade_data.csv",
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("trade_data.csv not found.")


def _load() -> pd.DataFrame:
    if "df" not in _DATA_CACHE:
        _DATA_CACHE["df"] = pd.read_csv(_data_path())
    return _DATA_CACHE["df"]


def expected_pnl(client: str, tau: List[int]) -> dict:
    """
    Parameters
    ----------
    client : str
        Client identifier.
    tau : List[int]
        Horizons in seconds, e.g. [5, 10, 15, 20, 25, 30].

    Returns
    -------
    dict
        {
          'per_horizon': List[float]  expected PnL per trade at each tau (eq. 5),
          'aggregate'  : float        expected aggregate PnL per trade (eq. 6,
                                       uniform weights over ALL six horizons).
        }
    """
    df = _load()
    sub = df[df["Name"] == client]

    side = sub["Side"].to_numpy(dtype=float)
    vol = sub["Volume"].to_numpy(dtype=float)
    tp = sub["Trade Price"].to_numpy(dtype=float)

    per_horizon: List[float] = []
    for t in tau:
        col = _MID_COL.get(int(t))
        if col is None:
            raise ValueError(f"Unsupported horizon tau={t}; expected one of {_ALL_TAU}")
        m_tau = sub[col].to_numpy(dtype=float)
        pnl = side * vol * (m_tau - tp)
        per_horizon.append(float(np.mean(pnl)) if len(sub) else float("nan"))

    # Aggregate (eq. 6) always uses the full set of six horizons with weight 1/6.
    mids = np.column_stack([sub[_MID_COL[t]].to_numpy(dtype=float) for t in _ALL_TAU])
    agg_per_trade = side * vol * (mids.mean(axis=1) - tp)
    aggregate = float(np.mean(agg_per_trade)) if len(sub) else float("nan")

    return {"per_horizon": per_horizon, "aggregate": aggregate}


def classify_client(client: str) -> str:
    """
    Returns 'profitable' or 'costly' based on the expected aggregate PnL (eq. 6).
    Net-profitable to trade with  <=>  expected aggregate PnL >= 0.
    """
    agg = expected_pnl(client, _ALL_TAU)["aggregate"]
    return "profitable" if agg >= 0 else "costly"


def min_half_spread(client: str) -> float:
    """
    Minimum half-spread delta* (in data/price units) such that, had we quoted at
    M0 +/- delta* for all of this client's trades, the expected aggregate PnL per
    trade (eq. 6) would be non-negative.

        delta* = max( 0, -A / E[V] ),
        A = E[ side * V * (Mbar - M0) ],  Mbar = mean of M5..M30.
    """
    df = _load()
    sub = df[df["Name"] == client]
    if len(sub) == 0:
        return float("nan")

    side = sub["Side"].to_numpy(dtype=float)
    vol = sub["Volume"].to_numpy(dtype=float)
    m0 = sub["M0"].to_numpy(dtype=float)
    mids = np.column_stack([sub[_MID_COL[t]].to_numpy(dtype=float) for t in _ALL_TAU])
    mbar = mids.mean(axis=1)

    a = float(np.mean(side * vol * (mbar - m0)))   # aggregate PnL at zero spread
    ev = float(np.mean(vol))
    delta_star = -a / ev
    return float(max(0.0, delta_star))


# Write out task2_results.csv for all clients.
def _build_results_csv(path: str = "task2_results.csv") -> pd.DataFrame:
    df = _load()
    clients = sorted(df["Name"].unique())

    rows = []
    for c in clients:
        ep = expected_pnl(c, _ALL_TAU)
        rows.append(
            [c] + ep["per_horizon"] + [ep["aggregate"], min_half_spread(c)]
        )

    cols = (
        ["client"]
        + [f"tau={t}" for t in _ALL_TAU]
        + ["agg_pnl", "delta_star"]
    )
    res = pd.DataFrame(rows, columns=cols)
    res.to_csv(path, index=False)
    return res


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    res = _build_results_csv(os.path.join(out_dir, "task2_results.csv"))
    pd.set_option("display.float_format", lambda v: f"{v:.6f}")
    print(res.to_string(index=False))
    print()
    for c in sorted(_load()["Name"].unique()):
        print(f"Client {c}: {classify_client(c):>10s}   delta* = {min_half_spread(c):.6f}")
