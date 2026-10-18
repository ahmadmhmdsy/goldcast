"""Research dashboard over the cached R-002..R-005 forecasts.

Run: streamlit run src/goldcast/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from goldcast.config import load_config
from goldcast.dashboard import data as D

st.set_page_config(page_title="Gold forecast research", layout="wide")
cfg = load_config()


@st.cache_data
def forecasts(freq):
    return D.load(freq)


@st.cache_data
def panel(freq):
    from goldcast.data.store import load_raw
    from goldcast.features.build import build_panel

    raw = load_raw(cfg)
    X, _ = build_panel(raw, cfg.base_series, cfg.freqs[freq], cfg.target)
    return X, raw[cfg.target]


# ---- sidebar
st.sidebar.title("Gold forecast research")
freq = st.sidebar.radio("Frequency", ["weekly", "monthly"], horizontal=True)
fc = forecasts(freq)
lo, hi = fc.y.index.min().date(), fc.y.index.max().date()
window = st.sidebar.radio("Window", ["full test (2015+)", "2023+ regime", "custom"])
if window == "custom":
    d0, d1 = st.sidebar.slider("Dates", lo, hi, (lo, hi))
elif window == "2023+ regime":
    d0, d1 = pd.Timestamp(cfg.regime_start).date(), hi
else:
    d0, d1 = lo, hi
mask = (fc.y.index >= pd.Timestamp(d0)) & (fc.y.index <= pd.Timestamp(d1))
st.sidebar.caption(f"{int(mask.sum())} origins, {d0} .. {d1}. Target: next-period gold log return.")
st.sidebar.warning("No model has beaten the random walk (points) or GARCH (intervals) after "
                   "multiple-testing correction in any study. Treat rankings as noise until a gate passes.")

pb = D.point_board(fc, mask, cfg.regime_start)
ib = D.interval_board(fc, mask, cfg.regime_start)

tab_lb, tab_fc, tab_err, tab_in, tab_next = st.tabs(["Leaderboard", "Forecast vs actual", "Error over time", "Input data", "Next period"])


def _style(df, ratio_col):
    def row(r):
        if r["rank"] == 1:
            return ["background-color: rgba(212,175,55,0.25); font-weight: bold"] * len(r)
        if r["model"] in ("rw", "garch"):
            return ["background-color: rgba(128,128,128,0.15)"] * len(r)
        return [""] * len(r)

    show = df.copy()
    show.insert(1, "", np.where(show["rank"] == 1, "🥇", np.where(show["rank"] == 2, "🥈", np.where(show["rank"] == 3, "🥉", ""))))
    show["beats ref"] = show["beats ref"].map({True: "✅", False: "—"})
    show["significant"] = show["significant"].map({True: "✅", False: "—"})
    if "coverage ok" in show:
        show["coverage ok"] = show["coverage ok"].map({True: "✅", False: "⚠️"})
    show["about"] = show["model"].map(lambda m: D.MODELS.get(m, ("", ""))[1])
    fmt = {c: "{:.4f}" for c in show.select_dtypes("float").columns}
    fmt.update({"hit": "{:.3f}", "coverage 80%": "{:.3f}"} if "hit" in show else {"coverage 80%": "{:.3f}"})
    return show.style.apply(row, axis=1).format(fmt, na_rep="ref")


with tab_lb:
    st.subheader("Point forecasts — ranked by MSE relative to the random walk")
    st.caption("Below 1 = lower squared error than forecasting zero. 'significant' = one-sided Diebold-Mariano "
               "vs the random walk, Holm-adjusted over the models shown, p < 0.05.")
    st.dataframe(_style(pb, "mse/rw"), hide_index=True, use_container_width=True)
    st.subheader("80% intervals — ranked by pinball loss relative to GARCH")
    st.caption("Below 1 = sharper and/or better calibrated than GARCH(1,1)-t. Coverage should sit in [0.75, 0.85].")
    st.dataframe(_style(ib, "pinball/garch"), hide_index=True, use_container_width=True)
    with st.expander("Pre-registered verdicts"):
        st.markdown("""
| study | question | verdict |
|---|---|---|
| R-002 | Do cross-asset/macro features beat a random walk? | refuted (weekly interval edge only, later explained by vol tracking) |
| R-003 | Are R-002's intervals just volatility tracking? | partial — GARCH/EWMA match or beat every learner |
| R-004 | Do flows, demand, uncertainty or attention blocks help? | refuted |
| R-005 | Do foundation models (TabPFN-2, TabFM, Chronos-2+cov) help? | refuted, 0/18 gates |

Scores on this page are recomputed for the chosen window; they match the reports on the full window only,
and the Holm family here is the models shown, not each study's frozen family.
""")

with tab_fc:
    c1, c2, c3 = st.columns([2, 1, 1])
    default = [m for m in [pb["model"].iloc[0], "lgbm", "tabpfn"] if m in fc.point]
    pick = c1.multiselect("Point models", list(fc.point), default=list(dict.fromkeys(default)))
    band = c2.selectbox("Interval band", ["none", *fc.quant], index=1 + list(fc.quant).index("garch"))
    view = c3.radio("Show as", ["return", "price"], horizontal=True)
    y = fc.y[mask]
    fig = go.Figure()
    if view == "return":
        if band != "none":
            q = fc.quant[band][mask]
            fig.add_trace(go.Scatter(x=q.index, y=q[0.9], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=q.index, y=q[0.1], fill="tonexty", line=dict(width=0),
                                     fillcolor="rgba(212,175,55,0.25)", name=f"{band} 80% band"))
        fig.add_trace(go.Bar(x=y.index, y=y, name="actual return", marker_color="rgba(120,120,120,0.6)"))
        for m in pick:
            fig.add_trace(go.Scatter(x=y.index, y=fc.point[m][mask], name=m, mode="lines"))
        fig.update_yaxes(title="log return", tickformat=".1%")
    else:
        _, price = panel(freq)
        p0 = price.reindex(y.index, method="ffill")
        nxt = y.index[1:].append(pd.DatetimeIndex([y.index[-1] + (y.index[-1] - y.index[-2])]))
        actual = p0 * np.exp(y)
        if band != "none":
            q = fc.quant[band][mask]
            fig.add_trace(go.Scatter(x=nxt, y=p0 * np.exp(q[0.9]), line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=nxt, y=p0 * np.exp(q[0.1]), fill="tonexty", line=dict(width=0),
                                     fillcolor="rgba(212,175,55,0.25)", name=f"{band} 80% band"))
        fig.add_trace(go.Scatter(x=nxt, y=actual, name="actual price", line=dict(color="goldenrod", width=3)))
        for m in pick:
            fig.add_trace(go.Scatter(x=nxt, y=p0 * np.exp(fc.point[m][mask]), name=f"{m} forecast", mode="lines"))
        fig.update_yaxes(title="gold price (forecast for the next period)")
    fig.update_layout(height=520, hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    if band != "none":
        q = fc.quant[band][mask]
        inside = (y >= q[0.1]) & (y <= q[0.9])
        st.caption(f"{band}: actual inside the 80% band {inside.mean():.1%} of the time in this window.")

with tab_err:
    kind = st.radio("Compare", ["point models vs random walk", "intervals vs GARCH"], horizontal=True)
    if kind.startswith("point"):
        ms = st.multiselect("Models", [m for m in fc.point if m != "rw"], default=[m for m in ["lgbm", "tabpfn", "drift"] if m in fc.point])
        fn, ref = D.cumulative_gain, "random walk"
    else:
        ms = st.multiselect("Models", [m for m in fc.quant if m != "garch"], default=[m for m in ["ewma", "tabpfn", "lgbm_q"] if m in fc.quant])
        fn, ref = D.cumulative_gain_interval, "GARCH"
    fig = go.Figure()
    for m in ms:
        s = fn(fc, m, mask)
        fig.add_trace(go.Scatter(x=s.index, y=s, name=m, mode="lines"))
    fig.add_hline(y=0, line_dash="dot")
    rs = pd.Timestamp(cfg.regime_start)
    if pd.Timestamp(d0) < rs < pd.Timestamp(d1):
        fig.add_vline(x=rs, line_dash="dash", line_color="goldenrod")
        fig.add_annotation(x=rs, y=1, yref="paper", text="2023+ regime", showarrow=False, xanchor="left")
    fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h"),
                      yaxis_title=f"cumulative loss saved vs {ref}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Rising line = the model is gaining on the {ref}; falling = losing. "
               "A real edge shows as a steady climb, not a few jumps.")
    if ms and kind.startswith("point"):
        roll = st.slider("Rolling window (periods)", 12, 156 if freq == "weekly" else 60, 52 if freq == "weekly" else 24)
        y = fc.y[mask]
        fig2 = go.Figure()
        for m in ms:
            r = ((y - fc.point[m][mask]) ** 2).rolling(roll).mean() / (y**2).rolling(roll).mean()
            fig2.add_trace(go.Scatter(x=r.index, y=r, name=m, mode="lines"))
        fig2.add_hline(y=1, line_dash="dot")
        fig2.update_layout(height=360, yaxis_title=f"rolling MSE / rw ({roll})", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

with tab_in:
    X, price = panel(freq)
    X = X.loc[: fc.y.index.max()]
    dates = [d.date() for d in X.index[(X.index >= pd.Timestamp(d0)) & (X.index <= pd.Timestamp(d1))]]
    when = pd.Timestamp(st.select_slider("Forecast origin", options=dates, value=dates[-1]))
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**Inputs the models saw at {when.date()}**")
        hist = X.loc[:when]
        row = X.loc[when]
        z = (row - hist.mean()) / hist.std()
        tbl = pd.DataFrame({"value": row, "z vs history": z}).sort_values("z vs history", key=abs, ascending=False)
        st.dataframe(tbl.style.format("{:.4f}").background_gradient(subset=["z vs history"], cmap="RdBu_r", vmin=-3, vmax=3),
                     use_container_width=True, height=480)
    with right:
        st.markdown(f"**Forecasts made at {when.date()}**")
        if when in fc.y.index:
            actual = fc.y.loc[when]
            f = pd.DataFrame({"forecast": fc.point.loc[when]}).dropna()
            f["error"] = actual - f["forecast"]
            f["right direction"] = np.sign(f["forecast"]) == np.sign(actual)
            st.metric("actual next-period return", f"{actual:.2%}")
            st.dataframe(f.sort_values("error", key=abs).style.format({"forecast": "{:.2%}", "error": "{:.2%}"}),
                         use_container_width=True)
            qs = pd.DataFrame({m: q.loc[when] for m, q in fc.quant.items()}).T
            qs.columns = ["q10", "q50", "q90"]
            st.dataframe(qs.style.format("{:.2%}"), use_container_width=True)
        else:
            st.info("Not a test origin (before 2015 or target not yet known).")
    st.markdown("**Series history**")
    cols = st.multiselect("Features", list(X.columns), default=[c for c in X.columns if c.startswith("gold_")][:2])
    if cols:
        norm = st.checkbox("z-score each series", value=len(cols) > 1)
        fig = go.Figure()
        for c in cols:
            s = X[c]
            fig.add_trace(go.Scatter(x=s.index, y=(s - s.mean()) / s.std() if norm else s, name=c, mode="lines"))
        fig.add_vline(x=when, line_dash="dash", line_color="goldenrod")
        fig.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

with tab_next:
    st.subheader(f"Next {freq[:-2]} forecast")
    st.warning("Research output, not advice. None of these point models beat the random walk in backtests; "
               "the GARCH/EWMA range is the only piece that held up (calibrated ~80%).")
    st.caption("Refits the cheap models on the stored data (run `goldcast fetch` first for fresh prices). "
               "Foundation models are left out: they take minutes to hours per fit.")
    if st.button("Refit and forecast", type="primary"):
        with st.spinner("Fitting..."):
            st.session_state[f"next_{freq}"] = D.next_period(cfg, freq)
    r = st.session_state.get(f"next_{freq}")
    if r:
        p0 = r["price"]
        st.markdown(f"Origin **{r['origin'].date()}** (last stored price {r['data_last'].date()}), "
                    f"gold **{p0:,.2f}**. Forecast is for the {freq[:-2]} after the origin.")
        g = r["quant"]["garch"]
        c = st.columns(4)
        for col, (lab, v) in zip(c, [("GARCH low (10%)", g[0]), ("GARCH median", g[1]), ("GARCH high (90%)", g[2])]):
            col.metric(lab, f"{p0 * np.exp(v):,.0f}", f"{v:+.2%}")
        mv = r.get("gvz_move", np.nan)
        if np.isfinite(mv):
            c[3].metric("GVZ expected move (1σ)", f"±{p0 * mv:,.0f}", f"±{mv:.2%}", delta_color="off",
                        help=f"GVZ {r['gvz']:.1f} at the origin, scaled to the period. Best size-of-move forecast "
                             "in R-006/R-007 (QLIKE), but its ranges run wide, so the range above stays GARCH.")
        rows = [{"model": m, "kind": "point", "return": v, "price": p0 * np.exp(v)} for m, v in r["point"].items()]
        rows += [{"model": m, "kind": "80% range", "return": q[1], "price": p0 * np.exp(q[1]),
                  "low": p0 * np.exp(q[0]), "high": p0 * np.exp(q[2])} for m, q in r["quant"].items()]
        rank = dict(zip(pb["model"], pb["rank"])) | dict(zip(ib["model"], ib["rank"]))
        t = pd.DataFrame(rows)
        t["backtest rank"] = [rank.get(m) for m in t["model"]]
        st.dataframe(t.style.format({"return": "{:+.2%}", "price": "{:,.0f}", "low": "{:,.0f}", "high": "{:,.0f}",
                                     "backtest rank": "{:.0f}"}, na_rep=""), hide_index=True, use_container_width=True)
        fig = go.Figure()
        for m, q in r["quant"].items():
            fig.add_trace(go.Scatter(x=p0 * np.exp(q), y=[m] * 3, mode="lines+markers", name=m,
                                     marker=dict(size=[8, 14, 8])))
        fig.add_vline(x=p0, line_dash="dot", annotation_text="today")
        fig.update_layout(height=320, showlegend=False, xaxis_title="gold price, 10% / 50% / 90%")
        st.plotly_chart(fig, use_container_width=True)
