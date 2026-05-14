import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required

import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt # type: ignore
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
USE_LIVE_DATA = True    # set False to use synthetic data instead
SYMBOL = "BTC/USDT"
TIMEFRAME = "1d"
LIMIT = 1500            # total candles to fetch (~4 years of daily data)
INITIAL_CAPITAL = 10_000.0
FAST_EMA = 12
SLOW_EMA = 26
SIGNAL_EMA = 9          # for MACD signal line
TREND_EMA = 200         # macro trend filter
RSI_LOW = 40            # RSI safe zone lower bound  (longs)
RSI_HIGH = 65           # RSI safe zone upper bound  (longs)
RSI_SHORT_LOW = 35      # RSI safe zone lower bound  (shorts)
RSI_SHORT_HIGH = 60     # RSI safe zone upper bound  (shorts)
# ── Risk Management Config ────────────────────────────────────────────────────
POSITION_SIZE  = 0.25   # Deploy only 25% of capital per trade
STOP_LOSS_PCT  = 0.05   # Close position if trade loses more than 5%

# ── Fetch OHLCV data via ccxt ─────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    if not USE_LIVE_DATA:
        print("USE_LIVE_DATA=False → using synthetic BTC-like data.")
        return _generate_synthetic_ohlcv(limit)
    try:
        exchange = ccxt.mexc({"enableRateLimit": True, "timeout": 10000})
        batch_size = 500
        tf_ms = exchange.parse_timeframe(timeframe) * 1000
        all_candles = []
        since = None

        # paginate backwards until we have enough candles
        while len(all_candles) < limit:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
            if not batch:
                break
            # prepend older batch
            all_candles = batch + all_candles
            since = batch[0][0] - batch_size * tf_ms   # step back one window
            if len(batch) < batch_size:
                break   # exchange has no more history

        all_candles = all_candles[-limit:]   # keep most recent `limit` candles
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        print(f"Live data fetched from {exchange.id} ({len(df)} candles).")
        return df
    except Exception as e:
        print(f"Network error ({type(e).__name__}). Falling back to synthetic data.")
        return _generate_synthetic_ohlcv(limit)


def _generate_synthetic_ohlcv(n: int) -> pd.DataFrame:
    """Geometric Brownian Motion with realistic BTC vol (~3% daily)."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D")
    dt = 1 / 252
    mu, sigma = 0.6, 0.55          # annualised drift & vol
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n)
    prices = 20_000 * np.exp(np.cumsum(log_returns))

    noise = np.random.uniform(0.005, 0.025, n)
    opens   = prices * (1 - noise / 2)
    highs   = prices * (1 + noise)
    lows    = prices * (1 - noise)
    volumes = np.random.uniform(1_000, 50_000, n)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
        index=dates,
    )


# ── Add indicators ────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Trend filter
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=TREND_EMA).ema_indicator()

    # MACD
    macd = ta.trend.MACD(df["close"], window_fast=FAST_EMA, window_slow=SLOW_EMA, window_sign=SIGNAL_EMA)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    return df.dropna()


# ── Triple-condition strategy (long + short) ──────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signal"] = 0

    macd_cross_up   = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    macd_cross_down = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))

    # ── Long entry: all 3 bull conditions ──
    above_trend = df["close"] > df["ema_200"]
    rsi_long    = (df["rsi"] > RSI_LOW) & (df["rsi"] < RSI_HIGH)
    df.loc[macd_cross_up & above_trend & rsi_long, "signal"] = 1

    # ── Short entry: all 3 bear conditions ──
    below_trend = df["close"] < df["ema_200"]
    rsi_short   = (df["rsi"] > RSI_SHORT_LOW) & (df["rsi"] < RSI_SHORT_HIGH)
    df.loc[macd_cross_down & below_trend & rsi_short, "signal"] = -1

    return df


# ── Backtest engine ───────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    capital = initial_capital
    position = 0.0      # +units = long BTC, -units = short BTC
    entry_price = 0.0
    trades = []
    equity_curve = []

    for ts, row in df.iterrows():
        price = row["close"]
        sig   = row["signal"]

        # ── Stop-loss check — runs before signal logic ────────────────────────
        if position > 0:
            loss_pct = (price - entry_price) / entry_price
            if loss_pct <= -STOP_LOSS_PCT:
                pnl      = (price - entry_price) * position
                capital += position * price          # add sale proceeds to cash
                trades.append({
                    "date":  ts,
                    "type":  "STOP",
                    "price": price,
                    "units": position,
                    "pnl":   pnl
                })
                position = 0.0
                equity_curve.append({"date": ts, "equity": capital, "price": price})
                continue

        elif position < 0:
            loss_pct = (price - entry_price) / entry_price
            if loss_pct >= STOP_LOSS_PCT:
                pnl      = (entry_price - price) * abs(position)
                capital += abs(position) * entry_price + pnl
                trades.append({
                    "date":  ts,
                    "type":  "STOP_SHORT",
                    "price": price,
                    "units": abs(position),
                    "pnl":   pnl
                })
                position = 0.0
                equity_curve.append({"date": ts, "equity": capital, "price": price})
                continue

        # ── Close any open position first ──
        if sig == 1 and position < 0:
            # cover short
            pnl      = (entry_price - price) * abs(position)
            capital += abs(position) * entry_price + pnl
            trades.append({"date": ts, "type": "COVER", "price": price, "units": abs(position), "pnl": pnl})
            position = 0.0

        elif sig == -1 and position > 0:
            # exit long: add sale proceeds to remaining cash
            pnl      = (price - entry_price) * position
            capital += position * price
            trades.append({"date": ts, "type": "SELL", "price": price, "units": position, "pnl": pnl})
            position = 0.0

        # ── Open new position ──
        if sig == 1 and position == 0:
            trade_capital = capital * POSITION_SIZE
            position      = trade_capital / price
            entry_price   = price
            capital      -= trade_capital            # remaining capital stays in cash
            trades.append({"date": ts, "type": "BUY", "price": price, "units": position})

        elif sig == -1 and position == 0:
            # short: size by POSITION_SIZE but capital stays as collateral
            position    = -(capital * POSITION_SIZE / price)
            entry_price = price
            trades.append({"date": ts, "type": "SHORT", "price": price, "units": abs(position)})

        # ── Mark-to-market equity ──
        if position > 0:
            equity = capital + (position * price)        # cash + open long value
        elif position < 0:
            unrealised_pnl = (entry_price - price) * abs(position)
            equity = capital + unrealised_pnl            # cash + short pnl
        else:
            equity = capital

        equity_curve.append({"date": ts, "equity": equity, "price": price})

    # Close any open position at last price
    last_price = df["close"].iloc[-1]
    if position > 0:
        capital += position * last_price             # add proceeds to cash
    elif position < 0:
        pnl      = (entry_price - last_price) * abs(position)
        capital += pnl

    df_equity = pd.DataFrame(equity_curve).set_index("date")
    return df_equity, trades, capital


# ── Performance metrics ───────────────────────────────────────────────────────
def print_metrics(equity_curve: pd.DataFrame, trades: list, final_capital: float, initial_capital: float):
    total_return = (final_capital - initial_capital) / initial_capital * 100
    buy_hold = (equity_curve["price"].iloc[-1] / equity_curve["price"].iloc[0] - 1) * 100

    daily_ret = equity_curve["equity"].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5) if daily_ret.std() > 0 else 0

    rolling_max = equity_curve["equity"].cummax()
    drawdown = (equity_curve["equity"] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    closed   = [t for t in trades if t["type"] in ("SELL", "COVER")]
    n_trades = len(closed)
    n_longs  = len([t for t in trades if t["type"] == "BUY"])
    n_shorts = len([t for t in trades if t["type"] == "SHORT"])
    winners  = [t for t in closed if t.get("pnl", 0) > 0]
    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0

    # ── Profit Factor ─────────────────────────────────────────────────────────
    gross_profit  = sum(t["pnl"] for t in closed if t.get("pnl", 0) > 0)
    gross_loss    = abs(sum(t["pnl"] for t in closed if t.get("pnl", 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # ── Calmar Ratio ──────────────────────────────────────────────────────────
    years  = (equity_curve.index[-1] - equity_curve.index[0]).days / 365
    cagr   = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
    max_dd_for_calmar = ((equity_curve["equity"] - equity_curve["equity"].cummax())
                         / equity_curve["equity"].cummax() * 100).min()
    calmar = cagr / abs(max_dd_for_calmar) if max_dd_for_calmar != 0 else float("inf")

    # ── Stop trades ───────────────────────────────────────────────────────────
    n_stops = len([t for t in trades if t["type"] in ("STOP", "STOP_SHORT")])

    print("\n" + "=" * 50)
    print(f"  BACKTEST RESULTS — Triple-Condition Strategy")
    print("=" * 50)
    print(f"  Symbol        : {SYMBOL} ({TIMEFRAME})")
    print(f"  Period        : {equity_curve.index[0].date()} → {equity_curve.index[-1].date()}")
    print(f"  Initial cap   : ${initial_capital:,.2f}")
    print(f"  Final equity  : ${final_capital:,.2f}")
    print(f"  Strategy ret  : {total_return:+.2f}%")
    print(f"  Buy & hold    : {buy_hold:+.2f}%")
    print(f"  Sharpe ratio  : {sharpe:.2f}")
    print(f"  Max drawdown  : {max_dd:.2f}%")
    print(f"  CAGR           : {cagr:+.2f}%")
    print(f"  Profit factor  : {profit_factor:.2f}")
    print(f"  Calmar ratio   : {calmar:.2f}")
    print(f"  Long trades   : {n_longs}")
    print(f"  Short trades  : {n_shorts}")
    print(f"  Win rate      : {win_rate:.1f}%")
    print(f"  Stop-outs      : {n_stops}")
    print("=" * 50 + "\n")


# ── Chart ─────────────────────────────────────────────────────────────────────
def plot_results(df: pd.DataFrame, equity_curve: pd.DataFrame, trades: list):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(5, 1, figsize=(16, 22), sharex=True)
    fig.suptitle(
        f"Triple-Condition Strategy  |  EMA 200 + MACD + RSI ({RSI_LOW}–{RSI_HIGH})  |  {SYMBOL} {TIMEFRAME}",
        fontsize=13, fontweight="bold", y=0.995
    )

    date_fmt = mdates.DateFormatter("%b '%y")
    date_loc = mdates.MonthLocator(interval=3)

    # ── Panel 1: Price + EMA 200 + trade signals ──
    ax1 = axes[0]
    ax1.plot(df.index, df["close"], color="#aab4be", linewidth=1, label="BTC/USDT Close")
    ax1.plot(df.index, df["ema_200"], color="#f43f5e", linewidth=1.5, linestyle="--", label="EMA 200 (trend filter)")

    buys   = [t for t in trades if t["type"] == "BUY"]
    sells  = [t for t in trades if t["type"] == "SELL"]
    shorts = [t for t in trades if t["type"] == "SHORT"]
    covers = [t for t in trades if t["type"] == "COVER"]
    stops  = [t for t in trades if t["type"] in ("STOP", "STOP_SHORT")]
    ax1.scatter([t["date"] for t in buys],   [t["price"] for t in buys],   marker="^", color="lime",   s=100, zorder=5, label="Long entry")
    ax1.scatter([t["date"] for t in sells],  [t["price"] for t in sells],  marker="v", color="red",    s=100, zorder=5, label="Long exit")
    ax1.scatter([t["date"] for t in shorts], [t["price"] for t in shorts], marker="v", color="orange", s=100, zorder=5, label="Short entry")
    ax1.scatter([t["date"] for t in covers], [t["price"] for t in covers], marker="^", color="cyan",   s=100, zorder=5, label="Short cover")
    ax1.scatter(
        [t["date"] for t in stops],
        [t["price"] for t in stops],
        marker="x", color="orange", s=120, zorder=6,
        linewidths=2, label="Stop-out"
    )

    ax1.set_ylabel("Price (USDT)", fontsize=10)
    ax1.set_title("Price Chart with Trade Signals", fontsize=10, pad=4)
    ax1.yaxis.set_major_formatter(mdates.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.grid(alpha=0.3)

    # ── Panel 2: MACD ──
    ax2 = axes[1]
    ax2.plot(df.index, df["macd"], color="#f0a500", linewidth=1, label=f"MACD ({FAST_EMA},{SLOW_EMA})")
    ax2.plot(df.index, df["macd_signal"], color="#5c85d6", linewidth=1, label=f"Signal ({SIGNAL_EMA})")
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["macd_hist"]]
    ax2.bar(df.index, df["macd_hist"], color=colors, alpha=0.6, width=1.5, label="Histogram")
    ax2.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("MACD Value", fontsize=10)
    ax2.set_title("MACD — Momentum & Crossover Signal", fontsize=10, pad=4)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: RSI ──
    ax3 = axes[2]
    ax3.plot(df.index, df["rsi"], color="#a78bfa", linewidth=1, label="RSI (14)")
    ax3.axhline(RSI_HIGH, color="#ef4444", linewidth=0.8, linestyle="--", label=f"Upper bound ({RSI_HIGH})")
    ax3.axhline(RSI_LOW,  color="#22c55e", linewidth=0.8, linestyle="--", label=f"Lower bound ({RSI_LOW})")
    ax3.fill_between(df.index, RSI_LOW, RSI_HIGH, alpha=0.08, color="lime", label="Safe zone (buy filter)")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI (0 – 100)", fontsize=10)
    ax3.set_title("RSI 14 — Overbought / Oversold Filter", fontsize=10, pad=4)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.3)

    # ── Panel 4: Equity curve ──
    ax4 = axes[3]
    ax4.plot(equity_curve.index, equity_curve["equity"], color="#22c55e", linewidth=1.5, label="Strategy equity")
    rolling_max = equity_curve["equity"].cummax()
    ax4.fill_between(equity_curve.index, equity_curve["equity"], rolling_max, alpha=0.25, color="#ef4444", label="Drawdown period")
    ax4.set_ylabel("Portfolio Value (USDT)", fontsize=10)
    ax4.set_title("Equity Curve — Portfolio Growth Over Time", fontsize=10, pad=4)
    ax4.yaxis.set_major_formatter(mdates.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax4.xaxis.set_major_formatter(date_fmt)
    ax4.xaxis.set_major_locator(date_loc)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(alpha=0.3)

    # ── Panel 5: Drawdown ─────────────────────────────────────────────────────
    ax5 = axes[4]
    rolling_max_eq = equity_curve["equity"].cummax()
    drawdown_pct   = (equity_curve["equity"] - rolling_max_eq) / rolling_max_eq * 100
    ax5.fill_between(equity_curve.index, drawdown_pct, 0,
                     color="#ef4444", alpha=0.5, label="Drawdown %")
    ax5.axhline(drawdown_pct.min(), color="#ef4444", linewidth=1,
                linestyle="--", label=f"Max DD: {drawdown_pct.min():.1f}%")
    ax5.set_ylabel("Drawdown (%)", fontsize=10)
    ax5.set_xlabel("Date", fontsize=10)
    ax5.set_title("Drawdown — Peak to Trough Loss Over Time", fontsize=10, pad=4)
    ax5.xaxis.set_major_formatter(date_fmt)
    ax5.xaxis.set_major_locator(date_loc)
    ax5.legend(loc="lower left", fontsize=8)
    ax5.grid(alpha=0.3)

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()
    out_path = "backtest_ep3_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved → {out_path}")

    _save_ema200_chart(df, trades)
    _save_macd_chart(df)
    _save_rsi_chart(df)


def _save_ema200_chart(df: pd.DataFrame, trades: list):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(f"EMA 200 — Trend Filter  |  {SYMBOL} {TIMEFRAME}", fontsize=13, fontweight="bold")

    ax.plot(df.index, df["close"], color="#aab4be", linewidth=1, label="BTC/USDT Close")
    ax.plot(df.index, df["ema_200"], color="#f43f5e", linewidth=2, linestyle="--", label="EMA 200")

    # shade above/below EMA to show bull/bear zones
    ax.fill_between(df.index, df["close"], df["ema_200"],
                    where=(df["close"] >= df["ema_200"]), alpha=0.12, color="#22c55e", label="Price above EMA (bull zone)")
    ax.fill_between(df.index, df["close"], df["ema_200"],
                    where=(df["close"] < df["ema_200"]),  alpha=0.12, color="#ef4444", label="Price below EMA (bear zone)")

    buys   = [t for t in trades if t["type"] == "BUY"]
    sells  = [t for t in trades if t["type"] == "SELL"]
    shorts = [t for t in trades if t["type"] == "SHORT"]
    covers = [t for t in trades if t["type"] == "COVER"]
    ax.scatter([t["date"] for t in buys],   [t["price"] for t in buys],   marker="^", color="lime",   s=100, zorder=5, label="Long entry")
    ax.scatter([t["date"] for t in sells],  [t["price"] for t in sells],  marker="v", color="red",    s=100, zorder=5, label="Long exit")
    ax.scatter([t["date"] for t in shorts], [t["price"] for t in shorts], marker="v", color="orange", s=100, zorder=5, label="Short entry")
    ax.scatter([t["date"] for t in covers], [t["price"] for t in covers], marker="^", color="cyan",   s=100, zorder=5, label="Short cover")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Price (USDT)", fontsize=10)
    ax.yaxis.set_major_formatter(mdates.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("indicator_ema200.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart saved → indicator_ema200.png")


def _save_macd_chart(df: pd.DataFrame):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(f"MACD ({FAST_EMA}, {SLOW_EMA}, {SIGNAL_EMA}) — Momentum & Crossover Signal  |  {SYMBOL} {TIMEFRAME}", fontsize=13, fontweight="bold")

    ax.plot(df.index, df["macd"],        color="#f0a500", linewidth=1.2, label=f"MACD line ({FAST_EMA},{SLOW_EMA})")
    ax.plot(df.index, df["macd_signal"], color="#5c85d6", linewidth=1.2, label=f"Signal line ({SIGNAL_EMA})")
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["macd_hist"]]
    ax.bar(df.index, df["macd_hist"], color=colors, alpha=0.6, width=1.5, label="Histogram (MACD − Signal)")
    ax.axhline(0, color="white", linewidth=0.6, linestyle="--", label="Zero line")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("MACD Value", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("indicator_macd.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart saved → indicator_macd.png")


def _save_rsi_chart(df: pd.DataFrame):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(f"RSI (14) — Overbought / Oversold Filter  |  {SYMBOL} {TIMEFRAME}", fontsize=13, fontweight="bold")

    ax.plot(df.index, df["rsi"], color="#a78bfa", linewidth=1.2, label="RSI (14)")
    ax.axhline(RSI_HIGH, color="#ef4444", linewidth=1,   linestyle="--", label=f"Long upper bound ({RSI_HIGH})")
    ax.axhline(RSI_LOW,  color="#22c55e", linewidth=1,   linestyle="--", label=f"Long lower bound ({RSI_LOW})")
    ax.axhline(70,       color="#ef4444", linewidth=0.5, linestyle=":",  label="Overbought (70)")
    ax.axhline(30,       color="#22c55e", linewidth=0.5, linestyle=":",  label="Oversold (30)")
    ax.fill_between(df.index, RSI_LOW, RSI_HIGH, alpha=0.08, color="lime",  label="Long safe zone")
    ax.fill_between(df.index, df["rsi"], 70, where=(df["rsi"] >= 70), alpha=0.15, color="#ef4444", label="Overbought region")
    ax.fill_between(df.index, df["rsi"], 30, where=(df["rsi"] <= 30), alpha=0.15, color="#22c55e", label="Oversold region")

    ax.set_ylim(0, 100)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("RSI (0 – 100)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("indicator_rsi.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart saved → indicator_rsi.png")


def compare_strategies(df: pd.DataFrame):
    """Run backtest twice and print side-by-side comparison."""
    print("\n" + "=" * 60)
    print("  STRATEGY COMPARISON — Original vs Risk-Managed")
    print("=" * 60)

    configs = [
        {"label": "Original (Ep 3)",     "pos": 1.0,  "sl": None},
        {"label": "Risk-Managed (Ep 4)", "pos": 0.25, "sl": 0.05},
    ]

    for cfg in configs:
        global POSITION_SIZE, STOP_LOSS_PCT
        POSITION_SIZE = cfg["pos"]
        STOP_LOSS_PCT = cfg["sl"] if cfg["sl"] else 999  # 999 = never triggers

        eq, tr, final = run_backtest(df.copy(), INITIAL_CAPITAL)

        total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        daily_ret = eq["equity"].pct_change().dropna()
        sharpe    = (daily_ret.mean() / daily_ret.std()) * (252**0.5) if daily_ret.std() > 0 else float("nan")
        roll_max  = eq["equity"].cummax()
        max_dd    = ((eq["equity"] - roll_max) / roll_max * 100).min()
        closed_t  = [t for t in tr if t["type"] in ("SELL", "COVER")]
        win_rate  = (len([t for t in closed_t if t.get("pnl", 0) > 0])
                     / len(closed_t) * 100 if closed_t else 0)
        n_stops   = len([t for t in tr if "STOP" in t["type"]])

        print(f"\n  [{cfg['label']}]")
        print(f"  Position size  : {cfg['pos']*100:.0f}%")
        print(f"  Stop-loss      : {cfg['sl']*100 if cfg['sl'] else 'None'}%")
        print(f"  Final equity   : ${final:,.2f}")
        print(f"  Total return   : {total_ret:+.2f}%")
        print(f"  Sharpe ratio   : {sharpe:.2f}")
        print(f"  Max drawdown   : {max_dd:.2f}%")
        print(f"  Win rate       : {win_rate:.1f}%")
        print(f"  Stop-outs      : {n_stops}")

    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Fetching {LIMIT} {TIMEFRAME} candles for {SYMBOL}...")
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, LIMIT)
    print(f"Got {len(df)} candles: {df.index[0].date()} → {df.index[-1].date()}")

    df = add_indicators(df)
    df = generate_signals(df)

    equity_curve, trades, final_capital = run_backtest(df, INITIAL_CAPITAL)
    print_metrics(equity_curve, trades, final_capital, INITIAL_CAPITAL)
    compare_strategies(df)
    plot_results(df, equity_curve, trades)
