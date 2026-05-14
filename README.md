# BTC/USDT Triple-Condition Backtest

A Python backtesting framework for a triple-condition trading strategy on BTC/USDT. Combines EMA trend filter, MACD momentum crossover, and RSI zone filter to generate long and short signals — with full risk management (position sizing + stop-loss) and side-by-side strategy comparison.

---

## Strategy Logic

A trade is only opened when **all three conditions** are met simultaneously:

### Long Entry
| Condition | Rule |
|---|---|
| Trend filter | Price is **above** EMA 200 |
| Momentum | MACD line crosses **above** signal line |
| RSI filter | RSI is between **40 and 65** (not overbought, not oversold) |

### Short Entry
| Condition | Rule |
|---|---|
| Trend filter | Price is **below** EMA 200 |
| Momentum | MACD line crosses **below** signal line |
| RSI filter | RSI is between **35 and 60** |

---

## Risk Management

| Feature | Value | Description |
|---|---|---|
| Position sizing | 25% | Only 25% of available capital is deployed per trade |
| Stop-loss | 5% | Position is closed automatically if it loses more than 5% |

The remaining 75% of capital sits in cash during any open trade, reducing exposure to single-trade blowups.

---

## Requirements

**Python 3.8+**

Install dependencies:

```bash
pip install ccxt pandas numpy ta matplotlib
```

Or if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/quant-course.git
cd quant-course

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install ccxt pandas numpy ta matplotlib
```

---

## Usage

```bash
python backtest_ep3.py
```

The script will:
1. Fetch live BTC/USDT OHLCV data from MEXC (falls back to synthetic data if offline)
2. Compute EMA 200, MACD, and RSI indicators
3. Generate trade signals
4. Run the backtest with risk management active
5. Print full performance metrics to the terminal
6. Print a side-by-side comparison of Original vs Risk-Managed
7. Save 5 chart images to disk

---

## Configuration

All parameters are at the top of `backtest_ep3.py`:

```python
# ── Data
USE_LIVE_DATA  = True       # False = use synthetic GBM data (no internet needed)
SYMBOL         = "BTC/USDT"
TIMEFRAME      = "1d"       # "1d", "4h", "1h", etc.
LIMIT          = 1500       # number of candles (~4 years on daily)
INITIAL_CAPITAL = 10_000.0

# ── Indicators
FAST_EMA    = 12            # MACD fast period
SLOW_EMA    = 26            # MACD slow period
SIGNAL_EMA  = 9             # MACD signal period
TREND_EMA   = 200           # macro trend filter
RSI_LOW     = 40            # long entry RSI lower bound
RSI_HIGH    = 65            # long entry RSI upper bound
RSI_SHORT_LOW  = 35         # short entry RSI lower bound
RSI_SHORT_HIGH = 60         # short entry RSI upper bound

# ── Risk Management
POSITION_SIZE = 0.25        # fraction of capital per trade (0.25 = 25%)
STOP_LOSS_PCT = 0.05        # stop-loss threshold (0.05 = 5%)
```

### Common tweaks

| Goal | Change |
|---|---|
| Test on 4h candles | `TIMEFRAME = "4h"`, `LIMIT = 8760` |
| Go all-in per trade | `POSITION_SIZE = 1.0` |
| Widen stop-loss | `STOP_LOSS_PCT = 0.08` |
| Use synthetic data (offline) | `USE_LIVE_DATA = False` |

---

## Output

### Terminal
```
==================================================
  BACKTEST RESULTS — Triple-Condition Strategy
==================================================
  Symbol        : BTC/USDT (1d)
  Period        : 2022-10-21 → 2026-05-13
  Initial cap   : $10,000.00
  Final equity  : $12,450.00
  Strategy ret  : +24.50%
  Buy & hold    : +312.72%
  Sharpe ratio  : 0.61
  Max drawdown  : -18.32%
  CAGR           : +6.18%
  Profit factor  : 1.85
  Calmar ratio   : 0.34
  Long trades   : 4
  Short trades  : 5
  Win rate      : 62.5%
  Stop-outs      : 3
==================================================

============================================================
  STRATEGY COMPARISON — Original vs Risk-Managed
============================================================

  [Original (Ep 3)]
  Position size  : 100%
  Stop-loss      : None%
  ...

  [Risk-Managed (Ep 4)]
  Position size  : 25%
  Stop-loss      : 5.0%
  ...
```

### Chart files saved

| File | Contents |
|---|---|
| `backtest_ep3_results.png` | 5-panel master chart |
| `indicator_ema200.png` | EMA 200 trend zones + trade markers |
| `indicator_macd.png` | MACD line, signal line, histogram |
| `indicator_rsi.png` | RSI with overbought/oversold bands |

### 5-panel chart breakdown
1. **Price** — BTC/USDT close price with EMA 200 and all trade entry/exit markers
2. **MACD** — momentum crossover signal
3. **RSI** — overbought/oversold filter with safe zone shading
4. **Equity curve** — portfolio value over time with drawdown shading
5. **Drawdown** — percentage peak-to-trough loss over time

### Trade markers on price chart
| Marker | Colour | Meaning |
|---|---|---|
| ▲ | Lime | Long entry (BUY) |
| ▼ | Red | Long exit (SELL) |
| ▼ | Orange | Short entry |
| ▲ | Cyan | Short cover |
| ✕ | Orange | Stop-out (long or short) |

---

## Performance Metrics Explained

| Metric | Description |
|---|---|
| **Strategy ret** | Total % return over the full period |
| **Buy & hold** | What you'd have made just holding BTC |
| **Sharpe ratio** | Risk-adjusted return (annualised). Above 1.0 is good |
| **Max drawdown** | Largest peak-to-trough loss. Closer to 0% is better |
| **CAGR** | Compound Annual Growth Rate |
| **Profit factor** | Gross profit ÷ gross loss. Above 1.5 is solid |
| **Calmar ratio** | CAGR ÷ Max drawdown. Measures return per unit of drawdown risk |
| **Win rate** | % of closed trades that were profitable |
| **Stop-outs** | Number of trades closed by the stop-loss |

---

## Project Structure

```
quant-course/
├── backtest_ep3.py     # main script
├── README.md
├── .gitignore
└── venv/               # local virtual environment (not committed)
```

---

## Disclaimer

This project is for **educational purposes only**. It is not financial advice. Past backtest performance does not guarantee future results. Always do your own research before trading real capital.
