# LEAPs Options Strategy Bot

Automated bot that executes a deep ITM LEAPs (Long-term Equity AnticiPation Securities) call options strategy on the S&P 500 (SPY or SPYM) via the [Alpaca](https://alpaca.markets) API. Designed to run on GitHub Actions on a daily schedule.

## Strategy

1. **Buy** 20% deep ITM call options with 12–18 months to expiry
2. **Hold** until 1/3 of original time remains (configurable)
3. **Sell** before expiry — never holds to expiration
4. **Roll** forward into new 20% ITM LEAPs after selling
5. **Allocate** new funds quarterly (Mar / Jun / Sep / Dec)

Example: Buy a 12-month deep ITM call, sell when ~4 months remain, immediately roll into a new 12–18 month contract.

## Why deep ITM LEAPs?

- High delta (~0.85+) — moves nearly 1:1 with the underlying
- Acts as a leveraged stock substitute with lower capital outlay
- Time decay is minimal far from expiry; selling at 1/3 remaining avoids accelerated theta
- Long expiry reduces noise from short-term volatility

## Features

- **Black-Scholes fair pricing** — calculates theoretical option prices using underlying price, IV, risk-free rate, and dividend yield (compensates for delayed indicative data on Alpaca's free tier)
- **Smart limit orders** — uses lower of theoretical/market for buys, higher for sells
- **First-hour avoidance** — no orders placed in the first 60 minutes after market open (volatile)
- **Quarterly allocation** — automatic deployment of new capital on schedule
- **Roll forward** — automatically buys replacement LEAPs after selling
- **Auto SPY/SPYM selection** — picks the underlying based on buying power
- **Paper trading default** — must explicitly opt into live trading
- **Dry-run mode** — simulate without placing real orders
- **Persistent state** — tracks positions, allocations, and pending orders across runs
- **GitHub Actions scheduling** — runs daily on a free-tier compatible schedule

## Setup

### Prerequisites

- Python 3.10+
- An [Alpaca](https://alpaca.markets) account (paper trading is fine)
- Options trading enabled (Level 2 minimum for buying calls)

### Install

```bash
git clone <your-repo-url>
cd leaps-bot
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` with your Alpaca API keys:

```
ALPACA_API_KEY=your-key
ALPACA_SECRET_KEY=your-secret
```

## Usage

```bash
leaps-bot status                            # show current positions and account state
leaps-bot dry-run                           # simulate today's actions without placing orders
leaps-bot run                               # execute the daily trade check (places orders)
leaps-bot monitor                           # reconcile pending orders + capture snapshot (never trades)
leaps-bot monitor --no-snapshot             # reconcile only, used by mid-day workflow
leaps-bot report                            # generate PDF performance report
leaps-bot export-trades                     # export full trade log as CSV
leaps-bot export-tax --year 2026                   # US calendar year tax CSV
leaps-bot export-tax --fy 2026                     # Australian FY (Jul 2025 – Jun 2026), USD only
leaps-bot export-tax --fy 2026 --fx-auto           # AU FY with per-trade AUD/USD conversion (ATO-recommended)
leaps-bot export-tax --fy 2026 --aud-rate 1.52     # AU FY with flat-rate conversion (offline fallback)
```

The first run will create `data/state.json` to track positions, allocations, trades, and daily snapshots. Generated reports go to `reports/` (gitignored).

### Report contents

- **PDF report** — summary stats, portfolio value chart with SPY benchmark, cumulative realized P&L chart, open positions table, allocation history, recent trade history
- **Trade CSV** — every fill (buy + sell) with timestamps, prices, underlying price at trade, realized P&L for closes
- **Tax CSV** — closed positions for tax filing. Two filter modes (mutually exclusive):
  - `--year YYYY`: US calendar year. Column `term` = `short` / `long` (>365 days)
  - `--fy YYYY`: Australian financial year (Jul 1 of YYYY-1 to Jun 30 of YYYY). Column `cgt_discount_eligible` = `yes` / `no` (held >12 months → eligible for 50% CGT discount)

  Two optional FX modes (mutually exclusive) add `fx_rate`, `proceeds_aud`, `cost_basis_aud`, `gain_loss_aud` columns:
  - `--fx-auto`: fetches the AUD/USD rate **on each trade's actual fill date** from [Frankfurter](https://www.frankfurter.app) (free, ECB-sourced). This is the ATO-recommended approach for foreign currency transactions. Rates are cached to `data/fx_cache.json`. Note: ECB rates differ from RBA's official 4 PM rate by ~0.1%; for material amounts verify against RBA's [F11 historical exchange rates](https://www.rba.gov.au/statistics/historical-data.html).
  - `--aud-rate RATE`: flat-rate conversion (e.g., `--aud-rate 1.52`). Quick offline approximation; not ATO-accurate for material amounts.

## Configuration

All strategy parameters live in [config.yaml](config.yaml). Key settings:

| Parameter | Default | Description |
|---|---|---|
| `paper_trading` | `true` | Use paper account (set `false` for live) |
| `strategy.itm_depth_pct` | `0.20` | Strike = underlying × (1 - this) |
| `strategy.min_expiry_months` | `12` | Minimum DTE for new buys |
| `strategy.max_expiry_months` | `18` | Maximum DTE for new buys |
| `strategy.sell_threshold_fraction` | `0.333` | Sell when this fraction of original DTE remains |
| `strategy.order_type` | `limit` | `limit` or `market` |
| `strategy.underlying_preference` | `auto` | `SPY`, `SPYM`, or `auto` |
| `pricing.risk_free_rate` | `0.045` | Fallback if treasury fetch fails |
| `pricing.dividend_yield` | `0.013` | Fallback SPY dividend yield |
| `allocation.quarterly_months` | `[3, 6, 9, 12]` | When to deploy new funds |
| `allocation.max_cash_deploy_pct` | `0.90` | Max % of cash to deploy per quarter |
| `allocation.min_cash_reserve` | `500.0` | Minimum cash reserve |
| `safety.no_trade_minutes_after_open` | `60` | Skip first hour of trading |
| `safety.emergency_sell_days` | `30` | Force sell at this many days to expiry |
| `safety.min_delta` | `0.80` | Minimum delta for deep ITM filter |
| `safety.max_bid_ask_spread_pct` | `0.10` | Reject contracts with wider spreads |
| `data.feed` | `indicative` | `indicative` (free) or `opra` (paid) |

## GitHub Actions Setup

The bot runs on GitHub Actions via **three weekday workflows**, all in Eastern Time (DST handled automatically via the `timezone` field):

| Workflow | Cron (ET) | What it does |
|---|---|---|
| [daily-check.yml](.github/workflows/daily-check.yml) | 10:33 AM | Trading run — sells, rolls, quarterly allocations |
| [midday-monitor.yml](.github/workflows/midday-monitor.yml) | 2:30 PM | Reconciles fills from morning orders (no snapshot, no trades) |
| [eod-monitor.yml](.github/workflows/eod-monitor.yml) | 4:30 PM | Reconciles + captures canonical EOD snapshot (no trades) |

All three share a single concurrency group (`leaps-bot-run`) so they can never overlap. Only the morning workflow originates orders; the mid-day and EOD runs only observe and reconcile broker state.

### Setup steps

1. Push the repo to GitHub
2. Go to **Settings → Secrets and variables → Actions**
3. Add two repository secrets:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
4. Go to **Actions** and enable workflows
5. State persists across runs on a dedicated `bot-state` branch (created automatically on first run)

You can also trigger any workflow manually via the **Run workflow** button in the Actions tab.

## Going Live

The bot defaults to paper trading. To go live:

1. Test thoroughly with `leaps-bot dry-run` for several days
2. Run on the paper account for at least one full quarterly cycle
3. Verify all logs make sense (sells happen at expected times, allocations on the right months)
4. Set `paper_trading: false` in `config.yaml`
5. Use **live API keys** in `.env` / GitHub Secrets (not paper keys)
6. Start with `min_cash_reserve` set high to limit initial deployment

## Project Structure

```
leaps-bot/
├── .github/workflows/daily-check.yml   # Daily cron schedule
├── src/leaps_bot/
│   ├── config.py            # YAML config + env loading
│   ├── models.py            # Dataclasses (OCC parsing, etc.)
│   ├── state.py             # JSON state persistence
│   ├── pricing.py           # Black-Scholes + rate fetching
│   ├── alpaca_client.py     # Alpaca SDK wrapper
│   ├── contract_finder.py   # Search & price LEAPs candidates
│   ├── position_manager.py  # Sell/roll decisions
│   ├── order_executor.py    # Order submission (with dry-run)
│   ├── allocator.py         # Quarterly fund deployment
│   ├── scheduler.py         # Daily orchestration
│   └── main.py              # CLI entry point
├── tests/                   # 37 unit tests
├── config.yaml              # Strategy parameters
├── .env.example             # API key template
└── pyproject.toml
```

## Testing

```bash
pytest tests/ -v
```

Tests cover Black-Scholes math, OCC symbol parsing, sell threshold logic, quarterly allocation logic, contract scoring, and scheduler orchestration. All external API calls are mocked.

## Safety Guardrails

- Emergency sell at 30 days to expiry (configurable)
- No orders in the first 60 minutes after market open
- Black-Scholes pricing as primary (compensates for delayed indicative feed)
- Extrinsic value cap rejects mispriced contracts
- Liquidity filters: max spread, min open interest, min delta
- Order submissions are never retried (prevents duplicate fills)
- Idempotent daily runs (won't trade twice on the same day)
- Account checks (trading not blocked, sufficient buying power)

## Free Tier Notes

This bot is designed to work with Alpaca's **free tier**:
- Uses the `indicative` options feed (delayed/modified data)
- Compensates by using Black-Scholes as the primary pricing tool
- Wider limit offsets to absorb data staleness
- Stale quote warnings when data is > 15 minutes old

For real-time OPRA data, set `data.feed: opra` in `config.yaml` (requires paid Alpaca subscription).

## Disclaimer

**This is not financial advice.** Options trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk. Test thoroughly on paper trading before deploying real capital. The author and contributors are not responsible for any financial losses incurred from using this bot.

## License

MIT
