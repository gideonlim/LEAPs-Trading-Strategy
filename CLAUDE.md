# LEAPs Options Strategy Bot

## Setup
```bash
pip install -e ".[dev]"
cp .env.example .env  # fill in Alpaca API keys
```

## Run
```bash
leaps-bot run          # daily check with real orders
leaps-bot dry-run      # show what would happen
leaps-bot status       # current positions and account info
```

## Test
```bash
pytest tests/
```

## Architecture
- `src/leaps_bot/` — all source modules
- `config.yaml` — strategy parameters (committed)
- `.env` — API secrets (gitignored)
- `data/state.json` — persistent bot state (gitignored locally, on bot-state branch in CI)
