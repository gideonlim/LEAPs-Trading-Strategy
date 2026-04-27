from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from leaps_bot.alpaca_client import AlpacaClient
from leaps_bot.config import load_config
from leaps_bot.logging_config import setup_logging
from leaps_bot.pricing import RateFetcher
from leaps_bot.scheduler import DailyScheduler
from leaps_bot.state import BotState

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="LEAPs Options Strategy Bot")
    parser.add_argument(
        "command",
        choices=["run", "dry-run", "status"],
        help="run=execute daily check, dry-run=simulate without orders, status=show current state",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--state", default="data/state.json", help="Path to state file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "dry-run":
        config.dry_run = True
    if args.verbose:
        config.log_level = "DEBUG"

    setup_logging(config.log_level)

    if not config.api_key or not config.secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or environment", file=sys.stderr)
        sys.exit(1)

    state_path = Path(args.state)
    state = BotState.load(state_path)
    client = AlpacaClient(config)
    rate_fetcher = RateFetcher(config.pricing)
    scheduler = DailyScheduler(config, client, state, rate_fetcher)

    if args.command == "status":
        status = scheduler.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    summary = {}
    exit_code = 0
    try:
        summary = scheduler.run()
        if summary.get("errors"):
            exit_code = 1
    except Exception as e:
        logger.exception("Unhandled exception during scheduler run: %s", e)
        exit_code = 2
    finally:
        # Always save state, even on partial failures. State updates from
        # confirmed fills must persist so we don't double-trade tomorrow.
        # Skip saving in dry-run to keep dry-run truly read-only.
        if not config.dry_run:
            try:
                state.save(state_path)
            except Exception as save_err:
                logger.error("Failed to save state: %s", save_err)
                exit_code = max(exit_code, 3)
        else:
            logger.info("[DRY-RUN] Skipping state save")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
