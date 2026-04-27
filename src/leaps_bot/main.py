from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from leaps_bot.alpaca_client import AlpacaClient
from leaps_bot.config import load_config
from leaps_bot.logging_config import setup_logging
from leaps_bot.pricing import RateFetcher
from leaps_bot.reporting import ReportGenerator
from leaps_bot.scheduler import DailyScheduler
from leaps_bot.state import BotState

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="LEAPs Options Strategy Bot")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="Path to config file")
    common.add_argument("--state", default="data/state.json", help="Path to state file")
    common.add_argument("--verbose", action="store_true", help="Enable debug logging")

    sub.add_parser("run", parents=[common], help="Execute the daily check")
    sub.add_parser("dry-run", parents=[common], help="Simulate without placing orders")
    sub.add_parser("status", parents=[common], help="Show current positions and account state")

    p_report = sub.add_parser("report", parents=[common], help="Generate a PDF performance report")
    p_report.add_argument("--output", default=None, help="Output PDF path (default: reports/leaps-report-<date>.pdf)")

    p_trades = sub.add_parser("export-trades", parents=[common], help="Export full trade log to CSV")
    p_trades.add_argument("--output", default=None, help="Output CSV path (default: reports/trades.csv)")

    p_tax = sub.add_parser("export-tax", parents=[common], help="Export tax-year closed positions (1099-B style) to CSV")
    p_tax.add_argument("--year", type=int, default=None, help="Tax year (default: all years)")
    p_tax.add_argument("--output", default=None, help="Output CSV path (default: reports/tax-<year>.csv)")

    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "dry-run":
        config.dry_run = True
    if getattr(args, "verbose", False):
        config.log_level = "DEBUG"

    setup_logging(config.log_level)

    state_path = Path(args.state)
    state = BotState.load(state_path)

    # Reporting commands don't need API keys
    if args.command in ("report", "export-trades", "export-tax"):
        _run_reporting(args, state)
        return

    if not config.api_key or not config.secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or environment", file=sys.stderr)
        sys.exit(1)

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
        if not config.dry_run:
            try:
                state.save(state_path)
            except Exception as save_err:
                logger.error("Failed to save state: %s", save_err)
                exit_code = max(exit_code, 3)
        else:
            logger.info("[DRY-RUN] Skipping state save")

    sys.exit(exit_code)


def _run_reporting(args, state: BotState) -> None:
    generator = ReportGenerator(state)

    if args.command == "report":
        out = Path(args.output) if args.output else Path(f"reports/leaps-report-{date.today().isoformat()}.pdf")
        path = generator.generate_pdf(out)
        print(f"PDF report: {path}")

    elif args.command == "export-trades":
        out = Path(args.output) if args.output else Path("reports/trades.csv")
        path = generator.export_trades_csv(out)
        print(f"Trade log: {path} ({len(state.trades)} trades)")

    elif args.command == "export-tax":
        if args.output:
            out = Path(args.output)
        else:
            year_part = f"-{args.year}" if args.year else ""
            out = Path(f"reports/tax{year_part}.csv")
        path = generator.export_tax_csv(out, year=args.year)
        print(f"Tax CSV: {path}")


if __name__ == "__main__":
    main()
