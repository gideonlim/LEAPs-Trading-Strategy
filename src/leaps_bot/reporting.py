"""Report generation: PDF performance reports and CSV exports for tax/trade history.

All output is derived from BotState — no API calls. Snapshots and trades must
have been captured by previous bot runs (see scheduler logging).
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from leaps_bot.fx import FlatFXProvider, FXProvider
from leaps_bot.models import (
    AllocationRecord,
    DailySnapshot,
    PositionSnapshot,
    TradeRecord,
    parse_timestamp,
)
from leaps_bot.state import BotState

logger = logging.getLogger(__name__)


@dataclass
class CashFlow:
    """An inferred external deposit (positive) or withdrawal (negative).

    Detected by comparing actual cash deltas between snapshots against the
    cash flow that would be expected from trade activity in the same period.
    """
    timestamp: str
    amount: float


@dataclass
class ReportSummary:
    """High-level stats shown on the cover page of the PDF."""
    report_date: str
    inception_date: str | None
    days_active: int
    cash: float
    portfolio_value: float
    positions_market_value: float
    num_open_positions: int
    total_invested: float           # net external capital contributed (initial + deposits - withdrawals)
    total_deployed: float           # sum of allocation amounts (capital pushed into options)
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_return_dollar: float      # portfolio_value - total_invested (cash-flow adjusted)
    total_return_pct: float         # time-weighted total return
    ytd_return_pct: float | None    # time-weighted YTD return
    spy_total_return_pct: float | None  # SPY buy-and-hold benchmark over inception → today
    num_trades: int
    num_external_flows: int         # how many deposits/withdrawals were detected


class ReportGenerator:
    # External flow detection: ignore residuals smaller than this (rounding,
    # fees, market data fuzz). $50 is well above typical noise but small
    # enough to catch any meaningful deposit or withdrawal.
    _FLOW_THRESHOLD_USD = 50.0

    def __init__(self, state: BotState):
        self._state = state

    # ------------------------------------------------------------------
    # Sorting helpers (so list order doesn't dictate time order)
    # ------------------------------------------------------------------

    def _sorted_snapshots(self) -> list[DailySnapshot]:
        # Sort by parsed UTC datetime, not lexical string. Snapshot timestamps
        # are naive ISO strings while trade timestamps may be tz-aware — lexical
        # comparison can mis-order them.
        return sorted(self._state.snapshots, key=lambda s: parse_timestamp(s.timestamp))

    def _sorted_trades(self) -> list[TradeRecord]:
        return sorted(self._state.trades, key=lambda t: parse_timestamp(t.timestamp))

    # ------------------------------------------------------------------
    # External cash flow detection
    # ------------------------------------------------------------------

    def compute_external_flows(self) -> list[CashFlow]:
        """Infer external deposits and withdrawals between snapshots.

        For each consecutive snapshot pair we know:
            actual_cash_change = s2.cash - s1.cash
            trade_cash_change  = sum(sell proceeds - buy costs in the period)
        The difference is what *cannot* be explained by bot activity, so it
        must be an external deposit (+) or withdrawal (-).

        Filters out small residuals below `_FLOW_THRESHOLD_USD` (rounding,
        commissions, fees). Returns flows in chronological order.
        """
        snapshots = self._sorted_snapshots()
        trades = self._sorted_trades()
        if len(snapshots) < 2:
            return []

        flows: list[CashFlow] = []
        for i in range(1, len(snapshots)):
            prev, curr = snapshots[i - 1], snapshots[i]
            prev_dt = parse_timestamp(prev.timestamp)
            curr_dt = parse_timestamp(curr.timestamp)
            period_trades = [
                t for t in trades
                if prev_dt < parse_timestamp(t.timestamp) <= curr_dt
            ]
            trade_cash_delta = sum(
                t.total_value if t.action == "sell" else -t.total_value
                for t in period_trades
            )
            actual_cash_delta = curr.cash - prev.cash
            external = actual_cash_delta - trade_cash_delta

            if abs(external) >= self._FLOW_THRESHOLD_USD:
                flows.append(CashFlow(timestamp=curr.timestamp, amount=external))
        return flows

    def compute_twr_series(self) -> list[tuple[str, float]]:
        """Time-weighted cumulative growth factor at each snapshot.

        Returns [(timestamp, growth_factor)] where 1.0 = 0% return,
        1.10 = +10%, 0.95 = -5%. Uses the Modified Dietz approximation:
        treats external flows as occurring at end of their period, so
        period_return = (end_value - flow) / start_value.

        This removes the effect of contributions/withdrawals from the
        return measurement — a deposit doesn't show as a "gain".
        """
        snapshots = self._sorted_snapshots()
        if len(snapshots) < 2:
            return [(s.timestamp, 1.0) for s in snapshots]

        flows = self.compute_external_flows()

        cum_growth = 1.0
        series: list[tuple[str, float]] = [(snapshots[0].timestamp, 1.0)]

        for i in range(1, len(snapshots)):
            prev, curr = snapshots[i - 1], snapshots[i]
            prev_dt = parse_timestamp(prev.timestamp)
            curr_dt = parse_timestamp(curr.timestamp)
            period_flow = sum(
                f.amount for f in flows
                if prev_dt < parse_timestamp(f.timestamp) <= curr_dt
            )
            if prev.portfolio_value > 0:
                period_return = (curr.portfolio_value - period_flow) / prev.portfolio_value
                cum_growth *= period_return
            series.append((curr.timestamp, cum_growth))
        return series

    # ------------------------------------------------------------------
    # Summary computation
    # ------------------------------------------------------------------

    def compute_summary(self) -> ReportSummary:
        snapshots = self._sorted_snapshots()
        trades = self._sorted_trades()
        allocations = self._state.allocations

        latest = snapshots[-1] if snapshots else None
        first = snapshots[0] if snapshots else None

        inception = first.date if first else (allocations[0].allocated_date if allocations else None)
        days_active = 0
        if inception:
            try:
                days_active = (date.today() - date.fromisoformat(inception)).days
            except ValueError:
                pass

        cash = latest.cash if latest else 0.0
        portfolio_value = latest.portfolio_value if latest else 0.0
        positions_market_value = latest.positions_market_value if latest else 0.0
        num_open_positions = latest.num_positions if latest else 0

        # total_deployed = capital pushed into options at allocation events.
        # This double-counts when realized gains are redeployed at the next
        # quarter — useful for "how much have we put through this strategy"
        # but NOT a return denominator.
        total_deployed = sum(a.amount for a in allocations)

        # total_invested = NET external capital. The correct denominator for
        # return calculations. Initial capital plus subsequent deposits, less
        # withdrawals, derived from snapshot+trade deltas.
        flows = self.compute_external_flows()
        if first is not None:
            total_invested = first.portfolio_value + sum(f.amount for f in flows)
        elif allocations:
            # No snapshots yet → fall back to allocation sum (less accurate)
            total_invested = total_deployed
        else:
            total_invested = 0.0

        realized_pnl = self._state.get_realized_pnl()
        unrealized_pnl = sum(p.unrealized_pl for p in latest.positions) if latest else 0.0

        # Total return: portfolio_value already reflects all gains/losses (in
        # cash) plus any deposits. Subtracting total external contributions
        # gives the actual return on capital.
        total_return_dollar = portfolio_value - total_invested

        # Total return % uses time-weighted return so contributions don't
        # inflate the percentage. Fall back to dollar-based ratio when we
        # don't have enough snapshots for TWR.
        twr_series = self.compute_twr_series()
        if len(twr_series) >= 2:
            final_growth = twr_series[-1][1]
            total_return_pct = (final_growth - 1.0) * 100
        elif total_invested > 0:
            total_return_pct = total_return_dollar / total_invested * 100
        else:
            total_return_pct = 0.0

        ytd_return_pct = self._compute_ytd_return(snapshots, flows)
        spy_return_pct = self._compute_spy_benchmark_return(snapshots)

        return ReportSummary(
            report_date=date.today().isoformat(),
            inception_date=inception,
            days_active=days_active,
            cash=cash,
            portfolio_value=portfolio_value,
            positions_market_value=positions_market_value,
            num_open_positions=num_open_positions,
            total_invested=total_invested,
            total_deployed=total_deployed,
            total_realized_pnl=realized_pnl,
            total_unrealized_pnl=unrealized_pnl,
            total_return_dollar=total_return_dollar,
            total_return_pct=total_return_pct,
            ytd_return_pct=ytd_return_pct,
            spy_total_return_pct=spy_return_pct,
            num_trades=len(trades),
            num_external_flows=len(flows),
        )

    def _compute_ytd_return(
        self,
        sorted_snapshots: list[DailySnapshot],
        flows: list[CashFlow],
    ) -> float | None:
        """Time-weighted YTD return — strips out cash flows so deposits don't
        inflate the percentage."""
        if not sorted_snapshots:
            return None
        current_year = date.today().year
        # Earliest snapshot in the current year is the YTD baseline
        ytd_snaps = [s for s in sorted_snapshots if s.date.startswith(str(current_year))]
        if len(ytd_snaps) < 2:
            return None

        cum = 1.0
        for i in range(1, len(ytd_snaps)):
            prev, curr = ytd_snaps[i - 1], ytd_snaps[i]
            prev_dt = parse_timestamp(prev.timestamp)
            curr_dt = parse_timestamp(curr.timestamp)
            period_flow = sum(
                f.amount for f in flows
                if prev_dt < parse_timestamp(f.timestamp) <= curr_dt
            )
            if prev.portfolio_value > 0:
                cum *= (curr.portfolio_value - period_flow) / prev.portfolio_value
        return (cum - 1.0) * 100

    def _compute_spy_benchmark_return(self, sorted_snapshots: list[DailySnapshot]) -> float | None:
        """Total return of SPY from first snapshot's price to latest snapshot's price.
        SPY has no external cash flows so this is just the price ratio."""
        spy_prices = [
            (s.timestamp, s.underlying_prices.get("SPY"))
            for s in sorted_snapshots
            if s.underlying_prices.get("SPY")
        ]
        if len(spy_prices) < 2:
            return None
        _, first_price = spy_prices[0]
        _, last_price = spy_prices[-1]
        if first_price <= 0:
            return None
        return (last_price - first_price) / first_price * 100

    # ------------------------------------------------------------------
    # Chart generation
    # ------------------------------------------------------------------

    def generate_portfolio_chart(self) -> bytes | None:
        """PNG bytes for cumulative-return chart vs SPY benchmark.

        Plots the portfolio's TIME-WEIGHTED return alongside SPY's price-ratio
        return. Both lines start at 0% and are directly comparable: deposits
        and withdrawals don't inflate the portfolio line, so the gap to SPY
        reflects actual investment performance, not contribution timing.
        """
        snapshots = self._sorted_snapshots()
        if len(snapshots) < 2:
            return None

        twr_series = self.compute_twr_series()
        if len(twr_series) < 2:
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        portfolio_dates = [datetime.fromisoformat(ts) for ts, _ in twr_series]
        portfolio_pct = [(g - 1.0) * 100 for _, g in twr_series]

        # SPY benchmark — price ratio from first to current snapshot
        spy_dates: list[datetime] = []
        spy_pct: list[float] = []
        first_spy = None
        for s in snapshots:
            spy = s.underlying_prices.get("SPY")
            if spy is None:
                continue
            if first_spy is None:
                first_spy = spy
            spy_dates.append(datetime.fromisoformat(s.timestamp))
            spy_pct.append((spy / first_spy - 1.0) * 100)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(portfolio_dates, portfolio_pct, label="Portfolio (TWR)",
                color="#2c3e50", linewidth=2)
        if len(spy_pct) >= 2:
            ax.plot(spy_dates, spy_pct, label="SPY benchmark",
                    color="#7f8c8d", linewidth=1.5, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.6)

        ax.set_title("Cumulative return (time-weighted, contributions excluded)")
        ax.set_ylabel("Return (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        return buf.getvalue()

    def generate_pnl_chart(self) -> bytes | None:
        """Cumulative realized P&L over trades, plotted in chronological order
        of fill (not insertion order). Late-arriving fills must not corrupt
        the cumulative path."""
        sells = sorted(
            (t for t in self._state.trades if t.action == "sell" and t.realized_pnl is not None),
            key=lambda t: parse_timestamp(t.timestamp),
        )
        if len(sells) < 2:
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        dates = [parse_timestamp(t.timestamp) for t in sells]
        cumulative = []
        running = 0.0
        for t in sells:
            running += t.realized_pnl or 0.0
            cumulative.append(running)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.fill_between(dates, cumulative, 0, alpha=0.3, color="#27ae60")
        ax.plot(dates, cumulative, color="#27ae60", linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Cumulative realized P&L")
        ax.set_ylabel("$")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # PDF generation
    # ------------------------------------------------------------------

    def generate_pdf(self, output_path: Path) -> Path:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        styles = getSampleStyleSheet()
        h1 = styles["Heading1"]
        h2 = styles["Heading2"]
        body = styles["BodyText"]
        small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)

        summary = self.compute_summary()
        elements: list = []

        # Title
        elements.append(Paragraph("LEAPs Bot Performance Report", h1))
        elements.append(Paragraph(f"Generated {summary.report_date}", small))
        elements.append(Spacer(1, 0.2 * inch))

        # Summary stats
        elements.append(Paragraph("Summary", h2))
        elements.append(self._summary_table(summary))
        elements.append(Spacer(1, 0.3 * inch))

        # Portfolio chart
        chart_png = self.generate_portfolio_chart()
        if chart_png:
            elements.append(Image(io.BytesIO(chart_png), width=7 * inch, height=3.5 * inch))
            elements.append(Spacer(1, 0.2 * inch))
        else:
            elements.append(Paragraph(
                "Not enough daily snapshots yet for a portfolio chart "
                "(need at least 2 snapshots).",
                small,
            ))
            elements.append(Spacer(1, 0.2 * inch))

        # P&L chart
        pnl_png = self.generate_pnl_chart()
        if pnl_png:
            elements.append(Image(io.BytesIO(pnl_png), width=7 * inch, height=2.5 * inch))
            elements.append(Spacer(1, 0.2 * inch))

        elements.append(PageBreak())

        # Open positions
        elements.append(Paragraph("Open positions", h2))
        elements.append(self._open_positions_table())
        elements.append(Spacer(1, 0.3 * inch))

        # Allocation history
        elements.append(Paragraph("Allocation history", h2))
        elements.append(self._allocations_table())
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(PageBreak())

        # Trade history
        elements.append(Paragraph("Trade history", h2))
        elements.append(self._trades_table(limit=40))

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        )
        doc.build(elements)
        logger.info("PDF report written to %s", output_path)
        return output_path

    # -- PDF table builders --

    def _summary_table(self, s: ReportSummary):
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle

        flow_note = ""
        if s.num_external_flows > 0:
            flow_note = f"  (incl. {s.num_external_flows} detected flow{'s' if s.num_external_flows != 1 else ''})"

        rows = [
            ["Inception", s.inception_date or "—"],
            ["Days active", str(s.days_active)],
            ["Cash", f"${s.cash:,.2f}"],
            ["Open positions market value", f"${s.positions_market_value:,.2f}"],
            ["Portfolio value", f"${s.portfolio_value:,.2f}"],
            ["Net contributions", f"${s.total_invested:,.2f}{flow_note}"],
            ["Total deployed (allocations)", f"${s.total_deployed:,.2f}"],
            ["Realized P&L", f"${s.total_realized_pnl:,.2f}"],
            ["Unrealized P&L", f"${s.total_unrealized_pnl:,.2f}"],
            ["Total return", f"${s.total_return_dollar:,.2f}  ({s.total_return_pct:+.2f}% TWR)"],
            ["YTD return (TWR)", f"{s.ytd_return_pct:+.2f}%" if s.ytd_return_pct is not None else "—"],
            ["SPY benchmark (inception → today)", f"{s.spy_total_return_pct:+.2f}%" if s.spy_total_return_pct is not None else "—"],
            ["Open positions", str(s.num_open_positions)],
            ["Total trades", str(s.num_trades)],
        ]
        t = Table(rows, colWidths=[2.2 * 72, 3.5 * 72])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return t

    def _open_positions_table(self):
        from reportlab.lib import colors
        from reportlab.platypus import Paragraph, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet

        sorted_snaps = self._sorted_snapshots()
        latest = sorted_snaps[-1] if sorted_snaps else None
        if latest is None or not latest.positions:
            return Paragraph("No open positions.", getSampleStyleSheet()["BodyText"])

        header = ["Symbol", "Qty", "Entry", "Mark", "Mkt value", "Unrealized P&L", "% Return", "Days left"]
        rows = [header]
        for p in latest.positions:
            rows.append([
                p.symbol,
                str(p.qty),
                f"${p.avg_entry_price:.2f}",
                f"${p.current_price:.2f}",
                f"${p.market_value:,.2f}",
                f"${p.unrealized_pl:,.2f}",
                f"{p.unrealized_plpc * 100:+.2f}%",
                str(p.days_remaining),
            ])

        return self._styled_table(rows)

    def _allocations_table(self):
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph

        if not self._state.allocations:
            return Paragraph("No allocations recorded yet.", getSampleStyleSheet()["BodyText"])

        # Sort by allocation date so the table presents a consistent timeline
        # regardless of insertion order. allocated_date is a YYYY-MM-DD string
        # so lexical sort matches calendar order.
        sorted_allocs = sorted(self._state.allocations, key=lambda a: a.allocated_date)
        header = ["Quarter", "Date", "Amount", "Contracts"]
        rows = [header]
        for a in sorted_allocs:
            rows.append([
                a.quarter,
                a.allocated_date,
                f"${a.amount:,.2f}",
                ", ".join(a.contracts_bought) or "—",
            ])
        return self._styled_table(rows)

    def _trades_table(self, limit: int = 40):
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph

        if not self._state.trades:
            return Paragraph("No trades recorded yet.", getSampleStyleSheet()["BodyText"])

        # Sort by parsed fill timestamp, most recent first. Use the normalized
        # parser so naive (internal) and tz-aware (broker) timestamps order
        # correctly together.
        trades = sorted(
            self._state.trades,
            key=lambda t: parse_timestamp(t.timestamp),
            reverse=True,
        )[:limit]
        header = ["Date", "Action", "Intent", "Symbol", "Qty", "Price", "Total", "Realized P&L"]
        rows = [header]
        for t in trades:
            try:
                date_str = t.timestamp.split("T")[0]
            except Exception:
                date_str = t.timestamp
            rows.append([
                date_str,
                t.action.upper(),
                t.intent,
                t.symbol,
                str(t.qty),
                f"${t.fill_price:.2f}",
                f"${t.total_value:,.2f}",
                f"${t.realized_pnl:,.2f}" if t.realized_pnl is not None else "—",
            ])
        return self._styled_table(rows)

    def _styled_table(self, rows: list[list[str]]):
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle

        t = Table(rows, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    # ------------------------------------------------------------------
    # CSV exports
    # ------------------------------------------------------------------

    def export_trades_csv(self, output_path: Path) -> Path:
        """Full trade log: every fill (buy + sell) with all logged fields."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "timestamp", "order_id", "action", "intent", "symbol", "underlying",
            "strike", "expiry", "qty", "fill_price", "total_value",
            "underlying_price", "avg_entry_price", "realized_pnl", "holding_days",
        ]
        with open(output_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for t in self._state.trades:
                w.writerow([
                    t.timestamp, t.order_id, t.action, t.intent, t.symbol, t.underlying,
                    f"{t.strike:.2f}", t.expiry, t.qty, f"{t.fill_price:.4f}",
                    f"{t.total_value:.2f}",
                    f"{t.underlying_price:.4f}" if t.underlying_price is not None else "",
                    f"{t.avg_entry_price:.4f}" if t.avg_entry_price is not None else "",
                    f"{t.realized_pnl:.2f}" if t.realized_pnl is not None else "",
                    t.holding_days if t.holding_days is not None else "",
                ])
        logger.info("Trade log CSV written to %s (%d trades)", output_path, len(self._state.trades))
        return output_path

    def export_tax_csv(
        self,
        output_path: Path,
        year: int | None = None,
        fy: int | None = None,
        aud_rate: float | None = None,
        fx_provider: FXProvider | None = None,
    ) -> Path:
        """Closed-position tax report.

        Two filter modes (mutually exclusive):
        - `year=YYYY`: US-style calendar year filter (Jan 1 – Dec 31). Column
          `term` is `short` / `long` against the 365-day threshold.
        - `fy=YYYY`: Australian financial year filter (Jul 1 of YYYY-1 to
          Jun 30 of YYYY). E.g., `fy=2026` → Jul 1, 2025 to Jun 30, 2026.
          Column `cgt_discount_eligible` is `yes` / `no` based on whether
          the position was held for more than 12 months (the AU 50% CGT
          discount threshold).

        FX conversion (mutually exclusive options):
        - `aud_rate`: flat AUD/USD rate applied to all rows. Quick
          approximation; not ATO-accurate for material amounts.
        - `fx_provider`: source of per-trade FX rates (e.g.,
          `FrankfurterFXProvider`). The rate on each trade's actual fill date
          is used. ATO-recommended for accurate filing.

        Either approach adds `proceeds_aud`, `cost_basis_aud`, `gain_loss_aud`
        columns and a `fx_rate` column showing the rate used per row.
        """
        if year is not None and fy is not None:
            raise ValueError("Pass either `year` (US calendar) or `fy` (AU financial), not both")
        if aud_rate is not None and fx_provider is not None:
            raise ValueError("Pass either `aud_rate` (flat) or `fx_provider` (per-trade), not both")

        # Normalize: a flat rate becomes a FlatFXProvider so the row loop has
        # one code path.
        if aud_rate is not None:
            fx_provider = FlatFXProvider(aud_rate)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        is_au_mode = fy is not None

        sells = [t for t in self._state.trades if t.action == "sell"]
        sells = self._filter_sells_for_period(sells, year=year, fy=fy)
        sells = sorted(sells, key=lambda t: parse_timestamp(t.timestamp))

        # If we're using a real per-trade provider, prefetch the date range
        # in one shot so we don't fire one HTTP request per sell.
        if fx_provider is not None and sells:
            self._maybe_prefetch_fx(fx_provider, sells)

        # Column layout depends on jurisdiction and FX mode
        period_term_field = "cgt_discount_eligible" if is_au_mode else "term"
        fields = [
            "description",
            "date_acquired",
            "date_sold",
            "qty_contracts",
            "proceeds",
            "cost_basis",
            "gain_loss",
            "holding_period_days",
            period_term_field,
            "underlying",
            "strike",
            "expiry",
            "symbol",
            "order_id",
        ]
        if fx_provider is not None:
            insert_at = fields.index("gain_loss") + 1
            fields[insert_at:insert_at] = [
                "fx_rate", "proceeds_aud", "cost_basis_aud", "gain_loss_aud",
            ]

        missing_basis_count = 0
        missing_fx_count = 0
        with open(output_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)

            for t in sells:
                date_sold = t.timestamp.split("T")[0]
                date_acquired = ""
                if t.holding_days is not None:
                    try:
                        from datetime import timedelta
                        sold_dt = date.fromisoformat(date_sold)
                        acquired_dt = sold_dt - timedelta(days=t.holding_days)
                        date_acquired = acquired_dt.isoformat()
                    except ValueError:
                        pass

                if t.holding_days is None:
                    period_term = ""
                elif is_au_mode:
                    period_term = "yes" if t.holding_days > 365 else "no"
                else:
                    period_term = "long" if t.holding_days > 365 else "short"

                proceeds = t.fill_price * t.qty * 100

                if t.avg_entry_price is None:
                    cost_basis = None
                    gain_loss = None
                    cost_basis_str = ""
                    gain_loss_str = ""
                    missing_basis_count += 1
                    logger.warning(
                        "Tax export: missing cost basis for %s (sold %s, order %s) — "
                        "row included with empty cost_basis/gain_loss for manual review",
                        t.symbol, date_sold, t.order_id,
                    )
                else:
                    cost_basis = t.avg_entry_price * t.qty * 100
                    gain_loss = (
                        t.realized_pnl if t.realized_pnl is not None
                        else (proceeds - cost_basis)
                    )
                    cost_basis_str = f"{cost_basis:.2f}"
                    gain_loss_str = f"{gain_loss:.2f}"

                description = self._option_description(t)

                row = [
                    description,
                    date_acquired,
                    date_sold,
                    t.qty,
                    f"{proceeds:.2f}",
                    cost_basis_str,
                    gain_loss_str,
                ]

                if fx_provider is not None:
                    # Use the FX rate on the trade's fill date (ATO-recommended:
                    # the rate at the time of the transaction, not a flat rate)
                    try:
                        sold_dt = date.fromisoformat(date_sold)
                        rate = fx_provider.get_rate(sold_dt)
                    except ValueError:
                        rate = None

                    if rate is None:
                        missing_fx_count += 1
                        logger.warning(
                            "Tax export: no FX rate for %s on %s — leaving AUD columns blank",
                            t.symbol, date_sold,
                        )
                        row.extend(["", "", "", ""])
                    else:
                        row.append(f"{rate:.6f}")
                        row.append(f"{proceeds * rate:.2f}")
                        row.append(f"{cost_basis * rate:.2f}" if cost_basis is not None else "")
                        row.append(f"{gain_loss * rate:.2f}" if gain_loss is not None else "")

                row.extend([
                    t.holding_days if t.holding_days is not None else "",
                    period_term,
                    t.underlying,
                    f"{t.strike:.2f}",
                    t.expiry,
                    t.symbol,
                    t.order_id,
                ])
                w.writerow(row)

        if missing_basis_count:
            logger.warning(
                "Tax export: %d row(s) had missing cost basis. Fill these in manually "
                "before filing — they appear in %s with empty cost_basis/gain_loss fields.",
                missing_basis_count, output_path,
            )
        if missing_fx_count:
            logger.warning(
                "Tax export: %d row(s) had no FX rate available — AUD columns are blank "
                "for those rows. Possible causes: offline, API down, or trade date "
                "outside the available rate range. Re-run with network access or use "
                "--aud-rate as a flat-rate fallback.",
                missing_fx_count,
            )

        if fx_provider is not None:
            if isinstance(fx_provider, FlatFXProvider):
                logger.warning(
                    "Tax export: AUD figures use a FLAT rate. The ATO generally expects "
                    "per-transaction conversion using the RBA daily AUD/USD rate "
                    "(or its annual average). Verify whether the flat rate is "
                    "acceptable for your filing.",
                )
            else:
                logger.info(
                    "Tax export: AUD figures use per-trade rates from %s. Note that "
                    "ECB-source rates differ from RBA's official 4 PM rate by ~0.1%%; "
                    "for material amounts, verify against RBA F11 data.",
                    fx_provider.describe(),
                )

        period_label = (
            f"FY{fy}" if fy is not None
            else (f"year={year}" if year is not None else "all years")
        )
        logger.info(
            "Tax CSV written to %s (%d closed trades, %s%s)",
            output_path, len(sells), period_label,
            f", FX={fx_provider.describe()}" if fx_provider is not None else "",
        )
        return output_path

    @staticmethod
    def _maybe_prefetch_fx(fx_provider: FXProvider, sells: list[TradeRecord]) -> None:
        """If the provider supports range prefetching (e.g., Frankfurter),
        fetch all needed dates in one HTTP request."""
        if not hasattr(fx_provider, "prefetch_range"):
            return
        try:
            trade_dates = []
            for t in sells:
                try:
                    trade_dates.append(date.fromisoformat(t.timestamp.split("T")[0]))
                except ValueError:
                    continue
            if not trade_dates:
                return
            fx_provider.prefetch_range(min(trade_dates), max(trade_dates))
        except Exception as e:
            logger.warning("FX prefetch failed: %s — falling back to per-row lookup", e)

    @staticmethod
    def _filter_sells_for_period(
        sells: list[TradeRecord],
        year: int | None = None,
        fy: int | None = None,
    ) -> list[TradeRecord]:
        """Filter sell trades to a tax-year period.

        - `year`: calendar year (Jan 1 – Dec 31)
        - `fy`: Australian FY (Jul 1 of fy-1 → Jun 30 of fy). E.g., fy=2026
          spans 2025-07-01 to 2026-06-30.
        """
        if year is None and fy is None:
            return sells

        if fy is not None:
            start = date(fy - 1, 7, 1)
            end = date(fy, 6, 30)
        else:
            start = date(year, 1, 1)
            end = date(year, 12, 31)

        result = []
        for t in sells:
            try:
                trade_date = parse_timestamp(t.timestamp).date()
            except (ValueError, AttributeError):
                continue
            if start <= trade_date <= end:
                result.append(t)
        return result

    @staticmethod
    def _option_description(t: TradeRecord) -> str:
        return f"{t.qty}x {t.underlying} {t.strike:g} CALL exp {t.expiry}"
