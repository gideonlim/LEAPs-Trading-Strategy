from datetime import date

from leaps_bot.models import OptionDetails


def test_occ_symbol_parsing_spy():
    details = OptionDetails.from_occ_symbol("SPY270417C00440000")
    assert details.underlying == "SPY"
    assert details.expiry == date(2027, 4, 17)
    assert details.option_type == "C"
    assert details.strike == 440.0


def test_occ_symbol_parsing_spym():
    details = OptionDetails.from_occ_symbol("SPYM261218C00050000")
    assert details.underlying == "SPYM"
    assert details.expiry == date(2026, 12, 18)
    assert details.option_type == "C"
    assert details.strike == 50.0


def test_occ_symbol_fractional_strike():
    details = OptionDetails.from_occ_symbol("SPY260918C00547500")
    assert details.strike == 547.5


def test_occ_symbol_put():
    details = OptionDetails.from_occ_symbol("SPY270417P00440000")
    assert details.option_type == "P"
