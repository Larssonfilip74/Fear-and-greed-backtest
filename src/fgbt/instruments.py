"""CME equity-index futures contract specifications and cost model (RESEARCH_PROTOCOL.md §6)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    point_value: float  # USD per 1.00 index point
    tick_size: float  # index points
    commission_rt: float  # USD per contract, round turn, incl. exchange/NFA fees

    @property
    def tick_value(self) -> float:
        return self.point_value * self.tick_size


INSTRUMENTS = {
    "ES": Instrument("ES", 50.0, 0.25, 5.00),
    "NQ": Instrument("NQ", 20.0, 0.25, 5.00),
    "MES": Instrument("MES", 5.0, 0.25, 1.50),
    "MNQ": Instrument("MNQ", 2.0, 0.25, 1.50),
}


def round_trip_cost_usd(symbol: str, slippage_ticks_per_side: float = 1.0) -> float:
    """Commission + fees + slippage for one contract, entry and exit."""
    inst = INSTRUMENTS[symbol]
    return inst.commission_rt + 2 * slippage_ticks_per_side * inst.tick_value
