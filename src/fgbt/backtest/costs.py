"""Fill-price and cost model (protocol §6)."""
from dataclasses import dataclass

from fgbt.instruments import INSTRUMENTS, Instrument


@dataclass(frozen=True)
class CostModel:
    inst: Instrument
    slippage_ticks: float = 1.0  # per side, adverse
    roll_ticks: float = 1.0  # spread paid per roll, per contract
    charge_roll_commission: bool = True

    @classmethod
    def for_symbol(cls, symbol: str = "ES", **kw) -> "CostModel":
        return cls(INSTRUMENTS[symbol], **kw)

    def slipped(self, raw: float, side: str) -> float:
        s = self.slippage_ticks * self.inst.tick_size
        return raw + s if side == "BUY" else raw - s

    @property
    def commission_per_side(self) -> float:
        return self.inst.commission_rt / 2

    @property
    def roll_cost_per_contract(self) -> float:
        return self.roll_ticks * self.inst.tick_value + (self.inst.commission_rt if self.charge_roll_commission else 0.0)
