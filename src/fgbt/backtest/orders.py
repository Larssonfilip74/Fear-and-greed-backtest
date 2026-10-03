"""Order types supported by the engine (RTH-only, long-only; see protocol §6)."""
from dataclasses import dataclass
from datetime import date

SIDES = ("BUY", "SELL")
KINDS = ("MOO", "MOC", "LIMIT", "STOP")


@dataclass(frozen=True)
class Order:
    side: str
    kind: str
    qty: int
    price: float | None = None  # LIMIT/STOP level (points)
    valid_sessions: int = 1  # LIMIT/STOP: sessions the order stays working, starting with the first
    tag: str = ""

    def __post_init__(self):
        if self.side not in SIDES or self.kind not in KINDS:
            raise ValueError(f"bad order {self}")
        if self.qty <= 0 or self.valid_sessions < 1:
            raise ValueError(f"qty and valid_sessions must be positive: {self}")
        if self.kind in ("LIMIT", "STOP"):
            if self.price is None:
                raise ValueError(f"{self.kind} needs a price")
            if self.side == "SELL":
                raise ValueError("SELL supports MOO/MOC only (no stop-losses by design)")


@dataclass(frozen=True)
class Fill:
    session: date
    side: str
    kind: str
    qty: int
    price: float  # includes slippage
    tag: str


@dataclass(frozen=True)
class Cancel:
    """Cancel working LIMIT/STOP orders: all of them (tag=None) or only those with this tag."""
    tag: str | None = None
