"""The adapter registry — one instance per dataset. Add a dataset = add a module + a line here."""

from .base import Adapter, melt, order_adapters
from .ca_libpas import CALibPAS
from .census import Equity
from .hud import HUD, CountyHomeless
from .imls import Core, CountyPanel, Metro, State

# Build order: equity & hud before county_homeless (it needs their tables); unify uses everyone.
REGISTRY: list[Adapter] = [
    Core(),
    Metro(),
    State(),
    CountyPanel(),
    Equity(),
    HUD(),
    CountyHomeless(),
    CALibPAS(),
]
BY_NAME = {a.name: a for a in REGISTRY}

__all__ = ["Adapter", "melt", "order_adapters", "REGISTRY", "BY_NAME"]
