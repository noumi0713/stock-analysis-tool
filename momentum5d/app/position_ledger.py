from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

Side = Literal["buy", "sell"]


class PositionLedgerError(ValueError):
    """Raised when an execution report cannot safely change the position ledger."""


@dataclass(frozen=True, slots=True)
class ConfirmedFill:
    execution_id: str
    ticker: str
    side: Side
    quantity: int
    price: float
    executed_at: str
    confirmation_status: Literal["confirmed"] = "confirmed"
    strategy: str | None = None


def _normalize_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not ticker:
        raise PositionLedgerError("ticker is required")
    return ticker if "." in ticker else f"{ticker}.T"


def _validate_fill(fill: ConfirmedFill) -> ConfirmedFill:
    if fill.confirmation_status != "confirmed":
        raise PositionLedgerError("Only confirmed executions may enter the ledger")
    if not fill.execution_id.strip():
        raise PositionLedgerError("execution_id is required")
    if fill.side not in {"buy", "sell"}:
        raise PositionLedgerError("side must be buy or sell")
    if isinstance(fill.quantity, bool) or int(fill.quantity) != fill.quantity:
        raise PositionLedgerError("quantity must be a whole number")
    if fill.quantity <= 0:
        raise PositionLedgerError("quantity must be positive")
    if fill.price <= 0:
        raise PositionLedgerError("price must be positive")
    try:
        executed_at = datetime.fromisoformat(fill.executed_at)
    except ValueError as exc:
        raise PositionLedgerError("executed_at must be ISO-8601") from exc
    if executed_at.tzinfo is None:
        raise PositionLedgerError("executed_at must be timezone-aware")
    return ConfirmedFill(
        execution_id=fill.execution_id.strip(),
        ticker=_normalize_ticker(fill.ticker),
        side=fill.side,
        quantity=int(fill.quantity),
        price=float(fill.price),
        executed_at=executed_at.isoformat(),
        confirmation_status="confirmed",
        strategy=fill.strategy,
    )


def load_fills(path: Path) -> list[ConfirmedFill]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise PositionLedgerError("Unsupported position ledger schema")
    events = payload.get("confirmed_fills")
    if not isinstance(events, list):
        raise PositionLedgerError("confirmed_fills must be a list")
    return [_validate_fill(ConfirmedFill(**event)) for event in events]


def open_positions(fills: list[ConfirmedFill]) -> dict[str, dict[str, float | int]]:
    positions: dict[str, dict[str, float | int]] = {}
    for fill in sorted(
        fills,
        key=lambda item: (datetime.fromisoformat(item.executed_at), item.execution_id),
    ):
        position = positions.setdefault(
            fill.ticker, {"quantity": 0, "average_entry_price": 0.0}
        )
        quantity = int(position["quantity"])
        average = float(position["average_entry_price"])
        if fill.side == "buy":
            new_quantity = quantity + fill.quantity
            position["average_entry_price"] = (
                quantity * average + fill.quantity * fill.price
            ) / new_quantity
            position["quantity"] = new_quantity
        else:
            if fill.quantity > quantity:
                raise PositionLedgerError(
                    f"Confirmed sell exceeds position: {fill.ticker} "
                    f"sell={fill.quantity} held={quantity}"
                )
            position["quantity"] = quantity - fill.quantity
            if position["quantity"] == 0:
                position["average_entry_price"] = 0.0
    return {
        ticker: position
        for ticker, position in positions.items()
        if int(position["quantity"]) > 0
    }


def record_confirmed_fill(path: Path, fill: ConfirmedFill) -> dict[str, object]:
    normalized = _validate_fill(fill)
    fills = load_fills(path)
    existing = {item.execution_id: item for item in fills}
    if normalized.execution_id in existing:
        if existing[normalized.execution_id] != normalized:
            raise PositionLedgerError(
                "execution_id already exists with different execution details"
            )
        return {
            "recorded": False,
            "reason": "duplicate_confirmed_execution",
            "positions": open_positions(fills),
        }

    candidate = [*fills, normalized]
    positions = open_positions(candidate)
    payload = {
        "schema_version": 1,
        "policy": "confirmed_executions_only",
        "confirmed_fills": [asdict(item) for item in candidate],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    return {"recorded": True, "positions": positions}
