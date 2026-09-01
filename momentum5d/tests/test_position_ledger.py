from __future__ import annotations

from pathlib import Path

import pytest

from app.position_ledger import (
    ConfirmedFill,
    PositionLedgerError,
    load_fills,
    open_positions,
    record_confirmed_fill,
)


def _fill(
    execution_id: str,
    *,
    side: str = "buy",
    quantity: int = 100,
    price: float = 1_000.0,
) -> ConfirmedFill:
    return ConfirmedFill(
        execution_id=execution_id,
        ticker="6302",
        side=side,  # type: ignore[arg-type]
        quantity=quantity,
        price=price,
        executed_at="2026-09-01T09:00:00+09:00",
    )


def test_only_confirmed_fill_creates_position(tmp_path: Path) -> None:
    ledger = tmp_path / "positions.json"

    result = record_confirmed_fill(ledger, _fill("broker-1"))

    assert result["recorded"] is True
    assert result["positions"] == {
        "6302.T": {"quantity": 100, "average_entry_price": 1_000.0}
    }
    assert len(load_fills(ledger)) == 1


def test_duplicate_execution_is_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "positions.json"
    record_confirmed_fill(ledger, _fill("broker-1"))

    result = record_confirmed_fill(ledger, _fill("broker-1"))

    assert result["recorded"] is False
    assert len(load_fills(ledger)) == 1


def test_unconfirmed_or_oversold_execution_is_rejected(tmp_path: Path) -> None:
    ledger = tmp_path / "positions.json"
    with pytest.raises(PositionLedgerError, match="confirmed"):
        record_confirmed_fill(
            ledger,
            ConfirmedFill(
                execution_id="broker-1",
                ticker="6302",
                side="buy",
                quantity=100,
                price=1_000.0,
                executed_at="2026-09-01T09:00:00+09:00",
                confirmation_status="pending",  # type: ignore[arg-type]
            ),
        )
    with pytest.raises(PositionLedgerError, match="exceeds position"):
        open_positions([_fill("broker-2", side="sell")])


def test_confirmed_sell_reduces_position(tmp_path: Path) -> None:
    ledger = tmp_path / "positions.json"
    record_confirmed_fill(ledger, _fill("broker-1", quantity=200))

    result = record_confirmed_fill(
        ledger, _fill("broker-2", side="sell", quantity=100, price=1_050.0)
    )

    assert result["positions"] == {
        "6302.T": {"quantity": 100, "average_entry_price": 1_000.0}
    }
