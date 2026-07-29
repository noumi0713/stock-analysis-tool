from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "intraday_return",
    "price_range",
    "close_to_ma5",
    "close_to_ma20",
    "volume_ratio_5_20",
    "turnover_ratio_5_20",
    "log_turnover_20",
    "return_5d_rank",
    "return_20d_rank",
    "volume_ratio_rank",
    "turnover_ratio_rank",
]


@dataclass(frozen=True, slots=True)
class LabelConfig:
    horizon_days: int = 5
    threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise ValueError("horizon_days は1以上である必要があります")
        if self.threshold <= 0:
            raise ValueError("threshold は0より大きい必要があります")


class FeatureBuilder:
    """引け時点で利用可能な特徴量と、検証専用の将来ラベルを作成する。"""

    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()

    def build(
        self,
        equities: pd.DataFrame,
        calendar: pd.DataFrame,
    ) -> pd.DataFrame:
        if equities.empty:
            raise ValueError("equities_daily が空です")
        required = {
            "date",
            "code",
            "adjusted_open",
            "adjusted_high",
            "adjusted_low",
            "adjusted_close",
            "adjusted_volume",
            "turnover_value",
        }
        missing = required.difference(equities.columns)
        if missing:
            raise ValueError(f"equities_daily の必須列がありません: {sorted(missing)}")

        frame = equities[
            [
                "date",
                "code",
                "adjusted_open",
                "adjusted_high",
                "adjusted_low",
                "adjusted_close",
                "adjusted_volume",
                "turnover_value",
            ]
        ].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["code"] = frame["code"].astype("string")
        numeric = required.difference({"date", "code"})
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.sort_values(["code", "date"]).reset_index(drop=True)

        group = frame.groupby("code", sort=False)
        close = frame["adjusted_close"]
        frame["return_1d"] = close / group["adjusted_close"].shift(1) - 1.0
        frame["return_5d"] = close / group["adjusted_close"].shift(5) - 1.0
        frame["return_20d"] = close / group["adjusted_close"].shift(20) - 1.0
        frame["volatility_20d"] = frame.groupby("code", sort=False)["return_1d"].transform(
            lambda values: values.rolling(20, min_periods=20).std()
        )
        frame["intraday_return"] = frame["adjusted_close"] / frame["adjusted_open"] - 1.0
        frame["price_range"] = frame["adjusted_high"] / frame["adjusted_low"] - 1.0

        close_ma5 = group["adjusted_close"].transform(
            lambda values: values.rolling(5, min_periods=5).mean()
        )
        close_ma20 = group["adjusted_close"].transform(
            lambda values: values.rolling(20, min_periods=20).mean()
        )
        volume_ma5 = group["adjusted_volume"].transform(
            lambda values: values.rolling(5, min_periods=5).mean()
        )
        volume_ma20 = group["adjusted_volume"].transform(
            lambda values: values.rolling(20, min_periods=20).mean()
        )
        turnover_ma5 = group["turnover_value"].transform(
            lambda values: values.rolling(5, min_periods=5).mean()
        )
        turnover_ma20 = group["turnover_value"].transform(
            lambda values: values.rolling(20, min_periods=20).mean()
        )
        frame["close_to_ma5"] = close / close_ma5 - 1.0
        frame["close_to_ma20"] = close / close_ma20 - 1.0
        frame["volume_ratio_5_20"] = volume_ma5 / volume_ma20
        frame["turnover_ratio_5_20"] = turnover_ma5 / turnover_ma20
        frame["turnover_mean_20"] = turnover_ma20
        frame["log_turnover_20"] = np.log1p(turnover_ma20.clip(lower=0))

        frame["return_5d_rank"] = frame.groupby("date")["return_5d"].rank(pct=True)
        frame["return_20d_rank"] = frame.groupby("date")["return_20d"].rank(pct=True)
        frame["volume_ratio_rank"] = frame.groupby("date")["volume_ratio_5_20"].rank(pct=True)
        frame["turnover_ratio_rank"] = frame.groupby("date")["turnover_ratio_5_20"].rank(pct=True)

        frame = self._attach_calendar_positions(frame, calendar)
        frame = self._attach_future_outcomes(frame)
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame.sort_values(["date", "code"]).reset_index(drop=True)

    def _attach_calendar_positions(
        self, frame: pd.DataFrame, calendar: pd.DataFrame
    ) -> pd.DataFrame:
        if calendar.empty:
            trading_dates = sorted(frame["date"].drop_duplicates())
        else:
            business = calendar.loc[calendar["holiday_division"].astype("string") == "1", "date"]
            trading_dates = sorted(pd.to_datetime(business).dt.date.drop_duplicates())
        positions = pd.DataFrame(
            {
                "date": trading_dates,
                "calendar_position": range(len(trading_dates)),
            }
        )
        merged = frame.merge(positions, how="left", on="date", validate="many_to_one")
        if merged["calendar_position"].isna().any():
            raise ValueError("株価日に対応する営業日カレンダーがありません")
        merged["calendar_position"] = merged["calendar_position"].astype("int64")
        merged.attrs["max_calendar_position"] = len(trading_dates) - 1
        return merged

    def _attach_future_outcomes(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame
        horizon = self.config.horizon_days
        lookup = frame[
            ["code", "calendar_position", "adjusted_open", "adjusted_high", "adjusted_close"]
        ]
        for offset in range(1, horizon + 1):
            selected_columns = ["code", "calendar_position", "adjusted_high"]
            rename = {"adjusted_high": f"future_high_{offset}"}
            if offset == 1:
                selected_columns.append("adjusted_open")
                rename["adjusted_open"] = "future_open_1"
            if offset == horizon:
                selected_columns.append("adjusted_close")
                rename["adjusted_close"] = f"future_close_{horizon}"
            future = lookup[selected_columns].copy()
            future["calendar_position"] = future["calendar_position"] - offset
            future = future.rename(columns=rename)
            result = result.merge(
                future,
                how="left",
                on=["code", "calendar_position"],
                validate="one_to_one",
            )

        high_columns = [f"future_high_{offset}" for offset in range(1, horizon + 1)]
        max_position = int(frame.attrs["max_calendar_position"])
        horizon_complete = result["calendar_position"] + horizon <= max_position
        result["horizon_complete"] = horizon_complete
        result["future_max_high"] = result[high_columns].max(axis=1, skipna=True)
        result["future_max_return"] = result["future_max_high"] / result["adjusted_close"] - 1.0
        result["target_5d"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
        label_available = horizon_complete & result["adjusted_close"].notna()
        result.loc[label_available, "target_5d"] = (
            result.loc[label_available, "future_max_return"] >= self.config.threshold
        ).astype("int8")

        entry = result["future_open_1"]
        target_price = entry * (1.0 + self.config.threshold)
        hit_day = pd.Series(pd.NA, index=result.index, dtype="Int8")
        for offset in range(1, horizon + 1):
            hit = hit_day.isna() & entry.notna() & (result[f"future_high_{offset}"] >= target_price)
            hit_day.loc[hit] = offset
        result["target_hit_day"] = hit_day
        result["entry_price"] = entry
        result["exit_price"] = target_price.where(
            hit_day.notna(), result[f"future_close_{horizon}"]
        )
        result["trade_gross_return"] = result["exit_price"] / entry - 1.0
        result["trade_outcome_available"] = (
            horizon_complete & entry.notna() & result["exit_price"].notna()
        )
        result["exit_calendar_position"] = (result["calendar_position"] + horizon).where(
            horizon_complete
        )
        exit_dates = (
            result[["date", "calendar_position"]]
            .drop_duplicates("calendar_position")
            .rename(
                columns={
                    "date": "exit_date",
                    "calendar_position": "exit_calendar_position",
                }
            )
        )
        result = result.merge(
            exit_dates,
            how="left",
            on="exit_calendar_position",
            validate="many_to_one",
        )
        return result
