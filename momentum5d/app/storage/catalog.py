from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from app.storage.schemas import DATASET_SCHEMAS


class DuckDBCatalog:
    def __init__(self, database_path: Path, processed_dir: Path) -> None:
        self.database_path = database_path
        self.processed_dir = processed_dir
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def refresh(self) -> None:
        with duckdb.connect(str(self.database_path)) as connection:
            for dataset, schema in DATASET_SCHEMAS.items():
                files = list((self.processed_dir / dataset).glob("year=*/data.parquet"))
                if files:
                    glob_path = (
                        (self.processed_dir / dataset / "year=*" / "data.parquet")
                        .resolve()
                        .as_posix()
                        .replace("'", "''")
                    )
                    connection.execute(
                        f'CREATE OR REPLACE VIEW "{dataset}" AS '
                        f"SELECT * FROM read_parquet('{glob_path}', "
                        "union_by_name=true, hive_partitioning=false)"
                    )
                else:
                    expressions = ", ".join(
                        f'CAST(NULL AS {sql_type}) AS "{name}"' for name, sql_type in schema.columns
                    )
                    connection.execute(
                        f'CREATE OR REPLACE VIEW "{dataset}" AS SELECT {expressions} WHERE FALSE'
                    )
            quality_path = self.processed_dir / "quality" / "latest.parquet"
            if quality_path.exists():
                sql_path = quality_path.resolve().as_posix().replace("'", "''")
                connection.execute(
                    "CREATE OR REPLACE VIEW quality_issues AS "
                    f"SELECT * FROM read_parquet('{sql_path}')"
                )
            else:
                connection.execute(
                    "CREATE OR REPLACE VIEW quality_issues AS "
                    "SELECT "
                    "CAST(NULL AS VARCHAR) AS checked_at, "
                    "CAST(NULL AS VARCHAR) AS severity, "
                    "CAST(NULL AS VARCHAR) AS check_name, "
                    "CAST(NULL AS VARCHAR) AS dataset, "
                    "CAST(NULL AS DATE) AS date, "
                    "CAST(NULL AS VARCHAR) AS code, "
                    "CAST(NULL AS VARCHAR) AS message, "
                    "CAST(NULL AS VARCHAR) AS observed_value "
                    "WHERE FALSE"
                )
            backtest_views = {
                "backtest_predictions": (
                    "latest_predictions.parquet",
                    (
                        ("date", "DATE"),
                        ("code", "VARCHAR"),
                        ("fold", "INTEGER"),
                        ("probability", "DOUBLE"),
                        ("rank", "INTEGER"),
                        ("selected", "BOOLEAN"),
                        ("target_5d", "INTEGER"),
                        ("future_max_return", "DOUBLE"),
                        ("turnover_mean_20", "DOUBLE"),
                    ),
                ),
                "backtest_trades": (
                    "latest_trades.parquet",
                    (
                        ("date", "DATE"),
                        ("code", "VARCHAR"),
                        ("probability", "DOUBLE"),
                        ("rank", "INTEGER"),
                        ("trade_net_return", "DOUBLE"),
                    ),
                ),
                "backtest_equity_curve": (
                    "latest_equity_curve.parquet",
                    (
                        ("exit_date", "DATE"),
                        ("cohort_return", "DOUBLE"),
                        ("portfolio_return", "DOUBLE"),
                        ("equity", "DOUBLE"),
                        ("drawdown", "DOUBLE"),
                    ),
                ),
            }
            for view, (filename, columns) in backtest_views.items():
                path = self.processed_dir / "backtest" / filename
                if path.exists():
                    sql_path = path.resolve().as_posix().replace("'", "''")
                    connection.execute(
                        f'CREATE OR REPLACE VIEW "{view}" AS '
                        f"SELECT * FROM read_parquet('{sql_path}')"
                    )
                else:
                    expressions = ", ".join(
                        f'CAST(NULL AS {sql_type}) AS "{name}"' for name, sql_type in columns
                    )
                    connection.execute(
                        f'CREATE OR REPLACE VIEW "{view}" AS SELECT {expressions} WHERE FALSE'
                    )

    def status(self) -> list[dict[str, Any]]:
        self.refresh()
        result: list[dict[str, Any]] = []
        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            for dataset, schema in DATASET_SCHEMAS.items():
                key_columns = ", ".join(f'"{column}"' for column in schema.primary_key)
                row = connection.execute(
                    f'SELECT COUNT(*) AS rows, MIN("date") AS min_date, '
                    f'MAX("date") AS max_date FROM "{dataset}"'
                ).fetchone()
                duplicate_groups = connection.execute(
                    f"SELECT COUNT(*) FROM (SELECT {key_columns}, COUNT(*) AS n "
                    f'FROM "{dataset}" GROUP BY {key_columns} HAVING COUNT(*) > 1)'
                ).fetchone()[0]
                result.append(
                    {
                        "dataset": dataset,
                        "rows": row[0],
                        "min_date": row[1],
                        "max_date": row[2],
                        "duplicate_key_groups": duplicate_groups,
                    }
                )
        return result
