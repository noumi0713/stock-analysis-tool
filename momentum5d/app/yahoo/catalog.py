from __future__ import annotations

from pathlib import Path

import duckdb

from app.yahoo.ingestion import YahooPaths


class YahooDuckDBCatalog:
    def __init__(self, paths: YahooPaths) -> None:
        self.paths = paths
        self.database_path = paths.metadata_dir / "market.duckdb"

    def refresh(self) -> None:
        self.paths.ensure()
        views = {
            "equities_daily": self.paths.prices_path,
            "latest_candidates": (
                self.paths.processed_dir / "analysis" / "latest_candidates.parquet"
            ),
            "historical_patterns": (
                self.paths.processed_dir / "analysis" / "historical_patterns.parquet"
            ),
            "quality_issues": self.paths.processed_dir / "quality" / "latest.parquet",
            "prime_universe": self.paths.universe_path,
        }
        with duckdb.connect(str(self.database_path)) as connection:
            for view, path in views.items():
                if not path.exists():
                    continue
                sql_path = _sql_path(path)
                connection.execute(
                    f'CREATE OR REPLACE VIEW "{view}" AS '
                    f"SELECT * FROM read_parquet('{sql_path}', union_by_name=true)"
                )


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")
