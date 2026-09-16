from swing_data.analysis_access import PUBLIC, access_document


def test_access_entry_links_and_scope():
    text = access_document()
    assert f"{PUBLIC}/manifest.json" in text
    assert f"{PUBLIC}/stocks/銘柄コード.csv" in text
    assert "GitHub連携" in text
    assert "掲示板ランキングで母集団を制限しない" in text
    assert "PARTIALは全銘柄取得済みを意味しない" in text
    assert "権限エラーは停止" in text


def test_publish_generates_entry_before_failed_run_return():
    from pathlib import Path
    source = Path("swing_data/publish.py").read_text(encoding="utf-8")
    assert source.index('target / "analysis_access.md"') < source.index("if not latest")
