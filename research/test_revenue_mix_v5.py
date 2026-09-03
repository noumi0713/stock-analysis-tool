from __future__ import annotations

from collect_revenue_mix_v5 import (
    map_theme_share,
    parse_explicit_percent_segments,
    parse_external_customer_amount_segments,
)


def close(a: float | None, b: float, tol: float = 0.15) -> None:
    assert a is not None, f"expected {b}, got None"
    assert abs(a - b) <= tol, f"expected {b}, got {a}"


def test_ikk() -> None:
    text = """
    事業概要 売上構成
    婚礼事業 92.5% 介護事業 3.0% 食品事業 2.0% フォト事業 4.0%
    主力のウェディング事業が売上の約92.5％を占める。
    """
    segs = parse_explicit_percent_segments(text)
    close(map_theme_share("介護関連", segs)[0], 3.0)
    close(map_theme_share("食品", segs)[0], 2.0)


def test_sata() -> None:
    text = """
    セグメント別売上高構成比（2026年3月期実績）
    土木事業 30.5% 建築事業 68.3% 兼業事業 1.2%
    下水道工事にも取り組む。
    """
    segs = parse_explicit_percent_segments(text)
    close(map_theme_share("建設", segs)[0], 98.8)
    assert map_theme_share("下水道", segs)[0] is None


def test_yondenko() -> None:
    text = """
    事業別連結売上高 2024年度 1,058億円
    電気・計装工事（34.4%） 情報通信工事（7.8%） 送電・土木工事（4.8%）
    リース（1.5%） 空調・管工事（14.8%） 配電工事（33.4%） 太陽光発電（2.0%） その他（1.3%）
    """
    segs = parse_explicit_percent_segments(text)
    close(map_theme_share("太陽光発電関連", segs)[0], 2.0)
    close(map_theme_share("リース", segs)[0], 1.5)


def test_meito() -> None:
    text = """
    報告セグメント 食品事業 化成品事業 不動産事業 計
    売上高 外部顧客への売上高 5,223 993 119 6,337
    セグメント間の内部売上高又は振替高 － － － －
    """
    segs = parse_external_customer_amount_segments(text)
    close(map_theme_share("食品", segs)[0], 82.42, 0.2)


def test_hibiya_order_share_not_sales() -> None:
    text = """
    受注高 顧客別/分野別（連結）
    民間の構成比は70.0%。大型データセンター案件の受注により、
    データセンター/情報の構成比が76.7%まで増加。
    """
    segs = parse_explicit_percent_segments(text)
    assert map_theme_share("データセンター", segs)[0] is None


def test_crosscat_nonsegment_percent_not_used() -> None:
    text = """
    Comprehensive income 1,367 million [(5.4)%]. ROE 24.1%. Operating profit to net sales ratio 11.3%.
    IT services and cloud modernization are core activities.
    """
    segs = parse_explicit_percent_segments(text)
    assert map_theme_share("IT関連", segs)[0] is None
    assert map_theme_share("クラウドコンピューティング", segs)[0] is None


def main() -> None:
    tests = [test_ikk, test_sata, test_yondenko, test_meito, test_hibiya_order_share_not_sales, test_crosscat_nonsegment_percent_not_used]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("PASS all", len(tests))


if __name__ == "__main__":
    main()
