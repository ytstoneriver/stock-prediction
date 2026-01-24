"""
Phase 1: データ取得とラベル確認
"""
import sys
from pathlib import Path

# srcをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import date
import pandas as pd

from config import (
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    LABEL_TARGET_RETURN, LABEL_HORIZON_DAYS,
    MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_VOLUME, LIQUIDITY_WINDOW
)
from data import fetch_ohlcv, fetch_topix, fetch_prime_tickers, apply_liquidity_filter
from model import create_labels, calculate_positive_rate, print_positive_rate_report


def main():
    print("=" * 60)
    print("Phase 1: データ取得とラベル確認")
    print("=" * 60)

    # 設定表示
    print(f"\n【設定】")
    print(f"  学習期間: {TRAIN_START} 〜 {TRAIN_END}")
    print(f"  検証期間: {TEST_START} 〜 {TEST_END}")
    print(f"  ラベル: +{LABEL_TARGET_RETURN:.0%} / {LABEL_HORIZON_DAYS}営業日")
    print(f"  流動性フィルタ:")
    print(f"    株価: {MIN_PRICE:,} 〜 {MAX_PRICE:,}円")
    print(f"    売買代金: {MIN_TURNOVER/1e8:.1f}億円/日以上")
    print(f"    出来高: {MIN_VOLUME/1e4:.0f}万株/日以上")

    # Step 1: 銘柄リスト取得
    print(f"\n【Step 1】銘柄リスト取得")
    tickers = fetch_prime_tickers()
    print(f"  取得銘柄数: {len(tickers)}")

    # Step 2: 株価データ取得
    print(f"\n【Step 2】株価データ取得")
    # 学習期間 + 検証期間 + ラベル用の余裕期間
    data_start = TRAIN_START
    data_end = TEST_END

    df = fetch_ohlcv(tickers, data_start, data_end)
    print(f"  取得データ: {df['ticker'].nunique()}銘柄, {len(df):,}行")
    print(f"  期間: {df['date'].min()} 〜 {df['date'].max()}")

    # Step 3: TOPIXデータ取得
    print(f"\n【Step 3】TOPIXデータ取得")
    topix = fetch_topix(data_start, data_end)
    print(f"  取得データ: {len(topix)}日分")

    # Step 4: ユニバースフィルタ適用
    print(f"\n【Step 4】ユニバースフィルタ適用")
    filtered_df = apply_liquidity_filter(
        df,
        min_price=MIN_PRICE,
        max_price=MAX_PRICE,
        min_turnover=MIN_TURNOVER,
        min_volume=MIN_VOLUME,
        window=LIQUIDITY_WINDOW
    )
    print(f"  フィルタ後: {filtered_df['ticker'].nunique()}銘柄, {len(filtered_df):,}行")

    # 日別の銘柄数を確認
    daily_count = filtered_df.groupby('date')['ticker'].nunique()
    print(f"  日別銘柄数: 平均{daily_count.mean():.0f}, 最小{daily_count.min()}, 最大{daily_count.max()}")

    # Step 5: ラベル生成
    print(f"\n【Step 5】ラベル生成")
    labeled_df = create_labels(
        filtered_df,
        target_return=LABEL_TARGET_RETURN,
        horizon_days=LABEL_HORIZON_DAYS
    )

    # 学習期間のみでpositive_rate計算（検証期間はラベルが不完全になる可能性）
    train_df = labeled_df[
        (labeled_df['date'] >= TRAIN_START) &
        (labeled_df['date'] <= TRAIN_END)
    ]
    print(f"  学習期間データ: {len(train_df):,}行")

    # Step 6: positive_rate確認
    print(f"\n【Step 6】positive_rate確認")
    stats = calculate_positive_rate(train_df)
    print_positive_rate_report(stats)

    # 判断基準
    rate = stats['overall']['rate']
    print(f"\n【判断】")
    if rate >= 0.10:
        print(f"  positive_rate = {rate:.2%} >= 10%")
        print(f"  → そのまま継続")
    elif rate >= 0.05:
        print(f"  positive_rate = {rate:.2%} (5%〜10%)")
        print(f"  → 継続（class_weight調整推奨）")
    else:
        print(f"  positive_rate = {rate:.2%} < 5%")
        print(f"  → 閾値を+8%に下げることを検討")

    # データ保存
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    labeled_df.to_parquet(output_dir / "labeled_data.parquet", index=False)
    topix.to_parquet(output_dir / "topix.parquet", index=False)
    print(f"\n【データ保存】")
    print(f"  {output_dir / 'labeled_data.parquet'}")
    print(f"  {output_dir / 'topix.parquet'}")

    return stats


if __name__ == "__main__":
    stats = main()
