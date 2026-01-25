"""
Phase 1: データ取得とラベル確認
"""
import sys
from pathlib import Path

# srcとプロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
import pandas as pd

from config import (
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    LABEL_TAKE_PROFIT, LABEL_STOP_LOSS, LABEL_HORIZON_DAYS,
    MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_VOLUME, LIQUIDITY_WINDOW
)
from data import fetch_ohlcv, fetch_topix, fetch_prime_tickers, apply_liquidity_filter
from model.labels import create_labels, analyze_label_distribution


def main():
    print("=" * 60)
    print("Phase 1: データ取得とラベル確認")
    print("=" * 60)

    # 設定表示
    print(f"\n【設定】")
    print(f"  学習期間: {TRAIN_START} 〜 {TRAIN_END}")
    print(f"  検証期間: {TEST_START} 〜 {TEST_END}")
    print(f"  ラベル: 利確+{LABEL_TAKE_PROFIT:.0%} / 損切-{LABEL_STOP_LOSS:.0%} / {LABEL_HORIZON_DAYS}営業日")
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
    try:
        topix = fetch_topix(data_start, data_end)
        print(f"  取得データ: {len(topix)}日分")
    except Exception as e:
        print(f"  警告: TOPIXデータ取得失敗 ({e})")
        print(f"  → TOPIXなしで続行します")
        topix = None

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
        take_profit=LABEL_TAKE_PROFIT,
        stop_loss=LABEL_STOP_LOSS,
        horizon_days=LABEL_HORIZON_DAYS
    )
    print(f"  ラベル付きデータ: {len(labeled_df):,}行")

    # Step 6: ラベル分布確認
    print(f"\n【Step 6】ラベル分布確認")
    stats = analyze_label_distribution(labeled_df)
    print(f"  正例（利確先着）: {stats['positive_count']:,} ({stats['positive_rate']:.1%})")
    print(f"  負例（損切り/タイムアウト）: {stats['negative_count']:,}")

    # 判断基準
    rate = stats['positive_rate']
    print(f"\n【判断】")
    if rate >= 0.20:
        print(f"  positive_rate = {rate:.1%} >= 20%")
        print(f"  → 良好なバランス")
    elif rate >= 0.10:
        print(f"  positive_rate = {rate:.1%} (10%〜20%)")
        print(f"  → 許容範囲")
    else:
        print(f"  positive_rate = {rate:.1%} < 10%")
        print(f"  → 不均衡、class_weight調整推奨")

    # データ保存
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    labeled_df.to_parquet(output_dir / "labeled_data.parquet", index=False)
    print(f"\n【データ保存】")
    print(f"  {output_dir / 'labeled_data.parquet'}")

    if topix is not None:
        topix.to_parquet(output_dir / "topix.parquet", index=False)
        print(f"  {output_dir / 'topix.parquet'}")
    else:
        print(f"  TOPIXデータなし（スキップ）")

    return stats


if __name__ == "__main__":
    stats = main()
