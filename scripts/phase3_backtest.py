"""
Phase 3: バックテスト
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from datetime import date
import time

from config import (
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    CAPITAL, TOP_N, TAKE_PROFIT, STOP_LOSS_ATR_MULT, MAX_HOLD_DAYS
)
from backtest import BacktestEngine, BacktestResult


def main():
    print("=" * 60)
    print("Phase 3: バックテスト")
    print("=" * 60)

    # 設定表示
    print(f"\n【設定】")
    print(f"  資金: ¥{CAPITAL:,}")
    print(f"  最大同時保有: {TOP_N}銘柄")
    print(f"  利確: +{TAKE_PROFIT:.0%}")
    print(f"  損切: -{STOP_LOSS_ATR_MULT}×ATR14")
    print(f"  最大保有期間: {MAX_HOLD_DAYS}営業日")

    # Step 1: データ読み込み
    print("\n【Step 1】データ読み込み")
    data_dir = Path(__file__).parent.parent / "data"

    predictions = pd.read_parquet(data_dir / "test_predictions.parquet")
    print(f"  予測データ: {len(predictions):,}行")

    labeled_data = pd.read_parquet(data_dir / "labeled_data.parquet")
    print(f"  OHLCVデータ: {len(labeled_data):,}行")

    # TOPIXデータ（あれば）
    topix_path = data_dir / "topix.parquet"
    topix_df = None
    if topix_path.exists():
        topix_df = pd.read_parquet(topix_path)
        print(f"  TOPIX: {len(topix_df)}日分")
    else:
        print("  TOPIX: なし")

    # Step 2: バックテスト実行（地合いフィルタなし）
    print("\n【Step 2】バックテスト（地合いフィルタなし）")

    engine = BacktestEngine(
        initial_capital=CAPITAL,
        position_size=CAPITAL // TOP_N,
        max_positions=TOP_N,
        take_profit=TAKE_PROFIT,
        stop_loss_atr_mult=STOP_LOSS_ATR_MULT,
        max_hold_days=MAX_HOLD_DAYS
    )

    result_no_filter = engine.run(
        predictions_df=predictions,
        ohlcv_df=labeled_data,
        topix_df=None,
        use_market_filter=False
    )

    print("\n--- 地合いフィルタなし ---")
    result_no_filter.print_report()

    # 決済理由の内訳
    print("\n【決済理由内訳】")
    reasons = {}
    for t in result_no_filter.trades:
        if t.exit_reason:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        pct = count / result_no_filter.n_trades * 100 if result_no_filter.n_trades > 0 else 0
        print(f"  {reason}: {count}回 ({pct:.1f}%)")

    # Step 3: TOPIXを取得して地合いフィルタありでバックテスト
    if topix_df is None:
        print("\n【Step 3】TOPIXデータ取得")
        try:
            from data import fetch_topix
            topix_df = fetch_topix(TEST_START, TEST_END)
            topix_df.to_parquet(data_dir / "topix.parquet", index=False)
            print(f"  TOPIX取得成功: {len(topix_df)}日分")
        except Exception as e:
            print(f"  TOPIX取得失敗: {e}")
            print("  → 地合いフィルタなしの結果のみで評価")
            topix_df = None

    if topix_df is not None:
        print("\n【Step 4】バックテスト（地合いフィルタあり）")

        result_with_filter = engine.run(
            predictions_df=predictions,
            ohlcv_df=labeled_data,
            topix_df=topix_df,
            use_market_filter=True
        )

        print("\n--- 地合いフィルタあり（TOPIX > MA25） ---")
        result_with_filter.print_report()

        # 比較
        print("\n【比較】")
        print(f"{'':20} {'フィルタなし':>15} {'フィルタあり':>15}")
        print("-" * 50)
        print(f"{'総損益':20} ¥{result_no_filter.total_pnl:>13,.0f} ¥{result_with_filter.total_pnl:>13,.0f}")
        print(f"{'総損益率':20} {result_no_filter.total_pnl_pct:>14.2f}% {result_with_filter.total_pnl_pct:>14.2f}%")
        print(f"{'取引回数':20} {result_no_filter.n_trades:>15} {result_with_filter.n_trades:>15}")
        print(f"{'勝率':20} {result_no_filter.win_rate:>14.1%} {result_with_filter.win_rate:>14.1%}")
        print(f"{'PF':20} {result_no_filter.profit_factor:>15.2f} {result_with_filter.profit_factor:>15.2f}")
        print(f"{'最大DD':20} {result_no_filter.max_drawdown_pct:>14.2f}% {result_with_filter.max_drawdown_pct:>14.2f}%")

    # 判断
    print("\n【判断】")
    if result_no_filter.total_pnl > 0 and result_no_filter.win_rate >= 0.5:
        print(f"  総損益 = ¥{result_no_filter.total_pnl:,.0f} > 0、勝率 = {result_no_filter.win_rate:.1%} >= 50%")
        print(f"  → 成功。パラメータ微調整で改善可能")
    elif result_no_filter.total_pnl < 0 and result_no_filter.win_rate >= 0.45:
        print(f"  総損益 = ¥{result_no_filter.total_pnl:,.0f} < 0、勝率 = {result_no_filter.win_rate:.1%} >= 45%")
        print(f"  → 損切り/利確調整で改善余地あり")
    else:
        print(f"  勝率 = {result_no_filter.win_rate:.1%} < 40%")
        print(f"  → モデル・特徴量の見直しが必要")

    # 取引履歴保存
    trades_df = pd.DataFrame([
        {
            'ticker': t.ticker,
            'entry_date': t.entry_date,
            'entry_price': t.entry_price,
            'exit_date': t.exit_date,
            'exit_price': t.exit_price,
            'exit_reason': t.exit_reason,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'hold_days': t.hold_days
        }
        for t in result_no_filter.trades
    ])
    trades_df.to_csv(data_dir / "backtest_trades.csv", index=False)
    print(f"\n【保存】")
    print(f"  取引履歴: {data_dir / 'backtest_trades.csv'}")

    return result_no_filter


if __name__ == "__main__":
    result = main()
