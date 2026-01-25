"""
Phase 2: モデル学習

ファンダメンタルフィルタを適用し、特徴量を生成して学習
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import date

from config import (
    ROLLING_TRAIN_MONTHS, TEST_START, TEST_END,
    LABEL_TAKE_PROFIT, LABEL_STOP_LOSS, LABEL_HORIZON_DAYS,
    TOP_N_STOCKS, SCORE_THRESHOLD, MIN_SCORE_GAP,
    FUNDAMENTAL_FILTER, TOP_N_FEATURES,
)
from data.universe_filter import (
    apply_fundamental_filter,
    get_sector_mapping,
    print_filter_report,
)
from model.labels import create_labels, analyze_label_distribution, print_label_report
from features.builder import build_features, get_feature_columns, get_feature_groups
from model.trainer import StockPredictor, print_evaluation_report


def main():
    print("=" * 60)
    print("Phase 2: モデル学習")
    print("=" * 60)

    # 設定表示
    print(f"\n【設定】")
    print(f"  ラベル: 利確+{LABEL_TAKE_PROFIT:.0%} / 損切-{LABEL_STOP_LOSS:.0%} / {LABEL_HORIZON_DAYS}営業日")
    print(f"  ローリング学習: 直近{ROLLING_TRAIN_MONTHS}ヶ月")
    print(f"  評価期間: {TEST_START} 〜 {TEST_END}")
    print(f"  ファンダフィルタ:")
    print(f"    ROE > {FUNDAMENTAL_FILTER['min_roe']*100:.0f}%")
    print(f"    時価総額 > {FUNDAMENTAL_FILTER['min_market_cap']/1e9:.0f}億円")

    data_dir = Path(__file__).parent.parent / "data"

    # Step 1: OHLCVデータ読み込み
    print("\n【Step 1】データ読み込み")

    labeled_path = data_dir / "labeled_data.parquet"
    if not labeled_path.exists():
        print(f"  エラー: {labeled_path} が見つかりません")
        print(f"  → 先に phase1_data_check.py を実行してください")
        return None

    ohlcv_df = pd.read_parquet(labeled_path)
    if 'label' in ohlcv_df.columns:
        ohlcv_df = ohlcv_df.drop(columns=['label'])

    original_tickers = ohlcv_df['ticker'].unique().tolist()
    print(f"  OHLCV: {len(original_tickers)}銘柄, {len(ohlcv_df):,}行")

    topix_path = data_dir / "topix.parquet"
    topix_df = pd.read_parquet(topix_path) if topix_path.exists() else None
    if topix_df is not None:
        print(f"  TOPIX: {len(topix_df)}日分")
    else:
        print(f"  TOPIX: なし（スキップ）")

    # Step 2: ファンダメンタルフィルタ
    print("\n【Step 2】ファンダメンタルフィルタ")

    fundamental_cache = data_dir / "fundamental_cache.parquet"
    if fundamental_cache.exists():
        print(f"  キャッシュ読み込み: {fundamental_cache}")
        fundamental_df = pd.read_parquet(fundamental_cache)
    else:
        fundamental_df = None

    filter_result = apply_fundamental_filter(
        original_tickers,
        fundamental_df=fundamental_df,
        min_roe=FUNDAMENTAL_FILTER['min_roe'],
        min_market_cap=FUNDAMENTAL_FILTER['min_market_cap'],
        show_progress=True,
    )

    # キャッシュ保存
    if not fundamental_cache.exists():
        filter_result['fundamental_df'].to_parquet(fundamental_cache, index=False)
        print(f"  キャッシュ保存: {fundamental_cache}")

    print_filter_report(filter_result)

    filtered_tickers = filter_result['filtered_tickers']
    sector_mapping = get_sector_mapping(filter_result['fundamental_df'])

    if len(filtered_tickers) == 0:
        print("  エラー: フィルタ後の銘柄数が0です")
        return None

    # OHLCVにフィルタ適用
    ohlcv_df = ohlcv_df[ohlcv_df['ticker'].isin(filtered_tickers)]
    print(f"\n  フィルタ後: {ohlcv_df['ticker'].nunique()}銘柄, {len(ohlcv_df):,}行")

    # Step 3: ラベル生成
    print("\n【Step 3】ラベル生成")
    print(f"  利確: +{LABEL_TAKE_PROFIT:.0%}, 損切: -{LABEL_STOP_LOSS:.0%}, 期間: {LABEL_HORIZON_DAYS}日")

    labeled_df = create_labels(
        ohlcv_df,
        take_profit=LABEL_TAKE_PROFIT,
        stop_loss=LABEL_STOP_LOSS,
        horizon_days=LABEL_HORIZON_DAYS
    )
    print(f"  ラベル付きデータ: {len(labeled_df):,}行")

    stats = analyze_label_distribution(labeled_df)
    print_label_report(stats)

    # Step 4: 特徴量生成
    print("\n【Step 4】特徴量生成")
    feature_df = build_features(labeled_df, topix_df, sector_mapping)
    feature_cols = get_feature_columns(feature_df)
    print(f"  特徴量付きデータ: {len(feature_df):,}行")
    print(f"  特徴量数: {len(feature_cols)}")

    # 特徴量グループ表示
    print(f"\n  【特徴量グループ】")
    for group, cols in get_feature_groups().items():
        present = [c for c in cols if c in feature_cols]
        print(f"    {group}: {len(present)}/{len(cols)}")

    # Step 5: ローリング学習
    print("\n【Step 5】ローリング学習")
    if TOP_N_FEATURES:
        print(f"  特徴量選択: Top {TOP_N_FEATURES}個を使用")
    predictor = StockPredictor(
        feature_columns=feature_cols,
        train_months=ROLLING_TRAIN_MONTHS,
        take_profit=LABEL_TAKE_PROFIT,
        stop_loss=LABEL_STOP_LOSS,
        top_n_features=TOP_N_FEATURES
    )

    results = predictor.fit_rolling(
        feature_df,
        start_date=TEST_START,
        end_date=TEST_END,
        verbose=True
    )

    if results['n_periods'] == 0:
        print("  エラー: 学習できる期間がありませんでした")
        return None

    # Step 6: 評価
    print("\n【Step 6】評価")
    print_evaluation_report(results['metrics'])

    # Step 7: 特徴量重要度
    print("\n【Step 7】特徴量重要度（Top 20）")
    try:
        importance = predictor.get_feature_importance(top_n=20)
        for _, row in importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.0f}")
    except ValueError as e:
        print(f"  {e}")

    # Step 8: 保存
    print("\n【Step 8】保存")
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    predictor.save(model_dir / "predictor.pkl")
    print(f"  モデル: {model_dir / 'predictor.pkl'}")

    # 予測結果（ランキング付き）
    predictions_df = results['predictions']
    predictions_df = predictor.predict_with_ranking(
        predictions_df,
        top_n=TOP_N_STOCKS,
        score_threshold=SCORE_THRESHOLD,
        min_score_gap=MIN_SCORE_GAP
    )

    # Step 8.5: 最新の日付まで予測を追加（ラベルがない期間）
    print("\n  【最新日付の予測追加】")
    latest_pred_date = pd.Timestamp(predictions_df['date'].max())
    latest_ohlcv_date = pd.Timestamp(ohlcv_df['date'].max())
    print(f"  予測データ最終日: {latest_pred_date.date()}")
    print(f"  OHLCVデータ最終日: {latest_ohlcv_date.date()}")

    if latest_ohlcv_date > latest_pred_date:
        # 特徴量生成には過去データも必要なので、全期間のOHLCVを使う
        ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])
        # ダミーラベルを設定（予測には使わない）
        ohlcv_for_future = ohlcv_df.copy()
        ohlcv_for_future['label'] = -1
        ohlcv_for_future['exit_reason'] = 'pending'
        ohlcv_for_future['entry_price'] = ohlcv_for_future['Open']

        # 特徴量生成（全期間）
        all_features = build_features(ohlcv_for_future, topix_df, sector_mapping)
        # latest_pred_date以降のデータだけを抽出
        future_features = all_features[all_features['date'] > latest_pred_date]

        if len(future_features) > 0:
            # 予測
            future_pred = predictor.predict_with_ranking(
                future_features,
                top_n=TOP_N_STOCKS,
                score_threshold=SCORE_THRESHOLD,
                min_score_gap=MIN_SCORE_GAP
            )
            # 結合
            predictions_df = pd.concat([predictions_df, future_pred], ignore_index=True)
            print(f"  追加: {future_pred['date'].nunique()}日分")
        else:
            print(f"  追加なし（特徴量生成できず）")
    else:
        print(f"  追加なし（データは最新）")

    predictions_df.to_parquet(data_dir / "predictions.parquet", index=False)
    print(f"  予測結果: {data_dir / 'predictions.parquet'}")

    # 特徴量付きデータも保存（バックテスト用）
    feature_df.to_parquet(data_dir / "features.parquet", index=False)
    print(f"  特徴量データ: {data_dir / 'features.parquet'}")

    # セクターマッピング保存
    sector_df = pd.DataFrame([
        {'ticker': k, 'sector': v} for k, v in sector_mapping.items()
    ])
    sector_df.to_parquet(data_dir / "sector_mapping.parquet", index=False)
    print(f"  セクター: {data_dir / 'sector_mapping.parquet'}")

    # 判断
    print("\n【判断】")
    ep = results['metrics'].get('expected_profit_per_trade', 0)
    p1 = results['metrics'].get('precision_at_1', 0)
    p2 = results['metrics'].get('precision_at_2', 0)

    if ep > 0.02:
        print(f"  期待利益 = {ep:.2%} > 2%")
        print(f"  Precision@1 = {p1:.2%}, Precision@2 = {p2:.2%}")
        print(f"  → 優秀。Phase 3（バックテスト）に進む")
    elif ep > 0:
        print(f"  期待利益 = {ep:.2%} > 0%")
        print(f"  Precision@1 = {p1:.2%}, Precision@2 = {p2:.2%}")
        print(f"  → 改善余地あり。Phase 3で検証")
    else:
        print(f"  期待利益 = {ep:.2%} <= 0%")
        print(f"  Precision@1 = {p1:.2%}, Precision@2 = {p2:.2%}")
        print(f"  → モデル・特徴量の根本的見直しが必要")

    return results


if __name__ == "__main__":
    results = main()
