"""
Phase 2: 特徴量生成とモデル学習
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from datetime import date

from config import TRAIN_START, TRAIN_END, TEST_START, TEST_END, TOP_N
from features import build_features
from model.trainer import StockPredictor, evaluate_precision_at_k


def main():
    print("=" * 60)
    print("Phase 2: 特徴量生成とモデル学習")
    print("=" * 60)

    # Step 1: データ読み込み
    print("\n【Step 1】データ読み込み")
    data_dir = Path(__file__).parent.parent / "data"

    labeled_df = pd.read_parquet(data_dir / "labeled_data.parquet")
    print(f"  データ: {labeled_df['ticker'].nunique()}銘柄, {len(labeled_df):,}行")

    # TOPIXデータ（あれば）
    topix_path = data_dir / "topix.parquet"
    topix_df = None
    if topix_path.exists():
        topix_df = pd.read_parquet(topix_path)
        print(f"  TOPIX: {len(topix_df)}日分")
    else:
        print("  TOPIX: なし（地合い特徴量はスキップ）")

    # Step 2: 特徴量生成
    print("\n【Step 2】特徴量生成")
    feature_df = build_features(labeled_df, topix_df)
    print(f"  特徴量生成後: {len(feature_df):,}行")

    # 特徴量カラム確認
    feature_cols = [c for c in feature_df.columns if c.startswith('feat_')]
    print(f"  特徴量数: {len(feature_cols)}")
    print(f"  特徴量: {feature_cols}")

    # Step 3: 学習/検証データ分割
    print("\n【Step 3】データ分割")
    train_df = feature_df[
        (feature_df['date'] >= TRAIN_START) &
        (feature_df['date'] <= TRAIN_END)
    ].copy()

    test_df = feature_df[
        (feature_df['date'] >= TEST_START) &
        (feature_df['date'] <= TEST_END)
    ].copy()

    print(f"  学習データ: {len(train_df):,}行 ({train_df['date'].min()} ~ {train_df['date'].max()})")
    print(f"  検証データ: {len(test_df):,}行 ({test_df['date'].min()} ~ {test_df['date'].max()})")

    # Step 4: モデル学習
    print("\n【Step 4】モデル学習")
    predictor = StockPredictor(feature_columns=feature_cols)
    cv_scores = predictor.fit(train_df, n_splits=3, verbose=True)

    # Step 5: 検証データで評価
    print("\n【Step 5】検証データで評価")
    test_result = predictor.predict_with_ranking(test_df, top_n=TOP_N)

    # AUC
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(test_result['label'], test_result['score'])
    print(f"  検証AUC: {test_auc:.4f}")

    # Precision@K
    eval_metrics = evaluate_precision_at_k(test_result, k=TOP_N)
    print(f"  Precision@{TOP_N}: {eval_metrics['precision_at_k']:.2%}")
    print(f"  Hit Rate（少なくとも1銘柄当たり）: {eval_metrics['hit_rate']:.2%}")
    print(f"  取引日数: {eval_metrics['n_days']}日")
    print(f"  総取引数: {eval_metrics['n_trades']}回")

    # Step 6: 特徴量重要度
    print("\n【Step 6】特徴量重要度（Top 10）")
    importance = predictor.get_feature_importance(top_n=10)
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")

    # Step 7: モデル保存
    print("\n【Step 7】モデル保存")
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    predictor.save(model_dir / "predictor_v1.pkl")
    print(f"  保存先: {model_dir / 'predictor_v1.pkl'}")

    # 予測結果も保存
    test_result.to_parquet(data_dir / "test_predictions.parquet", index=False)
    print(f"  予測結果: {data_dir / 'test_predictions.parquet'}")

    # 判断基準
    print("\n【判断】")
    if test_auc >= 0.55:
        print(f"  検証AUC = {test_auc:.4f} >= 0.55")
        print(f"  → Phase 3（バックテスト）に進む")
    else:
        print(f"  検証AUC = {test_auc:.4f} < 0.55")
        print(f"  → 特徴量追加やパラメータ調整を検討")

    return {
        'cv_auc': cv_scores['cv_auc_mean'],
        'test_auc': test_auc,
        'precision_at_k': eval_metrics['precision_at_k'],
        'hit_rate': eval_metrics['hit_rate']
    }


if __name__ == "__main__":
    results = main()
