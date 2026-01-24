"""
LightGBMモデルの学習・予測
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, precision_score
from pathlib import Path
import pickle


class StockPredictor:
    """株価上昇予測モデル"""

    def __init__(self, feature_columns: list[str] | None = None):
        self.model = None
        self.feature_columns = feature_columns
        self.feature_importance_ = None

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        n_splits: int = 3,
        verbose: bool = True
    ) -> dict:
        """
        時系列分割でモデルを学習

        Args:
            df: 特徴量付きデータ
            label_col: ラベルカラム名
            n_splits: 時系列分割数
            verbose: 詳細出力

        Returns:
            評価指標のdict
        """
        # 特徴量カラムを自動検出
        if self.feature_columns is None:
            self.feature_columns = [c for c in df.columns if c.startswith('feat_')]

        if verbose:
            print(f"特徴量数: {len(self.feature_columns)}")

        # データ準備
        df_sorted = df.sort_values('date').reset_index(drop=True)
        X = df_sorted[self.feature_columns].values
        y = df_sorted[label_col].values

        # クラス重み計算（不均衡対策）
        pos_ratio = y.mean()
        neg_ratio = 1 - pos_ratio
        scale_pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0

        if verbose:
            print(f"正例率: {pos_ratio:.2%}")
            print(f"scale_pos_weight: {scale_pos_weight:.2f}")

        # LightGBMパラメータ
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'seed': 42
        }

        # 時系列分割でCV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        models = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)  # ログ出力を抑制
                ]
            )

            # 検証スコア
            y_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            cv_scores.append(auc)
            models.append(model)

            if verbose:
                print(f"  Fold {fold + 1}: AUC = {auc:.4f}")

        # 最終モデルは全データで学習
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500
        )

        # 特徴量重要度
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        # 評価指標
        mean_auc = np.mean(cv_scores)
        std_auc = np.std(cv_scores)

        if verbose:
            print(f"\nCV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

        return {
            'cv_auc_mean': mean_auc,
            'cv_auc_std': std_auc,
            'cv_scores': cv_scores
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        予測スコアを算出

        Args:
            df: 特徴量付きデータ

        Returns:
            予測確率（0-1）
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません")

        X = df[self.feature_columns].values
        return self.model.predict(X)

    def predict_with_ranking(self, df: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
        """
        予測スコアを算出し、日次ランキングを付与

        Args:
            df: 特徴量付きデータ
            top_n: 上位N銘柄

        Returns:
            予測スコアとランキング付きDataFrame
        """
        result = df.copy()
        result['score'] = self.predict(df)

        # 日次ランキング
        result['rank'] = result.groupby('date')['score'].rank(ascending=False, method='first')
        result['is_top_n'] = (result['rank'] <= top_n).astype(int)

        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """特徴量重要度を取得"""
        if self.feature_importance_ is None:
            raise ValueError("モデルが学習されていません")
        return self.feature_importance_.head(top_n)

    def save(self, path: str | Path):
        """モデルを保存"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance_
            }, f)

    def load(self, path: str | Path):
        """モデルを読み込み"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.feature_importance_ = data['feature_importance']


def evaluate_precision_at_k(df: pd.DataFrame, k: int = 2) -> dict:
    """
    Precision@Kを評価

    Args:
        df: 予測結果（columns: date, label, score, rank）
        k: 上位K銘柄

    Returns:
        評価指標のdict
    """
    # 日次で上位K銘柄を抽出
    top_k = df[df['rank'] <= k].copy()

    # Precision@K（上位K銘柄のうち、実際に+10%到達した割合）
    precision = top_k['label'].mean()

    # 「少なくとも1銘柄が当たった日」の割合
    daily_hit = top_k.groupby('date')['label'].max()
    hit_rate = daily_hit.mean()

    # 日数
    n_days = df['date'].nunique()
    n_trades = len(top_k)

    return {
        'precision_at_k': precision,
        'hit_rate': hit_rate,
        'n_days': n_days,
        'n_trades': n_trades
    }


if __name__ == "__main__":
    # テスト
    import numpy as np
    from datetime import date, timedelta

    # ダミーデータ
    n_days = 200
    n_tickers = 10
    dates = sorted([date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)] * n_tickers)
    tickers = ['TICK' + str(i) for i in range(n_tickers)] * n_days

    test_df = pd.DataFrame({
        'date': dates,
        'ticker': tickers,
        'feat_ret_1d': np.random.randn(n_days * n_tickers),
        'feat_ret_5d': np.random.randn(n_days * n_tickers),
        'feat_volume_ratio': np.random.uniform(0.5, 2, n_days * n_tickers),
        'label': np.random.randint(0, 2, n_days * n_tickers)
    })

    # 学習
    predictor = StockPredictor()
    scores = predictor.fit(test_df)
    print(f"\nCV AUC: {scores['cv_auc_mean']:.4f}")

    # 予測
    result = predictor.predict_with_ranking(test_df)
    print(f"\nPredictions shape: {result.shape}")

    # 評価
    eval_result = evaluate_precision_at_k(result, k=2)
    print(f"Precision@2: {eval_result['precision_at_k']:.2%}")
    print(f"Hit Rate: {eval_result['hit_rate']:.2%}")

    # 特徴量重要度
    print("\n特徴量重要度:")
    print(predictor.get_feature_importance())
