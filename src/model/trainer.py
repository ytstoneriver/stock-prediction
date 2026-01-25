"""
v2モデル学習（ローリング学習・期待利益評価）

v1との主な違い:
1. ローリング学習: 直近N ヶ月のデータで毎月再学習
2. 期待利益評価: P(win)*take_profit - P(loss)*stop_loss
3. スコア差分析: rank1-rank2の差を監視
4. スコア閾値・最小差フィルタ
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score
from pathlib import Path
import pickle


class StockPredictor:
    """v2株価上昇予測モデル（ローリング学習対応）"""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        train_months: int = 6,
        take_profit: float = 0.10,
        stop_loss: float = 0.10,
        top_n_features: int | None = None
    ):
        self.model = None
        self.feature_columns = feature_columns
        self.feature_importance_ = None
        self.train_months = train_months
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.top_n_features = top_n_features
        self.selected_features = None  # 特徴量選択後のリスト
        self.models_by_period = {}  # 期間別モデル

    def fit_rolling(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        start_date: date | None = None,
        end_date: date | None = None,
        verbose: bool = True
    ) -> dict:
        """
        ローリング学習を実行

        毎月、直近N ヶ月のデータで学習し、翌月を予測

        Args:
            df: 特徴量・ラベル付きデータ
            label_col: ラベルカラム名
            start_date: 評価開始日
            end_date: 評価終了日
            verbose: 詳細出力

        Returns:
            評価指標のdict
        """
        if self.feature_columns is None:
            self.feature_columns = [c for c in df.columns if c.startswith('feat_')]

        df_sorted = df.sort_values('date').reset_index(drop=True)
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])

        # 特徴量選択（top_n_featuresが指定されている場合）
        if self.top_n_features and self.top_n_features < len(self.feature_columns):
            if verbose:
                print(f"特徴量選択: {len(self.feature_columns)}個 → Top {self.top_n_features}個")
            self.selected_features = self._select_top_features(
                df_sorted, label_col, self.top_n_features, verbose
            )
            self.feature_columns = self.selected_features
        else:
            self.selected_features = self.feature_columns

        if start_date is None:
            min_date = df_sorted['date'].min()
            start_date = (min_date + relativedelta(months=self.train_months)).date()
        if end_date is None:
            end_date = df_sorted['date'].max().date()

        # 月次でループ
        all_predictions = []
        current = pd.Timestamp(start_date).replace(day=1)
        end = pd.Timestamp(end_date)

        if verbose:
            print(f"ローリング学習: {current.date()} 〜 {end.date()}")
            print(f"学習期間: 直近 {self.train_months} ヶ月")

        while current <= end:
            # 学習期間: current - train_months ~ current - 1日
            train_end = current - pd.Timedelta(days=1)
            train_start = current - relativedelta(months=self.train_months)

            # 検証期間: current月
            test_start = current
            test_end = current + relativedelta(months=1) - pd.Timedelta(days=1)

            # データ抽出
            train_df = df_sorted[
                (df_sorted['date'] >= pd.Timestamp(train_start)) &
                (df_sorted['date'] <= train_end)
            ]
            test_df = df_sorted[
                (df_sorted['date'] >= test_start) &
                (df_sorted['date'] <= test_end)
            ]

            if len(train_df) < 100 or len(test_df) == 0:
                current += relativedelta(months=1)
                continue

            # モデル学習
            model = self._train_single_model(train_df, label_col)

            # 予測
            X_test = test_df[self.feature_columns].values
            y_pred = model.predict(X_test)

            test_result = test_df.copy()
            test_result['score'] = y_pred
            all_predictions.append(test_result)

            # モデルを保存
            period_key = current.strftime('%Y-%m')
            self.models_by_period[period_key] = model

            if verbose:
                try:
                    auc = roc_auc_score(test_df[label_col], y_pred)
                    print(f"  {period_key}: train={len(train_df):,}, test={len(test_df):,}, AUC={auc:.4f}")
                except ValueError:
                    print(f"  {period_key}: train={len(train_df):,}, test={len(test_df):,}, AUC=N/A")

            current += relativedelta(months=1)

        if not all_predictions:
            return {'predictions': pd.DataFrame(), 'metrics': {}, 'n_periods': 0}

        # 全期間の予測を結合
        predictions_df = pd.concat(all_predictions, ignore_index=True)

        # 最終モデル（最新期間のモデル）
        if self.models_by_period:
            self.model = list(self.models_by_period.values())[-1]
            self.feature_importance_ = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)

        # 評価
        metrics = self._evaluate(predictions_df, label_col)

        return {
            'predictions': predictions_df,
            'metrics': metrics,
            'n_periods': len(self.models_by_period)
        }

    def _train_single_model(
        self,
        df: pd.DataFrame,
        label_col: str
    ) -> lgb.Booster:
        """単一期間のモデルを学習"""
        X = df[self.feature_columns].values
        y = df[label_col].values

        pos_ratio = y.mean()
        scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0

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

        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data, num_boost_round=300)

        return model

    def _select_top_features(
        self,
        df: pd.DataFrame,
        label_col: str,
        top_n: int,
        verbose: bool = True
    ) -> list[str]:
        """
        特徴量重要度に基づいてTop N特徴量を選択

        全データの最初の部分で予備学習を行い、重要度を計算
        """
        # 予備学習用データ（最初の6ヶ月分）
        df_sample = df.head(len(df) // 3)
        if len(df_sample) < 1000:
            df_sample = df.head(min(len(df), 10000))

        X = df_sample[self.feature_columns].values
        y = df_sample[label_col].values

        pos_ratio = y.mean()
        scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbose': -1,
            'seed': 42
        }

        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data, num_boost_round=100)

        # 特徴量重要度を取得
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        selected = importance.head(top_n)['feature'].tolist()

        if verbose:
            print(f"  選択された特徴量:")
            for i, row in importance.head(top_n).iterrows():
                print(f"    {row['feature']}: {row['importance']:.0f}")

        return selected

    def _evaluate(self, df: pd.DataFrame, label_col: str) -> dict:
        """評価指標を計算"""
        # 日次ランキング
        df = df.copy()
        df['rank'] = df.groupby('date')['score'].rank(ascending=False, method='first')

        # AUC
        try:
            auc = roc_auc_score(df[label_col], df['score'])
        except ValueError:
            auc = 0.5

        # Precision@K
        precision_at_1 = df[df['rank'] == 1][label_col].mean()
        precision_at_2 = df[df['rank'] <= 2][label_col].mean()

        # 期待利益
        expected_profit = self._calculate_expected_profit(df, label_col)

        # スコア差分析
        score_gap = self._analyze_score_gap(df)

        return {
            'auc': auc,
            'precision_at_1': precision_at_1,
            'precision_at_2': precision_at_2,
            'expected_profit_per_trade': expected_profit,
            'avg_score_gap': score_gap['avg_gap'],
            'min_score_gap': score_gap['min_gap']
        }

    def _calculate_expected_profit(
        self,
        df: pd.DataFrame,
        label_col: str,
        top_k: int = 2
    ) -> float:
        """
        期待利益を計算

        期待利益 = P(win) * take_profit - P(loss) * stop_loss
        """
        top_k_df = df[df['rank'] <= top_k]

        if len(top_k_df) == 0:
            return 0.0

        win_rate = top_k_df[label_col].mean()
        loss_rate = 1 - win_rate

        expected = win_rate * self.take_profit - loss_rate * self.stop_loss

        return expected

    def _analyze_score_gap(self, df: pd.DataFrame) -> dict:
        """スコア差を分析"""
        gaps = []

        for dt in df['date'].unique():
            day_df = df[df['date'] == dt].sort_values('score', ascending=False)
            if len(day_df) >= 2:
                gap = day_df.iloc[0]['score'] - day_df.iloc[1]['score']
                gaps.append(gap)

        return {
            'avg_gap': np.mean(gaps) if gaps else 0,
            'min_gap': np.min(gaps) if gaps else 0,
            'max_gap': np.max(gaps) if gaps else 0
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """予測"""
        if self.model is None:
            raise ValueError("モデルが学習されていません")

        X = df[self.feature_columns].values
        return self.model.predict(X)

    def predict_with_ranking(
        self,
        df: pd.DataFrame,
        top_n: int = 2,
        score_threshold: float = 0.0,
        min_score_gap: float = 0.0
    ) -> pd.DataFrame:
        """
        予測とランキングを付与（フィルタリング機能付き）

        Args:
            df: 特徴量付きデータ
            top_n: 上位N銘柄
            score_threshold: 最小スコア閾値
            min_score_gap: rank1とrank2の最小スコア差

        Returns:
            予測結果DataFrame
        """
        result = df.copy()

        # スコアがすでにある場合はそのまま使用、なければ予測
        if 'score' not in result.columns:
            result['score'] = self.predict(df)

        # 日次ランキング
        result['rank'] = result.groupby('date')['score'].rank(
            ascending=False, method='first'
        )

        # スコア閾値フィルタ
        result['above_threshold'] = result['score'] >= score_threshold

        # スコア差計算
        def calc_gap(group):
            group = group.sort_values('score', ascending=False)
            if len(group) >= 2:
                gap = group.iloc[0]['score'] - group.iloc[1]['score']
            else:
                gap = 1.0  # 1銘柄しかない場合はフィルタを通過
            group['score_gap'] = gap
            group['meets_gap'] = gap >= min_score_gap
            return group

        result = result.groupby('date', group_keys=False).apply(calc_gap)

        # is_top_nフラグ
        result['is_top_n'] = (
            (result['rank'] <= top_n) &
            result['above_threshold']
        ).astype(int)

        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """特徴量重要度"""
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
                'feature_importance': self.feature_importance_,
                'models_by_period': self.models_by_period,
                'train_months': self.train_months,
                'take_profit': self.take_profit,
                'stop_loss': self.stop_loss,
                'top_n_features': self.top_n_features,
                'selected_features': self.selected_features
            }, f)

    def load(self, path: str | Path):
        """モデルを読み込み"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.feature_importance_ = data.get('feature_importance')
        self.models_by_period = data.get('models_by_period', {})
        self.train_months = data.get('train_months', 6)
        self.take_profit = data.get('take_profit', 0.10)
        self.stop_loss = data.get('stop_loss', 0.10)
        self.top_n_features = data.get('top_n_features')
        self.selected_features = data.get('selected_features')


def print_evaluation_report(metrics: dict):
    """v2評価レポートを表示"""
    print("=" * 60)
    print("v2 Model Evaluation Report")
    print("=" * 60)

    print(f"\n【予測精度】")
    print(f"  AUC: {metrics.get('auc', 0):.4f}")
    print(f"  Precision@1: {metrics.get('precision_at_1', 0):.2%}")
    print(f"  Precision@2: {metrics.get('precision_at_2', 0):.2%}")

    print(f"\n【期待利益】")
    ep = metrics.get('expected_profit_per_trade', 0)
    print(f"  1トレードあたり期待利益: {ep:.2%}")
    if ep > 0:
        annual = ep * 2 * 252 * 1_000_000
        print(f"  → 100万円×2銘柄×252日 = 年間約{annual:,.0f}円")

    print(f"\n【スコア差】")
    print(f"  平均スコア差: {metrics.get('avg_score_gap', 0):.4f}")
    print(f"  最小スコア差: {metrics.get('min_score_gap', 0):.4f}")

    print("=" * 60)


if __name__ == "__main__":
    # テスト
    from datetime import timedelta

    n_days = 500
    n_tickers = 10
    dates = sorted([date(2023, 1, 1) + timedelta(days=i) for i in range(n_days)] * n_tickers)
    tickers = [f'TICK{i % n_tickers}.T' for i in range(n_days * n_tickers)]

    test_df = pd.DataFrame({
        'date': dates,
        'ticker': tickers,
        'feat_ret_1d': np.random.randn(n_days * n_tickers),
        'feat_ret_5d': np.random.randn(n_days * n_tickers),
        'feat_volume_ratio': np.random.uniform(0.5, 2, n_days * n_tickers),
        'label': np.random.randint(0, 2, n_days * n_tickers)
    })

    predictor = StockPredictor(train_months=3)
    results = predictor.fit_rolling(
        test_df,
        start_date=date(2023, 6, 1),
        verbose=True
    )

    print(f"\n期間数: {results['n_periods']}")
    print_evaluation_report(results['metrics'])
