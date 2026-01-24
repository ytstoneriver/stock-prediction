"""
特徴量生成
"""
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame, topix_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    特徴量を生成

    Args:
        df: OHLCVデータ（columns: date, ticker, Open, High, Low, Close, Volume, label）
        topix_df: TOPIXデータ（columns: date, Open, High, Low, Close, Volume）

    Returns:
        特徴量付きDataFrame
    """
    result = df.copy()
    result = result.sort_values(['ticker', 'date']).reset_index(drop=True)

    # 銘柄ごとに特徴量を計算
    feature_dfs = []

    for ticker in result['ticker'].unique():
        ticker_df = result[result['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

        # 価格系特徴量
        ticker_df = _add_price_features(ticker_df)

        # 出来高系特徴量
        ticker_df = _add_volume_features(ticker_df)

        # ローソク足特徴量
        ticker_df = _add_candlestick_features(ticker_df)

        feature_dfs.append(ticker_df)

    result = pd.concat(feature_dfs, ignore_index=True)

    # 地合い特徴量（TOPIX）
    if topix_df is not None:
        result = _add_market_features(result, topix_df)

    # NaN行を削除（特徴量計算のウィンドウ分）
    feature_cols = [c for c in result.columns if c.startswith('feat_')]
    result = result.dropna(subset=feature_cols)

    return result


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """価格・トレンド系特徴量"""
    result = df.copy()

    # リターン
    result['feat_ret_1d'] = result['Close'].pct_change(1)
    result['feat_ret_5d'] = result['Close'].pct_change(5)
    result['feat_ret_20d'] = result['Close'].pct_change(20)

    # 移動平均
    result['ma5'] = result['Close'].rolling(5).mean()
    result['ma25'] = result['Close'].rolling(25).mean()

    # 移動平均乖離
    result['feat_ma5_dev'] = result['Close'] / result['ma5'] - 1
    result['feat_ma25_dev'] = result['Close'] / result['ma25'] - 1

    # 直近高値との距離
    result['high_20d'] = result['High'].rolling(20).max()
    result['feat_high_20d_dist'] = result['Close'] / result['high_20d'] - 1

    # 直近安値との距離
    result['low_20d'] = result['Low'].rolling(20).min()
    result['feat_low_20d_dist'] = result['Close'] / result['low_20d'] - 1

    # ボラティリティ（20日リターンの標準偏差）
    result['feat_volatility_20d'] = result['feat_ret_1d'].rolling(20).std()

    # 一時列を削除
    result = result.drop(columns=['ma5', 'ma25', 'high_20d', 'low_20d'])

    return result


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """出来高・売買代金系特徴量"""
    result = df.copy()

    # 売買代金
    result['turnover'] = result['Close'] * result['Volume']

    # 出来高異常度
    result['vol_ma20'] = result['Volume'].rolling(20).mean()
    result['feat_volume_ratio'] = result['Volume'] / result['vol_ma20']

    # 売買代金異常度
    result['turn_ma20'] = result['turnover'].rolling(20).mean()
    result['feat_turnover_ratio'] = result['turnover'] / result['turn_ma20']

    # 出来高の変化率
    result['feat_volume_change_5d'] = result['Volume'].pct_change(5)

    # 一時列を削除
    result = result.drop(columns=['turnover', 'vol_ma20', 'turn_ma20'])

    return result


def _add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """ローソク足形状特徴量"""
    result = df.copy()

    # ローソク足のレンジ
    candle_range = result['High'] - result['Low']
    candle_range = candle_range.replace(0, np.nan)  # ゼロ除算防止

    # 実体
    body = result['Close'] - result['Open']
    body_abs = body.abs()

    # 実体比率（実体 / レンジ）
    result['feat_body_ratio'] = body_abs / candle_range

    # 実体の方向（陽線: 1, 陰線: -1）
    result['feat_body_direction'] = np.sign(body)

    # 上ヒゲ比率
    upper_wick = result['High'] - result[['Open', 'Close']].max(axis=1)
    result['feat_upper_wick_ratio'] = upper_wick / candle_range

    # 下ヒゲ比率
    lower_wick = result[['Open', 'Close']].min(axis=1) - result['Low']
    result['feat_lower_wick_ratio'] = lower_wick / candle_range

    # ギャップ（前日終値との差）
    result['feat_gap'] = result['Open'] / result['Close'].shift(1) - 1

    # 連続陽線/陰線カウント
    result['is_positive'] = (result['Close'] > result['Open']).astype(int)
    result['feat_consecutive_positive'] = result['is_positive'].rolling(5).sum()

    result = result.drop(columns=['is_positive'])

    return result


def _add_market_features(df: pd.DataFrame, topix_df: pd.DataFrame) -> pd.DataFrame:
    """地合い特徴量（TOPIX）"""
    result = df.copy()

    # TOPIXの特徴量を計算
    topix = topix_df.copy()
    topix = topix.sort_values('date').reset_index(drop=True)

    # TOPIXリターン
    topix['topix_ret_1d'] = topix['Close'].pct_change(1)
    topix['topix_ret_5d'] = topix['Close'].pct_change(5)

    # TOPIX MA
    topix['topix_ma25'] = topix['Close'].rolling(25).mean()
    topix['topix_above_ma25'] = (topix['Close'] > topix['topix_ma25']).astype(int)

    # TOPIXボラティリティ
    topix['topix_vol_10d'] = topix['topix_ret_1d'].rolling(10).std()

    # マージ用に整形
    topix_features = topix[['date', 'topix_ret_1d', 'topix_ret_5d',
                            'topix_above_ma25', 'topix_vol_10d']].copy()
    topix_features.columns = ['date', 'feat_topix_ret_1d', 'feat_topix_ret_5d',
                              'feat_topix_above_ma25', 'feat_topix_vol_10d']

    # メインデータとマージ
    result = result.merge(topix_features, on='date', how='left')

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """特徴量カラム名を取得"""
    return [c for c in df.columns if c.startswith('feat_')]


if __name__ == "__main__":
    # テスト
    import numpy as np
    from datetime import date, timedelta

    # ダミーデータ作成
    n_days = 100
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    test_df = pd.DataFrame({
        'date': dates * 2,
        'ticker': ['7203.T'] * n_days + ['6758.T'] * n_days,
        'Open': np.random.uniform(2000, 2100, n_days * 2),
        'High': np.random.uniform(2050, 2200, n_days * 2),
        'Low': np.random.uniform(1900, 2000, n_days * 2),
        'Close': np.random.uniform(2000, 2100, n_days * 2),
        'Volume': np.random.uniform(1e6, 1e7, n_days * 2),
        'label': np.random.randint(0, 2, n_days * 2)
    })

    result = build_features(test_df)
    print(f"Shape: {result.shape}")
    print(f"Feature columns: {get_feature_columns(result)}")
    print(result.head())
