"""
特徴量生成モジュール

47特徴量を生成:
- 価格・トレンド系: 8個
- 出来高系: 3個
- ローソク足形状: 6個
- RSI: 3個
- MACD: 4個
- ボリンジャーバンド: 2個
- ATR: 1個
- 価格位置: 3個
- 需給代替指標: 6個
- セクターダミー: 11個
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict


# セクター一覧（yfinanceのsector）
SECTORS = [
    'Technology',
    'Financial Services',
    'Healthcare',
    'Consumer Cyclical',
    'Industrials',
    'Communication Services',
    'Consumer Defensive',
    'Energy',
    'Utilities',
    'Real Estate',
    'Basic Materials',
]


def build_features(
    df: pd.DataFrame,
    topix_df: Optional[pd.DataFrame] = None,
    sector_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    特徴量を生成

    Args:
        df: OHLCVデータ（columns: date, ticker, Open, High, Low, Close, Volume）
        topix_df: TOPIXデータ（optional）
        sector_mapping: 銘柄→セクターのマッピング（optional）

    Returns:
        特徴量付きDataFrame
    """
    result = df.copy()
    result = result.sort_values(['ticker', 'date']).reset_index(drop=True)

    feature_dfs = []

    for ticker in result['ticker'].unique():
        ticker_df = result[result['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

        # 価格・トレンド系
        ticker_df = _add_price_features(ticker_df)
        ticker_df = _add_volume_features(ticker_df)
        ticker_df = _add_candlestick_features(ticker_df)

        # テクニカル指標
        ticker_df = _add_rsi(ticker_df)
        ticker_df = _add_macd(ticker_df)
        ticker_df = _add_bollinger_band(ticker_df)
        ticker_df = _add_atr_normalized(ticker_df)
        ticker_df = _add_price_position(ticker_df)

        # 需給代替指標
        ticker_df = _add_supply_demand_features(ticker_df)

        # セクターダミー変数
        if sector_mapping is not None:
            sector = sector_mapping.get(ticker, 'Unknown')
            ticker_df = _add_sector_dummies(ticker_df, sector)
        else:
            for s in SECTORS:
                col_name = f"feat_sector_{s.lower().replace(' ', '_')}"
                ticker_df[col_name] = 0

        feature_dfs.append(ticker_df)

    result = pd.concat(feature_dfs, ignore_index=True)

    # 地合い特徴量
    if topix_df is not None:
        result = _add_market_features(result, topix_df)

    # NaN削除
    feature_cols = [c for c in result.columns if c.startswith('feat_')]
    result = result.dropna(subset=feature_cols)

    return result


# ===== 価格・トレンド系 =====

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

    # ボラティリティ
    result['feat_volatility_20d'] = result['feat_ret_1d'].rolling(20).std()

    # 一時列を削除
    result = result.drop(columns=['ma5', 'ma25', 'high_20d', 'low_20d'])

    return result


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """出来高系特徴量"""
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
    candle_range = candle_range.replace(0, np.nan)

    # 実体
    body = result['Close'] - result['Open']
    body_abs = body.abs()

    # 実体比率
    result['feat_body_ratio'] = body_abs / candle_range

    # 実体の方向
    result['feat_body_direction'] = np.sign(body)

    # 上ヒゲ比率
    upper_wick = result['High'] - result[['Open', 'Close']].max(axis=1)
    result['feat_upper_wick_ratio'] = upper_wick / candle_range

    # 下ヒゲ比率
    lower_wick = result[['Open', 'Close']].min(axis=1) - result['Low']
    result['feat_lower_wick_ratio'] = lower_wick / candle_range

    # ギャップ
    result['feat_gap'] = result['Open'] / result['Close'].shift(1) - 1

    # 連続陽線/陰線
    result['is_positive'] = (result['Close'] > result['Open']).astype(int)
    result['feat_consecutive_positive'] = result['is_positive'].rolling(5).sum()

    result = result.drop(columns=['is_positive'])

    return result


# ===== テクニカル指標 =====

def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI(14)を計算"""
    result = df.copy()
    delta = result['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    result['feat_rsi_14'] = rsi
    result['feat_rsi_oversold'] = (rsi < 30).astype(int)
    result['feat_rsi_overbought'] = (rsi > 70).astype(int)

    return result


def _add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """MACD (12, 26, 9)を計算"""
    result = df.copy()

    ema_fast = result['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = result['Close'].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    # 正規化
    result['feat_macd'] = macd_line / result['Close']
    result['feat_macd_signal'] = signal_line / result['Close']
    result['feat_macd_hist'] = histogram / result['Close']
    result['feat_macd_cross'] = np.sign(macd_line - signal_line)

    return result


def _add_bollinger_band(
    df: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0
) -> pd.DataFrame:
    """ボリンジャーバンドを計算"""
    result = df.copy()

    ma = result['Close'].rolling(period).mean()
    std = result['Close'].rolling(period).std()

    upper = ma + std_mult * std
    lower = ma - std_mult * std

    band_width = upper - lower
    result['feat_bb_position'] = (result['Close'] - lower) / band_width.replace(0, np.nan)
    result['feat_bb_width'] = band_width / ma

    return result


def _add_atr_normalized(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """正規化ATRを計算"""
    result = df.copy()

    high = result['High']
    low = result['Low']
    close = result['Close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    result['feat_atr_pct'] = atr / result['Close']

    return result


def _add_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """52週高値/安値からの位置を計算"""
    result = df.copy()

    high_52w = result['High'].rolling(250, min_periods=60).max()
    low_52w = result['Low'].rolling(250, min_periods=60).min()

    range_52w = high_52w - low_52w
    result['feat_price_position_52w'] = (result['Close'] - low_52w) / range_52w.replace(0, np.nan)
    result['feat_from_52w_high'] = result['Close'] / high_52w - 1
    result['feat_from_52w_low'] = result['Close'] / low_52w - 1

    return result


# ===== 需給代替指標 =====

def _add_supply_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """需給代替指標を追加"""
    result = df.copy()

    vol_ma5 = result['Volume'].rolling(5).mean()
    vol_ma20 = result['Volume'].rolling(20).mean()

    result['feat_volume_ratio_5d'] = result['Volume'] / vol_ma5.replace(0, np.nan)
    result['feat_volume_ratio_20d'] = result['Volume'] / vol_ma20.replace(0, np.nan)

    # OBV
    price_direction = np.sign(result['Close'].diff())
    obv = (price_direction * result['Volume']).cumsum()

    result['feat_obv_change_5d'] = obv.pct_change(5)
    result['feat_obv_change_20d'] = obv.pct_change(20)

    result['feat_volume_spike'] = (result['Volume'] > 2 * vol_ma20).astype(int)
    result['feat_volume_trend'] = vol_ma5 / vol_ma20.replace(0, np.nan)

    return result


# ===== セクター =====

def _add_sector_dummies(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """セクターダミー変数を追加"""
    result = df.copy()

    for s in SECTORS:
        col_name = f"feat_sector_{s.lower().replace(' ', '_')}"
        result[col_name] = 1 if s == sector else 0

    return result


# ===== 地合い =====

def _add_market_features(df: pd.DataFrame, topix_df: pd.DataFrame) -> pd.DataFrame:
    """地合い特徴量（TOPIX）"""
    result = df.copy()

    topix = topix_df.copy()
    topix = topix.sort_values('date').reset_index(drop=True)

    topix['topix_ret_1d'] = topix['Close'].pct_change(1)
    topix['topix_ret_5d'] = topix['Close'].pct_change(5)
    topix['topix_ma25'] = topix['Close'].rolling(25).mean()
    topix['topix_above_ma25'] = (topix['Close'] > topix['topix_ma25']).astype(int)
    topix['topix_vol_10d'] = topix['topix_ret_1d'].rolling(10).std()

    topix_features = topix[['date', 'topix_ret_1d', 'topix_ret_5d',
                            'topix_above_ma25', 'topix_vol_10d']].copy()
    topix_features.columns = ['date', 'feat_topix_ret_1d', 'feat_topix_ret_5d',
                              'feat_topix_above_ma25', 'feat_topix_vol_10d']

    result = result.merge(topix_features, on='date', how='left')

    return result


# ===== ユーティリティ =====

def get_feature_columns(df: pd.DataFrame) -> list:
    """特徴量カラム名を取得"""
    return [c for c in df.columns if c.startswith('feat_')]


def get_feature_groups() -> Dict[str, list]:
    """特徴量グループを取得"""
    return {
        'price': [
            'feat_ret_1d', 'feat_ret_5d', 'feat_ret_20d',
            'feat_ma5_dev', 'feat_ma25_dev',
            'feat_high_20d_dist', 'feat_low_20d_dist',
            'feat_volatility_20d',
        ],
        'volume': [
            'feat_volume_ratio', 'feat_turnover_ratio', 'feat_volume_change_5d',
        ],
        'candlestick': [
            'feat_body_ratio', 'feat_body_direction',
            'feat_upper_wick_ratio', 'feat_lower_wick_ratio',
            'feat_gap', 'feat_consecutive_positive',
        ],
        'rsi': [
            'feat_rsi_14', 'feat_rsi_oversold', 'feat_rsi_overbought',
        ],
        'macd': [
            'feat_macd', 'feat_macd_signal', 'feat_macd_hist', 'feat_macd_cross',
        ],
        'bollinger': [
            'feat_bb_position', 'feat_bb_width',
        ],
        'atr': [
            'feat_atr_pct',
        ],
        'price_position': [
            'feat_price_position_52w', 'feat_from_52w_high', 'feat_from_52w_low',
        ],
        'supply_demand': [
            'feat_volume_ratio_5d', 'feat_volume_ratio_20d',
            'feat_obv_change_5d', 'feat_obv_change_20d',
            'feat_volume_spike', 'feat_volume_trend',
        ],
        'sector': [f"feat_sector_{s.lower().replace(' ', '_')}" for s in SECTORS],
        'market': [
            'feat_topix_ret_1d', 'feat_topix_ret_5d',
            'feat_topix_above_ma25', 'feat_topix_vol_10d',
        ],
    }
