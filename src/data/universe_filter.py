"""
ユニバースフィルタ（v3用）
ファンダメンタル指標で銘柄をフィルタリング
"""
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
import time


def fetch_fundamental_data(tickers: List[str], show_progress: bool = True) -> pd.DataFrame:
    """
    複数銘柄のファンダメンタルデータを取得

    Parameters
    ----------
    tickers : List[str]
        銘柄コードリスト（例: ['7203.T', '6758.T']）
    show_progress : bool
        プログレスバー表示

    Returns
    -------
    pd.DataFrame
        ファンダメンタルデータ（ticker, sector, market_cap, roe, beta）
    """
    results = []

    iterator = tqdm(tickers, desc="ファンダメンタル取得") if show_progress else tickers

    for ticker in iterator:
        try:
            info = yf.Ticker(ticker).info

            # ROE計算（returnOnEquityがない場合はnetIncomeMarginで代替）
            roe = info.get('returnOnEquity')
            if roe is None:
                # 代替: 純利益率があれば使用
                roe = info.get('profitMargins', 0)

            results.append({
                'ticker': ticker,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'roe': roe if roe is not None else 0,
                'beta': info.get('beta', 1.0),
                'float_shares': info.get('floatShares', 0),
                'avg_volume': info.get('averageVolume', 0),
            })

            # API制限対策
            time.sleep(0.1)

        except Exception as e:
            # エラー時はスキップ
            results.append({
                'ticker': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'roe': 0,
                'beta': 1.0,
                'float_shares': 0,
                'avg_volume': 0,
            })

    return pd.DataFrame(results)


def apply_fundamental_filter(
    tickers: List[str],
    fundamental_df: Optional[pd.DataFrame] = None,
    min_roe: float = 0.0,
    min_market_cap: float = 50e9,
    max_beta: float = 2.5,
    show_progress: bool = True,
) -> Dict:
    """
    ファンダメンタルフィルタを適用

    Parameters
    ----------
    tickers : List[str]
        フィルタ対象の銘柄リスト
    fundamental_df : Optional[pd.DataFrame]
        既に取得済みのファンダメンタルデータ（Noneなら取得）
    min_roe : float
        最小ROE（デフォルト: 0 = 黒字企業のみ）
    min_market_cap : float
        最小時価総額（デフォルト: 500億円）
    max_beta : float
        最大Beta（デフォルト: 2.5）
    show_progress : bool
        プログレスバー表示

    Returns
    -------
    Dict
        {
            'filtered_tickers': フィルタ後の銘柄リスト,
            'fundamental_df': ファンダメンタルデータ,
            'removed': 除外された銘柄の詳細
        }
    """
    # ファンダメンタルデータ取得
    if fundamental_df is None:
        fundamental_df = fetch_fundamental_data(tickers, show_progress)

    # フィルタ適用
    original_count = len(fundamental_df)

    # ROEフィルタ
    roe_mask = fundamental_df['roe'] > min_roe
    roe_removed = fundamental_df[~roe_mask]['ticker'].tolist()

    # 時価総額フィルタ
    cap_mask = fundamental_df['market_cap'] > min_market_cap
    cap_removed = fundamental_df[~cap_mask]['ticker'].tolist()

    # Betaフィルタ
    beta_mask = fundamental_df['beta'] < max_beta
    beta_removed = fundamental_df[~beta_mask]['ticker'].tolist()

    # 全条件を満たす銘柄
    combined_mask = roe_mask & cap_mask & beta_mask
    filtered_df = fundamental_df[combined_mask]
    filtered_tickers = filtered_df['ticker'].tolist()

    return {
        'filtered_tickers': filtered_tickers,
        'fundamental_df': fundamental_df,
        'filtered_df': filtered_df,
        'stats': {
            'original': original_count,
            'after_filter': len(filtered_tickers),
            'removed_roe': len(roe_removed),
            'removed_cap': len(cap_removed),
            'removed_beta': len(beta_removed),
        },
        'removed': {
            'roe': roe_removed,
            'market_cap': cap_removed,
            'beta': beta_removed,
        }
    }


def get_sector_mapping(fundamental_df: pd.DataFrame) -> Dict[str, str]:
    """
    銘柄→セクターのマッピングを取得

    Parameters
    ----------
    fundamental_df : pd.DataFrame
        ファンダメンタルデータ

    Returns
    -------
    Dict[str, str]
        {ticker: sector}
    """
    mapping = dict(zip(fundamental_df['ticker'], fundamental_df['sector']))

    # yfinanceのセクター情報が間違っている銘柄を手動で修正
    # 参考: https://finance.yahoo.co.jp/ の業種情報
    SECTOR_OVERRIDES = {
        '5016.T': 'Basic Materials',   # JX金属（非鉄金属）
        '5706.T': 'Basic Materials',   # 三井金属（非鉄金属）
        '5801.T': 'Basic Materials',   # 古河電気工業（非鉄金属・電線）
        '5802.T': 'Basic Materials',   # 住友電気工業（非鉄金属・電線）
        '5803.T': 'Basic Materials',   # フジクラ（非鉄金属・電線）
        '6620.T': 'Industrials',       # 宮越ホールディングス（機械）
    }

    for ticker, sector in SECTOR_OVERRIDES.items():
        if ticker in mapping:
            mapping[ticker] = sector

    return mapping


def print_filter_report(result: Dict) -> None:
    """
    フィルタ結果のレポートを出力
    """
    stats = result['stats']

    print("=" * 60)
    print("【ファンダメンタルフィルタ結果】")
    print("=" * 60)
    print(f"  元銘柄数: {stats['original']}")
    print(f"  フィルタ後: {stats['after_filter']}")
    print(f"  除外数:")
    print(f"    - ROE <= 0: {stats['removed_roe']}銘柄")
    print(f"    - 時価総額 <= 500億: {stats['removed_cap']}銘柄")
    print(f"    - Beta >= 2.5: {stats['removed_beta']}銘柄")
    print(f"  通過率: {stats['after_filter']/stats['original']*100:.1f}%")

    # セクター分布
    if 'filtered_df' in result:
        sector_counts = result['filtered_df']['sector'].value_counts()
        print()
        print("【セクター分布】")
        for sector, count in sector_counts.items():
            print(f"  {sector}: {count}銘柄")
