"""
東証プライム銘柄リスト取得とユニバースフィルタ
"""
import pandas as pd
import requests
from io import StringIO
from pathlib import Path

# JPX上場銘柄一覧のURL
JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"


def fetch_prime_tickers() -> list[str]:
    """
    JPXから東証プライム銘柄リストを取得

    Returns:
        list[str]: ティッカーリスト（例: ['7203.T', '6758.T', ...]）
    """
    try:
        # JPXのExcelファイルを取得
        df = pd.read_excel(JPX_LIST_URL)

        # プライム市場のみフィルタ
        prime_df = df[df['市場・商品区分'] == 'プライム（内国株式）']

        # ティッカーコードを取得（4桁コード + .T）
        tickers = [f"{code}.T" for code in prime_df['コード'].astype(str)]

        return tickers
    except Exception as e:
        print(f"JPXからの取得に失敗: {e}")
        print("フォールバック: 主要銘柄リストを使用")
        return get_fallback_tickers()


def get_fallback_tickers() -> list[str]:
    """
    フォールバック用の主要銘柄リスト（TOPIX Core30 + α）
    """
    # 時価総額上位の主要銘柄
    major_tickers = [
        # 自動車
        "7203.T",  # トヨタ
        "7267.T",  # ホンダ
        "7201.T",  # 日産
        # 電機
        "6758.T",  # ソニー
        "6501.T",  # 日立
        "6502.T",  # 東芝
        "6752.T",  # パナソニック
        "6861.T",  # キーエンス
        "6954.T",  # ファナック
        # 通信
        "9432.T",  # NTT
        "9433.T",  # KDDI
        "9434.T",  # ソフトバンク
        # 金融
        "8306.T",  # 三菱UFJ
        "8316.T",  # 三井住友FG
        "8411.T",  # みずほFG
        # 商社
        "8058.T",  # 三菱商事
        "8031.T",  # 三井物産
        "8001.T",  # 伊藤忠
        # 製薬
        "4502.T",  # 武田
        "4503.T",  # アステラス
        "4568.T",  # 第一三共
        # その他
        "9984.T",  # ソフトバンクG
        "6098.T",  # リクルート
        "4063.T",  # 信越化学
        "8035.T",  # 東京エレクトロン
        "7974.T",  # 任天堂
        "9983.T",  # ファーストリテイリング
        "4661.T",  # OLC
        "6367.T",  # ダイキン
    ]
    return major_tickers


def apply_liquidity_filter(
    df: pd.DataFrame,
    min_price: float,
    max_price: float,
    min_turnover: float,
    min_volume: float,
    window: int = 20
) -> pd.DataFrame:
    """
    流動性フィルタを適用

    Args:
        df: OHLCVデータ（マルチインデックス: date, ticker）
        min_price: 最小株価
        max_price: 最大株価
        min_turnover: 最小売買代金（日次平均）
        min_volume: 最小出来高（日次平均）
        window: 平均計算の窓

    Returns:
        フィルタ後のDataFrame
    """
    result = df.copy()

    # 売買代金を計算
    result['turnover'] = result['Close'] * result['Volume']

    # 各銘柄ごとにrolling平均を計算
    result['avg_turnover'] = result.groupby('ticker')['turnover'].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    result['avg_volume'] = result.groupby('ticker')['Volume'].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )

    # フィルタ適用
    mask = (
        (result['Close'] >= min_price) &
        (result['Close'] <= max_price) &
        (result['avg_turnover'] >= min_turnover) &
        (result['avg_volume'] >= min_volume)
    )

    return result[mask]


if __name__ == "__main__":
    # テスト実行
    tickers = fetch_prime_tickers()
    print(f"取得銘柄数: {len(tickers)}")
    print(f"先頭10銘柄: {tickers[:10]}")
