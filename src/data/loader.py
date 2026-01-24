"""
yfinanceによる株価データ取得
"""
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# キャッシュディレクトリ
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


def fetch_ohlcv(
    tickers: list[str],
    start: date,
    end: date,
    use_cache: bool = True,
    max_workers: int = 10
) -> pd.DataFrame:
    """
    複数銘柄のOHLCVデータを取得

    Args:
        tickers: ティッカーリスト
        start: 開始日
        end: 終了日
        use_cache: キャッシュを使用するか
        max_workers: 並列ダウンロード数

    Returns:
        DataFrame with columns: date, ticker, Open, High, Low, Close, Volume
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_data = []
    failed_tickers = []

    # yfinanceで一括取得を試みる（効率的）
    print(f"データ取得中: {len(tickers)}銘柄...")

    try:
        # yfinanceの一括ダウンロード
        df = yf.download(
            tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            group_by='ticker',
            auto_adjust=True,
            progress=True,
            threads=True
        )

        if len(tickers) == 1:
            # 1銘柄の場合はカラム構造が異なる
            df.columns = pd.MultiIndex.from_product([[tickers[0]], df.columns])

        # マルチカラムを整形
        records = []
        for ticker in tickers:
            try:
                if ticker in df.columns.get_level_values(0):
                    ticker_df = df[ticker].copy()
                    ticker_df = ticker_df.dropna(subset=['Close'])
                    if len(ticker_df) > 0:
                        ticker_df['ticker'] = ticker
                        ticker_df = ticker_df.reset_index()
                        ticker_df = ticker_df.rename(columns={'Date': 'date'})
                        records.append(ticker_df)
            except Exception as e:
                failed_tickers.append(ticker)

        if records:
            result = pd.concat(records, ignore_index=True)
            result = result[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            result['date'] = pd.to_datetime(result['date']).dt.date
            print(f"取得成功: {result['ticker'].nunique()}銘柄, {len(result)}行")
            if failed_tickers:
                print(f"取得失敗: {len(failed_tickers)}銘柄")
            return result

    except Exception as e:
        print(f"一括取得に失敗: {e}")
        print("個別取得にフォールバック...")

    # フォールバック: 個別取得
    def fetch_single(ticker: str) -> pd.DataFrame | None:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end + timedelta(days=1), auto_adjust=True)
            if len(hist) > 0:
                hist = hist.reset_index()
                hist['ticker'] = ticker
                hist = hist.rename(columns={'Date': 'date'})
                hist['date'] = pd.to_datetime(hist['date']).dt.date
                return hist[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers), desc="個別取得"):
            result = future.result()
            if result is not None:
                all_data.append(result)
            else:
                failed_tickers.append(futures[future])

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"取得成功: {result['ticker'].nunique()}銘柄")
        if failed_tickers:
            print(f"取得失敗: {len(failed_tickers)}銘柄")
        return result

    raise ValueError("データ取得に失敗しました")


def fetch_topix(start: date, end: date) -> pd.DataFrame:
    """
    TOPIXデータを取得

    Args:
        start: 開始日
        end: 終了日

    Returns:
        DataFrame with columns: date, Open, High, Low, Close, Volume
    """
    print("TOPIX取得中...")
    ticker = "^TPX"  # TOPIXのyfinanceティッカー

    try:
        topix = yf.Ticker(ticker)
        hist = topix.history(start=start, end=end + timedelta(days=1), auto_adjust=True)

        if len(hist) == 0:
            raise ValueError("TOPIXデータが取得できません")

        hist = hist.reset_index()
        hist = hist.rename(columns={'Date': 'date'})
        hist['date'] = pd.to_datetime(hist['date']).dt.date
        result = hist[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"TOPIX取得成功: {len(result)}日分")
        return result

    except Exception as e:
        print(f"TOPIX取得失敗: {e}")
        raise


if __name__ == "__main__":
    # テスト
    from datetime import date

    # 少数銘柄でテスト
    test_tickers = ["7203.T", "6758.T", "9984.T"]
    df = fetch_ohlcv(test_tickers, date(2024, 1, 1), date(2024, 1, 31))
    print(df.head())
    print(f"Shape: {df.shape}")

    # TOPIX
    topix = fetch_topix(date(2024, 1, 1), date(2024, 1, 31))
    print(topix.head())
