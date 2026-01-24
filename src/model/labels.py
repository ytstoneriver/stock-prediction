"""
ラベル生成（+10%到達判定）
"""
import pandas as pd
import numpy as np


def create_labels(
    df: pd.DataFrame,
    target_return: float = 0.10,
    horizon_days: int = 20
) -> pd.DataFrame:
    """
    各日付・銘柄に対してラベルを生成

    ラベル定義:
    - 未来horizon_days営業日以内に高値がtarget_return以上上昇したら1、それ以外は0

    Args:
        df: OHLCVデータ（columns: date, ticker, Open, High, Low, Close, Volume）
        target_return: 目標リターン（例: 0.10 = +10%）
        horizon_days: ルックアヘッド期間（営業日）

    Returns:
        DataFrame with additional column: label
    """
    result = df.copy()
    result = result.sort_values(['ticker', 'date']).reset_index(drop=True)

    # 各銘柄ごとに処理
    labels = []

    for ticker in result['ticker'].unique():
        ticker_df = result[result['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

        n = len(ticker_df)
        ticker_labels = np.zeros(n, dtype=int)

        for i in range(n):
            # 基準価格
            base_price = ticker_df.loc[i, 'Close']

            # 未来期間の高値を取得（i+1 から i+horizon_days まで）
            future_start = i + 1
            future_end = min(i + horizon_days + 1, n)

            if future_start < n:
                future_highs = ticker_df.loc[future_start:future_end - 1, 'High']
                max_high = future_highs.max()

                # リターン計算
                max_return = (max_high / base_price) - 1

                if max_return >= target_return:
                    ticker_labels[i] = 1

        ticker_df['label'] = ticker_labels
        labels.append(ticker_df)

    result = pd.concat(labels, ignore_index=True)
    return result


def calculate_positive_rate(df: pd.DataFrame) -> dict:
    """
    positive_rateを計算

    Args:
        df: ラベル付きデータ（columns: date, ticker, label, ...）

    Returns:
        dict with positive_rate statistics
    """
    # 全体
    total = len(df)
    positive = df['label'].sum()
    overall_rate = positive / total if total > 0 else 0

    # 年別
    df_with_year = df.copy()
    df_with_year['year'] = pd.to_datetime(df_with_year['date']).dt.year
    yearly = df_with_year.groupby('year').agg(
        total=('label', 'count'),
        positive=('label', 'sum')
    )
    yearly['rate'] = yearly['positive'] / yearly['total']

    # 月別
    df_with_month = df.copy()
    df_with_month['year_month'] = pd.to_datetime(df_with_month['date']).dt.to_period('M')
    monthly = df_with_month.groupby('year_month').agg(
        total=('label', 'count'),
        positive=('label', 'sum')
    )
    monthly['rate'] = monthly['positive'] / monthly['total']

    return {
        'overall': {
            'total': total,
            'positive': positive,
            'rate': overall_rate
        },
        'yearly': yearly.to_dict('index'),
        'monthly': monthly.to_dict('index')
    }


def print_positive_rate_report(stats: dict):
    """
    positive_rateのレポートを表示
    """
    print("=" * 50)
    print("Positive Rate Report")
    print("=" * 50)

    # 全体
    overall = stats['overall']
    print(f"\n【全体】")
    print(f"  サンプル数: {overall['total']:,}")
    print(f"  正例数: {overall['positive']:,}")
    print(f"  正例率: {overall['rate']:.2%}")

    # 年別
    print(f"\n【年別】")
    for year, data in sorted(stats['yearly'].items()):
        print(f"  {year}: {data['rate']:.2%} ({data['positive']:,}/{data['total']:,})")

    # 月別（直近12ヶ月のみ表示）
    print(f"\n【月別（直近12ヶ月）】")
    monthly_items = sorted(stats['monthly'].items(), reverse=True)[:12]
    for month, data in reversed(monthly_items):
        print(f"  {month}: {data['rate']:.2%} ({data['positive']:,}/{data['total']:,})")

    print("=" * 50)


if __name__ == "__main__":
    # テスト用のダミーデータ
    import numpy as np
    from datetime import date, timedelta

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    test_df = pd.DataFrame({
        'date': dates * 2,
        'ticker': ['7203.T'] * 100 + ['6758.T'] * 100,
        'Open': np.random.uniform(2000, 2100, 200),
        'High': np.random.uniform(2050, 2200, 200),
        'Low': np.random.uniform(1900, 2000, 200),
        'Close': np.random.uniform(2000, 2100, 200),
        'Volume': np.random.uniform(1e6, 1e7, 200)
    })

    labeled = create_labels(test_df, target_return=0.05, horizon_days=10)
    stats = calculate_positive_rate(labeled)
    print_positive_rate_report(stats)
