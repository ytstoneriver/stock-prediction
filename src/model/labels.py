"""
v2ラベル生成（利確vs損切りの先着判定）

v1との主な違い:
1. エントリー価格 = 翌日始値（v1は当日終値）
2. 利確(+10%)が先に到達 → label=1
3. 損切り(-10%)が先に到達 or タイムアウト → label=0
4. 同日両方到達 → 損切り優先（保守的）
"""
import pandas as pd
import numpy as np
from enum import Enum


class ExitReason(Enum):
    """決済理由"""
    TAKE_PROFIT = 'take_profit'
    STOP_LOSS = 'stop_loss'
    STOP_LOSS_SAME_DAY = 'stop_loss_same_day'
    TIMEOUT = 'timeout'
    INSUFFICIENT_DATA = 'insufficient_data'


def create_labels(
    df: pd.DataFrame,
    take_profit: float = 0.10,
    stop_loss: float = 0.10,
    horizon_days: int = 20
) -> pd.DataFrame:
    """
    v2ラベルを生成

    ラベル定義:
    - シグナル日: d日目（当日終値時点で予測）
    - エントリー: d+1日目の始値
    - 判定: d+1 ~ d+horizon_daysの間に
        - 高値が entry_price * (1 + take_profit) に先に到達 → label=1
        - 安値が entry_price * (1 - stop_loss) に先に到達 → label=0
        - どちらにも到達しない → label=0（タイムアウト）

    Args:
        df: OHLCVデータ（columns: date, ticker, Open, High, Low, Close, Volume）
        take_profit: 利確ライン（例: 0.10 = +10%）
        stop_loss: 損切りライン（例: 0.10 = -10%）
        horizon_days: 最大保有期間（営業日）

    Returns:
        DataFrame with columns: label, exit_reason, entry_price
    """
    result = df.copy()
    result = result.sort_values(['ticker', 'date']).reset_index(drop=True)

    labels = []

    for ticker in result['ticker'].unique():
        ticker_df = result[result['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

        n = len(ticker_df)
        ticker_labels = np.full(n, -1, dtype=int)  # -1 = 未計算
        ticker_entry_prices = np.full(n, np.nan)
        ticker_exit_reasons = [''] * n

        for i in range(n - 1):  # 最終日は翌日データがないのでスキップ
            # エントリー価格 = 翌日始値
            entry_idx = i + 1
            entry_price = ticker_df.iloc[entry_idx]['Open']
            ticker_entry_prices[i] = entry_price

            # 利確・損切りライン
            tp_price = entry_price * (1 + take_profit)
            sl_price = entry_price * (1 - stop_loss)

            # 判定期間: entry_idx から entry_idx + horizon_days - 1 まで
            check_start = entry_idx
            check_end = min(entry_idx + horizon_days, n)

            if check_start >= n:
                ticker_exit_reasons[i] = ExitReason.INSUFFICIENT_DATA.value
                continue

            # 日次で判定（先着順）
            exit_reason = ExitReason.TIMEOUT
            label = 0

            for j in range(check_start, check_end):
                high = ticker_df.iloc[j]['High']
                low = ticker_df.iloc[j]['Low']

                # 判定
                hit_tp = high >= tp_price
                hit_sl = low <= sl_price

                if hit_tp and hit_sl:
                    # 両方到達: 損切り優先（保守的）
                    exit_reason = ExitReason.STOP_LOSS_SAME_DAY
                    label = 0
                    break
                elif hit_tp:
                    exit_reason = ExitReason.TAKE_PROFIT
                    label = 1
                    break
                elif hit_sl:
                    exit_reason = ExitReason.STOP_LOSS
                    label = 0
                    break

            ticker_labels[i] = label
            ticker_exit_reasons[i] = exit_reason.value

        ticker_df['label'] = ticker_labels
        ticker_df['entry_price'] = ticker_entry_prices
        ticker_df['exit_reason'] = ticker_exit_reasons
        labels.append(ticker_df)

    result = pd.concat(labels, ignore_index=True)

    # データ不足行を除外（label == -1）
    result = result[result['label'] >= 0].copy()

    return result


def analyze_label_distribution(df: pd.DataFrame) -> dict:
    """
    ラベル分布を分析

    Args:
        df: v2ラベル付きデータ

    Returns:
        統計情報のdict
    """
    total = len(df)
    if total == 0:
        return {'total': 0, 'positive': 0, 'negative': 0, 'positive_rate': 0}

    positive = (df['label'] == 1).sum()
    negative = (df['label'] == 0).sum()

    # exit_reasonの内訳
    exit_reason_counts = df['exit_reason'].value_counts().to_dict()

    # 年別
    df_copy = df.copy()
    df_copy['year'] = pd.to_datetime(df_copy['date']).dt.year
    yearly = df_copy.groupby('year').agg(
        total=('label', 'count'),
        positive=('label', 'sum')
    )
    yearly['rate'] = yearly['positive'] / yearly['total']
    yearly = yearly.to_dict('index')

    return {
        'total': total,
        'positive': int(positive),
        'negative': int(negative),
        'positive_rate': positive / total,
        'exit_reasons': exit_reason_counts,
        'yearly': yearly
    }


def print_label_report(stats: dict):
    """v2ラベルのレポートを表示"""
    print("=" * 60)
    print("v2 Label Distribution Report")
    print("=" * 60)

    print(f"\n【全体】")
    print(f"  サンプル数: {stats['total']:,}")
    print(f"  正例（利確先着）: {stats['positive']:,} ({stats['positive_rate']:.2%})")
    print(f"  負例（損切り/タイムアウト）: {stats['negative']:,}")

    if 'exit_reasons' in stats:
        print(f"\n【Exit Reason内訳】")
        for reason, count in sorted(stats['exit_reasons'].items()):
            pct = count / stats['total'] * 100
            print(f"  {reason}: {count:,} ({pct:.1f}%)")

    if 'yearly' in stats:
        print(f"\n【年別正例率】")
        for year, data in sorted(stats['yearly'].items()):
            print(f"  {year}: {data['rate']:.2%} ({data['positive']:,}/{data['total']:,})")

    print("=" * 60)


if __name__ == "__main__":
    # テスト
    from datetime import date, timedelta

    # ダミーデータ
    n_days = 50
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    test_df = pd.DataFrame({
        'date': dates,
        'ticker': ['TEST.T'] * n_days,
        'Open': [1000 + i * 5 for i in range(n_days)],
        'High': [1050 + i * 5 for i in range(n_days)],
        'Low': [950 + i * 5 for i in range(n_days)],
        'Close': [1020 + i * 5 for i in range(n_days)],
        'Volume': [100000] * n_days
    })

    result = create_labels(test_df, take_profit=0.10, stop_loss=0.10)
    stats = analyze_label_distribution(result)
    print_label_report(stats)
