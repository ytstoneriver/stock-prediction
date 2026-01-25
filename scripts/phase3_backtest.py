"""
Phase 3: バックテスト

予測結果を使用してバックテストを実行
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dataclasses import dataclass
from datetime import date

from config import (
    LABEL_TAKE_PROFIT, LABEL_STOP_LOSS, LABEL_HORIZON_DAYS,
    TOP_N_STOCKS, SCORE_THRESHOLD, MIN_SCORE_GAP,
)


# 資金設定
CAPITAL = 1_000_000  # 100万円


@dataclass
class Trade:
    """取引記録"""
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    exit_reason: str
    pnl: float
    pnl_pct: float
    hold_days: int


class BacktestEngine:
    """バックテストエンジン"""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        position_size: float = 500_000,
        max_positions: int = 2,
        take_profit: float = 0.10,
        stop_loss: float = 0.10,
        max_hold_days: int = 20,
        commission_rate: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days
        self.commission_rate = commission_rate

    def run(
        self,
        predictions_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame
    ) -> dict:
        """バックテストを実行"""
        predictions_df = predictions_df.copy()
        ohlcv_df = ohlcv_df.copy()
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])

        signal_dates = sorted(predictions_df['date'].unique())

        capital = self.initial_capital
        positions = []
        trades = []
        equity_curve = []

        for i, signal_date in enumerate(signal_dates[:-1]):
            next_dates = [d for d in signal_dates if d > signal_date]
            if not next_dates:
                continue
            entry_date = next_dates[0]

            # 既存ポジションのチェック
            new_positions = []
            for pos in positions:
                pos['hold_days'] += 1

                day_data = ohlcv_df[
                    (ohlcv_df['date'] == entry_date) &
                    (ohlcv_df['ticker'] == pos['ticker'])
                ]

                if day_data.empty:
                    new_positions.append(pos)
                    continue

                high = day_data.iloc[0]['High']
                low = day_data.iloc[0]['Low']
                close = day_data.iloc[0]['Close']

                tp_price = pos['entry_price'] * (1 + self.take_profit)
                sl_price = pos['entry_price'] * (1 - self.stop_loss)

                exit_price = None
                exit_reason = None

                if high >= tp_price and low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                elif low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                elif pos['hold_days'] >= self.max_hold_days:
                    exit_price = close
                    exit_reason = 'timeout'

                if exit_price:
                    shares = pos['shares']
                    pnl = (exit_price - pos['entry_price']) * shares
                    pnl -= self.position_size * self.commission_rate
                    pnl_pct = pnl / self.position_size * 100

                    trade = Trade(
                        ticker=pos['ticker'],
                        entry_date=pos['entry_date'],
                        entry_price=pos['entry_price'],
                        exit_date=entry_date,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        hold_days=pos['hold_days']
                    )
                    trades.append(trade)
                    capital += pnl
                else:
                    new_positions.append(pos)

            positions = new_positions

            # 新規エントリー
            if len(positions) < self.max_positions:
                day_signals = predictions_df[
                    (predictions_df['date'] == signal_date) &
                    (predictions_df['is_top_n'] == 1)
                ].sort_values('rank')

                for _, signal in day_signals.iterrows():
                    if len(positions) >= self.max_positions:
                        break

                    ticker = signal['ticker']

                    if any(p['ticker'] == ticker for p in positions):
                        continue

                    entry_data = ohlcv_df[
                        (ohlcv_df['date'] == entry_date) &
                        (ohlcv_df['ticker'] == ticker)
                    ]

                    if entry_data.empty:
                        continue

                    entry_price = entry_data.iloc[0]['Open']
                    shares = int(self.position_size / entry_price)

                    if shares <= 0:
                        continue

                    positions.append({
                        'ticker': ticker,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'hold_days': 0
                    })

            equity_curve.append({
                'date': entry_date,
                'capital': capital,
                'n_positions': len(positions)
            })

        result = self._calculate_metrics(trades, equity_curve)
        result['trades'] = trades
        result['equity_curve'] = pd.DataFrame(equity_curve)

        return result

    def _calculate_metrics(self, trades: list, equity_curve: list) -> dict:
        """評価指標を計算"""
        if not trades:
            return {
                'n_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'total_pnl_pct': 0, 'profit_factor': 0,
                'max_drawdown': 0, 'max_drawdown_pct': 0
            }

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_profit = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses))

        if equity_curve:
            capitals = [e['capital'] for e in equity_curve]
            peak = capitals[0]
            max_dd = 0
            for cap in capitals:
                if cap > peak:
                    peak = cap
                dd = peak - cap
                if dd > max_dd:
                    max_dd = dd
            max_dd_pct = max_dd / self.initial_capital * 100
        else:
            max_dd = 0
            max_dd_pct = 0

        total_pnl = sum(t.pnl for t in trades)

        return {
            'n_trades': len(trades),
            'n_wins': len(wins),
            'n_losses': len(losses),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl / self.initial_capital * 100,
            'avg_pnl': total_pnl / len(trades) if trades else 0,
            'avg_win': total_profit / len(wins) if wins else 0,
            'avg_loss': -total_loss / len(losses) if losses else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'avg_hold_days': sum(t.hold_days for t in trades) / len(trades) if trades else 0
        }


def print_backtest_report(result: dict):
    """バックテスト結果を表示"""
    print("=" * 60)
    print("Backtest Report")
    print("=" * 60)

    print(f"\n【サマリー】")
    print(f"  取引回数: {result['n_trades']}")
    print(f"  勝ち/負け: {result.get('n_wins', 0)} / {result.get('n_losses', 0)}")
    print(f"  勝率: {result['win_rate']:.1%}")

    print(f"\n【損益】")
    print(f"  総損益: ¥{result['total_pnl']:,.0f} ({result['total_pnl_pct']:+.2f}%)")
    print(f"  平均損益: ¥{result.get('avg_pnl', 0):,.0f}")
    print(f"  平均利益: ¥{result.get('avg_win', 0):,.0f}")
    print(f"  平均損失: ¥{result.get('avg_loss', 0):,.0f}")

    print(f"\n【リスク】")
    print(f"  プロフィットファクター: {result['profit_factor']:.2f}")
    print(f"  最大ドローダウン: ¥{result['max_drawdown']:,.0f} ({result['max_drawdown_pct']:.2f}%)")
    print(f"  平均保有日数: {result.get('avg_hold_days', 0):.1f}日")

    if 'trades' in result and result['trades']:
        print(f"\n【決済理由内訳】")
        reasons = {}
        for t in result['trades']:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        for reason, count in sorted(reasons.items()):
            pct = count / result['n_trades'] * 100
            print(f"  {reason}: {count}回 ({pct:.1f}%)")

    print("=" * 60)


def main():
    print("=" * 60)
    print("Phase 3: バックテスト")
    print("=" * 60)

    # 設定表示
    print(f"\n【設定】")
    print(f"  資金: ¥{CAPITAL:,}")
    print(f"  最大同時保有: {TOP_N_STOCKS}銘柄")
    print(f"  利確: +{LABEL_TAKE_PROFIT:.0%}")
    print(f"  損切: -{LABEL_STOP_LOSS:.0%}")
    print(f"  最大保有期間: {LABEL_HORIZON_DAYS}営業日")
    print(f"  スコア閾値: {SCORE_THRESHOLD}")
    print(f"  最小スコア差: {MIN_SCORE_GAP}")

    data_dir = Path(__file__).parent.parent / "data"

    # データ読み込み
    print("\n【Step 1】データ読み込み")

    predictions_path = data_dir / "predictions.parquet"
    if not predictions_path.exists():
        print(f"  エラー: {predictions_path} が見つかりません")
        print(f"  → 先に phase2_train.py を実行してください")
        return None

    predictions = pd.read_parquet(predictions_path)
    print(f"  予測データ: {len(predictions):,}行")

    features_path = data_dir / "features.parquet"
    if features_path.exists():
        ohlcv_df = pd.read_parquet(features_path)
    else:
        ohlcv_df = pd.read_parquet(data_dir / "labeled_data.parquet")
    print(f"  OHLCVデータ: {len(ohlcv_df):,}行")

    # セクター情報読み込み
    sector_path = data_dir / "sector_mapping.parquet"
    if sector_path.exists():
        sector_df = pd.read_parquet(sector_path)
        sector_mapping = dict(zip(sector_df['ticker'], sector_df['sector']))
        print(f"  セクター情報: {len(sector_mapping)}銘柄")
    else:
        sector_mapping = {}

    # バックテスト実行
    print("\n【Step 2】バックテスト実行")

    engine = BacktestEngine(
        initial_capital=CAPITAL,
        position_size=CAPITAL // max(TOP_N_STOCKS, 1),
        max_positions=TOP_N_STOCKS,
        take_profit=LABEL_TAKE_PROFIT,
        stop_loss=LABEL_STOP_LOSS,
        max_hold_days=LABEL_HORIZON_DAYS
    )

    result = engine.run(predictions, ohlcv_df)

    # 結果表示
    print("\n【Step 3】結果")
    print_backtest_report(result)

    # セクター別分析
    if result['trades'] and sector_mapping:
        print("\n【セクター別分析】")
        sector_stats = {}
        for t in result['trades']:
            sector = sector_mapping.get(t.ticker, 'Unknown')
            if sector not in sector_stats:
                sector_stats[sector] = {'n': 0, 'wins': 0, 'pnl': 0}
            sector_stats[sector]['n'] += 1
            sector_stats[sector]['wins'] += 1 if t.pnl > 0 else 0
            sector_stats[sector]['pnl'] += t.pnl

        print(f"  {'セクター':<25} {'取引':<6} {'勝率':<8} {'損益':<12}")
        print(f"  {'-'*55}")
        for sector in sorted(sector_stats.keys(), key=lambda x: sector_stats[x]['pnl'], reverse=True):
            stats = sector_stats[sector]
            win_rate = stats['wins'] / stats['n'] if stats['n'] > 0 else 0
            print(f"  {sector:<25} {stats['n']:<6} {win_rate:>6.1%}  ¥{stats['pnl']:>10,.0f}")

    # 取引履歴保存
    if result['trades']:
        trades_df = pd.DataFrame([
            {
                'ticker': t.ticker,
                'sector': sector_mapping.get(t.ticker, 'Unknown'),
                'entry_date': t.entry_date,
                'entry_price': t.entry_price,
                'exit_date': t.exit_date,
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'hold_days': t.hold_days
            }
            for t in result['trades']
        ])
        trades_df.to_csv(data_dir / "backtest_trades.csv", index=False)
        print(f"\n【保存】")
        print(f"  取引履歴: {data_dir / 'backtest_trades.csv'}")

    # 判断
    print("\n【判断】")
    if result['n_trades'] == 0:
        print(f"  取引なし")
    elif result['total_pnl'] > 0 and result['win_rate'] >= 0.55:
        print(f"  総損益 = ¥{result['total_pnl']:,.0f} > 0、勝率 = {result['win_rate']:.1%} >= 55%")
        print(f"  → 成功。本番運用を検討可能")
    elif result['total_pnl'] > 0:
        print(f"  総損益 = ¥{result['total_pnl']:,.0f} > 0、勝率 = {result['win_rate']:.1%}")
        print(f"  → プラスだが改善余地あり")
    else:
        print(f"  総損益 = ¥{result['total_pnl']:,.0f} < 0")
        print(f"  → モデル・特徴量の見直しが必要")

    return result


if __name__ == "__main__":
    result = main()
