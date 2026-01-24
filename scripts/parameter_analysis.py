"""
パラメータ感度分析
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from itertools import product

from config import CAPITAL, TOP_N
from backtest import BacktestEngine


def run_parameter_analysis():
    print("=" * 60)
    print("パラメータ感度分析")
    print("=" * 60)

    # データ読み込み
    data_dir = Path(__file__).parent.parent / "data"
    predictions = pd.read_parquet(data_dir / "test_predictions.parquet")
    labeled_data = pd.read_parquet(data_dir / "labeled_data.parquet")

    # パラメータグリッド
    take_profit_values = [0.08, 0.10, 0.12, 0.15]
    stop_loss_atr_values = [1.0, 1.5, 2.0, 2.5]
    max_hold_values = [10, 15, 20, 25]

    results = []

    # ベースライン
    print("\n【ベースライン】TP=10%, SL=1.5×ATR, Hold=20日")

    # 利確パラメータ分析
    print("\n【分析1】利確パラメータ（損切り=1.5×ATR, 保有=20日固定）")
    print("-" * 50)
    for tp in take_profit_values:
        engine = BacktestEngine(
            initial_capital=CAPITAL,
            position_size=CAPITAL // TOP_N,
            max_positions=TOP_N,
            take_profit=tp,
            stop_loss_atr_mult=1.5,
            max_hold_days=20
        )
        result = engine.run(predictions, labeled_data)
        results.append({
            'param': 'take_profit',
            'value': tp,
            'total_pnl': result.total_pnl,
            'win_rate': result.win_rate,
            'pf': result.profit_factor,
            'max_dd': result.max_drawdown_pct,
            'n_trades': result.n_trades
        })
        print(f"  TP={tp:4.0%}: 損益=¥{result.total_pnl:>10,.0f}, 勝率={result.win_rate:5.1%}, PF={result.profit_factor:.2f}, DD={result.max_drawdown_pct:5.1f}%")

    # 損切りパラメータ分析
    print("\n【分析2】損切りパラメータ（利確=10%, 保有=20日固定）")
    print("-" * 50)
    for sl in stop_loss_atr_values:
        engine = BacktestEngine(
            initial_capital=CAPITAL,
            position_size=CAPITAL // TOP_N,
            max_positions=TOP_N,
            take_profit=0.10,
            stop_loss_atr_mult=sl,
            max_hold_days=20
        )
        result = engine.run(predictions, labeled_data)
        results.append({
            'param': 'stop_loss_atr',
            'value': sl,
            'total_pnl': result.total_pnl,
            'win_rate': result.win_rate,
            'pf': result.profit_factor,
            'max_dd': result.max_drawdown_pct,
            'n_trades': result.n_trades
        })
        print(f"  SL={sl:.1f}×ATR: 損益=¥{result.total_pnl:>10,.0f}, 勝率={result.win_rate:5.1%}, PF={result.profit_factor:.2f}, DD={result.max_drawdown_pct:5.1f}%")

    # 保有期間パラメータ分析
    print("\n【分析3】保有期間パラメータ（利確=10%, 損切り=1.5×ATR固定）")
    print("-" * 50)
    for hold in max_hold_values:
        engine = BacktestEngine(
            initial_capital=CAPITAL,
            position_size=CAPITAL // TOP_N,
            max_positions=TOP_N,
            take_profit=0.10,
            stop_loss_atr_mult=1.5,
            max_hold_days=hold
        )
        result = engine.run(predictions, labeled_data)
        results.append({
            'param': 'max_hold_days',
            'value': hold,
            'total_pnl': result.total_pnl,
            'win_rate': result.win_rate,
            'pf': result.profit_factor,
            'max_dd': result.max_drawdown_pct,
            'n_trades': result.n_trades
        })
        print(f"  Hold={hold:2d}日: 損益=¥{result.total_pnl:>10,.0f}, 勝率={result.win_rate:5.1%}, PF={result.profit_factor:.2f}, DD={result.max_drawdown_pct:5.1f}%")

    # 最適組み合わせを探索（上位候補のみ）
    print("\n【分析4】組み合わせ最適化（主要候補）")
    print("-" * 50)

    best_combinations = []
    for tp, sl, hold in product([0.10, 0.12], [1.5, 2.0], [15, 20]):
        engine = BacktestEngine(
            initial_capital=CAPITAL,
            position_size=CAPITAL // TOP_N,
            max_positions=TOP_N,
            take_profit=tp,
            stop_loss_atr_mult=sl,
            max_hold_days=hold
        )
        result = engine.run(predictions, labeled_data)
        best_combinations.append({
            'tp': tp,
            'sl': sl,
            'hold': hold,
            'total_pnl': result.total_pnl,
            'win_rate': result.win_rate,
            'pf': result.profit_factor,
            'max_dd': result.max_drawdown_pct,
            'n_trades': result.n_trades
        })

    # 損益でソート
    best_combinations.sort(key=lambda x: x['total_pnl'], reverse=True)

    print("  [損益順 Top 5]")
    for i, c in enumerate(best_combinations[:5]):
        print(f"  {i+1}. TP={c['tp']:.0%}, SL={c['sl']:.1f}×ATR, Hold={c['hold']}日")
        print(f"     損益=¥{c['total_pnl']:>10,.0f}, 勝率={c['win_rate']:5.1%}, PF={c['pf']:.2f}, DD={c['max_dd']:5.1f}%")

    # PFでソート
    best_combinations.sort(key=lambda x: x['pf'], reverse=True)

    print("\n  [PF順 Top 5]")
    for i, c in enumerate(best_combinations[:5]):
        print(f"  {i+1}. TP={c['tp']:.0%}, SL={c['sl']:.1f}×ATR, Hold={c['hold']}日")
        print(f"     損益=¥{c['total_pnl']:>10,.0f}, 勝率={c['win_rate']:5.1%}, PF={c['pf']:.2f}, DD={c['max_dd']:5.1f}%")

    return results, best_combinations


if __name__ == "__main__":
    results, best = run_parameter_analysis()
