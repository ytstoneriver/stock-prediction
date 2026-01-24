"""
バックテストエンジン
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import date
from typing import Literal


@dataclass
class Trade:
    """個別取引"""
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: Literal['take_profit', 'stop_loss', 'timeout'] | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    hold_days: int = 0


@dataclass
class BacktestResult:
    """バックテスト結果"""
    trades: list[Trade]
    initial_capital: float
    final_capital: float

    # 集計指標
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_hold_days: float = 0.0
    n_trading_days: int = 0
    investment_ratio: float = 0.0  # ポジションを持っていた日の割合

    def __post_init__(self):
        self._calculate_metrics()

    def _calculate_metrics(self):
        if not self.trades:
            return

        completed_trades = [t for t in self.trades if t.pnl is not None]
        if not completed_trades:
            return

        # 基本集計
        self.n_trades = len(completed_trades)
        pnls = [t.pnl for t in completed_trades]
        pnl_pcts = [t.pnl_pct for t in completed_trades]

        self.total_pnl = sum(pnls)
        self.total_pnl_pct = self.total_pnl / self.initial_capital * 100

        # 勝敗
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        self.n_wins = len(wins)
        self.n_losses = len(losses)
        self.win_rate = self.n_wins / self.n_trades if self.n_trades > 0 else 0

        # 平均利益/損失
        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = np.mean(losses) if losses else 0

        # プロフィットファクター
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        self.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # 平均保有日数
        self.avg_hold_days = np.mean([t.hold_days for t in completed_trades])

        # 最大ドローダウン計算
        self._calculate_drawdown(completed_trades)

    def _calculate_drawdown(self, trades: list[Trade]):
        """最大ドローダウンを計算"""
        if not trades:
            return

        # 累積損益を計算
        cumulative_pnl = [0]
        for t in sorted(trades, key=lambda x: x.exit_date):
            cumulative_pnl.append(cumulative_pnl[-1] + t.pnl)

        # ドローダウン
        peak = cumulative_pnl[0]
        max_dd = 0
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            dd = peak - pnl
            if dd > max_dd:
                max_dd = dd

        self.max_drawdown = max_dd
        self.max_drawdown_pct = max_dd / self.initial_capital * 100 if self.initial_capital > 0 else 0

    def print_report(self):
        """結果レポートを出力"""
        print("=" * 50)
        print("バックテスト結果")
        print("=" * 50)
        print(f"初期資金: ¥{self.initial_capital:,.0f}")
        print(f"最終資金: ¥{self.final_capital:,.0f}")
        print(f"総損益: ¥{self.total_pnl:,.0f} ({self.total_pnl_pct:+.2f}%)")
        print("-" * 50)
        print(f"取引回数: {self.n_trades}")
        print(f"勝ち: {self.n_wins} / 負け: {self.n_losses}")
        print(f"勝率: {self.win_rate:.1%}")
        print(f"平均利益: ¥{self.avg_win:,.0f}")
        print(f"平均損失: ¥{self.avg_loss:,.0f}")
        print(f"プロフィットファクター: {self.profit_factor:.2f}")
        print("-" * 50)
        print(f"最大ドローダウン: ¥{self.max_drawdown:,.0f} ({self.max_drawdown_pct:.2f}%)")
        print(f"平均保有日数: {self.avg_hold_days:.1f}日")
        print("=" * 50)


class BacktestEngine:
    """バックテストエンジン"""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        position_size: float = 500_000,
        max_positions: int = 2,
        take_profit: float = 0.10,
        stop_loss_atr_mult: float = 1.5,
        max_hold_days: int = 20,
        commission_rate: float = 0.001  # 0.1%
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.take_profit = take_profit
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.max_hold_days = max_hold_days
        self.commission_rate = commission_rate

    def run(
        self,
        predictions_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        topix_df: pd.DataFrame | None = None,
        use_market_filter: bool = False
    ) -> BacktestResult:
        """
        バックテストを実行

        Args:
            predictions_df: 予測結果（columns: date, ticker, score, rank）
            ohlcv_df: OHLCVデータ
            topix_df: TOPIXデータ（地合いフィルタ用）
            use_market_filter: 地合いフィルタを使用するか

        Returns:
            BacktestResult
        """
        # データ準備
        predictions = predictions_df.copy()
        ohlcv = ohlcv_df.copy()

        # ATR計算
        ohlcv = self._calculate_atr(ohlcv)

        # 地合いフィルタ準備
        market_filter = None
        if use_market_filter and topix_df is not None:
            market_filter = self._prepare_market_filter(topix_df)

        # 取引日リスト
        trading_dates = sorted(predictions['date'].unique())

        # 状態管理
        capital = self.initial_capital
        open_positions: list[dict] = []
        trades: list[Trade] = []

        for i, signal_date in enumerate(trading_dates[:-1]):  # 最後の日はエントリーできない
            entry_date = trading_dates[i + 1]  # 翌営業日にエントリー

            # 既存ポジションの決済チェック
            open_positions, closed_trades, capital = self._check_exits(
                open_positions, ohlcv, entry_date, capital
            )
            trades.extend(closed_trades)

            # 地合いフィルタチェック
            if market_filter is not None:
                if signal_date in market_filter and not market_filter[signal_date]:
                    continue  # ノートレ

            # 新規エントリー
            if len(open_positions) < self.max_positions:
                # 上位銘柄を取得
                day_predictions = predictions[predictions['date'] == signal_date]
                top_picks = day_predictions.nsmallest(
                    self.max_positions - len(open_positions),
                    'rank'
                )

                for _, row in top_picks.iterrows():
                    ticker = row['ticker']

                    # エントリー日のOHLCVを取得
                    entry_data = ohlcv[
                        (ohlcv['ticker'] == ticker) &
                        (ohlcv['date'] == entry_date)
                    ]

                    if len(entry_data) == 0:
                        continue

                    entry_price = entry_data.iloc[0]['Open']
                    atr = entry_data.iloc[0].get('atr', entry_price * 0.03)  # ATRがなければ3%

                    # 資金チェック
                    if capital < self.position_size:
                        break

                    # ポジション作成
                    position = {
                        'ticker': ticker,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'take_profit_price': entry_price * (1 + self.take_profit),
                        'stop_loss_price': entry_price - self.stop_loss_atr_mult * atr,
                        'position_size': self.position_size,
                        'shares': self.position_size / entry_price,
                        'hold_days': 0
                    }
                    open_positions.append(position)
                    capital -= self.position_size * (1 + self.commission_rate)

        # 残りのポジションを強制決済
        for pos in open_positions:
            # 最終日の終値で決済
            last_data = ohlcv[ohlcv['ticker'] == pos['ticker']].sort_values('date').iloc[-1]
            exit_price = last_data['Close']
            pnl = (exit_price - pos['entry_price']) * pos['shares']
            pnl -= self.position_size * self.commission_rate  # 手数料

            trade = Trade(
                ticker=pos['ticker'],
                entry_date=pos['entry_date'],
                entry_price=pos['entry_price'],
                exit_date=last_data['date'],
                exit_price=exit_price,
                exit_reason='timeout',
                pnl=pnl,
                pnl_pct=pnl / self.position_size * 100,
                hold_days=pos['hold_days']
            )
            trades.append(trade)
            capital += self.position_size + pnl

        final_capital = capital

        return BacktestResult(
            trades=trades,
            initial_capital=self.initial_capital,
            final_capital=final_capital
        )

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATRを計算"""
        result = df.copy()

        for ticker in result['ticker'].unique():
            mask = result['ticker'] == ticker
            ticker_df = result[mask].sort_values('date')

            high = ticker_df['High']
            low = ticker_df['Low']
            close = ticker_df['Close'].shift(1)

            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()

            result.loc[mask, 'atr'] = atr.values

        return result

    def _prepare_market_filter(self, topix_df: pd.DataFrame) -> dict:
        """地合いフィルタを準備（TOPIX > MA25）"""
        topix = topix_df.copy()
        topix = topix.sort_values('date')
        topix['ma25'] = topix['Close'].rolling(25).mean()
        topix['above_ma25'] = topix['Close'] > topix['ma25']

        return dict(zip(topix['date'], topix['above_ma25']))

    def _check_exits(
        self,
        positions: list[dict],
        ohlcv: pd.DataFrame,
        current_date: date,
        capital: float
    ) -> tuple[list[dict], list[Trade], float]:
        """ポジションの決済をチェック"""
        remaining_positions = []
        closed_trades = []

        for pos in positions:
            pos['hold_days'] += 1

            # 当日のOHLCVを取得
            day_data = ohlcv[
                (ohlcv['ticker'] == pos['ticker']) &
                (ohlcv['date'] == current_date)
            ]

            exit_price = None
            exit_reason = None

            if len(day_data) == 0:
                # データがない場合、タイムアウトなら最後の終値で決済
                if pos['hold_days'] >= self.max_hold_days:
                    # 最後に利用可能なデータを取得
                    last_data = ohlcv[ohlcv['ticker'] == pos['ticker']].sort_values('date')
                    if len(last_data) > 0:
                        exit_price = last_data.iloc[-1]['Close']
                        exit_reason = 'timeout'
                    else:
                        # データが全くない場合はエントリー価格で決済
                        exit_price = pos['entry_price']
                        exit_reason = 'timeout'
                else:
                    remaining_positions.append(pos)
                    continue
            else:
                high = day_data.iloc[0]['High']
                low = day_data.iloc[0]['Low']
                close = day_data.iloc[0]['Close']

                # 利確チェック
                if high >= pos['take_profit_price']:
                    exit_price = pos['take_profit_price']
                    exit_reason = 'take_profit'
                # 損切りチェック
                elif low <= pos['stop_loss_price']:
                    exit_price = pos['stop_loss_price']
                    exit_reason = 'stop_loss'
                # タイムアウトチェック
                elif pos['hold_days'] >= self.max_hold_days:
                    exit_price = close
                    exit_reason = 'timeout'

            if exit_price is not None:
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                pnl -= self.position_size * self.commission_rate  # 手数料

                trade = Trade(
                    ticker=pos['ticker'],
                    entry_date=pos['entry_date'],
                    entry_price=pos['entry_price'],
                    exit_date=current_date,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    pnl_pct=pnl / self.position_size * 100,
                    hold_days=pos['hold_days']
                )
                closed_trades.append(trade)
                capital += self.position_size + pnl
            else:
                remaining_positions.append(pos)

        return remaining_positions, closed_trades, capital


if __name__ == "__main__":
    # テスト
    print("BacktestEngine test")
