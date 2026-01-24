"""
パラメータ設定
"""
from datetime import date

# 期間設定
TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)
TEST_END = date.today()

# ラベル設定
LABEL_TARGET_RETURN = 0.10  # +10%
LABEL_HORIZON_DAYS = 20     # 20営業日

# ユニバースフィルタ
MIN_PRICE = 300
MAX_PRICE = 10000
MIN_TURNOVER = 1e8          # 1億円/日
MIN_VOLUME = 100000         # 10万株/日
LIQUIDITY_WINDOW = 20       # 流動性計算の窓（営業日）

# 売買ルール
TAKE_PROFIT = 0.12          # 利確+12%
STOP_LOSS_ATR_MULT = 2.0    # 損切り ATR×2.0
MAX_HOLD_DAYS = 15          # 最大保有期間
TOP_N = 2                   # 上位N銘柄

# 運用設定
CAPITAL = 1_000_000         # 運用資金
