# Claude Code プロジェクトコンテキスト

このファイルはClaudeがセッション開始時に読み込む。プロジェクトの概要と重要なルールを記載。

## プロジェクト概要

**日本株短期上昇予測システム** - LightGBMで「20営業日以内に+10%上昇する銘柄」を予測

### 現在の設定（v8: 最良）
- 勝率: 66.7%
- 総リターン: +57.70%
- PF: 2.13

### 実行方法
```bash
# 仮想環境を使用
.venv/bin/python scripts/phase2_train.py   # 学習
.venv/bin/python scripts/phase3_backtest.py  # バックテスト
```

---

## 重要: やってはいけないこと

過去の実験で失敗が確認された変更。**絶対にやらないこと**:

1. **ボラティリティに上限を設ける** → タイムアウト増加
2. **ボラティリティに下限を設ける** → 損切り増加
3. **スコア閾値を上げる（0.55→0.60）** → 取引減少、精度向上せず
4. **スコア閾値を下げる（0.55→0.50）** → 質の悪いトレードを拾い損切り増加
5. **フィルタを厳しくしすぎる** → サンプル不足
6. **複数の変更を同時に行う** → 原因特定困難
7. **特徴量を増やしすぎる（47個以上）** → 過学習のリスク
8. **同時保有銘柄数を増やす（1→2以上）** → ランク2位以下は精度が低い
9. **RSI > 70 で除外する** → 取引機会減少、総損益悪化
10. **連続シグナルを除外する** → 勝率・PFは改善するが取引半減で総損益悪化
11. **Permutation Importanceで特徴量選択** → サンプリング依存で不安定、Gain-basedを使うこと

---

## 重要: やって良かったこと

1. ラベル定義を実トレードルールと一致させる（翌日始値→利確/損切り先着）
2. 緩やかなファンダメンタルフィルタ（ROE > 0, 時価総額 > 100億円）
3. 需給代替指標とセクターダミー変数の追加
4. ローリング学習（直近6ヶ月で毎月再学習）
5. **特徴量選択（Top 20）で過学習防止** ← 最重要

---

## 現在の設定値（config.py）

```python
LABEL_TAKE_PROFIT = 0.10     # 利確: +10%
LABEL_STOP_LOSS = 0.10       # 損切り: -10%
LABEL_HORIZON_DAYS = 20      # 最大保持期間: 20営業日
SCORE_THRESHOLD = 0.55       # スコア閾値
MIN_SCORE_GAP = 0.02         # 最小スコア差
TOP_N_STOCKS = 1             # 上位1銘柄のみ
TOP_N_FEATURES = 20          # 特徴量選択: Top 20
FUNDAMENTAL_FILTER = {
    'min_roe': 0.0,          # ROE > 0%
    'min_market_cap': 100e9, # 時価総額 > 100億円
}
```

---

## ファイル構成

```
config.py                    # 本番設定
app.py                       # Streamlit UI
scripts/
  phase1_data_check.py       # データ取得
  phase2_train.py            # 学習
  phase3_backtest.py         # バックテスト
  run_full_test.py           # フルテスト
src/
  backtest/engine.py         # バックテストエンジン
  data/
    loader.py                # yfinanceデータ取得
    universe.py              # 銘柄リスト
    universe_filter.py       # ファンダフィルタ
  features/builder.py        # 特徴量生成（47個→Top 20選択）
  model/
    labels.py                # ラベル生成（利確/損切り先着）
    trainer.py               # LightGBM学習（ローリング学習）
docs/
  SPECIFICATION.md           # 詳細仕様書
```

---

## 詳細ドキュメント

- `docs/SPECIFICATION.md` - 詳細な仕様、実験履歴、教訓
