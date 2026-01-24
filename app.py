"""
株価上昇予測システム - Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
import yfinance as yf

# ページ設定
st.set_page_config(
    page_title="株価上昇予測",
    page_icon="📈",
    layout="centered"
)

# タイトル
st.title("📈 株価上昇予測システム")
st.caption("LightGBMモデルによる短期上昇候補の検出")

# データディレクトリ
DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


@st.cache_data(ttl=3600)
def load_predictions():
    """予測データを読み込み"""
    # 軽量版を優先
    pred_path = DATA_DIR / "app_predictions.parquet"
    if not pred_path.exists():
        pred_path = DATA_DIR / "test_predictions.parquet"
    if not pred_path.exists():
        return None
    return pd.read_parquet(pred_path)


@st.cache_data(ttl=3600)
def load_labeled_data():
    """ラベル付きデータを読み込み"""
    data_path = DATA_DIR / "labeled_data.parquet"
    if not data_path.exists():
        return None
    return pd.read_parquet(data_path)


@st.cache_data(ttl=300)
def get_stock_info(ticker: str):
    """株価情報と買い理由を取得"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName') or info.get('shortName') or 'N/A'

        # 直近60日のデータを取得
        hist = stock.history(period='60d')
        if len(hist) < 20:
            return name, None, None, 'データ不足'

        last_row = hist.iloc[-1]
        open_price = last_row['Open']
        close_price = last_row['Close']

        # 買い理由を分析
        reasons = []

        # RSI計算
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        if current_rsi < 30:
            reasons.append(f'RSI={current_rsi:.0f}(売られ過ぎ)')
        elif current_rsi < 40:
            reasons.append(f'RSI={current_rsi:.0f}(低め)')

        # 52週安値からの位置
        low_52w = hist['Low'].min()
        high_52w = hist['High'].max()
        current = close_price
        position = (current - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50
        if position < 20:
            reasons.append('52週安値圏')
        elif position < 35:
            reasons.append('安値圏')

        # 連続下落
        returns = hist['Close'].pct_change()
        consecutive_down = 0
        for r in returns.iloc[-5:]:
            if r < 0:
                consecutive_down += 1
            else:
                consecutive_down = 0
        if consecutive_down >= 3:
            reasons.append(f'{consecutive_down}日続落')

        # 出来高急増
        vol_ma = hist['Volume'].rolling(20).mean()
        if len(vol_ma) > 0 and vol_ma.iloc[-1] > 0:
            vol_ratio = hist['Volume'].iloc[-1] / vol_ma.iloc[-1]
            if vol_ratio > 2.0:
                reasons.append(f'出来高{vol_ratio:.1f}倍')
            elif vol_ratio > 1.5:
                reasons.append('出来高増')

        # ボリンジャーバンド
        ma20 = hist['Close'].rolling(20).mean()
        std20 = hist['Close'].rolling(20).std()
        bb_lower = ma20 - 2 * std20
        if close_price < bb_lower.iloc[-1]:
            reasons.append('BB下限割れ')

        # 直近の下落率
        ret_5d = (close_price / hist['Close'].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
        if ret_5d < -10:
            reasons.append(f'5日で{ret_5d:.0f}%')
        elif ret_5d < -5:
            reasons.append(f'5日で{ret_5d:.0f}%')

        if not reasons:
            reasons.append('MLスコア高')

        reason_str = ', '.join(reasons[:2])
        return name, open_price, close_price, reason_str

    except Exception as e:
        return 'N/A', None, None, str(e)[:20]


def main():
    # データ読み込み
    predictions = load_predictions()

    if predictions is None:
        st.error("予測データが見つかりません。先にモデルを実行してください。")
        st.stop()

    # 利用可能な日付を取得
    available_dates = sorted(predictions['date'].unique())

    # サイドバー
    st.sidebar.header("設定")

    # 日付選択
    min_date = pd.Timestamp(available_dates[0]).date()
    max_date = pd.Timestamp(available_dates[-1]).date()

    selected_date = st.sidebar.date_input(
        "分析日（シグナル日）",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    # 表示件数
    top_n = st.sidebar.slider("表示件数", 5, 30, 20)

    # パラメータ表示
    st.sidebar.markdown("---")
    st.sidebar.subheader("売買ルール")
    st.sidebar.markdown("""
    - 利確: **+12%**
    - 損切り: **ATR×2.0**
    - 最大保有: **15日**
    """)

    # 分析実行
    st.markdown("---")

    # 選択された日付の予測を取得
    selected_ts = pd.Timestamp(selected_date)

    if selected_ts not in [pd.Timestamp(d) for d in available_dates]:
        # 最も近い日付を探す
        closest_date = min(available_dates, key=lambda x: abs(pd.Timestamp(x) - selected_ts))
        st.warning(f"選択された日付のデータがありません。最も近い日付 {closest_date.strftime('%Y-%m-%d')} を使用します。")
        selected_ts = pd.Timestamp(closest_date)

    day_predictions = predictions[predictions['date'] == selected_ts].copy()
    day_predictions = day_predictions.sort_values('rank')

    # エントリー日を計算（翌営業日）
    entry_date = selected_ts + pd.Timedelta(days=1)
    # 土日をスキップ
    while entry_date.weekday() >= 5:
        entry_date += pd.Timedelta(days=1)

    st.subheader(f"📊 分析結果")
    st.markdown(f"**シグナル日**: {selected_ts.strftime('%Y-%m-%d')}（{['月','火','水','木','金','土','日'][selected_ts.weekday()]}）")
    st.markdown(f"**エントリー**: {entry_date.strftime('%Y-%m-%d')}（{['月','火','水','木','金','土','日'][entry_date.weekday()]}）寄付き")

    st.markdown("---")

    # 結果表示
    if len(day_predictions) == 0:
        st.warning("この日の予測データがありません。")
    else:
        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        for i, (_, row) in enumerate(day_predictions.head(top_n).iterrows()):
            ticker = row['ticker']
            code = ticker.replace('.T', '')

            status_text.text(f"データ取得中... {code}")
            progress_bar.progress((i + 1) / top_n)

            name, open_price, close_price, reason = get_stock_info(ticker)

            results.append({
                '順位': i + 1,
                'コード': code,
                '会社名': name[:20] if len(name) > 20 else name,
                'スコア': f"{row['score']:.4f}",
                '終値': f"¥{close_price:,.0f}" if close_price else 'N/A',
                '買い理由': reason
            })

        progress_bar.empty()
        status_text.empty()

        # テーブル表示
        df_results = pd.DataFrame(results)
        st.dataframe(
            df_results,
            hide_index=True,
            use_container_width=True
        )

        # 注意書き
        st.markdown("---")
        st.caption("""
        **注意事項**
        - このシステムは過去データに基づく機械学習モデルの予測であり、将来の株価を保証するものではありません
        - 投資判断は自己責任でお願いします
        - 終値は直近取引日のデータです
        """)

        # 買い理由の凡例
        with st.expander("買い理由の説明"):
            st.markdown("""
            | 理由 | 説明 |
            |------|------|
            | RSI=XX(売られ過ぎ) | RSI30未満で売られ過ぎ水準 |
            | RSI=XX(低め) | RSI40未満で低め |
            | 52週安値圏 | 52週レンジの下位20%以内 |
            | 安値圏 | 52週レンジの下位35%以内 |
            | X日続落 | 連続下落日数 |
            | 5日で-X% | 直近5日間の下落率 |
            | 出来高X倍 | 20日平均比の出来高 |
            | BB下限割れ | ボリンジャーバンド下限を下回る |
            | MLスコア高 | 機械学習モデルのスコアが主因 |
            """)


if __name__ == "__main__":
    main()
