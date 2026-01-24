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
    page_title="StockSignal",
    page_icon="📈",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    /* フォント */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* メインコンテナ */
    .main > div {
        padding-top: 1rem;
        max-width: 1200px;
    }

    /* ヘッダー */
    .main-header {
        border-bottom: 1px solid #e5e5e5;
        padding: 0 0 1.5rem 0;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #171717;
        letter-spacing: -0.025em;
    }
    .main-header p {
        margin: 0.25rem 0 0 0;
        color: #737373;
        font-size: 0.875rem;
        font-weight: 400;
    }

    /* 情報カード */
    .info-card {
        background: #fafafa;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .info-card-label {
        color: #737373;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.375rem;
    }
    .info-card-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: #171717;
    }

    /* 銘柄カード */
    .stock-card {
        background: #fff;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border: 1px solid #e5e5e5;
        margin-bottom: 0.5rem;
    }
    .stock-card:hover {
        border-color: #d4d4d4;
    }
    .stock-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.75rem;
    }
    .stock-rank {
        background: #171717;
        color: #fff;
        min-width: 24px;
        height: 24px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .stock-code {
        font-size: 1rem;
        font-weight: 600;
        color: #171717;
    }
    .stock-name {
        color: #737373;
        font-size: 0.75rem;
        margin-top: 0.125rem;
    }
    .stock-score {
        background: #f0fdf4;
        color: #15803d;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .stock-meta {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        align-items: center;
    }
    .stock-price {
        font-weight: 600;
        color: #171717;
        font-size: 0.875rem;
    }
    .stock-price-label {
        color: #a3a3a3;
        font-size: 0.7rem;
        margin-right: 0.25rem;
    }
    .stock-reason {
        background: #fafafa;
        color: #525252;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 500;
        border: 1px solid #e5e5e5;
    }

    /* サイドバー */
    [data-testid="stSidebar"] {
        background: #fafafa;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 0.75rem;
        font-weight: 600;
        color: #525252;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* セクションタイトル */
    .section-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #171717;
        margin-bottom: 1rem;
    }

    /* モバイル対応 */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.25rem;
        }
        .stock-meta {
            gap: 0.75rem;
        }
        .stock-card {
            padding: 0.875rem 1rem;
        }
        .info-card {
            padding: 0.875rem 1rem;
        }
        .info-card-value {
            font-size: 1rem;
        }
    }

    /* プログレスバー */
    .stProgress > div > div {
        background: #171717;
    }

    /* 注意書き */
    .disclaimer {
        background: #fafafa;
        border-radius: 6px;
        padding: 1rem;
        font-size: 0.75rem;
        color: #737373;
        margin-top: 2rem;
        line-height: 1.6;
    }
    .disclaimer strong {
        color: #525252;
    }

    /* ルールカード */
    .rule-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e5e5e5;
        font-size: 0.8rem;
    }
    .rule-item:last-child {
        border-bottom: none;
    }
    .rule-label {
        color: #737373;
    }
    .rule-value {
        font-weight: 600;
        color: #171717;
    }

    /* Streamlitデフォルトの調整 */
    .stSlider label {
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# データディレクトリ
DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


@st.cache_data(ttl=3600)
def load_predictions():
    """予測データを読み込み"""
    pred_path = DATA_DIR / "app_predictions.parquet"
    if not pred_path.exists():
        pred_path = DATA_DIR / "test_predictions.parquet"
    if not pred_path.exists():
        return None
    df = pd.read_parquet(pred_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data(ttl=300)
def get_stock_info(ticker: str):
    """株価情報と買い理由を取得"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        name = info.get('shortName') or info.get('longName') or 'N/A'

        hist = stock.history(period='60d')
        if len(hist) < 20:
            return name, None, None, 'データ不足'

        last_row = hist.iloc[-1]
        open_price = last_row['Open']
        close_price = last_row['Close']

        reasons = []

        # RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        if current_rsi < 30:
            reasons.append(f'RSI {current_rsi:.0f}')
        elif current_rsi < 40:
            reasons.append(f'RSI {current_rsi:.0f}')

        # 52週安値
        low_52w = hist['Low'].min()
        high_52w = hist['High'].max()
        position = (close_price - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50
        if position < 20:
            reasons.append('安値圏')
        elif position < 35:
            reasons.append('低位置')

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

        # 出来高
        vol_ma = hist['Volume'].rolling(20).mean()
        if len(vol_ma) > 0 and vol_ma.iloc[-1] > 0:
            vol_ratio = hist['Volume'].iloc[-1] / vol_ma.iloc[-1]
            if vol_ratio > 2.0:
                reasons.append(f'出来高{vol_ratio:.1f}x')

        # 下落率
        ret_5d = (close_price / hist['Close'].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
        if ret_5d < -10:
            reasons.append(f'{ret_5d:.0f}% / 5d')
        elif ret_5d < -5:
            reasons.append(f'{ret_5d:.0f}% / 5d')

        if not reasons:
            reasons.append('ML Score')

        return name, open_price, close_price, ', '.join(reasons[:2])

    except Exception as e:
        return 'N/A', None, None, '-'


def render_stock_card(rank, code, name, score, price, reason):
    """銘柄カードをレンダリング"""
    price_str = f"¥{price:,.0f}" if price else "-"
    st.markdown(f"""
    <div class="stock-card">
        <div class="stock-header">
            <div style="display: flex; align-items: center; gap: 0.875rem;">
                <div class="stock-rank">{rank}</div>
                <div>
                    <div class="stock-code">{code}</div>
                    <div class="stock-name">{name}</div>
                </div>
            </div>
            <div class="stock-score">{score:.2f}</div>
        </div>
        <div class="stock-meta">
            <div>
                <span class="stock-price-label">終値</span>
                <span class="stock-price">{price_str}</span>
            </div>
            <div class="stock-reason">{reason}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>StockSignal</h1>
        <p>短期上昇シグナル検出</p>
    </div>
    """, unsafe_allow_html=True)

    # データ読み込み
    predictions = load_predictions()

    if predictions is None:
        st.error("予測データが見つかりません。")
        st.stop()

    available_dates = sorted(predictions['date'].unique())

    # サイドバー
    with st.sidebar:
        st.markdown("### 設定")

        min_date = pd.Timestamp(available_dates[0]).date()
        max_date = pd.Timestamp(available_dates[-1]).date()

        selected_date = st.date_input(
            "分析日",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

        top_n = st.slider("表示件数", 5, 30, 10)

        st.markdown("---")
        st.markdown("### 売買ルール")
        st.markdown("""
        <div class="rule-item">
            <span class="rule-label">利確</span>
            <span class="rule-value">+12%</span>
        </div>
        <div class="rule-item">
            <span class="rule-label">損切り</span>
            <span class="rule-value">ATR × 2.0</span>
        </div>
        <div class="rule-item">
            <span class="rule-label">最大保有</span>
            <span class="rule-value">15日</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("更新方法"):
            st.markdown("""
            **データ更新**
            ```
            python scripts/phase1_data_check.py
            python scripts/phase2_train.py
            python scripts/phase3_backtest.py
            ```

            **デプロイ更新**
            GitHubへpushで自動反映
            """)

    # メイン
    selected_ts = pd.Timestamp(selected_date)

    if selected_ts not in [pd.Timestamp(d) for d in available_dates]:
        closest_date = min(available_dates, key=lambda x: abs(pd.Timestamp(x) - selected_ts))
        st.warning(f"{closest_date.strftime('%Y-%m-%d')} を表示")
        selected_ts = pd.Timestamp(closest_date)

    day_predictions = predictions[predictions['date'] == selected_ts].copy()
    day_predictions = day_predictions.sort_values('rank')

    entry_date = selected_ts + pd.Timedelta(days=1)
    while entry_date.weekday() >= 5:
        entry_date += pd.Timedelta(days=1)

    weekdays = ['月', '火', '水', '木', '金', '土', '日']

    # 情報カード
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-label">シグナル日</div>
            <div class="info-card-value">{selected_ts.strftime('%Y/%m/%d')} ({weekdays[selected_ts.weekday()]})</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-label">エントリー日</div>
            <div class="info-card-value">{entry_date.strftime('%Y/%m/%d')} ({weekdays[entry_date.weekday()]}) 寄付</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-label">候補銘柄</div>
            <div class="info-card-value">{len(day_predictions)} 銘柄</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 結果
    if len(day_predictions) == 0:
        st.warning("この日の予測データがありません。")
    else:
        st.markdown('<div class="section-title">上昇予測ランキング</div>', unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        for i, (_, row) in enumerate(day_predictions.head(top_n).iterrows()):
            ticker = row['ticker']
            code = ticker.replace('.T', '')

            status_text.text(f"取得中: {code}")
            progress_bar.progress((i + 1) / top_n)

            name, _, close_price, reason = get_stock_info(ticker)

            results.append({
                'rank': i + 1,
                'code': code,
                'name': name[:20] if len(name) > 20 else name,
                'score': row['score'],
                'price': close_price,
                'reason': reason
            })

        progress_bar.empty()
        status_text.empty()

        col1, col2 = st.columns(2)
        for i, result in enumerate(results):
            with col1 if i % 2 == 0 else col2:
                render_stock_card(
                    result['rank'],
                    result['code'],
                    result['name'],
                    result['score'],
                    result['price'],
                    result['reason']
                )

        st.markdown("""
        <div class="disclaimer">
            <strong>注意事項</strong><br>
            本システムは過去データに基づく機械学習モデルの予測です。将来の株価を保証するものではありません。
            投資判断は自己責任でお願いします。
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
