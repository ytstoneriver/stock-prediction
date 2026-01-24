"""
StockSignal - 短期上昇シグナル検出
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

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
    }

    /* 統計カード */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    @media (max-width: 768px) {
        .stats-container {
            grid-template-columns: 1fr;
        }
    }
    .stat-card {
        background: #fafafa;
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }
    .stat-label {
        color: #737373;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: #171717;
    }

    /* セクション */
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #171717;
    }
    .section-subtitle {
        font-size: 0.75rem;
        color: #737373;
    }

    /* 銘柄カード */
    .stock-card {
        background: #fff;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border: 1px solid #e5e5e5;
        margin-bottom: 0.5rem;
        transition: border-color 0.15s;
    }
    .stock-card:hover {
        border-color: #d4d4d4;
    }
    .stock-card.top-1 {
        border-left: 3px solid #171717;
    }
    .stock-card.top-2 {
        border-left: 3px solid #525252;
    }
    .stock-card.top-3 {
        border-left: 3px solid #a3a3a3;
    }

    .stock-main {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.75rem;
    }
    .stock-info {
        display: flex;
        align-items: center;
        gap: 0.75rem;
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
    .stock-rank.top-1 { background: #171717; }
    .stock-rank.top-2 { background: #404040; }
    .stock-rank.top-3 { background: #737373; }

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

    /* スコア */
    .score-container {
        text-align: right;
    }
    .score-value {
        font-size: 0.875rem;
        font-weight: 600;
        color: #171717;
        margin-bottom: 0.25rem;
    }
    .score-bar {
        width: 60px;
        height: 4px;
        background: #e5e5e5;
        border-radius: 2px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        background: #171717;
        border-radius: 2px;
    }

    /* メタ情報 */
    .stock-meta {
        display: flex;
        gap: 1.5rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .meta-item {
        display: flex;
        align-items: center;
        gap: 0.375rem;
    }
    .meta-label {
        color: #a3a3a3;
        font-size: 0.7rem;
    }
    .meta-value {
        font-weight: 500;
        color: #171717;
        font-size: 0.8rem;
    }
    .tag {
        background: #fafafa;
        color: #525252;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 500;
        border: 1px solid #e5e5e5;
    }
    .link {
        color: #737373;
        font-size: 0.7rem;
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    .link:hover {
        color: #171717;
    }

    /* スケルトン */
    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 4px;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .skeleton-card {
        background: #fff;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border: 1px solid #e5e5e5;
        margin-bottom: 0.5rem;
    }
    .skeleton-line {
        height: 12px;
        margin-bottom: 0.5rem;
    }
    .skeleton-line.w-20 { width: 20%; }
    .skeleton-line.w-40 { width: 40%; }
    .skeleton-line.w-60 { width: 60%; }

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

    .rule-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e5e5e5;
        font-size: 0.8rem;
    }
    .rule-item:last-child { border-bottom: none; }
    .rule-label { color: #737373; }
    .rule-value { font-weight: 600; color: #171717; }

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

    /* プログレスバー */
    .stProgress > div > div {
        background: #171717;
    }

    /* モバイル */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.25rem; }
        .stock-meta { gap: 0.75rem; }
        .stock-card { padding: 0.875rem 1rem; }
        .score-bar { width: 48px; }
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(ttl=3600)
def load_predictions():
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
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('shortName') or info.get('longName') or 'N/A'

        hist = stock.history(period='60d')
        if len(hist) < 20:
            return name, None, None, 'データ不足'

        close_price = hist.iloc[-1]['Close']
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

        # 位置
        low_52w = hist['Low'].min()
        high_52w = hist['High'].max()
        position = (close_price - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50
        if position < 20:
            reasons.append('安値圏')
        elif position < 35:
            reasons.append('低位置')

        # 連続下落
        returns = hist['Close'].pct_change()
        consecutive_down = sum(1 for r in returns.iloc[-5:] if r < 0)
        if consecutive_down >= 3:
            reasons.append(f'{consecutive_down}日続落')

        # 下落率
        ret_5d = (close_price / hist['Close'].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
        if ret_5d < -5:
            reasons.append(f'{ret_5d:.0f}%/5d')

        if not reasons:
            reasons.append('ML判定')

        return name, close_price, ', '.join(reasons[:2]), None

    except Exception as e:
        return 'N/A', None, '-', str(e)


def render_skeleton():
    st.markdown("""
    <div class="skeleton-card">
        <div class="skeleton skeleton-line w-40"></div>
        <div class="skeleton skeleton-line w-60"></div>
        <div class="skeleton skeleton-line w-20"></div>
    </div>
    """, unsafe_allow_html=True)


def render_stock_card(rank, code, name, score, price, reason):
    price_str = f"¥{price:,.0f}" if price else "-"
    score_pct = min(score * 100, 100)
    top_class = f"top-{rank}" if rank <= 3 else ""
    rank_class = f"top-{rank}" if rank <= 3 else ""
    yahoo_url = f"https://finance.yahoo.co.jp/quote/{code}.T"

    st.markdown(f"""
    <div class="stock-card {top_class}">
        <div class="stock-main">
            <div class="stock-info">
                <div class="stock-rank {rank_class}">{rank}</div>
                <div>
                    <div class="stock-code">{code}</div>
                    <div class="stock-name">{name}</div>
                </div>
            </div>
            <div class="score-container">
                <div class="score-value">{score:.2f}</div>
                <div class="score-bar">
                    <div class="score-fill" style="width: {score_pct}%"></div>
                </div>
            </div>
        </div>
        <div class="stock-meta">
            <div class="meta-item">
                <span class="meta-label">終値</span>
                <span class="meta-value">{price_str}</span>
            </div>
            <div class="tag">{reason}</div>
            <a href="{yahoo_url}" target="_blank" class="link">
                Yahoo Finance →
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.markdown("""
    <div class="main-header">
        <h1>StockSignal</h1>
        <p>短期上昇シグナル検出</p>
    </div>
    """, unsafe_allow_html=True)

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
        <div class="rule-item"><span class="rule-label">利確</span><span class="rule-value">+12%</span></div>
        <div class="rule-item"><span class="rule-label">損切り</span><span class="rule-value">ATR × 2.0</span></div>
        <div class="rule-item"><span class="rule-label">最大保有</span><span class="rule-value">15日</span></div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("更新方法"):
            st.code("python scripts/phase1_data_check.py\npython scripts/phase2_train.py\npython scripts/phase3_backtest.py", language="bash")
            st.caption("GitHubへpushで自動デプロイ")

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

    # 統計カード
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-label">シグナル日</div>
            <div class="stat-value">{selected_ts.strftime('%Y/%m/%d')} ({weekdays[selected_ts.weekday()]})</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">エントリー</div>
            <div class="stat-value">{entry_date.strftime('%Y/%m/%d')} ({weekdays[entry_date.weekday()]}) 寄付</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">検出銘柄</div>
            <div class="stat-value">{len(day_predictions)} 銘柄</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(day_predictions) == 0:
        st.info("この日の予測データがありません。")
        return

    # セクションヘッダー
    st.markdown(f"""
    <div class="section-header">
        <span class="section-title">上昇予測ランキング</span>
        <span class="section-subtitle">スコア順 Top {top_n}</span>
    </div>
    """, unsafe_allow_html=True)

    # ローディング
    progress = st.progress(0)
    status = st.empty()

    results = []
    for i, (_, row) in enumerate(day_predictions.head(top_n).iterrows()):
        ticker = row['ticker']
        code = ticker.replace('.T', '')
        status.text(f"取得中: {code}")
        progress.progress((i + 1) / top_n)

        name, close_price, reason, _ = get_stock_info(ticker)
        results.append({
            'rank': i + 1,
            'code': code,
            'name': name[:18] if name and len(name) > 18 else name,
            'score': row['score'],
            'price': close_price,
            'reason': reason
        })

    progress.empty()
    status.empty()

    # 2カラムレイアウト
    col1, col2 = st.columns(2)
    for i, r in enumerate(results):
        with col1 if i % 2 == 0 else col2:
            render_stock_card(r['rank'], r['code'], r['name'], r['score'], r['price'], r['reason'])

    st.markdown("""
    <div class="disclaimer">
        <strong>注意</strong> — 本システムは機械学習モデルによる予測であり、将来の株価を保証するものではありません。投資判断は自己責任でお願いします。
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
