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

    /* ページ背景 */
    .stApp {
        background: #fcfcfc;
    }

    .main > div {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* 背景装飾 */
    .main::before {
        content: '';
        position: fixed;
        top: -50%;
        right: -20%;
        width: 800px;
        height: 800px;
        background: radial-gradient(circle, rgba(0,0,0,0.02) 0%, transparent 70%);
        pointer-events: none;
        z-index: -1;
    }
    .main::after {
        content: '';
        position: fixed;
        bottom: -30%;
        left: -10%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(0,0,0,0.015) 0%, transparent 70%);
        pointer-events: none;
        z-index: -1;
    }

    /* ヘッダー */
    .main-header {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        padding: 1rem 0 2.5rem 0;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid #ebebeb;
    }
    .logo-mark {
        position: relative;
        width: 48px;
        height: 48px;
        flex-shrink: 0;
    }
    .header-text h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #0a0a0a;
        letter-spacing: -0.03em;
    }
    .header-text p {
        margin: 0.375rem 0 0 0;
        color: #888;
        font-size: 0.8rem;
        letter-spacing: 0.02em;
    }

    /* 統計 */
    .stats-container {
        display: flex;
        gap: 3rem;
        margin-bottom: 2.5rem;
    }
    @media (max-width: 768px) {
        .stats-container {
            flex-direction: column;
            gap: 1.25rem;
        }
    }
    .stat-item {
        display: flex;
        flex-direction: column;
    }
    .stat-label {
        color: #999;
        font-size: 0.65rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.375rem;
    }
    .stat-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: #0a0a0a;
        white-space: nowrap;
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
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #eee;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .stock-card:hover {
        border-color: #ddd;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        transform: translateY(-1px);
    }
    .stock-card.top-1 {
        border-left: 3px solid #0a0a0a;
        background: linear-gradient(135deg, #fff 0%, #fafafa 100%);
    }
    .stock-card.top-2 {
        border-left: 3px solid #444;
    }
    .stock-card.top-3 {
        border-left: 3px solid #888;
    }

    .stock-main {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.75rem;
    }
    .stock-info {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        flex: 1;
        min-width: 0;
    }
    .stock-rank {
        background: #171717;
        color: #fff;
        min-width: 24px;
        width: 24px;
        height: 24px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.75rem;
        flex-shrink: 0;
    }
    .stock-rank.top-1 { background: #171717; }
    .stock-rank.top-2 { background: #404040; }
    .stock-rank.top-3 { background: #737373; }

    .stock-text {
        min-width: 0;
        flex: 1;
    }
    .stock-name-main {
        font-size: 0.95rem;
        font-weight: 600;
        color: #171717;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stock-code-sub {
        color: #737373;
        font-size: 0.75rem;
        margin-top: 0.125rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sector {
        color: #a3a3a3;
        font-size: 0.7rem;
    }
    .sector::before {
        content: '·';
        margin-right: 0.25rem;
    }

    /* スコア */
    .score-container {
        text-align: right;
        flex-shrink: 0;
        margin-left: 0.5rem;
    }
    .score-value {
        font-size: 0.875rem;
        font-weight: 600;
        color: #171717;
    }

    /* メタ情報 */
    .stock-meta {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        flex-wrap: wrap;
        margin-top: auto;
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
        cursor: help;
        position: relative;
    }
    .tag[title]:hover::after {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: #171717;
        color: #fff;
        padding: 0.375rem 0.5rem;
        border-radius: 4px;
        font-size: 0.65rem;
        white-space: nowrap;
        z-index: 100;
        margin-bottom: 4px;
    }
    .link {
        color: #737373;
        font-size: 0.7rem;
        text-decoration: none;
        margin-left: auto;
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
        background: transparent;
        border-top: 1px solid #eee;
        border-radius: 0;
        padding: 2rem 0 1rem 0;
        font-size: 0.7rem;
        color: #999;
        margin-top: 3rem;
        line-height: 1.7;
    }
    .disclaimer strong {
        color: #666;
        font-weight: 500;
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

# 業種の英語→日本語マッピング
SECTOR_MAP = {
    'Technology': 'テクノロジー',
    'Consumer Cyclical': '一般消費財',
    'Consumer Defensive': '生活必需品',
    'Financial Services': '金融',
    'Healthcare': 'ヘルスケア',
    'Industrials': '資本財',
    'Energy': 'エネルギー',
    'Basic Materials': '素材',
    'Communication Services': '通信',
    'Real Estate': '不動産',
    'Utilities': '公益',
}

# 理由タグの説明
REASON_HELP = {
    'RSI': 'RSI（相対力指数）が低い = 売られ過ぎの可能性',
    '安値圏': '52週レンジの下位20%',
    '低位置': '52週レンジの下位35%',
    '続落': '連続して下落している',
    '/5d': '直近5日間の下落率',
    'ML判定': 'モデルスコアによる判定',
}


@st.cache_data(ttl=86400)
def fetch_company_name_from_yahoo(code: str) -> str:
    """Yahoo Financeから日本語会社名を取得"""
    try:
        import urllib.request
        url = f"https://finance.yahoo.co.jp/quote/{code}.T"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as res:
            html = res.read().decode('utf-8')
            # タイトルから会社名を抽出
            import re
            match = re.search(r'<title>(.+?)【\d+】', html)
            if match:
                return match.group(1).strip()
    except:
        pass
    return None


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
    code = ticker.replace('.T', '')

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='60d')

        # 日本語名を取得（Yahoo Financeから）
        name = fetch_company_name_from_yahoo(code)
        if not name:
            name = info.get('shortName') or info.get('longName') or code

        # 業種を取得
        sector_en = info.get('sector', '')
        sector = SECTOR_MAP.get(sector_en, sector_en) if sector_en else ''

        if len(hist) < 20:
            return name, None, None, 'データ不足', sector

        open_price = hist.iloc[-1]['Open']
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

        return name, open_price, close_price, ', '.join(reasons[:2]), sector

    except Exception as e:
        name = fetch_company_name_from_yahoo(code) or code
        return name, None, None, '-', ''


def render_skeleton():
    st.markdown("""
    <div class="skeleton-card">
        <div class="skeleton skeleton-line w-40"></div>
        <div class="skeleton skeleton-line w-60"></div>
        <div class="skeleton skeleton-line w-20"></div>
    </div>
    """, unsafe_allow_html=True)


def render_stock_card(rank, code, name, score, open_price, close_price, reason, sector):
    open_str = f"¥{open_price:,.0f}" if open_price else "-"
    close_str = f"¥{close_price:,.0f}" if close_price else "-"
    top_class = f"top-{rank}" if rank <= 3 else ""
    rank_class = f"top-{rank}" if rank <= 3 else ""
    yahoo_url = f"https://finance.yahoo.co.jp/quote/{code}.T"
    display_name = name if name else code
    sector_html = f'<span class="sector">{sector}</span>' if sector else ''

    # 理由のツールチップ用説明を生成
    reason_parts = reason.split(', ')
    reason_tags = ''
    for r in reason_parts:
        tooltip = ''
        for key, desc in REASON_HELP.items():
            if key in r:
                tooltip = desc
                break
        if tooltip:
            reason_tags += f'<span class="tag" title="{tooltip}">{r}</span>'
        else:
            reason_tags += f'<span class="tag">{r}</span>'

    st.markdown(f"""
    <div class="stock-card {top_class}">
        <div class="stock-main">
            <div class="stock-info">
                <div class="stock-rank {rank_class}">{rank}</div>
                <div class="stock-text">
                    <div class="stock-name-main">{display_name}</div>
                    <div class="stock-code-sub">{code}{sector_html}</div>
                </div>
            </div>
            <div class="score-container">
                <div class="score-value">{score:.2f}</div>
            </div>
        </div>
        <div class="stock-meta">
            <div class="meta-item">
                <span class="meta-label">始値</span>
                <span class="meta-value">{open_str}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">終値</span>
                <span class="meta-value">{close_str}</span>
            </div>
            {reason_tags}
            <a href="{yahoo_url}" target="_blank" class="link">詳細 →</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.markdown("""
    <div class="main-header">
        <div class="logo-mark">
            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- フラクタル三角形（シェルピンスキー風） -->
                <polygon points="24,4 44,40 4,40" fill="none" stroke="#0a0a0a" stroke-width="1.5"/>
                <polygon points="24,16 34,34 14,34" fill="none" stroke="#0a0a0a" stroke-width="1"/>
                <polygon points="14,28 19,37 9,37" fill="#0a0a0a"/>
                <polygon points="24,22 29,31 19,31" fill="#666"/>
                <polygon points="34,28 39,37 29,37" fill="#aaa"/>
            </svg>
        </div>
        <div class="header-text">
            <h1>StockSignal</h1>
            <p>短期上昇シグナル検出</p>
        </div>
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

    # 統計
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-label">シグナル日</div>
            <div class="stat-value">{selected_ts.strftime('%Y/%m/%d')} ({weekdays[selected_ts.weekday()]})</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">エントリー</div>
            <div class="stat-value">{entry_date.strftime('%Y/%m/%d')} ({weekdays[entry_date.weekday()]}) 寄付</div>
        </div>
        <div class="stat-item">
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

        name, open_price, close_price, reason, sector = get_stock_info(ticker)
        results.append({
            'rank': i + 1,
            'code': code,
            'name': name[:18] if name and len(name) > 18 else name,
            'score': row['score'],
            'open': open_price,
            'close': close_price,
            'reason': reason,
            'sector': sector
        })

    progress.empty()
    status.empty()

    # 2カラムレイアウト
    col1, col2 = st.columns(2)
    for i, r in enumerate(results):
        with col1 if i % 2 == 0 else col2:
            render_stock_card(r['rank'], r['code'], r['name'], r['score'], r['open'], r['close'], r['reason'], r['sector'])

    st.markdown("""
    <div class="disclaimer">
        <strong>注意</strong> — 本システムは機械学習モデルによる予測であり、将来の株価を保証するものではありません。投資判断は自己責任でお願いします。
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
