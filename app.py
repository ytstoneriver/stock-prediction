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

    /* 価格 */
    .stock-prices {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 0.875rem;
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

    /* フッター */
    .stock-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: auto;
    }
    .stock-tags {
        display: flex;
        gap: 0.375rem;
        flex-wrap: wrap;
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

    /* 銘柄グリッド */
    .stock-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
    }
    .stock-grid .stock-card {
        margin-bottom: 0;
    }

    /* 高確度シグナルカード */
    .high-conf-card {
        border-left: 3px solid #16a34a;
        background: linear-gradient(135deg, #f0fdf4 0%, #fff 100%);
    }
    .high-conf-card:hover {
        border-color: #15803d;
    }
    .tag.high-conf {
        background: #dcfce7;
        color: #166534;
        border-color: #bbf7d0;
    }
    .high-conf-section {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e5e5e5;
    }
    .high-conf-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .high-conf-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #166534;
    }
    .high-conf-badge {
        background: #16a34a;
        color: #fff;
        padding: 0.125rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.65rem;
        font-weight: 600;
    }
    .high-conf-description {
        font-size: 0.75rem;
        color: #737373;
        margin-bottom: 1rem;
    }

    /* モバイル */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.25rem; }
        .stock-prices { gap: 1rem; }
        .stock-card { padding: 1rem; }
        .stock-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"

# 高確度セクター（勝率90%以上）
HIGH_CONFIDENCE_SECTORS = ['Financial Services', 'Basic Materials']

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


def _get_file_mtime(path):
    """ファイルの更新日時を取得（キャッシュキー用）"""
    try:
        return path.stat().st_mtime if path.exists() else 0
    except:
        return 0

@st.cache_data(ttl=300)
def load_sector_mapping(_mtime=None):
    """セクターマッピングを読み込み（ファイル更新時にキャッシュ無効化）"""
    sector_path = DATA_DIR / "sector_mapping.parquet"
    if sector_path.exists():
        return pd.read_parquet(sector_path)
    return None


@st.cache_data(ttl=60)
def load_predictions(_mtime=None):
    """予測データを読み込み（ファイル更新時にキャッシュ無効化）"""
    # 優先順位: predictions.parquet > app_predictions.parquet
    for filename in ["predictions.parquet", "app_predictions.parquet"]:
        pred_path = DATA_DIR / filename
        if pred_path.exists():
            df = pd.read_parquet(pred_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
    return None


def get_high_confidence_signals(predictions, sector_mapping, days=30):
    """直近N日の高確度シグナルを取得

    条件:
    1. Financial Services または Basic Materials セクター（勝率94%）
    2. 連続2回目のシグナル（勝率62%）
    3. スコア0.80以上（勝率58%）

    除外条件:
    - 連続3回目以降のシグナル（過学習傾向）
    - 直近5回中3回以上損切りの銘柄（実績不良）
    """
    if predictions is None or sector_mapping is None:
        return pd.DataFrame()

    # 直近の実績が悪い銘柄を特定（直近5回中3回以上損切り）
    rank1_all = predictions[predictions['rank'] == 1].copy()
    bad_tickers = set()
    for ticker in rank1_all['ticker'].unique():
        ticker_data = rank1_all[rank1_all['ticker'] == ticker].sort_values('date', ascending=False).head(5)
        if len(ticker_data) >= 3:
            # exit_reasonが0.0または'stop_loss'の場合は損切り
            stop_loss_count = sum(
                1 for _, row in ticker_data.iterrows()
                if row.get('exit_reason') in [0.0, 'stop_loss']
            )
            if stop_loss_count >= 3:
                bad_tickers.add(ticker)

    # 直近N日に絞る
    max_date = predictions['date'].max()
    min_date = max_date - pd.Timedelta(days=days)
    recent = predictions[predictions['date'] >= min_date].copy()

    # rank1のみ
    recent = recent[recent['rank'] == 1]

    # 実績不良銘柄を除外
    recent = recent[~recent['ticker'].isin(bad_tickers)]

    if recent.empty:
        return pd.DataFrame()

    # セクター情報を付加
    recent = recent.merge(sector_mapping, on='ticker', how='left')

    # 連続シグナルを検出
    recent = recent.sort_values(['ticker', 'date'])
    recent['prev_ticker'] = recent['ticker'].shift(1)
    recent['prev_date'] = recent['date'].shift(1)
    recent['is_consecutive'] = (
        (recent['ticker'] == recent['prev_ticker']) &
        ((recent['date'] - recent['prev_date']).dt.days <= 3)  # 土日考慮で3日以内
    )

    # 連続回数をカウント
    recent['consecutive_count'] = 0
    current_ticker = None
    count = 0
    for idx in recent.index:
        if recent.loc[idx, 'ticker'] != current_ticker:
            current_ticker = recent.loc[idx, 'ticker']
            count = 1
        else:
            count += 1
        recent.loc[idx, 'consecutive_count'] = count

    # 高確度条件を判定
    results = []
    for _, row in recent.iterrows():
        # 連続3回目以降は除外（過学習傾向）
        if row['consecutive_count'] >= 3:
            continue

        reasons = []
        confidence_score = 0

        # 条件1: 高確度セクター
        if row.get('sector') in HIGH_CONFIDENCE_SECTORS:
            sector_ja = SECTOR_MAP.get(row['sector'], row['sector'])
            reasons.append(f'{sector_ja}セクター（勝率94%）')
            confidence_score += 3

        # 条件2: 連続2回目シグナル
        if row['consecutive_count'] == 2:
            reasons.append('連続2回目シグナル（勝率62%）')
            confidence_score += 2

        # 条件3: 高スコア
        if row['score'] >= 0.80:
            reasons.append(f'高スコア {row["score"]:.2f}（勝率58%）')
            confidence_score += 1

        if reasons:
            results.append({
                'date': row['date'],
                'ticker': row['ticker'],
                'score': row['score'],
                'sector': row.get('sector', ''),
                'reasons': reasons,
                'confidence_score': confidence_score
            })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    # 信頼度スコア→日付の降順でソート
    result_df = result_df.sort_values(['confidence_score', 'date'], ascending=[False, False])
    # 同じ銘柄は最も信頼度の高い1件だけを残す
    result_df = result_df.drop_duplicates(subset='ticker', keep='first')
    return result_df


@st.cache_data(ttl=300)
def get_stock_info(ticker: str, signal_date: str = None):
    code = ticker.replace('.T', '')

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # 2年分のデータを取得（過去のsignal_dateに対応するため）
        hist = stock.history(period='2y')

        # 日本語名を取得（Yahoo Financeから）
        name = fetch_company_name_from_yahoo(code)
        if not name:
            name = info.get('shortName') or info.get('longName') or code

        # 業種を取得
        sector_en = info.get('sector', '')
        sector = SECTOR_MAP.get(sector_en, sector_en) if sector_en else ''

        # データが空の場合
        if hist.empty:
            return name, None, None, 'データ取得失敗', sector

        # タイムゾーン除去（tz_localizeはtzがない場合のみ、ある場合はtz_convert使用）
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)

        if len(hist) < 20:
            return name, None, None, 'データ不足', sector

        # シグナル日時点のデータに絞る
        if signal_date:
            target_date = pd.Timestamp(signal_date)
            # シグナル日以前のデータのみ使用
            hist_filtered = hist[hist.index <= target_date]
            # フィルタリング後も十分なデータがある場合のみ使用
            if len(hist_filtered) >= 20:
                hist = hist_filtered

        if len(hist) < 6:
            return name, None, None, 'データ不足', sector

        open_price = hist.iloc[-1]['Open']
        close_price = hist.iloc[-1]['Close']

        reasons = []

        # RSI（シグナル日時点）
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        if not pd.isna(current_rsi):
            if current_rsi < 30:
                reasons.append(f'RSI {current_rsi:.0f}')
            elif current_rsi < 40:
                reasons.append(f'RSI {current_rsi:.0f}')

        # 位置（シグナル日時点）
        low_52w = hist['Low'].min()
        high_52w = hist['High'].max()
        position = (close_price - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50
        if position < 20:
            reasons.append('安値圏')
        elif position < 35:
            reasons.append('低位置')

        # 連続下落（シグナル日時点）
        returns = hist['Close'].pct_change()
        consecutive_down = sum(1 for r in returns.iloc[-5:] if r < 0)
        if consecutive_down >= 3:
            reasons.append(f'{consecutive_down}日続落')

        # 下落率（シグナル日時点）
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


def get_high_confidence_card_html(code, name, score, sector, reasons, signal_date, entry_date):
    """高確度シグナル用のカードHTML"""
    import html
    yahoo_url = f"https://finance.yahoo.co.jp/quote/{code}.T"
    display_name = html.escape(name) if name else code
    sector_ja = SECTOR_MAP.get(sector, sector) if sector else ''
    sector_html = f'<span class="sector">{html.escape(sector_ja)}</span>' if sector_ja else ''

    # 理由タグを生成
    reason_tags = ''
    for r in reasons:
        r_escaped = html.escape(r)
        reason_tags += f'<span class="tag high-conf">{r_escaped}</span>'

    return f'''<div class="stock-card high-conf-card">
        <div class="stock-main">
            <div class="stock-info">
                <div class="stock-text">
                    <div class="stock-name-main">{display_name}</div>
                    <div class="stock-code-sub">{code}{sector_html}</div>
                </div>
            </div>
            <div class="score-container">
                <div class="score-value">{score:.2f}</div>
            </div>
        </div>
        <div class="stock-prices">
            <div class="meta-item">
                <span class="meta-label">シグナル日</span>
                <span class="meta-value">{signal_date}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">エントリー</span>
                <span class="meta-value">{entry_date}</span>
            </div>
        </div>
        <div class="stock-footer">
            <div class="stock-tags">{reason_tags}</div>
            <a href="{yahoo_url}" target="_blank" class="link">詳細 →</a>
        </div>
    </div>'''


def get_stock_card_html(rank, code, name, score, open_price, close_price, reason, sector):
    import html
    open_str = f"¥{open_price:,.0f}" if open_price else "-"
    close_str = f"¥{close_price:,.0f}" if close_price else "-"
    top_class = f"top-{rank}" if rank <= 3 else ""
    rank_class = f"top-{rank}" if rank <= 3 else ""
    yahoo_url = f"https://finance.yahoo.co.jp/quote/{code}.T"
    display_name = html.escape(name) if name else code
    sector_escaped = html.escape(sector) if sector else ''
    sector_html = f'<span class="sector">{sector_escaped}</span>' if sector else ''

    # 理由のツールチップ用説明を生成
    reason_parts = reason.split(', ')
    reason_tags = ''
    for r in reason_parts:
        r_escaped = html.escape(r)
        tooltip = ''
        for key, desc in REASON_HELP.items():
            if key in r:
                tooltip = html.escape(desc)
                break
        if tooltip:
            reason_tags += f'<span class="tag" title="{tooltip}">{r_escaped}</span>'
        else:
            reason_tags += f'<span class="tag">{r_escaped}</span>'

    return f'<div class="stock-card {top_class}"><div class="stock-main"><div class="stock-info"><div class="stock-rank {rank_class}">{rank}</div><div class="stock-text"><div class="stock-name-main">{display_name}</div><div class="stock-code-sub">{code}{sector_html}</div></div></div><div class="score-container"><div class="score-value">{score:.2f}</div></div></div><div class="stock-prices"><div class="meta-item"><span class="meta-label">始値</span><span class="meta-value">{open_str}</span></div><div class="meta-item"><span class="meta-label">終値</span><span class="meta-value">{close_str}</span></div></div><div class="stock-footer"><div class="stock-tags">{reason_tags}</div><a href="{yahoo_url}" target="_blank" class="link">詳細 →</a></div></div>'


def main():
    st.markdown("""
    <div class="main-header">
        <div class="logo-mark">
            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- 招き猫（全身） -->
                <!-- 体 -->
                <ellipse cx="24" cy="36" rx="10" ry="8" fill="#fef9f0" stroke="#e0d5c5" stroke-width="0.8"/>
                <!-- 足（左） -->
                <ellipse cx="17" cy="44" rx="4" ry="2.5" fill="#fef9f0" stroke="#e0d5c5" stroke-width="0.8"/>
                <line x1="15" y1="44" x2="15" y2="45.5" stroke="#e0d5c5" stroke-width="0.6"/>
                <line x1="17" y1="44" x2="17" y2="46" stroke="#e0d5c5" stroke-width="0.6"/>
                <line x1="19" y1="44" x2="19" y2="45.5" stroke="#e0d5c5" stroke-width="0.6"/>
                <!-- 足（右） -->
                <ellipse cx="31" cy="44" rx="4" ry="2.5" fill="#fef9f0" stroke="#e0d5c5" stroke-width="0.8"/>
                <line x1="29" y1="44" x2="29" y2="45.5" stroke="#e0d5c5" stroke-width="0.6"/>
                <line x1="31" y1="44" x2="31" y2="46" stroke="#e0d5c5" stroke-width="0.6"/>
                <line x1="33" y1="44" x2="33" y2="45.5" stroke="#e0d5c5" stroke-width="0.6"/>
                <!-- 左手（下げてる） -->
                <path d="M12,30 Q10,34 12,38 Q14,40 14,38 L14,32 Q14,30 12,30" fill="#fef9f0" stroke="#e0d5c5" stroke-width="0.8"/>
                <ellipse cx="12" cy="39" rx="3" ry="2" fill="#fef9f0" stroke="#e0d5c5" stroke-width="0.8"/>
                <line x1="10" y1="39" x2="10" y2="40.5" stroke="#e0d5c5" stroke-width="0.5"/>
                <line x1="12" y1="39" x2="12" y2="41" stroke="#e0d5c5" stroke-width="0.5"/>
                <line x1="14" y1="39" x2="14" y2="40.5" stroke="#e0d5c5" stroke-width="0.5"/>
                <!-- 右手（招いてる） -->
                <path d="M36,30 Q38,28 37,24 Q36,22 35,24 L34,28 Q34,30 36,30" fill="#fef9f0" stroke="#e0d5c5" stroke-width="0.8"/>
                <ellipse cx="36" cy="23" rx="2.5" ry="2" fill="#fef9f0" stroke="#e0d5c5" stroke-width="0.8" transform="rotate(-20 36 23)"/>
                <line x1="34.5" y1="22" x2="34" y2="20.5" stroke="#e0d5c5" stroke-width="0.5"/>
                <line x1="36" y1="21" x2="36" y2="19.5" stroke="#e0d5c5" stroke-width="0.5"/>
                <line x1="37.5" y1="22" x2="38" y2="20.5" stroke="#e0d5c5" stroke-width="0.5"/>
                <!-- 顔 -->
                <ellipse cx="24" cy="18" rx="11" ry="9" fill="#fef9f0" stroke="#e0d5c5" stroke-width="0.8"/>
                <!-- 耳（外側） -->
                <polygon points="14,12 10,3 19,9" fill="#1a1a1a"/>
                <polygon points="34,12 38,3 29,9" fill="#1a1a1a"/>
                <!-- 耳（内側ピンク） -->
                <polygon points="15,11 12,5 18,9" fill="#ffb6c1"/>
                <polygon points="33,11 36,5 30,9" fill="#ffb6c1"/>
                <!-- おでこのコイン -->
                <ellipse cx="24" cy="11" rx="4" ry="3" fill="#ffd700" stroke="#daa520" stroke-width="0.8"/>
                <text x="24" y="13" text-anchor="middle" font-size="4" font-weight="bold" fill="#8b6914">¥</text>
                <!-- 目 -->
                <ellipse cx="19" cy="17" rx="3" ry="3.5" fill="#fff"/>
                <ellipse cx="29" cy="17" rx="3" ry="3.5" fill="#fff"/>
                <ellipse cx="20" cy="18" rx="1.8" ry="2.2" fill="#2a2a2a"/>
                <ellipse cx="30" cy="18" rx="1.8" ry="2.2" fill="#2a2a2a"/>
                <!-- 目のハイライト -->
                <circle cx="20.5" cy="16.5" r="0.8" fill="#fff"/>
                <circle cx="30.5" cy="16.5" r="0.8" fill="#fff"/>
                <!-- 鼻 -->
                <ellipse cx="24" cy="21" rx="1.5" ry="1" fill="#ffb6c1"/>
                <!-- 口 -->
                <path d="M22,23 Q24,25 26,23" stroke="#2a2a2a" stroke-width="0.8" fill="none"/>
                <line x1="24" y1="22" x2="24" y2="23.5" stroke="#2a2a2a" stroke-width="0.6"/>
                <!-- ひげ -->
                <line x1="8" y1="18" x2="15" y2="19" stroke="#aaa" stroke-width="0.5"/>
                <line x1="8" y1="21" x2="15" y2="21" stroke="#aaa" stroke-width="0.5"/>
                <line x1="40" y1="18" x2="33" y2="19" stroke="#aaa" stroke-width="0.5"/>
                <line x1="40" y1="21" x2="33" y2="21" stroke="#aaa" stroke-width="0.5"/>
                <!-- 首輪 -->
                <ellipse cx="24" cy="26" rx="7" ry="1.5" fill="#e74c3c"/>
                <!-- 鈴 -->
                <circle cx="24" cy="28" r="2" fill="#ffd700" stroke="#daa520" stroke-width="0.5"/>
                <line x1="22.5" y1="28" x2="25.5" y2="28" stroke="#daa520" stroke-width="0.4"/>
            </svg>
        </div>
        <div class="header-text">
            <h1>StockSignal</h1>
            <p>東証プライム市場における短期上昇シグナル検出</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predictions = load_predictions(_mtime=_get_file_mtime(DATA_DIR / "predictions.parquet"))
    if predictions is None:
        st.error("予測データが見つかりません。")
        st.info(f"データディレクトリ: {DATA_DIR}")
        if DATA_DIR.exists():
            st.info(f"ファイル一覧: {list(DATA_DIR.glob('*.parquet'))}")
        else:
            st.warning("dataディレクトリが存在しません")
        st.stop()

    available_signal_dates = sorted(predictions['date'].unique())

    # シグナル日→エントリー日（翌営業日）を計算
    def signal_to_entry(signal_date):
        entry = pd.Timestamp(signal_date) + pd.Timedelta(days=1)
        while entry.weekday() >= 5:  # 土日をスキップ
            entry += pd.Timedelta(days=1)
        return entry

    # エントリー日→シグナル日（前営業日）を逆算
    def entry_to_signal(entry_date, available_signals):
        entry_ts = pd.Timestamp(entry_date)
        # エントリー日に対応するシグナル日を探す
        for signal in reversed(available_signals):
            signal_ts = pd.Timestamp(signal)
            expected_entry = signal_to_entry(signal_ts)
            if expected_entry.date() == entry_ts.date():
                return signal_ts
        # 見つからない場合は、最も近いシグナル日を返す
        # （選択されたエントリー日の前後で最も近いもの）
        signal_timestamps = [pd.Timestamp(s) for s in available_signals]
        closest = min(signal_timestamps, key=lambda s: abs(signal_to_entry(s) - entry_ts))
        return closest

    # 利用可能なエントリー日のリストを作成
    available_entry_dates = [signal_to_entry(d).date() for d in available_signal_dates]

    # サイドバー
    with st.sidebar:
        st.markdown("### 設定")
        min_entry_date = min(available_entry_dates)
        max_entry_date = max(available_entry_dates)

        selected_entry_date = st.date_input(
            "エントリー日",
            value=max_entry_date,
            min_value=min_entry_date,
            max_value=max_entry_date,
            help="この日の寄付でエントリー（前営業日の終値で判定）"
        )
        top_n = st.slider("表示件数", 5, 30, 10)

        st.markdown("---")
        st.markdown("### 売買ルール")
        st.markdown("""
        <div class="rule-item"><span class="rule-label">利確</span><span class="rule-value">+10%</span></div>
        <div class="rule-item"><span class="rule-label">損切り</span><span class="rule-value">-10%</span></div>
        <div class="rule-item"><span class="rule-label">最大保有</span><span class="rule-value">20営業日</span></div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("更新方法"):
            st.code("python scripts/phase1_data_check.py\npython scripts/phase2_train.py\npython scripts/phase3_backtest.py", language="bash")
            st.caption("GitHubへpushで自動デプロイ")

        # デバッグ情報
        with st.expander("データ情報"):
            st.caption(f"エントリー日範囲: {min_entry_date} 〜 {max_entry_date}")

    # メイン
    # 選択されたエントリー日から対応するシグナル日を逆算
    signal_ts = entry_to_signal(selected_entry_date, available_signal_dates)
    actual_entry_date = signal_to_entry(signal_ts)  # 実際のエントリー日
    selected_entry_ts = pd.Timestamp(selected_entry_date)

    # 選択日と実際のエントリー日が異なる場合は通知
    if actual_entry_date.date() != selected_entry_ts.date():
        st.info(f"📅 {selected_entry_date} は休場日のため、{actual_entry_date.strftime('%Y/%m/%d')} のデータを表示しています")

    day_predictions = predictions[predictions['date'] == signal_ts].copy()
    # スコア閾値(0.55)以上のみ表示
    if 'score' in day_predictions.columns:
        day_predictions = day_predictions[day_predictions['score'] >= 0.55]
    day_predictions = day_predictions.sort_values('rank')

    weekdays = ['月', '火', '水', '木', '金', '土', '日']

    # 統計
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-label">エントリー</div>
            <div class="stat-value">{actual_entry_date.strftime('%Y/%m/%d')} ({weekdays[actual_entry_date.weekday()]}) 寄付</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">シグナル日</div>
            <div class="stat-value">{signal_ts.strftime('%Y/%m/%d')} ({weekdays[signal_ts.weekday()]})</div>
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

        name, open_price, close_price, reason, sector = get_stock_info(ticker, str(signal_ts.date()))
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

    # CSSグリッドレイアウト
    cards_html = '<div class="stock-grid">'
    for r in results:
        cards_html += get_stock_card_html(r['rank'], r['code'], r['name'], r['score'], r['open'], r['close'], r['reason'], r['sector'])
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # 高確度シグナルセクション
    sector_mapping = load_sector_mapping(_mtime=_get_file_mtime(DATA_DIR / "sector_mapping.parquet"))
    high_conf_signals = get_high_confidence_signals(predictions, sector_mapping, days=30)

    if not high_conf_signals.empty:
        st.markdown("""
        <div class="high-conf-section">
            <div class="high-conf-header">
                <span class="high-conf-title">高確度シグナル</span>
                <span class="high-conf-badge">直近30日</span>
            </div>
            <div class="high-conf-description">
                過去データ分析に基づく高勝率条件に合致したシグナル（金融・素材セクター、連続2回目、スコア0.80以上）
            </div>
        </div>
        """, unsafe_allow_html=True)

        # シグナル日→エントリー日を計算する関数（既存のものを再利用）
        def calc_entry_date(signal_date):
            entry = pd.Timestamp(signal_date) + pd.Timedelta(days=1)
            while entry.weekday() >= 5:
                entry += pd.Timedelta(days=1)
            return entry.strftime('%m/%d')

        high_conf_cards = '<div class="stock-grid">'
        for _, row in high_conf_signals.head(10).iterrows():
            code = row['ticker'].replace('.T', '')
            name = fetch_company_name_from_yahoo(code) or code
            if name and len(name) > 18:
                name = name[:18]
            signal_date_str = row['date'].strftime('%m/%d')
            entry_date_str = calc_entry_date(row['date'])
            high_conf_cards += get_high_confidence_card_html(
                code, name, row['score'], row['sector'],
                row['reasons'], signal_date_str, entry_date_str
            )
        high_conf_cards += '</div>'
        st.markdown(high_conf_cards, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>注意</strong> — 本システムは機械学習モデルによる予測であり、将来の株価を保証するものではありません。投資判断は自己責任でお願いします。
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
