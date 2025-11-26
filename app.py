import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import textwrap

# === [앱 보안 설정] ===
APP_PASSWORD = "1979"

# === [페이지 기본 설정] ===
st.set_page_config(
    page_title="HK 옵션투자자문 (Expert v18.0 - Dynamic Exit)",
    page_icon="📊",
    layout="wide"
)

# 차트 스타일
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'

# === [0] 로그인 화면 ===
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    st.title("🔒 HK Advisory 보안 접속")
    password = st.text_input("비밀번호를 입력하세요", type="password")
    
    if st.button("로그인"):
        if password == APP_PASSWORD:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    return False

if not check_password():
    st.stop()

# === [1] 데이터 수집 ===
@st.cache_data(ttl=1800)
def get_market_data():
    qqq = yf.Ticker("QQQ")
    hist = qqq.history(period="2y")
    
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    
    hist['BB_Mid'] = hist['MA20']
    hist['BB_Std'] = hist['Close'].rolling(window=20).std()
    hist['BB_Upper'] = hist['BB_Mid'] + (hist['BB_Std'] * 2)
    hist['BB_Lower'] = hist['BB_Mid'] - (hist['BB_Std'] * 2)
    
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    hist['Vol_MA20'] = hist['Volume'].rolling(window=20).mean()
    
    vix_hist = yf.Ticker("^VIX").history(period="1y")
    
    curr = hist.iloc[-1]
    prev = hist.iloc[-2]
    curr_vix = vix_hist['Close'].iloc[-1]
    prev_vix = vix_hist['Close'].iloc[-2]
    
    vol_pct = (curr['Volume'] / curr['Vol_MA20']) * 100

    try:
        dates = qqq.options
        chain = qqq.option_chain(dates[1])
        current_iv = chain.calls['impliedVolatility'].mean()
    except:
        current_iv = curr_vix / 100.0

    return {
        'price': curr['Close'], 'price_prev': prev['Close'], 'open': curr['Open'],
        'ma20': curr['MA20'], 'ma50': curr['MA50'], 'ma200': curr['MA200'],
        'rsi': curr['RSI'], 'rsi_prev': prev['RSI'],
        'bb_upper': curr['BB_Upper'], 'bb_lower': curr['BB_Lower'], 'bb_lower_prev': prev['BB_Lower'],
        'macd': curr['MACD'], 'signal': curr['Signal'],
        'macd_prev': prev['MACD'], 'signal_prev': prev['Signal'],
        'volume': curr['Volume'], 'vol_ma20': curr['Vol_MA20'], 'vol_pct': vol_pct,
        'vix': curr_vix, 'vix_prev': prev_vix,
        'iv': current_iv,
        'hist': hist, 'vix_hist': vix_hist
    }

# === [2] 전문가 로직 (RSI Time-Decay & VIX 전략) ===
def analyze_expert_logic(d):
    if d['price'] > d['ma50'] and d['price'] > d['ma200']: season = "SUMMER"
    elif d['price'] < d['ma50'] and d['price'] > d['ma200']: season = "AUTUMN"
    elif d['price'] < d['ma50'] and d['price'] < d['ma200']: season = "WINTER"
    else: season = "SPRING"
    
    score = 0
    log = {}
    
    # --- RSI Logic ---
    hist_rsi = d['hist']['RSI']
    curr_rsi = d['rsi']
    days_since_escape = 0
    is_escape_mode = False

    if curr_rsi >= 30:
        for i in range(1, 10):
            check_idx = -1 - i
            if abs(check_idx) > len(hist_rsi): break
            if hist_rsi.iloc[check_idx] < 30:
                days_since_escape = i
                is_escape_mode = True
                break
    
    if curr_rsi < 30:
        pts = 5 if season == "SUMMER" else 4 if season == "AUTUMN" or season == "SPRING" else 0
        score += pts
        log['rsi'] = 'under'
    elif is_escape_mode and days_since_escape <= 7:
        score_map = {1: 3, 2: 4, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
        pts = score_map.get(days_since_escape, 0)
        score += pts
        log['rsi'] = f'escape_day_{days_since_escape}'
    elif curr_rsi >= 70:
        pts = -1 if season == "SUMMER" else -3 if season == "AUTUMN" else -5 if season == "WINTER" else -2
        score += pts
        log['rsi'] = 'over'
    else:
        pts = 1 if season == "SUMMER" or season == "SPRING" else 0 if season == "AUTUMN" else -1
        score += pts
        log['rsi'] = 'neutral'

    # --- VIX Logic ---
    if d['vix'] > 35:
        if d['vix'] < d['vix_prev']:
            pts = 7 if season == "WINTER" else 0
            score += pts
            log['vix'] = 'peak_out'
        else:
            pts = -5 if season == "WINTER" else -6 if season == "AUTUMN" else -5
            score += pts
            log['vix'] = 'panic_rise'
    elif d['vix'] < 20:
        pts = 2 if season == "SUMMER" else 1 if season == "SPRING" else -2 if season == "WINTER" else 0
        score += pts
        log['vix'] = 'stable'
    elif 20 <= d['vix'] <= 35:
        pts = 2 if season == "WINTER" else -1 if season == "SPRING" else -3 if season == "SUMMER" else -4
        score += pts
        log['vix'] = 'fear'
    else:
        log['vix'] = 'none'

    # Bollinger
    if d['price_prev'] < d['bb_lower_prev'] and d['price'] >= d['bb_lower']:
        pts = 5 if season == "WINTER" else 4
        score += pts
        log['bb'] = 'return'
    elif d['price'] < d['bb_lower']:
        pts = -2 if season == "WINTER" else 3 if season == "SUMMER" else 2 if season == "AUTUMN" else 1
        score += pts
        log['bb'] = 'out'
    else:
        log['bb'] = 'in'

    # Trend
    if d['price'] > d['ma20']:
        pts = 3 if season == "WINTER" or season == "SPRING" else 2
        score += pts
        log['trend'] = 'up'
    else:
        log['trend'] = 'down'

    # Volume
    if d['volume'] > d['vol_ma20'] * 1.5:
        pts = 3 if season == "WINTER" or season == "AUTUMN" else 2
        score += pts
        log['vol'] = 'explode'
    else:
        log['vol'] = 'normal'

    # MACD
    if d['macd_prev'] < 0 and d['macd'] >= 0:
        pts = 3
        score += pts
        log['macd'] = 'break_up'
    elif d['macd_prev'] > 0 and d['macd'] <= 0:
        pts = -3
        score += pts
        log['macd'] = 'break_down'
    elif d['macd'] > 0:
        pts = 1
        score += pts
        log['macd'] = 'above'
    else:
        pts = -1
        score += pts
        log['macd'] = 'below'

    return season, score, log

# === [3] 전략 탐색 및 행동 결정 (수정됨: Dynamic Exit Matrix) ===
def determine_action(score, season, data):
    """
    점수에 따라 달라지는 청산(Exit) 전략을 반환합니다.
    입력: 점수, 계절, 데이터(VIX)
    """
    # 1. VIX 급등 감지 (안전장치)
    vix_pct_change = ((data['vix'] - data['vix_prev']) / data['vix_prev']) * 100
    
    # 기본 고정 값
    TARGET_DELTA = -0.10
    
    # 반환 형식: (Target Delta, Verdict Text, Profit Target, Stop Loss, Matrix_ID)
    
    # 1. Panic Condition
    if vix_pct_change > 15.0:
        return TARGET_DELTA, "⛔ 매매 중단 (VIX 급등)", "-", "-", "panic"

    # 2. Strong Trend (강세장)
    if score >= 12:
        return TARGET_DELTA, "💎 추세 추종 (Strong)", "75%", "300%", "strong"
    
    # 3. Standard (평범)
    elif 8 <= score < 12:
        return TARGET_DELTA, "✅ 표준 대응 (Standard)", "50%", "200%", "standard"
        
    # 4. Hit & Run (약세/주의)
    elif 5 <= score < 8:
        return TARGET_DELTA, "⚠️ 속전 속결 (Hit & Run)", "30%", "150%", "weak"
        
    # 5. No Entry (관망)
    else:
        return None, "🛡️ 진입 보류", "-", "-", "no_entry"

def calculate_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return -0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

def find_best_option(price, iv, target_delta):
    if target_delta is None: return None
    
    TARGET_DTE_MIN = 45  # [고정] 만기 45일 선호
    SPREAD_WIDTH = 5     # [고정] 스프레드 폭 5달러
    
    qqq = yf.Ticker("QQQ")
    try:
        options = qqq.options
        valid_dates = []
        now = datetime.now()
        for d_str in options:
            d_date = datetime.strptime(d_str, "%Y-%m-%d")
            days_left = (d_date - now).days
            # 45일 이상이면서 가장 가까운 날짜 찾기
            if days_left >= TARGET_DTE_MIN:
                valid_dates.append((d_str, days_left))
        
        if not valid_dates: return None
        # DTE가 가장 작은 것(즉, 45일에 가장 근접한 것) 선택
        expiry, dte = min(valid_dates, key=lambda x: x[1])
        
        T = dte / 365.0
        r = 0.045
        best_strike = 0
        min_diff = 1.0
        found_delta = 0
        
        # 풋 옵션 탐색
        for strike in range(int(price * 0.5), int(price)):
            d = calculate_put_delta(price, strike, T, r, iv)
            diff = abs(d - target_delta)
            if diff < min_diff:
                min_diff = diff
                best_strike = strike
                found_delta = d
                
        return {
            'expiry': expiry, 'dte': dte,
            'short': best_strike, 'long': best_strike - SPREAD_WIDTH,
            'delta': found_delta,
            'width': SPREAD_WIDTH
        }
    except:
        return None

# === [4] 차트 (RSI 그래프 포함) ===
def create_charts(data):
    hist = data['hist']
    fig = plt.figure(figsize=(10, 16))
    
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 0.6, 1, 1, 1])
    
    # 1. Price
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(hist.index, hist['Close'], label='QQQ', color='black', alpha=0.7)
    ax1.plot(hist.index, hist['MA20'], label='20MA', color='green', ls='--', lw=1)
    ax1.plot(hist.index, hist['MA50'], label='50MA', color='blue', ls='-', lw=1.5)
    ax1.plot(hist.index, hist['MA200'], label='200MA', color='red', ls='-', lw=2)
    ax1.fill_between(hist.index, hist['BB_Upper'], hist['BB_Lower'], color='gray', alpha=0.1, label='Bollinger')
    ax1.set_title('QQQ Price Trend', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # 2. Volume
    ax_vol = fig.add_subplot(gs[1], sharex=ax1)
    colors = ['red' if c < o else 'green' for c, o in zip(hist['Close'], hist['Open'])]
    ax_vol.bar(hist.index, hist['Volume'], color=colors, alpha=0.5)
    ax_vol.plot(hist.index, hist['Vol_MA20'], color='black', lw=1)
    ax_vol.set_title(f"Volume ({data['vol_pct']:.1f}%)", fontsize=10, fontweight='bold')
    ax_vol.grid(True, alpha=0.3)
    plt.setp(ax_vol.get_xticklabels(), visible=False)

    # 3. RSI
    ax_rsi = fig.add_subplot(gs[2], sharex=ax1)
    ax_rsi.plot(hist.index, hist['RSI'], color='purple', label='RSI')
    ax_rsi.axhline(70, color='red', ls='--', alpha=0.7)
    ax_rsi.axhline(30, color='green', ls='--', alpha=0.7)
    ax_rsi.axhline(50, color='black', lw=0.5, alpha=0.5)
    ax_rsi.fill_between(hist.index, hist['RSI'], 70, where=(hist['RSI'] >= 70), color='red', alpha=0.3)
    ax_rsi.fill_between(hist.index, hist['RSI'], 30, where=(hist['RSI'] <= 30), color='green', alpha=0.3)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI (14)', fontsize=12, fontweight='bold')
    ax_rsi.grid(True, alpha=0.3)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # 4. MACD
    ax2 = fig.add_subplot(gs[3], sharex=ax1)
    ax2.plot(hist.index, hist['MACD'], label='MACD', color='blue')
    ax2.plot(hist.index, hist['Signal'], label='Signal', color='orange')
    ax2.bar(hist.index, hist['MACD']-hist['Signal'], color='gray', alpha=0.3)
    ax2.axhline(0, color='black', lw=0.8)
    
    crosses = np.sign(hist['MACD'] - hist['Signal']).diff()
    golden = hist[crosses == 2]
    death = hist[crosses == -2]
    
    ax2.scatter(golden.index, golden['MACD'], color='red', marker='^', s=100, zorder=5)
    ax2.scatter(death.index, death['MACD'], color='blue', marker='v', s=100, zorder=5)
    ax2.set_title('MACD', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # 5. VIX
    ax3 = fig.add_subplot(gs[4], sharex=ax1)
    ax3.plot(data['vix_hist'].index, data['vix_hist']['Close'], color='purple', label='VIX')
    ax3.axhline(30, color='red', ls='--')
    ax3.axhline(20, color='green', ls='--')
    ax3.set_title('VIX', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# === [메인 화면] ===
def main():
    st.title("📊 QQQ Expert Advisory (Dynamic Exit v18.0)")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with st.spinner('분석 중...'):
        try:
            data = get_market_data()
            season, score, log = analyze_expert_logic(data)
            target_delta, verdict_text, profit_target, stop_loss, matrix_id = determine_action(score, season, data)
            strategy = find_best_option(data['price'], data['iv'], target_delta)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            return

    # === [수정됨] Escape 모드 하이라이트를 위한 로직 개선 ===
    def hl_score(category, row_state, col_season):
        base = 'style="border: 1px solid #ddd; padding: 8px; color: black; background-color: white;"'
        
        current_val = log.get(category, '')
        
        is_match = False
        if category == 'rsi' and row_state == 'escape':
            if 'escape' in current_val:
                is_match = True
        else:
            if current_val == row_state:
                is_match = True
        
        if is_match and season == col_season:
            return 'style="border: 3px solid #FF5722; background-color: #FFF8E1; font-weight: bold; color: #D84315; padding: 8px;"'
        return base

    def hl_season(row_season):
        if season == row_season:
            return 'style="border: 3px solid #2196F3; background-color: #E3F2FD; font-weight: bold; color: black; padding: 8px;"'
        return 'style="border: 1px solid #ddd; padding: 8px; color: black; background-color: white;"'

    td_style = 'style="border: 1px solid #ddd; padding: 8px; color: black; background-color: white;"'
    th_style = 'style="border: 1px solid #ddd; padding: 8px; color: black; background-color: #f2f2f2;"'

    html_season = f"""
    <h3>1. Market Season Matrix</h3>
    <table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; text-align: center;">
        <tr>
            <th {th_style}>Season</th><th {th_style}>Condition</th><th {th_style}>Character</th>
        </tr>
        <tr><td {hl_season('SUMMER')}>☀️ SUMMER</td><td {hl_season('SUMMER')}>Price > 50MA & 200MA</td><td {hl_season('SUMMER')}>강세장</td></tr>
        <tr><td {hl_season('AUTUMN')}>🍂 AUTUMN</td><td {hl_season('AUTUMN')}>Price < 50MA but > 200MA</td><td {hl_season('AUTUMN')}>조정기</td></tr>
        <tr><td {hl_season('WINTER')}>❄️ WINTER</td><td {hl_season('WINTER')}>Price < 50MA & 200MA</td><td {hl_season('WINTER')}>약세장</td></tr>
        <tr><td {hl_season('SPRING')}>🌱 SPRING</td><td {hl_season('SPRING')}>Price > 50MA but < 200MA</td><td {hl_season('SPRING')}>회복기</td></tr>
    </table>
    <p>※ QQQ: <b>${data['price']:.2f}</b> (Vol: {data['vol_pct']:.1f}% of 20MA)</p>
    """
    st.markdown(textwrap.dedent(html_season), unsafe_allow_html=True)

    # HTML 2: Scorecard
    html_score = f"""
    <h3>2. Expert Matrix Scorecard</h3>
    <table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; text-align: center;">
        <tr>
            <th {th_style}>지표</th><th {th_style}>상태</th>
            <th {th_style}>☀️</th><th {th_style}>🍂</th><th {th_style}>❄️</th><th {th_style}>🌱</th>
            <th {th_style}>Logic</th>
        </tr>
        <tr><td rowspan="4" {td_style}>RSI</td>
            <td {td_style}>과열 (>70)</td>
            <td {hl_score('rsi', 'over', 'SUMMER')}>-1</td><td {hl_score('rsi', 'over', 'AUTUMN')}>-3</td><td {hl_score('rsi', 'over', 'WINTER')}>-5</td><td {hl_score('rsi', 'over', 'SPRING')}>-2</td>
            <td align="left" {td_style}>가짜 반등</td></tr>
        <tr><td {td_style}>중립 (45-65)</td>
            <td {hl_score('rsi', 'neutral', 'SUMMER')}>+1</td><td {hl_score('rsi', 'neutral', 'AUTUMN')}>0</td><td {hl_score('rsi', 'neutral', 'WINTER')}>-1</td><td {hl_score('rsi', 'neutral', 'SPRING')}>+1</td>
            <td align="left" {td_style}>-</td></tr>
        <tr><td {td_style}>과매도 (<30)</td>
            <td {hl_score('rsi', 'under', 'SUMMER')}>+5</td><td {hl_score('rsi', 'under', 'AUTUMN')}>+4</td><td {hl_score('rsi', 'under', 'WINTER')}>0</td><td {hl_score('rsi', 'under', 'SPRING')}>+4</td>
            <td align="left" {td_style}>겨울 바닥 X</td></tr>
        <tr><td {td_style}>🚀 탈출 (1~7일)</td>
            <td {hl_score('rsi', 'escape', 'SUMMER')}>3~5</td><td {hl_score('rsi', 'escape', 'AUTUMN')}>3~5</td><td {hl_score('rsi', 'escape', 'WINTER')}>3~5</td><td {hl_score('rsi', 'escape', 'SPRING')}>3~5</td>
            <td align="left" {td_style}><b>Best Timing</b></td></tr>
        <tr><td rowspan="4" {td_style}>VIX</td>
            <td {td_style}>안정 (<20)</td>
            <td {hl_score('vix', 'stable', 'SUMMER')}>+2</td><td {hl_score('vix', 'stable', 'AUTUMN')}>0</td><td {hl_score('vix', 'stable', 'WINTER')}>-2</td><td {hl_score('vix', 'stable', 'SPRING')}>+1</td>
            <td align="left" {td_style}>저변동성</td></tr>
        <tr><td {td_style}>공포 (20-35)</td>
            <td {hl_score('vix', 'fear', 'SUMMER')}>-3</td><td {hl_score('vix', 'fear', 'AUTUMN')}>-4</td><td {hl_score('vix', 'fear', 'WINTER')}>+2</td><td {hl_score('vix', 'fear', 'SPRING')}>-1</td>
            <td align="left" {td_style}>기회 탐색</td></tr>
        <tr><td {td_style}>패닉 상승</td>
            <td {hl_score('vix', 'panic_rise', 'SUMMER')}>-5</td><td {hl_score('vix', 'panic_rise', 'AUTUMN')}>-6</td><td {hl_score('vix', 'panic_rise', 'WINTER')}>-5</td><td {hl_score('vix', 'panic_rise', 'SPRING')}>-4</td>
            <td align="left" {td_style}>칼날</td></tr>
        <tr><td {td_style}>📉 꺾임</td>
            <td {hl_score('vix', 'peak_out', 'SUMMER')}>-</td><td {hl_score('vix', 'peak_out', 'AUTUMN')}>-</td><td {hl_score('vix', 'peak_out', 'WINTER')}>+7</td><td {hl_score('vix', 'peak_out', 'SPRING')}>-</td>
            <td align="left" {td_style}><b>Sniper</b></td></tr>
        <tr><td rowspan="3" {td_style}>BB</td>
            <td {td_style}>밴드 내부</td>
            <td {hl_score('bb', 'in', 'SUMMER')}>0</td><td {hl_score('bb', 'in', 'AUTUMN')}>0</td><td {hl_score('bb', 'in', 'WINTER')}>0</td><td {hl_score('bb', 'in', 'SPRING')}>0</td>
            <td align="left" {td_style}>대기</td></tr>
        <tr><td {td_style}>하단 이탈</td>
            <td {hl_score('bb', 'out', 'SUMMER')}>+3</td><td {hl_score('bb', 'out', 'AUTUMN')}>+2</td><td {hl_score('bb', 'out', 'WINTER')}>-2</td><td {hl_score('bb', 'out', 'SPRING')}>+1</td>
            <td align="left" {td_style}>가속화</td></tr>
        <tr><td {td_style}>↩️ 복귀</td>
            <td {hl_score('bb', 'return', 'SUMMER')}>+4</td><td {hl_score('bb', 'return', 'AUTUMN')}>+3</td><td {hl_score('bb', 'return', 'WINTER')}>+5</td><td {hl_score('bb', 'return', 'SPRING')}>+4</td>
            <td align="left" {td_style}><b>Close In</b></td></tr>
        <tr><td {td_style}>추세</td><td {td_style}>20일선 위</td>
            <td {hl_score('trend', 'up', 'SUMMER')}>+2</td><td {hl_score('trend', 'up', 'AUTUMN')}>+2</td><td {hl_score('trend', 'up', 'WINTER')}>+3</td><td {hl_score('trend', 'up', 'SPRING')}>+3</td>
            <td align="left" {td_style}>회복</td></tr>
        <tr><td {td_style}>거래량</td><td {td_style}>폭증 (>150%)</td>
            <td {hl_score('vol', 'explode', 'SUMMER')}>+2</td><td {hl_score('vol', 'explode', 'AUTUMN')}>+3</td><td {hl_score('vol', 'explode', 'WINTER')}>+3</td><td {hl_score('vol', 'explode', 'SPRING')}>+2</td>
            <td align="left" {td_style}><b>손바뀜</b></td></tr>
        <tr><td {td_style}>거래량</td><td {td_style}>일반</td>
            <td {hl_score('vol', 'normal', 'SUMMER')}>0</td><td {hl_score('vol', 'normal', 'AUTUMN')}>0</td><td {hl_score('vol', 'normal', 'WINTER')}>0</td><td {hl_score('vol', 'normal', 'SPRING')}>0</td>
            <td align="left" {td_style}>-</td></tr>
        <tr><td rowspan="4" {td_style}>MACD</td>
            <td {td_style}>🚀 수면 돌파</td>
            <td {hl_score('macd', 'break_up', 'SUMMER')}>+3</td><td {hl_score('macd', 'break_up', 'AUTUMN')}>+3</td><td {hl_score('macd', 'break_up', 'WINTER')}>+3</td><td {hl_score('macd', 'break_up', 'SPRING')}>+3</td>
            <td align="left" {td_style}><b>강력 매수</b></td></tr>
        <tr><td {td_style}>수면 위 (>0)</td>
            <td {hl_score('macd', 'above', 'SUMMER')}>+1</td><td {hl_score('macd', 'above', 'AUTUMN')}>+1</td><td {hl_score('macd', 'above', 'WINTER')}>+1</td><td {hl_score('macd', 'above', 'SPRING')}>+1</td>
            <td align="left" {td_style}>순풍</td></tr>
        <tr><td {td_style}>🌊 수면 추락</td>
            <td {hl_score('macd', 'break_down', 'SUMMER')}>-3</td><td {hl_score('macd', 'break_down', 'AUTUMN')}>-3</td><td {hl_score('macd', 'break_down', 'WINTER')}>-3</td><td {hl_score('macd', 'break_down', 'SPRING')}>-3</td>
            <td align="left" {td_style}><b>강력 매도</b></td></tr>
        <tr><td {td_style}>수면 아래 (<0)</td>
            <td {hl_score('macd', 'below', 'SUMMER')}>-1</td><td {hl_score('macd', 'below', 'AUTUMN')}>-1</td><td {hl_score('macd', 'below', 'WINTER')}>-1</td><td {hl_score('macd', 'below', 'SPRING')}>-1</td>
            <td align="left" {td_style}>역풍</td></tr>
    </table>
    """
    st.markdown(textwrap.dedent(html_score), unsafe_allow_html=True)

    # === [수정됨] HTML 3: Final Verdict & Dynamic Exit Matrix ===
    # 스타일 헬퍼 함수
    def get_matrix_style(current_id, row_id, bg_color):
        if current_id == row_id:
            # Highlight: 배경색 적용, 테두리 진하게, 글씨 두껍게
            return f'style="background-color: {bg_color}; border: 3px solid #666; font-weight: bold; color: #333; height: 50px;"'
        else:
            # Normal: 흰 배경, 연한 테두리
            return 'style="background-color: white; border: 1px solid #eee; color: #999;"'

    html_verdict = f"""
    <h3>3. Final Verdict: <span style="color:blue;">{score}점</span> - Dynamic Exit Matrix</h3>
    <div style="border: 2px solid #ccc; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; text-align: center;">
            <tr style="background-color: #333; color: white;">
                <th {th_style} style="color:white;">점수 구간</th>
                <th {th_style} style="color:white;">최종 판정</th>
                <th {th_style} style="color:white;">🎯 익절 목표</th>
                <th {th_style} style="color:white;">🛑 손절 라인</th>
            </tr>
            <tr {get_matrix_style(matrix_id, 'panic', '#ffebee')}>
                <td>VIX 급등</td>
                <td>⛔ 매매 중단 (Panic)</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr {get_matrix_style(matrix_id, 'strong', '#dff0d8')}>
                <td>12점 이상</td>
                <td>💎 추세 추종 (Strong)</td>
                <td style="color:green;">+75%</td>
                <td style="color:red;">-300% (원금 3배)</td>
            </tr>
            <tr {get_matrix_style(matrix_id, 'standard', '#ffffff')}>
                <td>8 ~ 11점</td>
                <td>✅ 표준 대응 (Standard)</td>
                <td style="color:green;">+50%</td>
                <td style="color:red;">-200% (원금 3배)</td>
            </tr>
            <tr {get_matrix_style(matrix_id, 'weak', '#fff9c4')}>
                <td>5 ~ 7점</td>
                <td>⚠️ 속전 속결 (Hit & Run)</td>
                <td style="color:green;">+30%</td>
                <td style="color:red;">-150% (원금 2.5배)</td>
            </tr>
            <tr {get_matrix_style(matrix_id, 'no_entry', '#f2dede')}>
                <td>5점 미만</td>
                <td>🛡️ 진입 보류 (No Entry)</td>
                <td>-</td>
                <td>-</td>
            </tr>
        </table>
        <div style="padding: 10px; background-color: #f9f9f9; text-align: center; color: #555; font-size: 13px;">
            ※ <b>설정:</b> Delta -0.10 (Fixed) / DTE 45일 / Spread $5<br>
            ※ 손절 라인은 프리미엄 가격 기준입니다. (예: $1.0 진입 시, 200% 손절은 $3.0 도달 시 청산)
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(html_verdict), unsafe_allow_html=True)

    if strategy and matrix_id != 'no_entry' and matrix_id != 'panic':
        html_manual = f"""
        <div style="border: 2px solid #2196F3; padding: 15px; margin-top: 20px; border-radius: 10px; background-color: #ffffff; color: black;">
            <h3 style="color: #2196F3; margin-top: 0;">👮‍♂️ 주문 상세 매뉴얼</h3>
            <ul style="line-height: 1.6; list-style-type: none; padding-left: 0; color: black;">
                <li>✅ <b>종목:</b> QQQ (Put Credit Spread)</li>
                <li>✅ <b>만기:</b> {strategy['expiry']} (DTE {strategy['dte']}일)</li>
                <li>✅ <b>Strike:</b> Short <b style="color:red">${strategy['short']}</b> / Long <b style="color:green">${strategy['long']}</b> (Width ${strategy['width']})</li>
                <li>✅ <b>Delta:</b> {strategy['delta']:.3f} (Target: {target_delta})</li>
            </ul>
            <hr>
            <h4 style="margin-bottom: 5px; color: black;">🛑 적용된 청산 원칙</h4>
            <ul style="line-height: 1.6; color: black;">
                <li><b>익절 (Take Profit):</b> 진입 프리미엄의 <b>{profit_target}</b> 이익 시 청산</li>
                <li style="color: red; font-weight: bold;">손절 (Stop Loss): 진입 프리미엄의 {stop_loss} 도달 시 즉시 청산</li>
                <li><b>주의:</b> 90% 비중 투자 시, 반드시 진입과 동시에 <u>감시 주문(Stop Limit)</u>을 설정하세요.</li>
            </ul>
        </div>
        """
    else:
        html_manual = """
        <div style="border: 2px solid red; padding: 15px; margin-top: 20px; border-radius: 10px; background-color: #ffebee;">
            <h3 style="color: red; margin-top: 0;">⛔ 진입 금지 (No Entry)</h3>
            <p style="color: black;">현재 점수 또는 시장 상황(VIX)이 신규 진입에 적합하지 않습니다.<br>
            기존 포지션 관리(청산/롤오버)에만 집중하십시오.</p>
        </div>
        """
    st.markdown(textwrap.dedent(html_manual), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 기술적 분석 차트")
    st.pyplot(create_charts(data))

if __name__ == "__main__":
    main()
