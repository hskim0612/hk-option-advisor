import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === [앱 보안 설정] ===
APP_PASSWORD = "1979"

# === [페이지 기본 설정] ===
st.set_page_config(
    page_title="HK 옵션투자자문 (Grand Master v21.0)",
    page_icon="🦅",
    layout="wide"
)

# 차트 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

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

# === [1] 데이터 수집 및 전처리 (CRITICAL: 동기화 로직 적용) ===
def calculate_rsi(series, period=14):
    delta = series.diff()
    # EMA 적용 (Wilder's Smoothing과 유사)
    gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def sync_all_data(qqq_hist, vix_hist, vix3m_hist, vvix_hist):
    """
    모든 데이터를 QQQ의 거래일 인덱스로 통일 (Timezone 제거 -> 날짜 정규화 -> Inner Join)
    """
    # 1. 공통 날짜 인덱스 생성 (QQQ 기준)
    master_index = qqq_hist.index.tz_localize(None).normalize()
    qqq_hist.index = master_index
    
    # 2. 각 데이터프레임 전처리 (Timezone 제거 및 정규화 함수)
    def prep_df(df):
        if df is None or df.empty: return pd.DataFrame()
        df = df.copy()
        df.index = df.index.tz_localize(None).normalize()
        # 중복 인덱스 제거
        df = df[~df.index.duplicated(keep='last')]
        return df

    vix_clean = prep_df(vix_hist)
    vix3m_clean = prep_df(vix3m_hist)
    vvix_clean = prep_df(vvix_hist)
    
    # 3. 병합 (Inner Join 효과를 위해 concat 후 dropna)
    # QQQ는 전체 컬럼 유지, 나머지는 Close만 가져와서 이름 변경
    merged = pd.concat([
        qqq_hist,
        vix_clean[['Close']].rename(columns={'Close': 'VIX'}),
        vix3m_clean[['Close']].rename(columns={'Close': 'VIX3M'}),
        vvix_clean[['Close']].rename(columns={'Close': 'VVIX'})
    ], axis=1).dropna()
    
    return merged

@st.cache_data(ttl=1800)
def get_market_data():
    # 1. 데이터 가져오기
    qqq = yf.Ticker("QQQ")
    hist = qqq.history(period="2y")
    
    vix = yf.Ticker("^VIX")
    vix_hist = vix.history(period="2y")
    
    vix3m = yf.Ticker("^VIX3M")
    vix3m_hist = vix3m.history(period="2y")
    
    vvix = yf.Ticker("^VVIX")
    vvix_hist = vvix.history(period="2y")

    # 2. 데이터 동기화 (가장 중요)
    merged_df = sync_all_data(hist, vix_hist, vix3m_hist, vvix_hist)
    
    if len(merged_df) < 200:
        st.error("데이터 부족: 주요 지수 데이터를 충분히 확보하지 못했습니다.")
        return None

    # 3. 지표 계산 (동기화된 데이터프레임 위에서 수행)
    df = merged_df.copy()
    
    # 이동평균선
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # 볼린저 밴드
    df['BB_Mid'] = df['MA20']
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (14 & 2)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['RSI_2'] = calculate_rsi(df['Close'], 2)
    
    # Volume MA
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # VIX Ratio
    df['VIX_Ratio'] = df['VIX'] / df['VIX3M']
    
    # 현재 상태값 추출
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # IV (Implied Volatility) - Fallback to VIX if API fails
    try:
        dates = qqq.options
        chain = qqq.option_chain(dates[1])
        current_iv = chain.calls['impliedVolatility'].mean()
    except:
        current_iv = curr['VIX'] / 100.0

    return {
        'price': curr['Close'], 'price_prev': prev['Close'], 'open': curr['Open'],
        'ma20': curr['MA20'], 'ma50': curr['MA50'], 'ma200': curr['MA200'],
        'rsi': curr['RSI'], 'rsi_prev': prev['RSI'],
        'rsi2': curr['RSI_2'], 'rsi2_prev': prev['RSI_2'],
        'bb_upper': curr['BB_Upper'], 'bb_lower': curr['BB_Lower'], 'bb_lower_prev': prev['BB_Lower'],
        'macd': curr['MACD'], 'signal': curr['Signal'],
        'macd_prev': prev['MACD'], 'signal_prev': prev['Signal'],
        'volume': curr['Volume'], 'vol_ma20': curr['Vol_MA20'],
        'vix': curr['VIX'], 'vix_prev': prev['VIX'],
        'vix3m': curr['VIX3M'],
        'vvix': curr['VVIX'], 'vvix_prev': prev['VVIX'],
        'vix_ratio': curr['VIX_Ratio'],
        'iv': current_iv,
        'hist': df,
    }

# === [2] 전문가 로직 (PART 2: 신규 배점표 반영) ===
def detect_capitulation(hist):
    """투매 감지: 2일 연속 (VIX Ratio > 1.0 AND Vol > 1.5배)"""
    if len(hist) < 2: return 0
    
    curr = hist.iloc[-1]
    prev = hist.iloc[-2]
    
    cond_curr = (curr['VIX_Ratio'] > 1.0) and (curr['Volume'] > curr['Vol_MA20'] * 1.5)
    cond_prev = (prev['VIX_Ratio'] > 1.0) and (prev['Volume'] > prev['Vol_MA20'] * 1.5)
    
    return 15 if (cond_curr and cond_prev) else 0

def detect_vvix_trap(hist):
    """VVIX Trap: VIX는 횡보/하락하는데 VVIX는 급등"""
    if len(hist) < 4: return 0
    
    # VIX 3일 이동평균 변화율
    vix_ma3 = hist['VIX'].rolling(3).mean()
    vix_change = (vix_ma3.iloc[-1] - vix_ma3.iloc[-4]) / vix_ma3.iloc[-4]
    
    # VVIX 전일 대비 변화율
    vvix_change = (hist['VVIX'].iloc[-1] - hist['VVIX'].iloc[-2]) / hist['VVIX'].iloc[-2]
    
    # VIX 횡보 (±2%) + VVIX 급등 (>5%)
    if abs(vix_change) < 0.02 and vvix_change > 0.05:
        return -5
    return 0

def analyze_expert_logic(d):
    hist = d['hist']
    score = 0
    log = {}
    
    # 2.1 계절 판정 (WINTER 감점 강화)
    if d['price'] > d['ma50'] and d['price'] > d['ma200']: season = "SUMMER"
    elif d['price'] < d['ma50'] and d['price'] > d['ma200']: season = "AUTUMN"
    elif d['price'] < d['ma50'] and d['price'] < d['ma200']: season = "WINTER"
    else: season = "SPRING"
    
    if season == "WINTER":
        score += -5
        log['season'] = 'winter_penalty'
    else:
        log['season'] = 'normal'

    # 2.2 VIX Term Structure (최우선 순위 - 붕괴 경보 강화)
    vix_ratio = d['vix_ratio']
    if vix_ratio > 1.0:
        score += -20 # 시스템 차단급 감점
        log['term'] = 'backwardation'
    elif vix_ratio < 0.9:
        score += 3
        log['term'] = 'contango'
    else:
        log['term'] = 'neutral'
    
    log['vix_ratio'] = vix_ratio

    # 2.3 투매 신호 (Capitulation)
    cap_score = detect_capitulation(hist)
    score += cap_score
    if cap_score > 0: log['capitulation'] = True
    else: log['capitulation'] = False

    # 2.4 VVIX Trap
    trap_score = detect_vvix_trap(hist)
    score += trap_score
    if trap_score < 0: log['vvix_trap'] = True
    else: log['vvix_trap'] = False

    # 2.5 RSI(14) 로직 (겨울철 강화)
    curr_rsi = d['rsi']
    # 탈출 로직 체크
    days_since_escape = 0
    is_escape_mode = False
    if curr_rsi >= 30:
        for i in range(1, 4): # 최근 3일 이내 탈출 체크
            check_idx = -1 - i
            if abs(check_idx) > len(hist): break
            if hist['RSI'].iloc[check_idx] < 30:
                days_since_escape = i
                is_escape_mode = True
                break
    
    if curr_rsi > 70:
        pts = -10 if season == "WINTER" else -3
        score += pts
        log['rsi'] = 'over'
    elif curr_rsi < 30:
        pts = 0 if season == "WINTER" else 4
        score += pts
        log['rsi'] = 'under'
    elif is_escape_mode:
        pts = 5 # 1-3일차는 강력 매수
        score += pts
        log['rsi'] = 'escape'
    else:
        log['rsi'] = 'neutral'

    # 2.6 RSI(2) 눌림목 신호
    vvix_change = (d['vvix'] - d['vvix_prev']) / d['vvix_prev']
    if d['rsi2'] < 10:
        # 조건: 구조 안정(<1.0) AND VVIX 감소세
        if vix_ratio < 1.0 and vvix_change < 0:
            score += 3
            log['rsi2'] = 'dip_buy'
        else:
            log['rsi2'] = 'dip_risk'
    else:
        log['rsi2'] = 'normal'

    # 2.7 MACD (감점 강화)
    if d['macd_prev'] < 0 and d['macd'] >= 0: # Golden Cross
        score += 3
        log['macd'] = 'golden'
    elif d['macd_prev'] > 0 and d['macd'] <= 0: # Dead Cross
        score += -5
        log['macd'] = 'dead'
    elif d['macd'] < 0:
        score += -2
        log['macd'] = 'below'
    else:
        score += 1
        log['macd'] = 'above'

    # 2.8 볼린저 밴드
    if d['price_prev'] < d['bb_lower_prev'] and d['price'] >= d['bb_lower']:
        pts = 5 if season == "WINTER" else 4
        score += pts
        log['bb'] = 'return'
    elif d['price'] < d['bb_lower']:
        pts = -2 if season == "WINTER" else 3
        score += pts
        log['bb'] = 'out'
    else:
        log['bb'] = 'in'
        
    # 2.9 Volume
    vol_pct = (d['volume'] / d['vol_ma20']) * 100
    if vol_pct > 150:
        # 투매와 중복되지 않게 단순 볼륨 증가는 소폭 가산
        if not log['capitulation']:
            score += 2
        log['vol'] = 'explode'
    else:
        log['vol'] = 'normal'

    return season, score, log

# === [3] 전략 탐색 및 행동 결정 (PART 3) ===
def determine_action(score, season, data, log):
    vix_pct_change = ((data['vix'] - data['vix_prev']) / data['vix_prev']) * 100
    
    # [Phase 0] 우선순위 차단 (Blocking)
    if log.get('term') == 'backwardation':
        return -0.10, "⛔ 매매 중단 (System Collapse)", "-", "-", "panic", "관망"
    if vix_pct_change > 15.0:
        return -0.10, "⛔ 매매 중단 (VIX 급등)", "-", "-", "panic", "관망"
    if log.get('vvix_trap'):
        return -0.10, "⛔ 매매 중단 (VVIX Trap 감지)", "-", "-", "panic", "관망"

    # [Phase 1] 전략 선택 매트릭스
    matrix_id = "no_entry"
    verdict_text = "🛡️ 진입 보류"
    profit_target = "-"
    stop_loss = "-"
    
    if score >= 15:
        matrix_id = "strong"
        verdict_text = "💎 Strong (Sniper Mode)"
        profit_target = "100%"
        stop_loss = "300%"
    elif 12 <= score < 15:
        matrix_id = "standard"
        verdict_text = "✅ Standard (표준 대응)"
        profit_target = "50%"
        stop_loss = "200%"
    elif 8 <= score < 12:
        matrix_id = "weak"
        verdict_text = "⚠️ Hit & Run (속전속결)"
        profit_target = "30%"
        stop_loss = "150%"
    
    # [Phase 2] 전략 구조 선택
    strategy_type = "관망"
    if score >= 8:
        # 눌림목이거나 구조가 매우 좋을 때
        if data['vix_ratio'] < 0.9 and not log.get('vvix_trap'):
            strategy_type = "Call Debit Spread (Bullish)"
        elif 0.9 <= data['vix_ratio'] < 1.0 and data['rsi2'] < 10:
             strategy_type = "Put Credit Spread (Neutral/Bullish)"
        else:
             strategy_type = "Put Credit Spread (Neutral)"
             
    target_delta = -0.10 # 기본값 (PCS 기준)

    return target_delta, verdict_text, profit_target, stop_loss, matrix_id, strategy_type

def calculate_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return -0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

def find_best_option(price, iv, target_delta):
    if target_delta is None: return None
    TARGET_DTE_MIN = 45
    SPREAD_WIDTH = 5
    
    qqq = yf.Ticker("QQQ")
    try:
        options = qqq.options
        valid_dates = []
        now = datetime.now()
        for d_str in options:
            d_date = datetime.strptime(d_str, "%Y-%m-%d")
            days_left = (d_date - now).days
            if days_left >= TARGET_DTE_MIN:
                valid_dates.append((d_str, days_left))
        
        if not valid_dates: return None
        expiry, dte = min(valid_dates, key=lambda x: x[1])
        
        T = dte / 365.0
        r = 0.045
        best_strike = 0
        min_diff = 1.0
        found_delta = 0
        
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

# === [4] 차트 시각화 (PART 4: 8단 동기화 차트 - The Radar) ===
def create_charts(data):
    hist = data['hist']
    
    # 8단 그리드 구성
    fig = plt.figure(figsize=(10, 22))
    gs = fig.add_gridspec(8, 1, height_ratios=[2, 0.6, 1, 1, 1, 1, 1, 1])
    
    # [1] Price (Master Axis)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(hist.index, hist['Close'], label='QQQ', color='black', alpha=0.7)
    ax1.plot(hist.index, hist['MA20'], label='20MA', color='green', ls='--', lw=1)
    ax1.plot(hist.index, hist['MA50'], label='50MA', color='blue', ls='-', lw=1.5)
    ax1.plot(hist.index, hist['MA200'], label='200MA', color='red', ls='-', lw=2)
    ax1.fill_between(hist.index, hist['BB_Upper'], hist['BB_Lower'], color='gray', alpha=0.1)
    ax1.set_title('QQQ Price Trend', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # [2] Volume (Sharex)
    ax_vol = fig.add_subplot(gs[1], sharex=ax1)
    colors = ['red' if c < o else 'green' for c, o in zip(hist['Close'], hist['Open'])]
    ax_vol.bar(hist.index, hist['Volume'], color=colors, alpha=0.5)
    ax_vol.plot(hist.index, hist['Vol_MA20'], color='black', lw=1)
    ax_vol.set_title(f"Volume", fontsize=10, fontweight='bold')
    ax_vol.grid(True, alpha=0.3)
    plt.setp(ax_vol.get_xticklabels(), visible=False)

    # [3] VIX Term Structure Ratio (Sharex)
    ax_ratio = fig.add_subplot(gs[2], sharex=ax1)
    ax_ratio.plot(hist.index, hist['VIX_Ratio'], color='black', lw=1.2, label='Ratio')
    ax_ratio.axhline(1.0, color='red', ls='--', alpha=0.8)
    ax_ratio.axhline(0.9, color='green', ls='--', alpha=0.8)
    ax_ratio.fill_between(hist.index, hist['VIX_Ratio'], 1.0, where=(hist['VIX_Ratio']>1.0), color='red', alpha=0.2, label='Backwardation')
    ax_ratio.fill_between(hist.index, hist['VIX_Ratio'], 0.9, where=(hist['VIX_Ratio']<0.9), color='green', alpha=0.2, label='Contango')
    ax_ratio.set_title('Structure (VIX/VIX3M)', fontsize=10, fontweight='bold')
    ax_ratio.grid(True, alpha=0.3)
    plt.setp(ax_ratio.get_xticklabels(), visible=False)

    # [4] VIX vs VVIX Divergence (New)
    ax_div = fig.add_subplot(gs[3], sharex=ax1)
    ax_div.plot(hist.index, hist['VIX'], color='purple', label='VIX', linewidth=1.5)
    ax_div.set_ylabel("VIX", color='purple')
    
    ax_vvix = ax_div.twinx()
    ax_vvix.plot(hist.index, hist['VVIX'], color='orange', linestyle='--', label='VVIX', linewidth=1.2)
    ax_vvix.set_ylabel("VVIX", color='orange')
    
    ax_div.set_title('VIX vs VVIX Divergence', fontsize=10, fontweight='bold')
    ax_div.grid(True, alpha=0.3)
    plt.setp(ax_div.get_xticklabels(), visible=False)

    # [5] RSI(14)
    ax_rsi = fig.add_subplot(gs[4], sharex=ax1)
    ax_rsi.plot(hist.index, hist['RSI'], color='purple')
    ax_rsi.axhline(70, color='red', ls='--')
    ax_rsi.axhline(30, color='green', ls='--')
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI (14)', fontsize=10, fontweight='bold')
    ax_rsi.grid(True, alpha=0.3)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)
    
    # [6] RSI(2) - Dip Buying (New)
    ax_rsi2 = fig.add_subplot(gs[5], sharex=ax1)
    ax_rsi2.plot(hist.index, hist['RSI_2'], color='blue', lw=1)
    ax_rsi2.axhline(10, color='green', ls='--')
    ax_rsi2.fill_between(hist.index, hist['RSI_2'], 10, where=(hist['RSI_2'] < 10), color='green', alpha=0.3)
    ax_rsi2.set_ylim(0, 100)
    ax_rsi2.set_title('RSI (2) - Dip Signal', fontsize=10, fontweight='bold')
    ax_rsi2.grid(True, alpha=0.3)
    plt.setp(ax_rsi2.get_xticklabels(), visible=False)

    # [7] MACD
    ax_macd = fig.add_subplot(gs[6], sharex=ax1)
    ax_macd.plot(hist.index, hist['MACD'], color='blue', lw=1)
    ax_macd.plot(hist.index, hist['Signal'], color='orange', lw=1)
    ax_macd.bar(hist.index, hist['MACD']-hist['Signal'], color='gray', alpha=0.3)
    ax_macd.axhline(0, color='black', lw=0.8)
    ax_macd.set_title('MACD', fontsize=10, fontweight='bold')
    ax_macd.grid(True, alpha=0.3)
    plt.setp(ax_macd.get_xticklabels(), visible=False)
    
    # [8] Capitulation Detector (New)
    ax_cap = fig.add_subplot(gs[7], sharex=ax1)
    vol_ratio = hist['Volume'] / hist['Vol_MA20']
    ax_cap.bar(hist.index, vol_ratio, color='gray', alpha=0.5)
    ax_cap.axhline(1.5, color='red', ls='--')
    
    # Highlight Capitulation Zones
    # 반복문 최소화를 위해 벡터 연산 사용 권장되나 가독성을 위해 순회
    for i in range(1, len(hist)):
        curr_ratio = hist['VIX_Ratio'].iloc[i]
        curr_vol = vol_ratio.iloc[i]
        prev_ratio = hist['VIX_Ratio'].iloc[i-1]
        prev_vol = vol_ratio.iloc[i-1]
        
        if (curr_ratio > 1.0 and curr_vol > 1.5) and (prev_ratio > 1.0 and prev_vol > 1.5):
            ax_cap.axvspan(hist.index[i-1], hist.index[i], color='yellow', alpha=0.5)
            
    ax_cap.set_title('Capitulation (2-Day Panic)', fontsize=10, fontweight='bold')
    ax_cap.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# === [메인 화면] ===
def main():
    st.title("🦅 HK Advisory (Grand Master v21.0)")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System: Institutional Grade (Conservative)")

    with st.spinner('시장 구조 및 변동성 정밀 분석 중...'):
        try:
            data = get_market_data()
            if data is None: return
            
            season, score, log = analyze_expert_logic(data)
            target_delta, verdict_text, profit_target, stop_loss, matrix_id, strat_type = determine_action(score, season, data, log)
            strategy = find_best_option(data['price'], data['iv'], target_delta)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            import traceback
            st.text(traceback.format_exc())
            return

    # [Sidebar Summary]
    st.sidebar.title("🛠️ Risk Dashboard")
    st.sidebar.markdown("---")
    
    # 주요 위험 지표
    ratio_val = data['vix_ratio']
    vvix_chg = (data['vvix'] - data['vvix_prev']) / data['vvix_prev'] * 100
    rsi2_val = data['rsi2']
    
    st.sidebar.metric("VIX Ratio", f"{ratio_val:.3f}", delta="Collapse" if ratio_val > 1.0 else "Stable", delta_color="inverse")
    st.sidebar.metric("VVIX Change", f"{vvix_chg:.1f}%", delta="Spike" if vvix_chg > 5 else "Normal", delta_color="inverse")
    st.sidebar.metric("RSI(2)", f"{rsi2_val:.1f}", delta="Dip Buy" if rsi2_val < 10 else "Neutral")
    
    if log.get('capitulation'):
        st.sidebar.error("🚨 투매 신호 감지!")
    if log.get('vvix_trap'):
        st.sidebar.error("🪤 VVIX 함정 감지!")

    # 스타일 헬퍼
    def hl_score(category, row_state, col_season):
        base = "style='border: 1px solid #ddd; padding: 8px; color: black; background-color: white;'"
        current_val = log.get(category, '')
        is_match = False
        
        # 특수 케이스 처리
        if category == 'rsi' and row_state == 'escape' and 'escape' in str(current_val): is_match = True
        elif str(current_val) == str(row_state): is_match = True
        elif category == 'capitulation' and row_state == 'True' and current_val: is_match = True
        elif category == 'vvix_trap' and row_state == 'True' and current_val: is_match = True
        
        if is_match and (season == col_season or col_season == 'ALL'):
            return "style='border: 3px solid #FF5722; background-color: #FFF8E1; font-weight: bold; color: #D84315; padding: 8px;'"
        return base

    td_style = "style='border: 1px solid #ddd; padding: 8px; color: black; background-color: white;'"
    th_style = "style='border: 1px solid #ddd; padding: 8px; color: black; background-color: #f2f2f2;'"

    # 1. Scorecard (HTML)
    html_score_list = [
        "<h3>1. Expert Matrix Scorecard (Conservative)</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; text-align: center;'>",
        "<tr>",
        f"<th {th_style}>지표</th><th {th_style}>상태</th>",
        f"<th {th_style}>점수</th><th {th_style}>비고</th>",
        "</tr>",
        
        # VIX Term Structure
        f"<tr><td rowspan='3' {td_style}><b>VIX Term</b></td>",
        f"<td {td_style}>Collapse (>1.0)</td>",
        f"<td {hl_score('term', 'backwardation', 'ALL')}>-20 (Block)</td><td {td_style}>시스템 중단</td></tr>",
        f"<tr><td {td_style}>Neutral</td>",
        f"<td {hl_score('term', 'neutral', 'ALL')}>0</td><td {td_style}>-</td></tr>",
        f"<tr><td {td_style}>Contango (<0.9)</td>",
        f"<td {hl_score('term', 'contango', 'ALL')}>+3</td><td {td_style}>기회</td></tr>",
        
        # Capitulation (New)
        f"<tr><td {td_style}><b>투매 신호</b></td>",
        f"<td {td_style}>2일 연속 공포+투매</td>",
        f"<td {hl_score('capitulation', 'True', 'ALL')}>+15</td><td {td_style}>Sniper Mode</td></tr>",
        
        # VVIX Trap (New)
        f"<tr><td {td_style}><b>VVIX Trap</b></td>",
        f"<td {td_style}>VIX 안정+VVIX 급등</td>",
        f"<td {hl_score('vvix_trap', 'True', 'ALL')}>-5 (Block)</td><td {td_style}>숨겨진 위험</td></tr>",
        
        # RSI(14)
        f"<tr><td rowspan='3' {td_style}>RSI(14)</td>",
        f"<td {td_style}>과열 (>70)</td>",
        f"<td {hl_score('rsi', 'over', season)}>-3 / -10(W)</td><td {td_style}>겨울철 금지</td></tr>",
        f"<tr><td {td_style}>과매도 (<30)</td>",
        f"<td {hl_score('rsi', 'under', season)}>+4 / 0(W)</td><td {td_style}>바닥 확인 필요</td></tr>",
        f"<tr><td {td_style}>탈출 (1-3일)</td>",
        f"<td {hl_score('rsi', 'escape', 'ALL')}>+5</td><td {td_style}>골든 타임</td></tr>",
        
        # RSI(2) (New)
        f"<tr><td {td_style}>RSI(2)</td>",
        f"<td {td_style}>눌림목 (<10)</td>",
        f"<td {hl_score('rsi2', 'dip_buy', 'ALL')}>+3</td><td {td_style}>구조 안정 시</td></tr>",
        
        # Season
        f"<tr><td {td_style}>계절</td>",
        f"<td {td_style}>WINTER</td>",
        f"<td {hl_score('season', 'winter_penalty', 'winter_penalty')}>-5 (Penalty)</td><td {td_style}>역추세 방지</td></tr>",
        
        "</table>"
    ]
    st.markdown("".join(html_score_list), unsafe_allow_html=True)

    # 2. Final Verdict
    def get_matrix_style(current_id, row_id, bg_color):
        if current_id == row_id:
            return f"style='background-color: {bg_color}; border: 3px solid #666; font-weight: bold; color: #333; height: 50px;'"
        else:
            return "style='background-color: white; border: 1px solid #eee; color: #999;'"

    html_verdict_list = [
        f"<h3>2. Final Verdict: <span style='color:blue;'>{score}점</span> - {strat_type}</h3>",
        "<div style='border: 2px solid #ccc; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; text-align: center;'>",
        f"<tr style='background-color: #333; color: white;'>",
        f"<th {th_style} style='color:white;'>점수 구간</th>",
        f"<th {th_style} style='color:white;'>판정</th>",
        f"<th {th_style} style='color:white;'>전략</th>",
        f"<th {th_style} style='color:white;'>익절/손절</th>",
        "</tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'panic', '#ffebee')}>",
        "<td>위험 감지</td><td>⛔ 매매 중단</td><td>관망</td><td>-</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'strong', '#dff0d8')}>",
        "<td>15점 이상</td><td>💎 Strong</td><td>Aggressive</td><td>+100% / -300%</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'standard', '#ffffff')}>",
        "<td>12 ~ 14점</td><td>✅ Standard</td><td>Balanced</td><td>+50% / -200%</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'weak', '#fff9c4')}>",
        "<td>8 ~ 11점</td><td>⚠️ Hit & Run</td><td>Conservative</td><td>+30% / -150%</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'no_entry', '#f2dede')}>",
        "<td>8점 미만</td><td>🛡️ No Entry</td><td>관망</td><td>-</td></tr>",
        
        "</table></div>"
    ]
    st.markdown("".join(html_verdict_list), unsafe_allow_html=True)

    # 3. Action Plan (Manual)
    if strategy and matrix_id != 'no_entry' and matrix_id != 'panic':
        st.info(f"💡 추천 전략: **{strat_type}** | 만기: {strategy['expiry']} (DTE {strategy['dte']}) | Strike: {strategy['short']}/{strategy['long']}")
    elif matrix_id == 'panic':
        st.error(f"⛔ 경고: {verdict_text} - 현재 시장은 진입하기에 너무 위험합니다.")
    else:
        st.warning("🛡️ 관망: 현재 점수가 진입 기준(8점)에 미치지 못합니다.")

    st.markdown("---")
    st.subheader("📈 The Radar (8-Sync Chart)")
    st.pyplot(create_charts(data))

if __name__ == "__main__":
    main()
   
