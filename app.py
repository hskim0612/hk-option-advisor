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
    gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean() # EMA 적용
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
    
    # 2. 각 데이터프레임 전처리 (Timezone 제거 및 정규화)
    def prep_df(df):
        if df is None or df.empty: return pd.DataFrame()
        df = df.copy()
        df.index = df.index.tz_localize(None).normalize()
        # 중복 인덱스 제거 (가끔 야간 선물 데이터 등 혼입 시)
        df = df[~df.index.duplicated(keep='last')]
        return df

    vix_clean = prep_df(vix_hist)
    vix3m_clean = prep_df(vix3m_hist)
    vvix_clean = prep_df(vvix_hist)
    
    # 3. 병합 (NaN 제거 - Inner Join 효과)
    # QQQ는 전체 컬럼 유지, 나머지는 Close만 가져옴
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
        'rsi2': curr['RSI_2'], 'rsi2_prev': prev['RSI_2'], # RSI 2 추가
        'bb_upper': curr['BB_Upper'], 'bb_lower': curr['BB_Lower'], 'bb_lower_prev': prev['BB_Lower'],
        'macd': curr['MACD'], 'signal': curr['Signal'],
        'macd_prev': prev['MACD'], 'signal_prev': prev['Signal'],
        'volume': curr['Volume'], 'vol_ma20': curr['Vol_MA20'],
        'vix': curr['VIX'], 'vix_prev': prev['VIX'],
        'vix3m': curr['VIX3M'],
        'vvix': curr['VVIX'], 'vvix_prev': prev['VVIX'], # VVIX 추가
        'vix_ratio': curr['VIX_Ratio'],
        'iv': current_iv,
        'hist': df, # 통합된 히스토리
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
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    return False

if not check_password():
    st.stop()

# === [1] 데이터 수집 (수정: 인덱스 정규화 및 안전 병합) ===
@st.cache_data(ttl=1800)
def get_market_data():
    # 1. QQQ 데이터
    qqq = yf.Ticker("QQQ")
    hist = qqq.history(period="2y")
    
    # 이동평균선 및 보조지표
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    
    # 볼린저 밴드
    hist['BB_Mid'] = hist['MA20']
    hist['BB_Std'] = hist['Close'].rolling(window=20).std()
    hist['BB_Upper'] = hist['BB_Mid'] + (hist['BB_Std'] * 2)
    hist['BB_Lower'] = hist['BB_Mid'] - (hist['BB_Std'] * 2)
    
    # MACD
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    hist['Vol_MA20'] = hist['Volume'].rolling(window=20).mean()
    
    # 2. VIX & VIX3M 데이터 처리 (핵심 수정 구간)
    vix_ticker = yf.Ticker("^VIX")
    vix_hist = vix_ticker.history(period="1y")
    
    vix3m_val = None
    vix3m_hist = None
    vix_term_df = None  # 초기화

    try:
        vix3m_ticker = yf.Ticker("^VIX3M")
        vix3m_hist = vix3m_ticker.history(period="1y")
        
        if not vix3m_hist.empty and not vix_hist.empty:
            vix3m_val = vix3m_hist['Close'].iloc[-1]
            
            # [CRITICAL FIX] Timezone 제거 및 날짜 정규화
            # df.copy()를 사용하여 원본 보존
            df_vix = vix_hist[['Close']].copy()
            df_vix3m = vix3m_hist[['Close']].copy()
            
            # Timezone 정보를 날리고(naive), 시간(00:00:00)으로 정규화
            df_vix.index = df_vix.index.tz_localize(None).normalize()
            df_vix3m.index = df_vix3m.index.tz_localize(None).normalize()
            
            # pd.merge 사용 (Inner Join)
            merged_df = pd.merge(
                df_vix, 
                df_vix3m, 
                left_index=True, 
                right_index=True, 
                suffixes=('_VIX', '_VIX3M')
            )
            
            # 데이터 개수 검증 (30일 이상일 때만 유효)
            if len(merged_df) >= 30:
                merged_df['Ratio'] = merged_df['Close_VIX'] / merged_df['Close_VIX3M']
                vix_term_df = merged_df
            else:
                # 데이터가 너무 적음
                vix_term_df = None

    except Exception as e:
        # 에러 발생 시 None 유지 (앱 중단 방지)
        vix3m_val = None
        vix_term_df = None
        print(f"Error fetching VIX3M: {e}")
    
    # 현재 상태값
    curr = hist.iloc[-1]
    prev = hist.iloc[-2]
    curr_vix = vix_hist['Close'].iloc[-1]
    prev_vix = vix_hist['Close'].iloc[-2]
    
    vol_pct = (curr['Volume'] / curr['Vol_MA20']) * 100

    # IV (Implied Volatility)
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
        'vix3m': vix3m_val,
        'iv': current_iv,
        'hist': hist, 'vix_hist': vix_hist, 'vix3m_hist': vix3m_hist,
        'vix_term_df': vix_term_df
    }

# === [2] 전문가 로직 ===
def analyze_expert_logic(d):
    if d['price'] > d['ma50'] and d['price'] > d['ma200']: season = "SUMMER"
    elif d['price'] < d['ma50'] and d['price'] > d['ma200']: season = "AUTUMN"
    elif d['price'] < d['ma50'] and d['price'] < d['ma200']: season = "WINTER"
    else: season = "SPRING"
    
    score = 0
    log = {}
    
    # 1. VIX Term Structure Logic
    vix_ratio = 1.0
    if d['vix3m'] and d['vix3m'] > 0:
        vix_ratio = d['vix'] / d['vix3m']
    
    if vix_ratio > 1.0:
        pts = -10
        score += pts
        log['term'] = 'backwardation'
    elif vix_ratio < 0.9:
        pts = 3
        score += pts
        log['term'] = 'contango'
    else:
        pts = 0
        score += pts
        log['term'] = 'normal'
    
    log['vix_ratio'] = vix_ratio

    # 2. RSI Logic
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

    # 3. VIX Level Logic
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

    # 4. Bollinger Logic
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

    # 5. Trend Logic
    if d['price'] > d['ma20']:
        pts = 3 if season == "WINTER" or season == "SPRING" else 2
        score += pts
        log['trend'] = 'up'
    else:
        log['trend'] = 'down'

    # 6. Volume Logic
    if d['volume'] > d['vol_ma20'] * 1.5:
        pts = 3 if season == "WINTER" or season == "AUTUMN" else 2
        score += pts
        log['vol'] = 'explode'
    else:
        log['vol'] = 'normal'

    # 7. MACD Logic
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

# === [3] 전략 탐색 및 행동 결정 ===
def determine_action(score, season, data, log):
    vix_pct_change = ((data['vix'] - data['vix_prev']) / data['vix_prev']) * 100
    TARGET_DELTA = -0.10
    
    # [PRIORITY 0] Backwardation Check
    if log.get('term') == 'backwardation':
        return TARGET_DELTA, "⛔ 매매 중단 (System Collapse)", "-", "-", "panic"

    # [PRIORITY 1] Panic Condition
    if vix_pct_change > 15.0:
        return TARGET_DELTA, "⛔ 매매 중단 (VIX 급등)", "-", "-", "panic"
    
    # 2. Strong
    if score >= 12:
        return TARGET_DELTA, "💎 추세 추종 (Strong)", "75%", "300%", "strong"
    # 3. Standard
    elif 8 <= score < 12:
        return TARGET_DELTA, "✅ 표준 대응 (Standard)", "50%", "200%", "standard"
    # 4. Hit & Run
    elif 5 <= score < 8:
        return TARGET_DELTA, "⚠️ 속전 속결 (Hit & Run)", "30%", "150%", "weak"
    # 5. No Entry
    else:
        return None, "🛡️ 진입 보류", "-", "-", "no_entry"

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

# === [4] 차트 (수정: 에러 핸들링 및 시각화 개선) ===
def create_charts(data):
    hist = data['hist']
    fig = plt.figure(figsize=(10, 18))
    gs = fig.add_gridspec(6, 1, height_ratios=[2, 0.6, 1, 1, 1, 1])
    
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
    ax2.set_title('MACD', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # 5. VIX Level
    ax3 = fig.add_subplot(gs[4], sharex=ax1)
    ax3.plot(data['vix_hist'].index, data['vix_hist']['Close'], color='purple', label='VIX (Spot)')
    if data['vix3m_hist'] is not None and not data['vix3m_hist'].empty:
         ax3.plot(data['vix3m_hist'].index, data['vix3m_hist']['Close'], color='gray', ls=':', label='VIX3M (Future)')
    
    ax3.axhline(30, color='red', ls='--')
    ax3.axhline(20, color='green', ls='--')
    ax3.set_title('VIX Level (Absolute)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # 6. [NEW] VIX Term Structure Ratio (수정)
    ax4 = fig.add_subplot(gs[5], sharex=ax1)
    term_data = data.get('vix_term_df')
    
    if term_data is not None and not term_data.empty:
        # Ratio Line
        ax4.plot(term_data.index, term_data['Ratio'], color='black', lw=1.2, label='Ratio (VIX/VIX3M)')
        
        # Guidelines
        ax4.axhline(1.0, color='red', ls='--', alpha=0.8, lw=1)
        ax4.axhline(0.9, color='green', ls='--', alpha=0.8, lw=1)
        
        # Fill Areas (Explicitly handling index)
        # Danger Zone
        ax4.fill_between(term_data.index, term_data['Ratio'], 1.0, 
                         where=(term_data['Ratio'] > 1.0), 
                         color='red', alpha=0.2, label='Backwardation')
        # Opportunity Zone
        ax4.fill_between(term_data.index, term_data['Ratio'], 0.9, 
                         where=(term_data['Ratio'] < 0.9), 
                         color='green', alpha=0.2, label='Contango')
        
        ax4.legend(loc='upper right')
    else:
        # 데이터 부족 시 메시지 표시
        ax4.text(0.5, 0.5, "데이터 부족: Ratio 그래프를 그릴 수 없습니다.\n(VIX/VIX3M 병합 실패)", 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=ax4.transAxes, fontsize=12, color='red')
        
    ax4.set_title('Structure of Volatility (Ratio = VIX / VIX3M)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# === [메인 화면] ===
def main():
    st.title("🦅 HK Advisory (Grand Master v20.0)")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System: Institutional Grade")

    with st.spinner('시장 구조 및 변동성 정밀 분석 중...'):
        try:
            data = get_market_data()
            season, score, log = analyze_expert_logic(data)
            target_delta, verdict_text, profit_target, stop_loss, matrix_id = determine_action(score, season, data, log)
            strategy = find_best_option(data['price'], data['iv'], target_delta)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            return

    # [NEW] Sidebar Debugging Panel (수정)
    st.sidebar.title("🛠️ 시스템 상태")
    st.sidebar.markdown("---")
    
    # 1. 데이터 카운트
    vix_count = len(data['vix_hist']) if not data['vix_hist'].empty else 0
    vix3m_count = len(data['vix3m_hist']) if data['vix3m_hist'] is not None and not data['vix3m_hist'].empty else 0
    
    term_df = data.get('vix_term_df')
    ratio_count = len(term_df) if term_df is not None else 0
    
    st.sidebar.metric("VIX Raw Data", f"{vix_count} rows")
    st.sidebar.metric("VIX3M Raw Data", f"{vix3m_count} rows")
    
    # Ratio 데이터 상태에 따른 색상 표시
    if ratio_count > 0:
        st.sidebar.success(f"Ratio Merged: {ratio_count} rows")
        curr_ratio = term_df['Ratio'].iloc[-1]
        st.sidebar.metric("Current Ratio", f"{curr_ratio:.4f}")
    else:
        st.sidebar.error("Ratio Merged: 0 rows (Error)")
        st.sidebar.warning("체크 포인트: 날짜 형식 불일치 또는 데이터 부족")

    # 스타일 헬퍼
    def hl_score(category, row_state, col_season):
        base = "style='border: 1px solid #ddd; padding: 8px; color: black; background-color: white;'"
        current_val = log.get(category, '')
        is_match = False
        if category == 'rsi' and row_state == 'escape':
            if 'escape' in current_val: is_match = True
        else:
            if current_val == row_state: is_match = True
        
        if is_match and (season == col_season or col_season == 'ALL'):
            return "style='border: 3px solid #FF5722; background-color: #FFF8E1; font-weight: bold; color: #D84315; padding: 8px;'"
        return base

    def hl_season(row_season):
        if season == row_season:
            return "style='border: 3px solid #2196F3; background-color: #E3F2FD; font-weight: bold; color: black; padding: 8px;'"
        return "style='border: 1px solid #ddd; padding: 8px; color: black; background-color: white;'"

    td_style = "style='border: 1px solid #ddd; padding: 8px; color: black; background-color: white;'"
    th_style = "style='border: 1px solid #ddd; padding: 8px; color: black; background-color: #f2f2f2;'"
    
    vix_ratio_disp = f"{log.get('vix_ratio', 0):.2f}"

    # 1. Season Matrix
    html_season_list = [
        "<h3>1. Market Season Matrix</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; text-align: center;'>",
        "<tr>",
        f"<th {th_style}>Season</th><th {th_style}>Condition</th><th {th_style}>Character</th>",
        "</tr>",
        f"<tr><td {hl_season('SUMMER')}>☀️ SUMMER</td><td {hl_season('SUMMER')}>Price > 50MA & 200MA</td><td {hl_season('SUMMER')}>강세장</td></tr>",
        f"<tr><td {hl_season('AUTUMN')}>🍂 AUTUMN</td><td {hl_season('AUTUMN')}>Price < 50MA but > 200MA</td><td {hl_season('AUTUMN')}>조정기</td></tr>",
        f"<tr><td {hl_season('WINTER')}>❄️ WINTER</td><td {hl_season('WINTER')}>Price < 50MA & 200MA</td><td {hl_season('WINTER')}>약세장</td></tr>",
        f"<tr><td {hl_season('SPRING')}>🌱 SPRING</td><td {hl_season('SPRING')}>Price > 50MA but < 200MA</td><td {hl_season('SPRING')}>회복기</td></tr>",
        "</table>",
        f"<p>※ QQQ: <b>${data['price']:.2f}</b> (Vol: {data['vol_pct']:.1f}% of 20MA)</p>"
    ]
    st.markdown("".join(html_season_list), unsafe_allow_html=True)

    # 2. Scorecard
    html_score_list = [
        "<h3>2. Expert Matrix Scorecard</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; text-align: center;'>",
        "<tr>",
        f"<th {th_style}>지표</th><th {th_style}>상태</th>",
        f"<th {th_style}>☀️</th><th {th_style}>🍂</th><th {th_style}>❄️</th><th {th_style}>🌱</th>",
        f"<th {th_style}>Logic</th>",
        "</tr>",
        
        # VIX Term Structure Row (Universal)
        f"<tr><td rowspan='3' {td_style}><b>VIX Term</b><br><span style='font-size:11px; color:blue;'>Ratio: {vix_ratio_disp}</span></td>",
        f"<td {td_style}><b>Easy Money</b><br>(Contango &lt;0.9)</td>",
        f"<td colspan='4' {hl_score('term', 'contango', 'ALL')}>+3 (Universal)</td>",
        f"<td align='left' {td_style}><b>Green Light</b></td></tr>",
        
        f"<tr><td {td_style}>Normal<br>(0.9 ~ 1.0)</td>",
        f"<td colspan='4' {hl_score('term', 'normal', 'ALL')}>0</td>",
        f"<td align='left' {td_style}>-</td></tr>",
        
        f"<tr><td {td_style}><b>Collapse</b><br>(Backwardation &gt;1.0)</td>",
        f"<td colspan='4' {hl_score('term', 'backwardation', 'ALL')}><b>-10 (Block)</b></td>",
        f"<td align='left' {td_style}><b style='color:red;'>🚨 붕괴 경보</b></td></tr>",
        
        # RSI
        f"<tr><td rowspan='4' {td_style}>RSI<br><span style='font-size:11px; color:#888; font-weight:normal'>지금 싼가? 비싼가?</span></td>",
        f"<td {td_style}>과열 (>70)</td>",
        f"<td {hl_score('rsi', 'over', 'SUMMER')}>-1</td><td {hl_score('rsi', 'over', 'AUTUMN')}>-3</td><td {hl_score('rsi', 'over', 'WINTER')}>-5</td><td {hl_score('rsi', 'over', 'SPRING')}>-2</td>",
        f"<td align='left' {td_style}>가짜 반등</td></tr>",
        
        f"<tr><td {td_style}>중립 (45-65)</td>",
        f"<td {hl_score('rsi', 'neutral', 'SUMMER')}>+1</td><td {hl_score('rsi', 'neutral', 'AUTUMN')}>0</td><td {hl_score('rsi', 'neutral', 'WINTER')}>-1</td><td {hl_score('rsi', 'neutral', 'SPRING')}>+1</td>",
        f"<td align='left' {td_style}>-</td></tr>",
        
        f"<tr><td {td_style}>과매도 (<30)</td>",
        f"<td {hl_score('rsi', 'under', 'SUMMER')}>+5</td><td {hl_score('rsi', 'under', 'AUTUMN')}>+4</td><td {hl_score('rsi', 'under', 'WINTER')}>0</td><td {hl_score('rsi', 'under', 'SPRING')}>+4</td>",
        f"<td align='left' {td_style}>겨울 바닥 X</td></tr>",
        
        f"<tr><td {td_style}>🚀 탈출 (1~7일)</td>",
        f"<td {hl_score('rsi', 'escape', 'SUMMER')}>3~5</td><td {hl_score('rsi', 'escape', 'AUTUMN')}>3~5</td><td {hl_score('rsi', 'escape', 'WINTER')}>3~5</td><td {hl_score('rsi', 'escape', 'SPRING')}>3~5</td>",
        f"<td align='left' {td_style}><b>Best Timing</b></td></tr>",
        
        # VIX
        f"<tr><td rowspan='4' {td_style}>VIX (Level)</td>",
        f"<td {td_style}>안정 (<20)</td>",
        f"<td {hl_score('vix', 'stable', 'SUMMER')}>+2</td><td {hl_score('vix', 'stable', 'AUTUMN')}>0</td><td {hl_score('vix', 'stable', 'WINTER')}>-2</td><td {hl_score('vix', 'stable', 'SPRING')}>+1</td>",
        f"<td align='left' {td_style}>저변동성</td></tr>",
        
        f"<tr><td {td_style}>공포 (20-35)</td>",
        f"<td {hl_score('vix', 'fear', 'SUMMER')}>-3</td><td {hl_score('vix', 'fear', 'AUTUMN')}>-4</td><td {hl_score('vix', 'fear', 'WINTER')}>+2</td><td {hl_score('vix', 'fear', 'SPRING')}>-1</td>",
        f"<td align='left' {td_style}>기회 탐색</td></tr>",
        
        f"<tr><td {td_style}>패닉 상승</td>",
        f"<td {hl_score('vix', 'panic_rise', 'SUMMER')}>-5</td><td {hl_score('vix', 'panic_rise', 'AUTUMN')}>-6</td><td {hl_score('vix', 'panic_rise', 'WINTER')}>-5</td><td {hl_score('vix', 'panic_rise', 'SPRING')}>-4</td>",
        f"<td align='left' {td_style}>칼날</td></tr>",
        
        f"<tr><td {td_style}>📉 꺾임</td>",
        f"<td {hl_score('vix', 'peak_out', 'SUMMER')}>-</td><td {hl_score('vix', 'peak_out', 'AUTUMN')}>-</td><td {hl_score('vix', 'peak_out', 'WINTER')}>+7</td><td {hl_score('vix', 'peak_out', 'SPRING')}>-</td>",
        f"<td align='left' {td_style}><b>Sniper</b></td></tr>",
        
        # Bollinger
        f"<tr><td rowspan='3' {td_style}>BB</td>",
        f"<td {td_style}>밴드 내부</td>",
        f"<td {hl_score('bb', 'in', 'SUMMER')}>0</td><td {hl_score('bb', 'in', 'AUTUMN')}>0</td><td {hl_score('bb', 'in', 'WINTER')}>0</td><td {hl_score('bb', 'in', 'SPRING')}>0</td>",
        f"<td align='left' {td_style}>대기</td></tr>",
        
        f"<tr><td {td_style}>하단 이탈</td>",
        f"<td {hl_score('bb', 'out', 'SUMMER')}>+3</td><td {hl_score('bb', 'out', 'AUTUMN')}>+2</td><td {hl_score('bb', 'out', 'WINTER')}>-2</td><td {hl_score('bb', 'out', 'SPRING')}>+1</td>",
        f"<td align='left' {td_style}>가속화</td></tr>",
        
        f"<tr><td {td_style}>↩️ 복귀</td>",
        f"<td {hl_score('bb', 'return', 'SUMMER')}>+4</td><td {hl_score('bb', 'return', 'AUTUMN')}>+3</td><td {hl_score('bb', 'return', 'WINTER')}>+5</td><td {hl_score('bb', 'return', 'SPRING')}>+4</td>",
        f"<td align='left' {td_style}><b>Close In</b></td></tr>",
        
        # Trend
        f"<tr><td {td_style}>추세 (20MA)<br><span style='font-size:11px; color:#888; font-weight:normal'>지금 당장의 추세모습</span></td><td {td_style}>20일선 위</td>",
        f"<td {hl_score('trend', 'up', 'SUMMER')}>+2</td><td {hl_score('trend', 'up', 'AUTUMN')}>+2</td><td {hl_score('trend', 'up', 'WINTER')}>+3</td><td {hl_score('trend', 'up', 'SPRING')}>+3</td>",
        f"<td align='left' {td_style}>회복</td></tr>",
        
        # Volume
        f"<tr><td {td_style}>거래량</td><td {td_style}>폭증 (>150%)</td>",
        f"<td {hl_score('vol', 'explode', 'SUMMER')}>+2</td><td {hl_score('vol', 'explode', 'AUTUMN')}>+3</td><td {hl_score('vol', 'explode', 'WINTER')}>+3</td><td {hl_score('vol', 'explode', 'SPRING')}>+2</td>",
        f"<td align='left' {td_style}><b>손바뀜</b></td></tr>",
        
        f"<tr><td {td_style}>거래량</td><td {td_style}>일반</td>",
        f"<td {hl_score('vol', 'normal', 'SUMMER')}>0</td><td {hl_score('vol', 'normal', 'AUTUMN')}>0</td><td {hl_score('vol', 'normal', 'WINTER')}>0</td><td {hl_score('vol', 'normal', 'SPRING')}>0</td>",
        f"<td align='left' {td_style}>-</td></tr>",
        
        # MACD
        f"<tr><td rowspan='4' {td_style}>MACD<br><span style='font-size:11px; color:#888; font-weight:normal'>상승장? 하락장?<br>(방향을 이끄는 힘)</span></td>",
        f"<td {td_style}>📈 상승 전환<br>(골든크로스)</td>",
        f"<td {hl_score('macd', 'break_up', 'SUMMER')}>+3</td><td {hl_score('macd', 'break_up', 'AUTUMN')}>+3</td><td {hl_score('macd', 'break_up', 'WINTER')}>+3</td><td {hl_score('macd', 'break_up', 'SPRING')}>+3</td>",
        f"<td align='left' {td_style}><b>강력 매수</b></td></tr>",
        
        f"<tr><td {td_style}>☁️ 상승 추세<br>(에너지 강)</td>",
        f"<td {hl_score('macd', 'above', 'SUMMER')}>+1</td><td {hl_score('macd', 'above', 'AUTUMN')}>+1</td><td {hl_score('macd', 'above', 'WINTER')}>+1</td><td {hl_score('macd', 'above', 'SPRING')}>+1</td>",
        f"<td align='left' {td_style}>순풍</td></tr>",
        
        f"<tr><td {td_style}>📉 하락 전환<br>(데드크로스)</td>",
        f"<td {hl_score('macd', 'break_down', 'SUMMER')}>-3</td><td {hl_score('macd', 'break_down', 'AUTUMN')}>-3</td><td {hl_score('macd', 'break_down', 'WINTER')}>-3</td><td {hl_score('macd', 'break_down', 'SPRING')}>-3</td>",
        f"<td align='left' {td_style}><b>강력 매도</b></td></tr>",
        
        f"<tr><td {td_style}>☔ 하락 추세<br>(에너지 약)</td>",
        f"<td {hl_score('macd', 'below', 'SUMMER')}>-1</td><td {hl_score('macd', 'below', 'AUTUMN')}>-1</td><td {hl_score('macd', 'below', 'WINTER')}>-1</td><td {hl_score('macd', 'below', 'SPRING')}>-1</td>",
        f"<td align='left' {td_style}>역풍</td></tr>",
        
        "</table>"
    ]
    st.markdown("".join(html_score_list), unsafe_allow_html=True)

    # 3. Final Verdict
    def get_matrix_style(current_id, row_id, bg_color):
        if current_id == row_id:
            return f"style='background-color: {bg_color}; border: 3px solid #666; font-weight: bold; color: #333; height: 50px;'"
        else:
            return "style='background-color: white; border: 1px solid #eee; color: #999;'"

    html_verdict_list = [
        f"<h3>3. Final Verdict: <span style='color:blue;'>{score}점</span> - Dynamic Exit Matrix</h3>",
        "<div style='border: 2px solid #ccc; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; text-align: center;'>",
        f"<tr style='background-color: #333; color: white;'>",
        f"<th {th_style} style='color:white;'>점수 구간</th>",
        f"<th {th_style} style='color:white;'>최종 판정</th>",
        f"<th {th_style} style='color:white;'>🎯 익절 목표</th>",
        f"<th {th_style} style='color:white;'>🛑 손절 라인</th>",
        "</tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'panic', '#ffebee')}>",
        "<td>VIX 급등 / 구조 붕괴</td><td>⛔ 매매 중단 (System Collapse)</td><td>-</td><td>-</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'strong', '#dff0d8')}>",
        "<td>12점 이상</td><td>💎 추세 추종 (Strong)</td><td style='color:green;'>+75%</td><td style='color:red;'>-300% (원금 3배)</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'standard', '#ffffff')}>",
        "<td>8 ~ 11점</td><td>✅ 표준 대응 (Standard)</td><td style='color:green;'>+50%</td><td style='color:red;'>-200% (원금 3배)</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'weak', '#fff9c4')}>",
        "<td>5 ~ 7점</td><td>⚠️ 속전 속결 (Hit & Run)</td><td style='color:green;'>+30%</td><td style='color:red;'>-150% (원금 2.5배)</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'no_entry', '#f2dede')}>",
        "<td>5점 미만</td><td>🛡️ 진입 보류 (No Entry)</td><td>-</td><td>-</td></tr>",
        
        "</table>",
        "<div style='padding: 10px; background-color: #f9f9f9; text-align: center; color: #555; font-size: 13px;'>",
        "※ <b>설정:</b> Delta -0.10 (Fixed) / DTE 45일 / Spread $5<br>",
        "※ 손절 라인은 프리미엄 가격 기준입니다. (예: $1.0 진입 시, 200% 손절은 $3.0 도달 시 청산)",
        "</div></div>"
    ]
    st.markdown("".join(html_verdict_list), unsafe_allow_html=True)

    # 4. Manual / Warning (테이블 적용)
    if strategy and matrix_id != 'no_entry' and matrix_id != 'panic':
        html_manual_list = [
            "<div style='border: 2px solid #2196F3; padding: 15px; margin-top: 20px; border-radius: 10px; background-color: #ffffff; color: black;'>",
            "<h3 style='color: #2196F3; margin-top: 0;'>👮‍♂️ 주문 상세 매뉴얼 (Action Plan)</h3>",
            
            # --- Table Start ---
            "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; text-align: center; font-size: 13px; margin-bottom: 15px;'>",
            
            # Header
            "<tr style='background-color: #e3f2fd; border: 1px solid #ddd;'>",
            "<th style='padding: 8px; border: 1px solid #ddd;'>구분</th>",
            "<th style='padding: 8px; border: 1px solid #ddd;'>행동</th>",
            "<th style='padding: 8px; border: 1px solid #ddd;'>시간</th>",
            "<th style='padding: 8px; border: 1px solid #ddd;'>방식</th>",
            "</tr>",
            
            # Row 1: Entry
            "<tr>",
            "<td style='padding: 8px; border: 1px solid #ddd; font-weight:bold;'>진입 (Entry)</td>",
            "<td style='padding: 8px; border: 1px solid #ddd;'>신규 포지션 구축</td>",
            "<td style='padding: 8px; border: 1px solid #ddd;'>🕒 <b>마감 30분 전</b><br><span style='font-size:11px; color:#666;'>(한국 아침 05:30)</span></td>",
            "<td style='padding: 8px; border: 1px solid #ddd;'><b>수동 진입</b><br><span style='font-size:11px; color:#666;'>(앱 점수 확인 후)</span></td>",
            "</tr>",
            
            # Row 2: Loss
            "<tr>",
            "<td style='padding: 8px; border: 1px solid #ddd; font-weight:bold; color:red;'>손절 (Loss)</td>",
            "<td style='padding: 8px; border: 1px solid #ddd;'>위기 탈출</td>",
            "<td style='padding: 8px; border: 1px solid #ddd;'>🚨 <b>언제든지</b><br><span style='font-size:11px; color:#666;'>(장중 내내)</span></td>",
            "<td style='padding: 8px; border: 1px solid #ddd;'><b>자동 감시 주문</b><br><span style='font-size:11px; color:#666;'>(진입 즉시 세팅)</span></td>",
            "</tr>",
            
            # Row 3: Win
            "<tr>",
            "<td style='padding: 8px; border: 1px solid #ddd; font-weight:bold; color:green;'>익절 (Win)</td>",
            "<td style='padding: 8px; border: 1px solid #ddd;'>수익 실현</td>",
            "<td style='padding: 8px; border: 1px solid #ddd;'>💰 <b>장중 아무 때나</b><br><span style='font-size:11px; color:#666;'>(목표가 도달 시)</span></td>",
            "<td style='padding: 8px; border: 1px solid #ddd;'><b>GTC 지정가 주문</b><br><span style='font-size:11px; color:#666;'>(미리 걸어두기)</span></td>",
            "</tr>",
            "</table>",
            
            # --- Summary Text ---
            "<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px; font-size: 14px;'>",
            f"<b>✅ 현재 포지션 목표 (Spec):</b><br>",
            f"• <b>종목:</b> QQQ Put Credit Spread (만기 {strategy['expiry']}, DTE {strategy['dte']}일)<br>",
            f"• <b>Strike:</b> Short ${strategy['short']} / Long ${strategy['long']} (Width ${strategy['width']})<br>",
            "<hr style='margin: 8px 0; border: 0; border-top: 1px solid #ddd;'>",
            f"• <b>익절 (Target):</b> 진입가 대비 <b style='color:green;'>{profit_target}</b> 도달 시<br>",
            f"• <b>손절 (Stop):</b> 진입가 대비 <b style='color:red;'>{stop_loss}</b> 도달 시 (즉시 청산)",
            "</div>",
            
            "</div>"
        ]
        st.markdown("".join(html_manual_list), unsafe_allow_html=True)
    else:
        # Warning Message Logic
        if matrix_id == 'panic':
            reason = "VIX 급등 또는 변동성 구조 붕괴(Backwardation)가 감지되었습니다."
        else:
            reason = "현재 점수가 신규 진입에 적합하지 않습니다."

        html_warning_list = [
            "<div style='border: 2px solid red; padding: 15px; margin-top: 20px; border-radius: 10px; background-color: #ffebee;'>",
            "<h3 style='color: red; margin-top: 0;'>⛔ 진입 금지 (No Entry)</h3>",
            f"<p style='color: black;'>{reason}<br>",
            "기존 포지션 관리(청산/롤오버)에만 집중하십시오.</p></div>"
        ]
        st.markdown("".join(html_warning_list), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 기술적 분석 차트")
    st.pyplot(create_charts(data))

if __name__ == "__main__":
    main()


