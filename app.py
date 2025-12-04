import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === [앱 보안 설정] ===
APP_PASSWORD = "1979"

# === [페이지 기본 설정] ===
st.set_page_config(
    page_title="HK 옵션투자자문 (Grand Master v22.3 - Smart Money)",
    page_icon="🦅",
    layout="wide"
)

# 차트 스타일 설정
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

# === [1] 데이터 수집 및 처리 ===
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
    
    # RSI(14)
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    # RSI(2)
    gain_2 = (delta.where(delta > 0, 0)).rolling(window=2).mean()
    loss_2 = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
    rs_2 = gain_2 / loss_2
    hist['RSI_2'] = 100 - (100 / (1 + rs_2))
    
    hist['Vol_MA20'] = hist['Volume'].rolling(window=20).mean()

    # [수정됨] ADL (Advance-Decline Line) 데이터 추가
    # 전략: ^ADD(나스닥 등락) 데이터를 우선 시도하고, 실패 시 QQQ 가격 변화로 대체(Fallback)
    try:
        # 방법 1: ^ADD 티커 시도
        add_ticker = yf.Ticker("^ADD")
        add_hist = add_ticker.history(period="2y")
        
        # 디버깅용 출력 (터미널)
        # print(f"^ADD Data Length: {len(add_hist)}")
        
        if not add_hist.empty and len(add_hist) > 10:
            # 인덱스 시간대 제거 및 정규화 (병합 오류 방지)
            hist.index = hist.index.tz_localize(None).normalize()
            add_hist.index = add_hist.index.tz_localize(None).normalize()
            
            # QQQ 데이터프레임에 병합 (Left Join)
            hist = hist.join(add_hist['Close'].rename('Net_Issues'), how='left')
            
            # 결측치 처리 (중요: 전방채움 후 0으로 채움)
            # 주말/공휴일 등으로 데이터가 비는 경우 직전 데이터 사용
            hist['Net_Issues'] = hist['Net_Issues'].ffill().fillna(0)
            
            # ADL 계산 (누적합)
            hist['ADL'] = hist['Net_Issues'].cumsum()
            
            # ADL 이동평균선
            hist['ADL_MA20'] = hist['ADL'].rolling(window=20).mean()
            
        else:
            raise ValueError("^ADD 데이터 부족 또는 없음")
            
    except Exception as e:
        print(f"⚠️ ADL 데이터 수집 실패 (^ADD): {e}")
        print("📊 대체 방법 사용: QQQ Price 변화 기반 ADL 근사치 생성")
        
        # 방법 2: 대체 로직 (Fallback)
        # 실제 등락 주선 데이터가 없을 때, QQQ가 오르면 +1, 내리면 -1로 가정하여 추세선 생성
        hist['Net_Issues'] = np.where(hist['Close'] > hist['Close'].shift(1), 1, -1)
        hist['Net_Issues'].iloc[0] = 0  # 첫 날은 0
        
        # ADL 계산
        hist['ADL'] = hist['Net_Issues'].cumsum()
        
        # 시각적 편의를 위해 스케일 조정
        hist['ADL'] = hist['ADL'] * 100
        
        # 이동평균선
        hist['ADL_MA20'] = hist['ADL'].rolling(window=20).mean()
    
    # 2. VIX, VIX3M, VVIX 데이터 처리
    vix_ticker = yf.Ticker("^VIX")
    vix_hist = vix_ticker.history(period="1y")
    
    vvix_ticker = yf.Ticker("^VVIX")
    vvix_hist = vvix_ticker.history(period="1y")

    vix3m_val = None
    vix3m_hist = None
    vix_term_df = None

    try:
        vix3m_ticker = yf.Ticker("^VIX3M")
        vix3m_hist = vix3m_ticker.history(period="1y")
        
        if not vix3m_hist.empty and not vix_hist.empty:
            vix3m_val = vix3m_hist['Close'].iloc[-1]
            
            df_vix = vix_hist[['Close']].copy()
            df_vix3m = vix3m_hist[['Close']].copy()
            
            df_vix.index = df_vix.index.tz_localize(None).normalize()
            df_vix3m.index = df_vix3m.index.tz_localize(None).normalize()
            
            merged_df = pd.merge(
                df_vix, 
                df_vix3m, 
                left_index=True, 
                right_index=True, 
                suffixes=('_VIX', '_VIX3M')
            )
            
            if len(merged_df) >= 30:
                merged_df['Ratio'] = merged_df['Close_VIX'] / merged_df['Close_VIX3M']
                vix_term_df = merged_df
            else:
                vix_term_df = None

    except Exception as e:
        vix3m_val = None
        vix_term_df = None
        print(f"Error fetching VIX/VIX3M: {e}")
    
    try:
        if not vvix_hist.empty:
            vvix_clean = vvix_hist[['Close']].copy()
            vvix_clean.index = vvix_clean.index.tz_localize(None).normalize()
    except Exception as e:
        print(f"Error processing VVIX: {e}")

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
        'rsi2': curr['RSI_2'], 
        'bb_upper': curr['BB_Upper'], 'bb_lower': curr['BB_Lower'], 'bb_lower_prev': prev['BB_Lower'],
        'macd': curr['MACD'], 'signal': curr['Signal'],
        'macd_prev': prev['MACD'], 'signal_prev': prev['Signal'],
        'volume': curr['Volume'], 'vol_ma20': curr['Vol_MA20'], 'vol_pct': vol_pct,
        'vix': curr_vix, 'vix_prev': prev_vix,
        'vix3m': vix3m_val,
        'iv': current_iv,
        'hist': hist, 'vix_hist': vix_hist, 'vix3m_hist': vix3m_hist, 'vvix_hist': vvix_hist,
        'vix_term_df': vix_term_df
    }

# === [2] 신규 로직 함수 ===

def detect_capitulation(data, log):
    if data['vix_term_df'] is None:
        log['capitulation'] = 'none'
        return 0

    ratio = data['vix'] / data['vix3m'] if data['vix3m'] else 0
    vol_ratio = data['volume'] / data['vol_ma20']
    
    try:
        term_df = data['vix_term_df']
        if len(term_df) < 2: return 0
        ratio_prev = term_df['Ratio'].iloc[-2] 
        
        vol_prev = data['hist']['Volume'].iloc[-2]
        vol_ma20_prev = data['hist']['Vol_MA20'].iloc[-2]
        vol_ratio_prev = vol_prev / vol_ma20_prev
        
        cond_today = (ratio > 1.0) and (vol_ratio > 1.5)
        cond_yesterday = (ratio_prev > 1.0) and (vol_ratio_prev > 1.5)
        
        if cond_today and cond_yesterday:
            log['capitulation'] = 'detected'
            return 15
    except Exception as e:
        print(f"Capitulation Check Error: {e}")

    log['capitulation'] = 'none'
    return 0

def detect_vvix_trap(data, log):
    try:
        vix_hist = data['vix_hist']['Close']
        if len(vix_hist) < 5: return 0
        vix_ma3 = vix_hist.rolling(3).mean()
        
        vix_change_pct = ((vix_ma3.iloc[-1] - vix_ma3.iloc[-4]) / vix_ma3.iloc[-4]) * 100
        
        vvix_hist = data['vvix_hist']['Close']
        if vvix_hist.empty: return 0
        vvix_change_pct = ((vvix_hist.iloc[-1] - vvix_hist.iloc[-2]) / vvix_hist.iloc[-2]) * 100
        
        if abs(vix_change_pct) < 2.0 and vvix_change_pct > 5.0:
            log['vvix_trap'] = 'detected'
            return -10
    except Exception as e:
        print(f"VVIX Trap Error: {e}")

    log['vvix_trap'] = 'none'
    return 0

def detect_rsi2_dip(data, log):
    try:
        rsi2 = data['rsi2']
        ratio = data['vix'] / data['vix3m'] if data['vix3m'] else 1.1
        
        vvix_hist = data['vvix_hist']['Close']
        if len(vvix_hist) < 2: return 0
        vvix_falling = vvix_hist.iloc[-1] < vvix_hist.iloc[-2]
        
        if rsi2 < 10 and ratio < 1.0 and vvix_falling:
            log['rsi2_dip'] = 'detected'
            return 8
    except:
        pass

    log['rsi2_dip'] = 'none'
    return 0

# === [3] 전문가 로직 (MACD 4-Zone Matrix 적용) ===
def analyze_expert_logic(d):
    if d['price'] > d['ma50'] and d['price'] > d['ma200']: season = "SUMMER"
    elif d['price'] < d['ma50'] and d['price'] > d['ma200']: season = "AUTUMN"
    elif d['price'] < d['ma50'] and d['price'] < d['ma200']: season = "WINTER"
    else: season = "SPRING"
    
    score = 0
    log = {}
    
    # 1. VIX Term Structure
    vix_ratio = 1.0
    if d['vix3m'] and d['vix3m'] > 0:
        vix_ratio = d['vix'] / d['vix3m']
    
    if vix_ratio > 1.0:
        score += -10
        log['term'] = 'backwardation'
    elif vix_ratio < 0.9:
        score += 3
        log['term'] = 'contango'
    else:
        score += 0
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
        pts = -1 if season == "SUMMER" else -3 if season == "AUTUMN" else -10 if season == "WINTER" else -2
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

    # 4. Bollinger Logic (Z-Score)
    numerator = d['price'] - d['ma20']
    denominator = (d['bb_upper'] - d['ma20']) / 2.0
    
    if denominator == 0:
        z_score = 0
    else:
        z_score = numerator / denominator
        
    log['z_score'] = z_score

    if z_score > 1.8:
        pts = -3
        score += pts
        log['bb'] = 'overbought_danger'
    elif 0.5 < z_score <= 1.8:
        pts = 1
        score += pts
        log['bb'] = 'uptrend'
    elif -0.5 <= z_score <= 0.5:
        pts = 0
        score += pts
        log['bb'] = 'neutral'
    elif -1.8 < z_score < -0.5:
        pts = 2
        score += pts
        log['bb'] = 'dip_buying'
    else: 
        pts = 1 
        score += pts
        log['bb'] = 'oversold_guard'

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

    # 7. MACD Logic (4-Zone Strategy 적용)
    macd_val = d['macd']
    signal_val = d['signal']
    
    if macd_val > signal_val:
        if macd_val >= 0:
            pts = 3
            score += pts
            log['macd'] = 'zero_up_golden' 
        else:
            pts = 0
            score += pts
            log['macd'] = 'zero_down_golden'

    else:
        if macd_val >= 0:
            pts = -3
            score += pts
            log['macd'] = 'zero_up_dead'
        else:
            pts = -5
            score += pts
            log['macd'] = 'zero_down_dead'

    # === [신규 항목 점수 누적] ===
    pts_cap = detect_capitulation(d, log)
    score += pts_cap
    
    pts_vvix = detect_vvix_trap(d, log)
    score += pts_vvix
    
    pts_rsi2 = detect_rsi2_dip(d, log)
    score += pts_rsi2

    return season, score, log

# === [4] 행동 결정 (수정됨: PCS vs CDS 분기) ===
def determine_action(score, season, data, log):
    vix_pct_change = ((data['vix'] - data['vix_prev']) / data['vix_prev']) * 100
    current_vix = data['vix']
    
    # 1. Panic Check
    if log.get('term') == 'backwardation':
        return None, "⛔ 매매 중단 (System Collapse)", "-", "-", "panic", "-", "-"
    if vix_pct_change > 15.0:
        return None, "⛔ 매매 중단 (VIX 급등)", "-", "-", "panic", "-", "-"
    if log.get('vvix_trap') == 'detected':
        return None, "⛔ 매매 중단 (VVIX Trap)", "-", "-", "panic", "-", "-"
    
    # 2. Score Grade & Strategy Selection
    verdict_text = ""
    profit_target = ""
    stop_loss = ""
    matrix_id = ""
    target_delta = None
    
    # 등급 결정
    if score >= 20:
        verdict_text = "💎💎 극강 추세 (Super Strong)"
        matrix_id = "super_strong"
        profit_target = "100%+"
        stop_loss = "-300%"
    elif score >= 12:
        verdict_text = "💎 추세 추종 (Strong)"
        matrix_id = "strong"
        profit_target = "75%"
        stop_loss = "-300%"
    elif 8 <= score < 12:
        verdict_text = "✅ 표준 대응 (Standard)"
        matrix_id = "standard"
        profit_target = "50%"
        stop_loss = "-200%"
    elif 5 <= score < 8:
        verdict_text = "⚠️ 속전 속결 (Hit & Run)"
        matrix_id = "weak"
        profit_target = "30%"
        stop_loss = "-150%"
    else:
        verdict_text = "🛡️ 진입 보류"
        matrix_id = "no_entry"
        return None, verdict_text, "-", "-", matrix_id, "-", "-"

    # 3. Strategy Logic (PCS vs CDS)
    # 전문가 로직:
    # A. Call Debit Spread (CDS): VIX < 18 (저변동성) AND Score >= 12 (강한 추세)
    # B. Put Credit Spread (PCS): 그 외 (VIX >= 18 OR Score < 12)
    
    strategy_type = ""
    strategy_basis = ""

    if current_vix < 18.0 and score >= 12:
        strategy_type = "CDS"
        strategy_basis = f"VIX {current_vix:.1f} (저변동성) + 점수 {score} (강세) 👉 방향성 베팅(가성비)"
        target_delta = 0.55 # CDS는 보통 ATM 근처 매수 (Delta ~0.50-0.60)
    else:
        strategy_type = "PCS"
        if current_vix >= 18.0:
            strategy_basis = f"VIX {current_vix:.1f} (고변동성) 👉 프리미엄 매도 유리"
        else:
            strategy_basis = f"점수 {score} (중립/완만) 👉 시간가치(Theta) 확보 유리"
        target_delta = -0.10 # PCS는 OTM Put 매도 (Delta -0.10 ~ -0.15)

    return target_delta, verdict_text, profit_target, stop_loss, matrix_id, strategy_type, strategy_basis

# === [5] 옵션 찾기 (수정됨: CDS/PCS 구분) ===
def calculate_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return -0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

def calculate_call_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def find_best_option(price, iv, target_delta, strategy_type):
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
        
        # CDS (Call Debit) vs PCS (Put Credit)
        if strategy_type == "CDS":
            # CDS: Long Call (Target Delta ~0.55) / Short Call (Higher)
            # 검색 범위: 현재가 주변 (ATM)
            start_k = int(price * 0.9)
            end_k = int(price * 1.1)
            
            for strike in range(start_k, end_k):
                d = calculate_call_delta(price, strike, T, r, iv)
                diff = abs(d - target_delta)
                if diff < min_diff:
                    min_diff = diff
                    best_strike = strike
                    found_delta = d
            
            long_strike = best_strike
            short_strike = best_strike + SPREAD_WIDTH
            return {
                'type': 'CDS',
                'expiry': expiry, 'dte': dte,
                'long': long_strike, 'short': short_strike,
                'delta': found_delta,
                'width': SPREAD_WIDTH
            }
            
        else: # PCS
            # PCS: Short Put (Target Delta ~-0.10) / Long Put (Lower)
            start_k = int(price * 0.5)
            end_k = int(price)
            
            for strike in range(start_k, end_k):
                d = calculate_put_delta(price, strike, T, r, iv)
                diff = abs(d - target_delta)
                if diff < min_diff:
                    min_diff = diff
                    best_strike = strike
                    found_delta = d
            
            short_strike = best_strike
            long_strike = best_strike - SPREAD_WIDTH
            return {
                'type': 'PCS',
                'expiry': expiry, 'dte': dte,
                'short': short_strike, 'long': long_strike,
                'delta': found_delta,
                'width': SPREAD_WIDTH
            }

    except Exception as e:
        print(f"Option Search Error: {e}")
        return None

# === [6] 차트 (10개 서브플롯) - 수정됨: Trend 차트 위치 및 ADL 추가 ===
def create_charts(data):
    hist = data['hist'].copy()  # 원본 데이터 보호를 위해 복사
    
    # === [배경색 로직] 4계절 계산 ===
    cond_summer = (hist['Close'] > hist['MA50']) & (hist['Close'] > hist['MA200'])
    cond_autumn = (hist['Close'] < hist['MA50']) & (hist['Close'] > hist['MA200'])
    cond_winter = (hist['Close'] < hist['MA50']) & (hist['Close'] < hist['MA200'])
    # Spring은 나머지 경우
    
    conditions = [cond_summer, cond_autumn, cond_winter]
    choices = ['SUMMER', 'AUTUMN', 'WINTER']
    
    # 'Season' 컬럼 생성 (기본값 SPRING)
    hist['Season'] = np.select(conditions, choices, default='SPRING')
    
    # 시즌별 배경 색상 설정 (파스텔 톤)
    season_colors = {
        'SUMMER': '#FFEBEE',  # 연한 붉은색 (상승확산)
        'AUTUMN': '#FFF3E0',  # 연한 주황색 (조정)
        'WINTER': '#E3F2FD',  # 연한 파란색 (하락)
        'SPRING': '#E8F5E9'   # 연한 초록색 (회복)
    }
    
    # === 차트 그리기 시작 ===
    # 높이를 늘리고 10행으로 변경 (ADL 추가됨)
    fig = plt.figure(figsize=(10, 30))
    # 높이 비율 조정: [9] ADL(1) 추가
    gs = fig.add_gridspec(10, 1, height_ratios=[2, 0.6, 1.5, 1, 1, 1, 1, 1, 1, 1])
    
    # 1. Price Chart (Main) - Index 0
    ax1 = fig.add_subplot(gs[0])
    
    # 기존 라인 플롯 (zorder 설정 유지)
    ax1.plot(hist.index, hist['Close'], label='QQQ', color='black', alpha=0.9, zorder=2)
    ax1.plot(hist.index, hist['MA20'], label='20MA', color='green', ls='--', lw=1, zorder=2)
    ax1.plot(hist.index, hist['MA50'], label='50MA', color='blue', ls='-', lw=1.5, zorder=2)
    ax1.plot(hist.index, hist['MA200'], label='200MA', color='red', ls='-', lw=2, zorder=2)
    ax1.fill_between(hist.index, hist['BB_Upper'], hist['BB_Lower'], color='gray', alpha=0.1, label='Bollinger', zorder=1)
    
    ax1.set_title('QQQ Price Trend with Market Seasons', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # 2. Volume (Moved to 2nd position) - Index 1
    ax_vol = fig.add_subplot(gs[1], sharex=ax1)
    colors = ['red' if c < o else 'green' for c, o in zip(hist['Close'], hist['Open'])]
    ax_vol.bar(hist.index, hist['Volume'], color=colors, alpha=0.5, zorder=2)
    ax_vol.plot(hist.index, hist['Vol_MA20'], color='black', lw=1, zorder=2)
    ax_vol.set_title(f"Volume ({data['vol_pct']:.1f}%)", fontsize=10, fontweight='bold')
    ax_vol.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_vol.get_xticklabels(), visible=False)
    
    # 3. QQQ Trend Graph (Moved to 3rd position) - Index 2
    # 배경: MACD 데드크로스(MACD < Signal) 구간을 다른 색으로 표시
    ax_trend = fig.add_subplot(gs[2], sharex=ax1)
    ax_trend.plot(hist.index, hist['Close'], label='QQQ', color='black', alpha=0.8, zorder=2)
    ax_trend.plot(hist.index, hist['MA20'], label='20MA', color='green', ls='--', lw=1, zorder=2)
    ax_trend.plot(hist.index, hist['MA50'], label='50MA', color='blue', ls='-', lw=1, zorder=2)
    
    # MACD 데드크로스(MACD < Signal) 구간 배경 칠하기
    dead_cross_mask = hist['MACD'] < hist['Signal']
    # 그룹화하여 연속된 구간 찾기
    hist['dc_group'] = (dead_cross_mask != dead_cross_mask.shift()).cumsum()
    
    for _, group in hist[dead_cross_mask].groupby('dc_group'):
        start = group.index[0]
        end = group.index[-1]
        # Light Red/Pink color specifically for Dead Cross
        ax_trend.axvspan(start, end, color='#FFCDD2', alpha=0.4, zorder=0, label='MACD < Signal (Dead)')

    # 중복 라벨 제거를 위한 범례 처리
    handles, labels = ax_trend.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_trend.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    ax_trend.set_title('QQQ Trend Check (Background: MACD Dead Cross)', fontsize=10, fontweight='bold')
    ax_trend.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_trend.get_xticklabels(), visible=False)

    # 4. VIX Level (Absolute) - Index 3
    ax_vix_abs = fig.add_subplot(gs[3], sharex=ax1)
    ax_vix_abs.plot(data['vix_hist'].index, data['vix_hist']['Close'], color='purple', label='VIX (Spot)', zorder=2)
    if data['vix3m_hist'] is not None and not data['vix3m_hist'].empty:
         ax_vix_abs.plot(data['vix3m_hist'].index, data['vix3m_hist']['Close'], color='gray', ls=':', label='VIX3M', zorder=2)
    
    ax_vix_abs.axhline(35, color='red', ls='--', zorder=2)
    ax_vix_abs.axhline(20, color='green', ls='--', zorder=2)
    ax_vix_abs.set_title('VIX vs VIX3M (Absolute Level)', fontsize=12, fontweight='bold')
    ax_vix_abs.legend(loc='upper right')
    ax_vix_abs.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_vix_abs.get_xticklabels(), visible=False)

    # 5. VIX Term Structure (Ratio) - Index 4
    ax_ratio = fig.add_subplot(gs[4], sharex=ax1)
    term_data = data.get('vix_term_df')
    
    if term_data is not None and not term_data.empty:
        ax_ratio.plot(term_data.index, term_data['Ratio'], color='black', lw=1.2, label='Ratio (VIX/VIX3M)', zorder=2)
        ax_ratio.axhline(1.0, color='red', ls='--', alpha=0.8, lw=1.5, label='Threshold (1.0)', zorder=2)
        
        ax_ratio.fill_between(term_data.index, term_data['Ratio'], 1.0, 
                             where=(term_data['Ratio'] > 1.0), 
                             color='red', alpha=0.2, interpolate=True, label='Danger (Back.)', zorder=1)
        
        ax_ratio.fill_between(term_data.index, term_data['Ratio'], 1.0, 
                             where=(term_data['Ratio'] <= 1.0), 
                             color='green', alpha=0.15, interpolate=True, label='Safe (Contango)', zorder=1)
        
        ax_ratio.fill_between(term_data.index, term_data['Ratio'], 0.9, 
                             where=(term_data['Ratio'] < 0.9), 
                             color='green', alpha=0.3, interpolate=True, label='Super Contango', zorder=1)
        
        ax_ratio.legend(loc='upper right', fontsize=8)
    else:
        ax_ratio.text(0.5, 0.5, "데이터 부족 (Data Insufficient)", transform=ax_ratio.transAxes, ha='center', color='red', zorder=2)
        
    ax_ratio.set_title('VIX Term Structure (Ratio = VIX / VIX3M)', fontsize=12, fontweight='bold')
    ax_ratio.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_ratio.get_xticklabels(), visible=False)

    # 6. RSI(14) - Index 5
    ax_rsi = fig.add_subplot(gs[5], sharex=ax1)
    ax_rsi.plot(hist.index, hist['RSI'], color='purple', label='RSI(14)', zorder=2)
    ax_rsi.axhline(70, color='red', ls='--', alpha=0.7, zorder=2)
    ax_rsi.axhline(30, color='green', ls='--', alpha=0.7, zorder=2)
    ax_rsi.fill_between(hist.index, hist['RSI'], 70, where=(hist['RSI'] >= 70), color='red', alpha=0.3, zorder=1)
    ax_rsi.fill_between(hist.index, hist['RSI'], 30, where=(hist['RSI'] <= 30), color='green', alpha=0.3, zorder=1)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI (14)', fontsize=12, fontweight='bold')
    ax_rsi.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # 7. MACD - Index 6
    ax2 = fig.add_subplot(gs[6], sharex=ax1)
    ax2.plot(hist.index, hist['MACD'], label='MACD', color='blue', zorder=2)
    ax2.plot(hist.index, hist['Signal'], label='Signal', color='orange', zorder=2)
    ax2.bar(hist.index, hist['MACD']-hist['Signal'], color='gray', alpha=0.3, zorder=2)
    ax2.axhline(0, color='black', lw=0.8, zorder=2)
    ax2.set_title('MACD', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # 8. VVIX / VIX Ratio - Index 7
    ax_ratio_vvix = fig.add_subplot(gs[7], sharex=ax1)
    try:
        df_v = data['vix_hist'][['Close']].copy()
        df_vv = data['vvix_hist'][['Close']].copy()
        df_v.index = df_v.index.tz_localize(None).normalize()
        df_vv.index = df_vv.index.tz_localize(None).normalize()
        
        merged_ratio = pd.merge(df_v, df_vv, left_index=True, right_index=True, suffixes=('_VIX', '_VVIX'))
        merged_ratio['Ratio'] = merged_ratio['Close_VVIX'] / merged_ratio['Close_VIX']
        
        if not merged_ratio.empty:
            ax_ratio_vvix.plot(merged_ratio.index, merged_ratio['Ratio'], color='#333333', lw=1.2, label='VVIX/VIX Ratio', zorder=2)
            ax_ratio_vvix.axhline(7.0, color='red', ls=':', alpha=0.5, zorder=2)
            ax_ratio_vvix.axhline(4.0, color='green', ls=':', alpha=0.5, zorder=2)
            ax_ratio_vvix.axhline(5.5, color='gray', ls='--', alpha=0.5, lw=0.8, zorder=2)
            ax_ratio_vvix.fill_between(merged_ratio.index, merged_ratio['Ratio'], 7.0, 
                                     where=(merged_ratio['Ratio'] > 7.0), color='red', alpha=0.2, label='Complacency', zorder=1)
            ax_ratio_vvix.fill_between(merged_ratio.index, merged_ratio['Ratio'], 4.0, 
                                     where=(merged_ratio['Ratio'] < 4.0), color='green', alpha=0.2, label='Panic', zorder=1)
            ax_ratio_vvix.legend(loc='upper left', fontsize=8)
        else:
            ax_ratio_vvix.text(0.5, 0.5, "No Data", transform=ax_ratio_vvix.transAxes, ha='center', zorder=2)
    except Exception as e:
        ax_ratio_vvix.text(0.5, 0.5, f"Error: {e}", transform=ax_ratio_vvix.transAxes, ha='center', color='red', zorder=2)

    ax_ratio_vvix.set_title('VVIX / VIX Ratio', fontsize=12, fontweight='bold')
    ax_ratio_vvix.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_ratio_vvix.get_xticklabels(), visible=False)

    # 9. RSI(2) - Index 8
    ax_rsi2 = fig.add_subplot(gs[8], sharex=ax1)
    ax_rsi2.plot(hist.index, hist['RSI_2'], color='gray', label='RSI(2)', linewidth=1.2, zorder=2)
    ax_rsi2.axhline(10, color='green', linestyle='--', alpha=0.7, zorder=2)
    ax_rsi2.axhline(90, color='red', linestyle='--', alpha=0.7, zorder=2)
    ax_rsi2.fill_between(hist.index, hist['RSI_2'], 10, where=(hist['RSI_2'] < 10), color='green', alpha=0.3, label='Buy Zone', zorder=1)
    ax_rsi2.fill_between(hist.index, hist['RSI_2'], 90, where=(hist['RSI_2'] > 90), color='red', alpha=0.3, label='Danger', zorder=1)
    ax_rsi2.scatter(hist.index[-1], hist['RSI_2'].iloc[-1], color='red', s=50, zorder=5)
    ax_rsi2.set_ylim(0, 100)
    ax_rsi2.set_title('RSI(2) - Short-term Pullback', fontsize=12, fontweight='bold')
    ax_rsi2.legend(loc='upper right')
    ax_rsi2.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_rsi2.get_xticklabels(), visible=False)

    # [수정됨] 10. ADL (Advance-Decline Line) - Index 9 (Last)
    ax_adl = fig.add_subplot(gs[9], sharex=ax1)
    
    # 데이터 안전 장치: 컬럼이 존재하고 데이터가 유효한 경우에만 플롯
    if 'ADL' in hist.columns and not hist['ADL'].isna().all():
        ax_adl.plot(hist.index, hist['ADL'], color='black', label='ADL (Breath)', linewidth=1.5, zorder=2)
        ax_adl.plot(hist.index, hist['ADL_MA20'], color='orange', ls='--', label='ADL 20MA', linewidth=1, zorder=2)
        
        # 마지막 값 텍스트 표시
        if not hist['ADL'].empty:
            last_adl = hist['ADL'].iloc[-1]
            ax_adl.text(hist.index[-1], last_adl, f"{last_adl:.0f}", 
                       color='black', fontsize=9, fontweight='bold', ha='left', va='center')
        
        # 기준선 (0)
        ax_adl.axhline(0, color='gray', ls=':', alpha=0.5, zorder=1)
        
        ax_adl.set_title('Advance-Decline Line (Market Breadth)', fontsize=12, fontweight='bold')
        ax_adl.legend(loc='upper left')
        
    else:
        # 데이터가 없을 경우 빈 화면 대신 경고 메시지 표시
        ax_adl.text(0.5, 0.5, "⚠️ ADL Data Not Available", 
                   transform=ax_adl.transAxes, ha='center', va='center', 
                   fontsize=12, color='red', fontweight='bold')
        ax_adl.set_title('Advance-Decline Line (No Data)', fontsize=12, fontweight='bold')

    ax_adl.grid(True, alpha=0.3, zorder=1)
    ax_adl.set_xlabel('Date', fontsize=10)
    
    # === [모든 서브플롯에 배경색 일괄 적용] ===
    # 배경색 칠하기를 위한 그룹화 (연속된 구간 찾기)
    hist['group'] = (hist['Season'] != hist['Season'].shift()).cumsum()
    
    # 모든 axes를 리스트로 묶음 (Trend 차트는 제외 - 별도 MACD 배경 적용됨)
    # 순서: Price, Volume, Trend(X), VIX_Abs, Ratio, RSI, MACD, Ratio_VVIX, RSI2, ADL
    all_axes_except_trend = [ax1, ax_vol, ax_vix_abs, ax_ratio, ax_rsi, ax2, ax_ratio_vvix, ax_rsi2, ax_adl]
    
    # 반복문으로 차트에 계절 배경색 적용 (Trend 차트 제외)
    for ax in all_axes_except_trend:
        for _, group_data in hist.groupby('group'):
            season = group_data['Season'].iloc[0]
            start_date = group_data.index[0]
            end_date = group_data.index[-1]
            # zorder=0으로 설정하여 모든 데이터(라인, 바, 그리드 등) 뒤에 배경이 오도록 함
            # alpha=0.4로 설정하여 가시성 확보
            ax.axvspan(start_date, end_date, color=season_colors[season], alpha=0.4, zorder=0)

    plt.tight_layout()
    return fig

# === [메인 화면] ===
def main():
    st.title("🦅 HK Advisory (Grand Master v22.3 - Smart Money)")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Logic: MACD 4-Zone & Expert Strategy Selector")

    with st.spinner('시장 구조 분석 및 전략 최적화 중...'):
        try:
            data = get_market_data()
            season, score, log = analyze_expert_logic(data)
            # return 값 추가됨 (strategy_type, strategy_basis)
            target_delta, verdict_text, profit_target, stop_loss, matrix_id, strat_type, strat_basis = determine_action(score, season, data, log)
            # find_best_option에 strat_type 전달
            strategy = find_best_option(data['price'], data['iv'], target_delta, strat_type)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            import traceback
            st.text(traceback.format_exc())
            return

    # [Sidebar]
    st.sidebar.title("🛠️ 시스템 상태")
    st.sidebar.markdown("---")
    
    term_df = data.get('vix_term_df')
    if term_df is not None:
        curr_ratio = term_df['Ratio'].iloc[-1]
        st.sidebar.metric("Current Ratio", f"{curr_ratio:.4f}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 실시간 위험 지표")

    # Ratio
    ratio_val = data['vix'] / data['vix3m'] if data['vix3m'] else 1.0
    if ratio_val > 1.0: st.sidebar.error(f"Ratio: {ratio_val:.4f} ⚠️")
    elif ratio_val < 0.9: st.sidebar.success(f"Ratio: {ratio_val:.4f} ✅")
    else: st.sidebar.info(f"Ratio: {ratio_val:.4f}")

    # VVIX Change
    vvix_hist = data['vvix_hist']['Close']
    if len(vvix_hist) > 1:
        vvix_change = ((vvix_hist.iloc[-1] - vvix_hist.iloc[-2]) / vvix_hist.iloc[-2]) * 100
        if vvix_change > 5.0: st.sidebar.error(f"VVIX Change: +{vvix_change:.1f}% ⚠️")
        else: st.sidebar.success(f"VVIX Change: {vvix_change:.1f}%")

    # RSI(2)
    rsi2_val = data['rsi2']
    if rsi2_val < 10: st.sidebar.success(f"RSI(2): {rsi2_val:.1f} (눌림목) ✅")
    else: st.sidebar.info(f"RSI(2): {rsi2_val:.1f}")

    # Signals
    if log.get('capitulation') == 'detected': st.sidebar.success("투매 신호: ✅ 발생")
    else: st.sidebar.info("투매 신호: ❌ 미발생")
    
    if log.get('vvix_trap') == 'detected': st.sidebar.error("VVIX Trap: ⚠️ 감지됨")
    else: st.sidebar.success("VVIX Trap: ✅ 없음")

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📊 총점: {score}점")
    st.sidebar.markdown(f"**판정:** {verdict_text}")
    if strat_type:
        st.sidebar.info(f"전략: {strat_type}")

    # 스타일 헬퍼
    def hl_score(category, row_state, col_season):
        base = "style='border: 1px solid #ddd; padding: 4px; color: black; background-color: white;'"
        current_val = log.get(category, '')
        is_match = False
        if category == 'rsi' and row_state == 'escape':
            if 'escape' in current_val: is_match = True
        else:
            if current_val == row_state: is_match = True
        
        if is_match and (season == col_season or col_season == 'ALL'):
            return "style='border: 3px solid #FF5722; background-color: #FFF8E1; font-weight: bold; color: #D84315; padding: 4px;'"
        return base

    def hl_season(row_season):
        if season == row_season:
            return "style='border: 3px solid #2196F3; background-color: #E3F2FD; font-weight: bold; color: black; padding: 4px;'"
        return "style='border: 1px solid #ddd; padding: 4px; color: black; background-color: white;'"

    td_style = "style='border: 1px solid #ddd; padding: 4px; color: black; background-color: white;'"
    th_style = "style='border: 1px solid #ddd; padding: 4px; color: black; background-color: #f2f2f2;'"
    vix_ratio_disp = f"{log.get('vix_ratio', 0):.2f}"
    z_disp = f"{log.get('z_score', 0):.2f}"

    # 1. Season Matrix
    html_season_list = [
        "<h3>1. Market Season Matrix</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; text-align: center;'>",
        "<tr>",
        f"<th {th_style}>Season</th><th {th_style}>Condition</th><th {th_style}>Character</th>",
        "</tr>",
        f"<tr><td {hl_season('SUMMER')}>☀️ SUMMER</td><td {hl_season('SUMMER')}>Price > 50MA & 200MA</td><td {hl_season('SUMMER')}>강세장</td></tr>",
        f"<tr><td {hl_season('AUTUMN')}>🍂 AUTUMN</td><td {hl_season('AUTUMN')}>Price < 50MA but > 200MA</td><td {hl_season('AUTUMN')}>조정기</td></tr>",
        f"<tr><td {hl_season('WINTER')}>❄️ WINTER</td><td {hl_season('WINTER')}>Price < 50MA & 200MA</td><td {hl_season('WINTER')}>약세장 (-5점)</td></tr>",
        f"<tr><td {hl_season('SPRING')}>🌱 SPRING</td><td {hl_season('SPRING')}>Price > 50MA but < 200MA</td><td {hl_season('SPRING')}>회복기</td></tr>",
        "</table>",
        f"<p>※ QQQ: <b>${data['price']:.2f}</b> (Vol: {data['vol_pct']:.1f}% of 20MA)</p>"
    ]
    st.markdown("".join(html_season_list), unsafe_allow_html=True)

    # 2. Scorecard
    html_score_list = [
        "<h3>2. Expert Matrix (Mobile Ver.)</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; text-align: center;'>",
        "<tr>",
        f"<th {th_style}>지표</th><th {th_style}>상태</th>",
        f"<th {th_style}>☀️</th><th {th_style}>🍂</th><th {th_style}>❄️</th><th {th_style}>🌱</th>",
        "</tr>",
        
        # 1. VIX Term
        f"<tr><td rowspan='3' {td_style}><b>VIX Term</b><br><span style='font-size:10px; color:blue;'>Ratio:{vix_ratio_disp}</span></td>",
        f"<td {td_style}><b>Easy</b><br>(&lt;0.9)</td>",
        f"<td colspan='4' {hl_score('term', 'contango', 'ALL')}>+3</td></tr>",
        
        f"<tr><td {td_style}>Normal<br>(0.9~1)</td>",
        f"<td colspan='4' {hl_score('term', 'normal', 'ALL')}>0</td></tr>",
        
        f"<tr><td {td_style}><b>붕괴</b><br>(&gt;1.0)</td>",
        f"<td colspan='4' {hl_score('term', 'backwardation', 'ALL')}><b>-10</b></td></tr>",
        
        # 2. Capitulation
        f"<tr><td {td_style}><b>투매</b></td>",
        f"<td {td_style}><b>2일연속</b><br>R&gt;1,V&gt;1.5</td>",
        f"<td colspan='4' {hl_score('capitulation', 'detected', 'ALL')}><b style='color:green;'>+15</b></td></tr>",
        
        # 3. VVIX Trap
        f"<tr><td {td_style}><b>VVIX 함정</b></td>",
        f"<td {td_style}><b>위험</b><br>VIX↓VVIX↑</td>",
        f"<td colspan='4' {hl_score('vvix_trap', 'detected', 'ALL')}><b style='color:red;'>-10</b></td></tr>",

        # 4. RSI(14)
        f"<tr><td rowspan='4' {td_style}>RSI(14)</td>",
        f"<td {td_style}>과열 (>70)</td>",
        f"<td {hl_score('rsi', 'over', 'SUMMER')}>-1</td><td {hl_score('rsi', 'over', 'AUTUMN')}>-3</td><td {hl_score('rsi', 'over', 'WINTER')}><b style='color:red;'>-10</b></td><td {hl_score('rsi', 'over', 'SPRING')}>-2</td></tr>",
        
        f"<tr><td {td_style}>중립</td>",
        f"<td {hl_score('rsi', 'neutral', 'SUMMER')}>+1</td><td {hl_score('rsi', 'neutral', 'AUTUMN')}>0</td><td {hl_score('rsi', 'neutral', 'WINTER')}>-1</td><td {hl_score('rsi', 'neutral', 'SPRING')}>+1</td></tr>",
        
        f"<tr><td {td_style}>과매도 (<30)</td>",
        f"<td {hl_score('rsi', 'under', 'SUMMER')}>+5</td><td {hl_score('rsi', 'under', 'AUTUMN')}>+4</td><td {hl_score('rsi', 'under', 'WINTER')}>0</td><td {hl_score('rsi', 'under', 'SPRING')}>+4</td></tr>",
        
        f"<tr><td {td_style}>🚀 탈출</td>",
        f"<td {hl_score('rsi', 'escape', 'SUMMER')}>3~5</td><td {hl_score('rsi', 'escape', 'AUTUMN')}>3~5</td><td {hl_score('rsi', 'escape', 'WINTER')}>3~5</td><td {hl_score('rsi', 'escape', 'SPRING')}>3~5</td></tr>",
        
        # 5. RSI(2)
        f"<tr><td {td_style}><b>RSI(2)</b></td>",
        f"<td {td_style}><b>눌림목</b><br>(&lt;10)</td>",
        f"<td colspan='4' {hl_score('rsi2_dip', 'detected', 'ALL')}><b style='color:green;'>+8</b></td></tr>",

        # 6. VIX Level
        f"<tr><td rowspan='4' {td_style}>VIX</td>",
        f"<td {td_style}>안정 (<20)</td>",
        f"<td {hl_score('vix', 'stable', 'SUMMER')}>+2</td><td {hl_score('vix', 'stable', 'AUTUMN')}>0</td><td {hl_score('vix', 'stable', 'WINTER')}>-2</td><td {hl_score('vix', 'stable', 'SPRING')}>+1</td></tr>",
        
        f"<tr><td {td_style}>공포 (20-35)</td>",
        f"<td {hl_score('vix', 'fear', 'SUMMER')}>-3</td><td {hl_score('vix', 'fear', 'AUTUMN')}>-4</td><td {hl_score('vix', 'fear', 'WINTER')}>+2</td><td {hl_score('vix', 'fear', 'SPRING')}>-1</td></tr>",
        
        f"<tr><td {td_style}>패닉 상승</td>",
        f"<td {hl_score('vix', 'panic_rise', 'SUMMER')}>-5</td><td {hl_score('vix', 'panic_rise', 'AUTUMN')}>-6</td><td {hl_score('vix', 'panic_rise', 'WINTER')}>-5</td><td {hl_score('vix', 'panic_rise', 'SPRING')}>-4</td></tr>",
        
        f"<tr><td {td_style}>📉 꺾임</td>",
        f"<td {hl_score('vix', 'peak_out', 'SUMMER')}>-</td><td {hl_score('vix', 'peak_out', 'AUTUMN')}>-</td><td {hl_score('vix', 'peak_out', 'WINTER')}>+7</td><td {hl_score('vix', 'peak_out', 'SPRING')}>-</td></tr>",
        
        # 7. Bollinger
        f"<tr><td rowspan='5' {td_style}>BB Z-Score<br><span style='font-size:10px; color:blue;'>{z_disp}</span></td>",
        f"<td {td_style} style='color:red;'><b>과열</b><br>(&gt;1.8)</td>",
        f"<td colspan='4' {hl_score('bb', 'overbought_danger', 'ALL')}><b style='color:red;'>-3</b></td></tr>",
        
        f"<tr><td {td_style}><b>상승</b><br>(0.5~1.8)</td>",
        f"<td colspan='4' {hl_score('bb', 'uptrend', 'ALL')}>+1</td></tr>",
        
        f"<tr><td {td_style}>중립</td>",
        f"<td colspan='4' {hl_score('bb', 'neutral', 'ALL')}>0</td></tr>",
        
        f"<tr><td {td_style}><b>저평가</b><br>(-1.8~-0.5)</td>",
        f"<td colspan='4' {hl_score('bb', 'dip_buying', 'ALL')}>+2</td></tr>",
        
        f"<tr><td {td_style}><b>바닥</b><br>(Z&le;-1.8)</td>",
        f"<td colspan='4' {hl_score('bb', 'oversold_guard', 'ALL')}><b>+1</b></td></tr>",
        
        # 8. Trend & Vol
        f"<tr><td {td_style}>추세</td><td {td_style}>20일선 위</td>",
        f"<td {hl_score('trend', 'up', 'SUMMER')}>+2</td><td {hl_score('trend', 'up', 'AUTUMN')}>+2</td><td {hl_score('trend', 'up', 'WINTER')}>+3</td><td {hl_score('trend', 'up', 'SPRING')}>+3</td></tr>",
        
        f"<tr><td {td_style}>거래량</td><td {td_style}>폭증</td>",
        f"<td {hl_score('vol', 'explode', 'SUMMER')}>+2</td><td {hl_score('vol', 'explode', 'AUTUMN')}>+3</td><td {hl_score('vol', 'explode', 'WINTER')}>+3</td><td {hl_score('vol', 'explode', 'SPRING')}>+2</td></tr>",
        
        # 9. MACD (4-Zone)
        f"<tr><td rowspan='4' {td_style}>MACD</td>",
        
        # Case 1
        f"<td {td_style}>📈 <b>가속</b><br><span style='font-size:10px;'>(위+골든)</span></td>",
        f"<td colspan='4' {hl_score('macd', 'zero_up_golden', 'ALL')}><b style='color:green;'>+3</b></td></tr>",
        
        # Case 2
        f"<td {td_style}>📉 <b>조정</b><br><span style='font-size:10px;'>(위+데드)</span></td>",
        f"<td colspan='4' {hl_score('macd', 'zero_up_dead', 'ALL')}><b style='color:orange;'>-3</b></td></tr>",
        
        # Case 3
        f"<td {td_style}>🎣 <b>함정</b><br><span style='font-size:10px;'>(아래+골든)</span></td>",
        f"<td colspan='4' {hl_score('macd', 'zero_down_golden', 'ALL')}><b style='color:gray;'>0</b></td></tr>",
        
        # Case 4
        f"<td {td_style}>☔ <b>폭락</b><br><span style='font-size:10px;'>(아래+데드)</span></td>",
        f"<td colspan='4' {hl_score('macd', 'zero_down_dead', 'ALL')}><b style='color:red;'>-5</b></td></tr>",
        
        "</table>"
    ]
    st.markdown("".join(html_score_list), unsafe_allow_html=True)

    # 3. Final Verdict (수정됨: 색상 변경 및 전략 로직 추가)
    def get_matrix_style(current_id, row_id, bg_color):
        if current_id == row_id:
            return f"style='background-color: {bg_color}; border: 3px solid #666; font-weight: bold; color: #333; height: 50px;'"
        else:
            return "style='background-color: white; border: 1px solid #eee; color: #999;'"
            
    strat_display = f"""
    <div style='background-color:#f1f8e9; padding:15px; border-left:5px solid #4caf50; margin-bottom:15px;'>
        <div style='font-size:18px; font-weight:bold; color:#2e7d32;'>🔔 추천 전략: {strat_type if strat_type else '-'}</div>
        <div style='font-size:14px; color:#555; margin-top:5px;'>💡 <b>선택 근거:</b> {strat_basis if strat_basis else '-'}</div>
    </div>
    """

    html_verdict_list = [
        # 점수 색상 Blue -> Black 변경
        f"<h3>3. Final Verdict: <span style='color:white;'>{score}점</span> - Dynamic Exit Matrix</h3>",
        strat_display, # 전략 추천 박스 추가
        "<div style='border: 2px solid #ccc; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; text-align: center;'>",
        f"<tr style='background-color: #333; color: white;'>",
        f"<th {th_style} style='color:white;'>점수 구간</th>",
        f"<th {th_style} style='color:white;'>최종 판정</th>",
        f"<th {th_style} style='color:white;'>🎯 익절 목표</th>",
        f"<th {th_style} style='color:white;'>🛑 손절 라인</th>",
        "</tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'panic', '#ffebee')}>",
        "<td>VIX 급등 / 구조 붕괴 / VVIX Trap</td><td>⛔ 매매 중단 (System Collapse)</td><td>-</td><td>-</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'super_strong', '#c8e6c9')}>",
        "<td>20점 이상 (투매 신호 포함)</td><td>💎💎 극강 추세 (Super Strong)</td><td style='color:green;'>+100%</td><td style='color:red;'>-300% (원금 4배)</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'strong', '#dff0d8')}>",
        "<td>12 ~ 19점</td><td>💎 추세 추종 (Strong)</td><td style='color:green;'>+75%</td><td style='color:red;'>-300% (원금 4배)</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'standard', '#ffffff')}>",
        "<td>8 ~ 11점</td><td>✅ 표준 대응 (Standard)</td><td style='color:green;'>+50%</td><td style='color:red;'>-200% (원금 3배)</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'weak', '#fff9c4')}>",
        "<td>5 ~ 7점</td><td>⚠️ 속전 속결 (Hit & Run)</td><td style='color:green;'>+30%</td><td style='color:red;'>-150% (원금 2.5배)</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'no_entry', '#f2dede')}>",
        "<td>5점 미만</td><td>🛡️ 진입 보류 (No Entry)</td><td>-</td><td>-</td></tr>",
        
        "</table>",
        "</div>"
    ]
    st.markdown("".join(html_verdict_list), unsafe_allow_html=True)

    # 4. Manual / Warning (매뉴얼 삭제됨) - 이 부분이 제거되었습니다
    # "진입 금지 (No Entry)" 메시지 표시 코드가 삭제됨

    st.markdown("---")
    st.subheader("📈 기술적 분석 차트")
    st.pyplot(create_charts(data))

if __name__ == "__main__":
    main()
