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
    page_title="HK 옵션투자자문 (Grand Master v21.1 - Safety First)",
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
    
    # RSI(14) - 기존
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    # [신규] RSI(2) - 단기 눌림목용
    gain_2 = (delta.where(delta > 0, 0)).rolling(window=2).mean()
    loss_2 = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
    rs_2 = gain_2 / loss_2
    hist['RSI_2'] = 100 - (100 / (1 + rs_2))
    
    hist['Vol_MA20'] = hist['Volume'].rolling(window=20).mean()
    
    # 2. VIX, VIX3M, VVIX 데이터 처리
    vix_ticker = yf.Ticker("^VIX")
    vix_hist = vix_ticker.history(period="1y")
    
    # [신규] VVIX 데이터 수집
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
            
            # Timezone 제거 및 날짜 정규화
            df_vix = vix_hist[['Close']].copy()
            df_vix3m = vix3m_hist[['Close']].copy()
            
            df_vix.index = df_vix.index.tz_localize(None).normalize()
            df_vix3m.index = df_vix3m.index.tz_localize(None).normalize()
            
            # VIX Term Structure 병합
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
    
    # [신규] VVIX 데이터 정규화 및 병합 (동기화)
    try:
        if not vvix_hist.empty:
            vvix_clean = vvix_hist[['Close']].copy()
            vvix_clean.index = vvix_clean.index.tz_localize(None).normalize()
    except Exception as e:
        print(f"Error processing VVIX: {e}")

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
    """
    [신규 1] 투매 감지: 2일 연속 공포 구조(Ratio>1.0) + 거래량 폭증(>1.5배)
    """
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
    """
    [신규 2] VVIX Trap: VIX 안정(횡보) + VVIX 급등
    """
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
    """
    [신규 3] RSI(2) 눌림목: 과매도(<10) + 구조 안정 + VVIX 하락
    """
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

# === [3] 전문가 로직 (수정된 Bollinger Logic 적용) ===
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

    # 4. Bollinger Logic (Z-Score & Risk Managed) - [수정됨]
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
    else: # z_score <= -1.8
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

    # 7. MACD Logic
    if d['macd_prev'] < 0 and d['macd'] >= 0:
        pts = 3
        score += pts
        log['macd'] = 'break_up'
    elif d['macd_prev'] > 0 and d['macd'] <= 0:
        if season == "WINTER": pts = -8
        else: pts = -5
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

    # === [신규 항목 점수 누적] ===
    pts_cap = detect_capitulation(d, log)
    score += pts_cap
    
    pts_vvix = detect_vvix_trap(d, log)
    score += pts_vvix
    
    pts_rsi2 = detect_rsi2_dip(d, log)
    score += pts_rsi2

    return season, score, log

# === [4] 행동 결정 ===
def determine_action(score, season, data, log):
    vix_pct_change = ((data['vix'] - data['vix_prev']) / data['vix_prev']) * 100
    TARGET_DELTA = -0.10
    
    if log.get('term') == 'backwardation':
        return TARGET_DELTA, "⛔ 매매 중단 (System Collapse)", "-", "-", "panic"

    if vix_pct_change > 15.0:
        return TARGET_DELTA, "⛔ 매매 중단 (VIX 급등)", "-", "-", "panic"
    
    if log.get('vvix_trap') == 'detected':
        return TARGET_DELTA, "⛔ 매매 중단 (VVIX Trap)", "-", "-", "panic"
    
    if score >= 20:
        return TARGET_DELTA, "💎💎 극강 추세 (Super Strong)", "100%", "300%", "super_strong"
    elif score >= 12:
        return TARGET_DELTA, "💎 추세 추종 (Strong)", "75%", "300%", "strong"
    elif 8 <= score < 12:
        return TARGET_DELTA, "✅ 표준 대응 (Standard)", "50%", "200%", "standard"
    elif 5 <= score < 8:
        return TARGET_DELTA, "⚠️ 속전 속결 (Hit & Run)", "30%", "150%", "weak"
    else:
        return None, "🛡️ 진입 보류", "-", "-", "no_entry"

# === [5] 옵션 찾기 ===
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

# === [6] 차트 (8개 서브플롯 - Capitulation 제거) ===
def create_charts(data):
    hist = data['hist']
    
    # 높이와 행 개수 수정 (9 -> 8)
    fig = plt.figure(figsize=(10, 24))
    gs = fig.add_gridspec(8, 1, height_ratios=[2, 0.6, 1, 1, 1, 1, 1, 1])
    
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

    # 3. VIX Term Structure
    ax_ratio = fig.add_subplot(gs[2], sharex=ax1)
    term_data = data.get('vix_term_df')
    
    if term_data is not None and not term_data.empty:
        ax_ratio.plot(term_data.index, term_data['Ratio'], color='black', lw=1.2, label='Ratio (VIX/VIX3M)')
        ax_ratio.axhline(1.0, color='red', ls='--', alpha=0.8, lw=1)
        ax_ratio.axhline(0.9, color='green', ls='--', alpha=0.8, lw=1)
        ax_ratio.fill_between(term_data.index, term_data['Ratio'], 1.0, 
                         where=(term_data['Ratio'] > 1.0), 
                         color='red', alpha=0.2, label='Backwardation')
        ax_ratio.fill_between(term_data.index, term_data['Ratio'], 0.9, 
                         where=(term_data['Ratio'] < 0.9), 
                         color='green', alpha=0.2, label='Contango')
        ax_ratio.legend(loc='upper right')
    else:
        ax_ratio.text(0.5, 0.5, "데이터 부족", transform=ax_ratio.transAxes, color='red')
        
    ax_ratio.set_title('VIX Term Structure (Ratio = VIX / VIX3M)', fontsize=12, fontweight='bold')
    ax_ratio.grid(True, alpha=0.3)
    plt.setp(ax_ratio.get_xticklabels(), visible=False)

    # 4. RSI(14)
    ax_rsi = fig.add_subplot(gs[3], sharex=ax1)
    ax_rsi.plot(hist.index, hist['RSI'], color='purple', label='RSI(14)')
    ax_rsi.axhline(70, color='red', ls='--', alpha=0.7)
    ax_rsi.axhline(30, color='green', ls='--', alpha=0.7)
    ax_rsi.fill_between(hist.index, hist['RSI'], 70, where=(hist['RSI'] >= 70), color='red', alpha=0.3)
    ax_rsi.fill_between(hist.index, hist['RSI'], 30, where=(hist['RSI'] <= 30), color='green', alpha=0.3)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI (14)', fontsize=12, fontweight='bold')
    ax_rsi.grid(True, alpha=0.3)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # 5. MACD
    ax2 = fig.add_subplot(gs[4], sharex=ax1)
    ax2.plot(hist.index, hist['MACD'], label='MACD', color='blue')
    ax2.plot(hist.index, hist['Signal'], label='Signal', color='orange')
    ax2.bar(hist.index, hist['MACD']-hist['Signal'], color='gray', alpha=0.3)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title('MACD', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # 6. VIX Level
    ax3 = fig.add_subplot(gs[5], sharex=ax1)
    ax3.plot(data['vix_hist'].index, data['vix_hist']['Close'], color='purple', label='VIX (Spot)')
    if data['vix3m_hist'] is not None and not data['vix3m_hist'].empty:
         ax3.plot(data['vix3m_hist'].index, data['vix3m_hist']['Close'], color='gray', ls=':', label='VIX3M')
    
    ax3.axhline(35, color='red', ls='--')
    ax3.axhline(20, color='green', ls='--')
    ax3.set_title('VIX Level (Absolute)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # 7. VIX vs VVIX Divergence
    ax_div = fig.add_subplot(gs[6], sharex=ax1)
    line1 = ax_div.plot(data['vix_hist'].index, data['vix_hist']['Close'], 
                       color='purple', label='VIX', linewidth=1.5)
    ax_div.set_ylabel('VIX', color='purple')
    ax_div.tick_params(axis='y', labelcolor='purple')
    
    ax_vvix = ax_div.twinx()
    line2 = ax_vvix.plot(data['vvix_hist'].index, data['vvix_hist']['Close'], 
                        color='orange', linestyle='--', label='VVIX', linewidth=1.2)
    ax_vvix.set_ylabel('VVIX', color='orange')
    ax_vvix.tick_params(axis='y', labelcolor='orange')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_div.legend(lines, labels, loc='upper left')
    ax_div.set_title('VIX vs VVIX Divergence (Trap Detector)', fontsize=12, fontweight='bold')
    ax_div.grid(True, alpha=0.3)
    plt.setp(ax_div.get_xticklabels(), visible=False)

    # 8. RSI(2) (수정됨: 색상 변경 및 포인트 강조)
    ax_rsi2 = fig.add_subplot(gs[7], sharex=ax1)
    ax_rsi2.plot(hist.index, hist['RSI_2'], color='gray', label='RSI(2)', linewidth=1.2)
    ax_rsi2.axhline(10, color='green', linestyle='--', alpha=0.7)
    ax_rsi2.axhline(90, color='red', linestyle='--', alpha=0.7)
    
    ax_rsi2.fill_between(hist.index, hist['RSI_2'], 10, where=(hist['RSI_2'] < 10),
                        color='green', alpha=0.3, label='Buy Zone')
    ax_rsi2.fill_between(hist.index, hist['RSI_2'], 90, where=(hist['RSI_2'] > 90),
                        color='red', alpha=0.3, label='Danger')
    
    # 마지막 시점 빨간색 동그라미 마커 추가
    ax_rsi2.scatter(hist.index[-1], hist['RSI_2'].iloc[-1], color='red', s=50, zorder=5)

    ax_rsi2.set_ylim(0, 100)
    ax_rsi2.set_title('RSI(2) - Short-term Pullback', fontsize=12, fontweight='bold')
    ax_rsi2.legend(loc='upper right')
    ax_rsi2.grid(True, alpha=0.3)
    # 마지막 차트이므로 X축 라벨 표시
    ax_rsi2.set_xlabel('Date', fontsize=10)
    
    plt.tight_layout()
    return fig

# === [메인 화면] ===
def main():
    st.title("🦅 HK Advisory (Grand Master v21.1 - Safety First)")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Logic: Bollinger Safe Mode")

    with st.spinner('시장 구조 및 신규 위험 지표(VVIX) 정밀 분석 중...'):
        try:
            data = get_market_data()
            season, score, log = analyze_expert_logic(data)
            target_delta, verdict_text, profit_target, stop_loss, matrix_id = determine_action(score, season, data, log)
            strategy = find_best_option(data['price'], data['iv'], target_delta)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            import traceback
            st.text(traceback.format_exc())
            return

    # [Sidebar] 시스템 상태 및 실시간 지표
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
    
    # Z-Score display for table
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

    # 2. Scorecard (확장판)
    html_score_list = [
        "<h3>2. Expert Matrix Scorecard (확장판 v21)</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; text-align: center;'>",
        "<tr>",
        f"<th {th_style}>지표</th><th {th_style}>상태</th>",
        f"<th {th_style}>☀️</th><th {th_style}>🍂</th><th {th_style}>❄️</th><th {th_style}>🌱</th>",
        f"<th {th_style}>Logic</th>",
        "</tr>",
        
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
        
        f"<tr><td {td_style}><b>투매 신호</b><br><span style='font-size:11px; color:#888;'>Capitulation</span></td>",
        f"<td {td_style}><b>2일 연속</b><br>Ratio&gt;1.0 + Vol&gt;1.5x</td>",
        f"<td colspan='4' {hl_score('capitulation', 'detected', 'ALL')}><b style='color:green;'>+15 (스나이퍼)</b></td>",
        f"<td align='left' {td_style}><b>💎 극강 바닥</b></td></tr>",
        
        f"<tr><td {td_style}><b>VVIX Trap</b><br><span style='font-size:11px; color:#888;'>변동성 함정</span></td>",
        f"<td {td_style}><b>위험 경보</b><br>VIX 안정 + VVIX 급등</td>",
        f"<td colspan='4' {hl_score('vvix_trap', 'detected', 'ALL')}><b style='color:red;'>-10 (차단)</b></td>",
        f"<td align='left' {td_style}><b>🚨 폭등 예고</b></td></tr>",

        f"<tr><td rowspan='4' {td_style}>RSI(14)<br><span style='font-size:11px; color:#888; font-weight:normal'>지금 싼가? 비싼가?</span></td>",
        f"<td {td_style}>과열 (>70)</td>",
        f"<td {hl_score('rsi', 'over', 'SUMMER')}>-1</td><td {hl_score('rsi', 'over', 'AUTUMN')}>-3</td><td {hl_score('rsi', 'over', 'WINTER')}><b style='color:red;'>-10</b></td><td {hl_score('rsi', 'over', 'SPRING')}>-2</td>",
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
        
        f"<tr><td {td_style}><b>RSI(2)</b><br><span style='font-size:11px; color:#888;'>단기 눌림목</span></td>",
        f"<td {td_style}><b>과매도</b><br>(&lt;10 + 구조안정)</td>",
        f"<td colspan='4' {hl_score('rsi2_dip', 'detected', 'ALL')}><b style='color:green;'>+8 (반등)</b></td>",
        f"<td align='left' {td_style}><b>✅ 눌림목 매수</b></td></tr>",

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
        
        # Bollinger Band (Z-Score) Section [수정됨]
        f"<tr><td rowspan='5' {td_style}>BB (Z-Score)<br><span style='font-size:11px; color:blue;'>Z: {z_disp}</span></td>",
        f"<td {td_style} style='color:red;'><b>과열/위험</b><br>(Z &gt; 1.8)</td>",
        f"<td colspan='4' {hl_score('bb', 'overbought_danger', 'ALL')}><b style='color:red;'>-3 (감점)</b></td>",
        f"<td align='left' {td_style}><b>Mean Reversion</b></td></tr>",
        
        f"<tr><td {td_style}><b>상승 추세</b><br>(0.5 &lt; Z &le; 1.8)</td>",
        f"<td colspan='4' {hl_score('bb', 'uptrend', 'ALL')}>+1</td>",
        f"<td align='left' {td_style}>추세 지속</td></tr>",
        
        f"<tr><td {td_style}><b>중립/횡보</b><br>(-0.5 &le; Z &le; 0.5)</td>",
        f"<td colspan='4' {hl_score('bb', 'neutral', 'ALL')}>0</td>",
        f"<td align='left' {td_style}>방향 탐색</td></tr>",
        
        f"<tr><td {td_style}><b>저평가/매수</b><br>(-1.8 &lt; Z &lt; -0.5)</td>",
        f"<td colspan='4' {hl_score('bb', 'dip_buying', 'ALL')}>+2</td>",
        f"<td align='left' {td_style}>저점 매수</td></tr>",
        
        f"<tr><td {td_style}><b>과매도/바닥</b><br>(Z &le; -1.8)</td>",
        f"<td colspan='4' {hl_score('bb', 'oversold_guard', 'ALL')}><b>+1 (보수적)</b></td>",
        f"<td align='left' {td_style}><b>안전마진 확보</b></td></tr>",
        
        f"<tr><td {td_style}>추세 (20MA)</td><td {td_style}>20일선 위</td>",
        f"<td {hl_score('trend', 'up', 'SUMMER')}>+2</td><td {hl_score('trend', 'up', 'AUTUMN')}>+2</td><td {hl_score('trend', 'up', 'WINTER')}>+3</td><td {hl_score('trend', 'up', 'SPRING')}>+3</td>",
        f"<td align='left' {td_style}>회복</td></tr>",
        
        f"<tr><td {td_style}>거래량</td><td {td_style}>폭증 (>150%)</td>",
        f"<td {hl_score('vol', 'explode', 'SUMMER')}>+2</td><td {hl_score('vol', 'explode', 'AUTUMN')}>+3</td><td {hl_score('vol', 'explode', 'WINTER')}>+3</td><td {hl_score('vol', 'explode', 'SPRING')}>+2</td>",
        f"<td align='left' {td_style}><b>손바뀜</b></td></tr>",
        
        f"<tr><td rowspan='4' {td_style}>MACD</td>",
        f"<td {td_style}>📈 상승 전환<br>(골든크로스)</td>",
        f"<td {hl_score('macd', 'break_up', 'SUMMER')}>+3</td><td {hl_score('macd', 'break_up', 'AUTUMN')}>+3</td><td {hl_score('macd', 'break_up', 'WINTER')}>+3</td><td {hl_score('macd', 'break_up', 'SPRING')}>+3</td>",
        f"<td align='left' {td_style}><b>강력 매수</b></td></tr>",
        
        f"<tr><td {td_style}>☁️ 상승 추세</td>",
        f"<td {hl_score('macd', 'above', 'SUMMER')}>+1</td><td {hl_score('macd', 'above', 'AUTUMN')}>+1</td><td {hl_score('macd', 'above', 'WINTER')}>+1</td><td {hl_score('macd', 'above', 'SPRING')}>+1</td>",
        f"<td align='left' {td_style}>순풍</td></tr>",
        
        f"<tr><td {td_style}>📉 하락 전환<br>(데드크로스)</td>",
        f"<td {hl_score('macd', 'break_down', 'SUMMER')}>-5</td><td {hl_score('macd', 'break_down', 'AUTUMN')}>-5</td><td {hl_score('macd', 'break_down', 'WINTER')}><b style='color:red;'>-8</b></td><td {hl_score('macd', 'break_down', 'SPRING')}>-5</td>",
        f"<td align='left' {td_style}><b>강력 매도</b></td></tr>",
        
        f"<tr><td {td_style}>☔ 하락 추세</td>",
        f"<td {hl_score('macd', 'below', 'SUMMER')}>-1</td><td {hl_score('macd', 'below', 'AUTUMN')}>-1</td><td {hl_score('macd', 'below', 'WINTER')}>-1</td><td {hl_score('macd', 'below', 'SPRING')}>-1</td>",
        f"<td align='left' {td_style}>역풍</td></tr>",
        
        "</table>"
    ]
    st.markdown("".join(html_score_list), unsafe_allow_html=True)

    # 3. Final Verdict (확장판)
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
        "<div style='padding: 10px; background-color: #f9f9f9; text-align: center; color: #555; font-size: 13px;'>",
        "※ <b>설정:</b> Delta -0.10 (Fixed) / DTE 45일 / Spread $5<br>",
        "※ 손절 라인은 프리미엄 가격 기준입니다. (예: $1.0 진입 시, 200% 손절은 $3.0 도달 시 청산)<br>",
        "<b style='color:red;'>※ 신규:</b> 투매 신호(+15) + RSI(2) 눌림목(+8) 시 최대 23점 초과 가능",
        "</div></div>"
    ]
    st.markdown("".join(html_verdict_list), unsafe_allow_html=True)

    # 4. Manual / Warning
    if strategy and matrix_id != 'no_entry' and matrix_id != 'panic':
        html_manual_list = [
            "<div style='border: 2px solid #2196F3; padding: 15px; margin-top: 20px; border-radius: 10px; background-color: #ffffff; color: black;'>",
            "<h3 style='color: #2196F3; margin-top: 0;'>👮‍♂️ 주문 상세 매뉴얼 (Action Plan)</h3>",
            "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; text-align: center; font-size: 13px; margin-bottom: 15px;'>",
            "<tr style='background-color: #e3f2fd; border: 1px solid #ddd;'>",
            "<th style='padding: 8px; border: 1px solid #ddd;'>구분</th><th style='padding: 8px; border: 1px solid #ddd;'>행동</th><th style='padding: 8px; border: 1px solid #ddd;'>시간</th><th style='padding: 8px; border: 1px solid #ddd;'>방식</th></tr>",
            
            "<tr><td style='padding: 8px; border: 1px solid #ddd; font-weight:bold;'>진입 (Entry)</td><td style='padding: 8px; border: 1px solid #ddd;'>신규 포지션 구축</td><td style='padding: 8px; border: 1px solid #ddd;'>🕒 <b>마감 30분 전</b></td><td style='padding: 8px; border: 1px solid #ddd;'><b>수동 진입</b></td></tr>",
            "<tr><td style='padding: 8px; border: 1px solid #ddd; font-weight:bold; color:red;'>손절 (Loss)</td><td style='padding: 8px; border: 1px solid #ddd;'>위기 탈출</td><td style='padding: 8px; border: 1px solid #ddd;'>🚨 <b>언제든지</b></td><td style='padding: 8px; border: 1px solid #ddd;'><b>자동 감시 주문</b></td></tr>",
            "<tr><td style='padding: 8px; border: 1px solid #ddd; font-weight:bold; color:green;'>익절 (Win)</td><td style='padding: 8px; border: 1px solid #ddd;'>수익 실현</td><td style='padding: 8px; border: 1px solid #ddd;'>💰 <b>장중 아무 때나</b></td><td style='padding: 8px; border: 1px solid #ddd;'><b>GTC 지정가 주문</b></td></tr>",
            "</table>",
            
            "<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px; font-size: 14px;'>",
            f"<b>✅ 현재 포지션 목표 (Spec):</b><br>",
            f"• <b>종목:</b> QQQ Put Credit Spread (만기 {strategy['expiry']}, DTE {strategy['dte']}일)<br>",
            f"• <b>Strike:</b> Short ${strategy['short']} / Long ${strategy['long']} (Width ${strategy['width']})<br>",
            "<hr style='margin: 8px 0; border: 0; border-top: 1px solid #ddd;'>",
            f"• <b>익절 (Target):</b> 진입가 대비 <b style='color:green;'>{profit_target}</b> 도달 시<br>",
            f"• <b>손절 (Stop):</b> 진입가 대비 <b style='color:red;'>{stop_loss}</b> 도달 시 (즉시 청산)",
            "</div></div>"
        ]
        st.markdown("".join(html_manual_list), unsafe_allow_html=True)
    else:
        if matrix_id == 'panic':
            reason = "VIX 급등, 구조 붕괴(Back.), 또는 VVIX Trap이 감지되었습니다."
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
