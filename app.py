import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === [앱 보안 설정] ===
APP_PASSWORD = "1979"

# === [페이지 기본 설정] ===
st.set_page_config(
    page_title="HK 옵션투자자문 (Grand Master v22.6 - ADL Momentum)",
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
    
    # 이동평균선
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
    
    # RSI(2)
    gain_2 = (delta.where(delta > 0, 0)).rolling(window=2).mean()
    loss_2 = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
    rs_2 = gain_2 / loss_2
    hist['RSI_2'] = 100 - (100 / (1 + rs_2))
    
    hist['Vol_MA20'] = hist['Volume'].rolling(window=20).mean()

    # ADL 데이터 추가 및 5MA 기울기 계산
    try:
        add_ticker = yf.Ticker("^ADD")
        add_hist = add_ticker.history(period="2y")
        
        if not add_hist.empty and len(add_hist) > 10:
            hist.index = hist.index.tz_localize(None).normalize()
            add_hist.index = add_hist.index.tz_localize(None).normalize()
            
            hist = hist.join(add_hist['Close'].rename('Net_Issues'), how='left')
            hist['Net_Issues'] = hist['Net_Issues'].ffill().fillna(0)
            hist['ADL'] = hist['Net_Issues'].cumsum()
            
            # [핵심 추가] ADL 5일 이동평균 및 기울기(Slope)
            hist['ADL_MA5'] = hist['ADL'].rolling(window=5).mean()
            hist['ADL_Slope'] = hist['ADL_MA5'].diff() # 전일 대비 증감분
            
        else:
            raise ValueError("^ADD 데이터 부족")
            
    except Exception as e:
        print(f"⚠️ ADL 데이터 수집 실패: {e}")
        # Fallback
        hist['Net_Issues'] = np.where(hist['Close'] > hist['Close'].shift(1), 1, -1)
        hist['Net_Issues'].iloc[0] = 0
        hist['ADL'] = hist['Net_Issues'].cumsum() * 100
        hist['ADL_MA5'] = hist['ADL'].rolling(window=5).mean()
        hist['ADL_Slope'] = hist['ADL_MA5'].diff()
    
    # VIX Data processing (기존과 동일)
    vix_ticker = yf.Ticker("^VIX")
    vix_hist = vix_ticker.history(period="1y")
    vvix_ticker = yf.Ticker("^VVIX")
    vvix_hist = vvix_ticker.history(period="1y")

    vix3m_val = None
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
            merged_df = pd.merge(df_vix, df_vix3m, left_index=True, right_index=True, suffixes=('_VIX', '_VIX3M'))
            if len(merged_df) >= 30:
                merged_df['Ratio'] = merged_df['Close_VIX'] / merged_df['Close_VIX3M']
                vix_term_df = merged_df
    except:
        pass

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
        'price': curr['Close'], 'price_prev': prev['Close'],
        'ma20': curr['MA20'], 'ma50': curr['MA50'], 'ma200': curr['MA200'],
        'rsi': curr['RSI'], 'rsi2': curr['RSI_2'], 
        'bb_upper': curr['BB_Upper'], 'bb_lower': curr['BB_Lower'],
        'macd': curr['MACD'], 'signal': curr['Signal'],
        'volume': curr['Volume'], 'vol_ma20': curr['Vol_MA20'], 'vol_pct': vol_pct,
        'vix': curr_vix, 'vix_prev': prev_vix, 'vix3m': vix3m_val, 'iv': current_iv,
        'hist': hist, 'vix_hist': vix_hist, 'vix3m_hist': vix3m_hist, 'vvix_hist': vvix_hist,
        'vix_term_df': vix_term_df
    }

# === [2] 로직 함수들 (기존 유지) ===
def detect_capitulation(data, log):
    if data['vix_term_df'] is None:
        log['capitulation'] = 'none'; return 0
    ratio = data['vix'] / data['vix3m'] if data['vix3m'] else 0
    vol_ratio = data['volume'] / data['vol_ma20']
    try:
        term_df = data['vix_term_df']
        if len(term_df) < 2: return 0
        ratio_prev = term_df['Ratio'].iloc[-2] 
        vol_prev = data['hist']['Volume'].iloc[-2]
        vol_ma20_prev = data['hist']['Vol_MA20'].iloc[-2]
        vol_ratio_prev = vol_prev / vol_ma20_prev
        if (ratio > 1.0 and vol_ratio > 1.5) and (ratio_prev > 1.0 and vol_ratio_prev > 1.5):
            log['capitulation'] = 'detected'; return 15
    except: pass
    log['capitulation'] = 'none'; return 0

def detect_vvix_trap(data, log):
    try:
        vix_hist = data['vix_hist']['Close']
        if len(vix_hist) < 5: return 0
        vix_ma3 = vix_hist.rolling(3).mean()
        vix_chg = ((vix_ma3.iloc[-1] - vix_ma3.iloc[-4])/vix_ma3.iloc[-4])*100
        vvix_hist = data['vvix_hist']['Close']
        if vvix_hist.empty: return 0
        vvix_chg = ((vvix_hist.iloc[-1] - vvix_hist.iloc[-2])/vvix_hist.iloc[-2])*100
        if abs(vix_chg) < 2.0 and vvix_chg > 5.0:
            log['vvix_trap'] = 'detected'; return -10
    except: pass
    log['vvix_trap'] = 'none'; return 0

def detect_rsi2_dip(data, log):
    try:
        rsi2 = data['rsi2']
        ratio = data['vix']/data['vix3m'] if data['vix3m'] else 1.1
        vvix_hist = data['vvix_hist']['Close']
        if len(vvix_hist) < 2: return 0
        if rsi2 < 10 and ratio < 1.0 and (vvix_hist.iloc[-1] < vvix_hist.iloc[-2]):
            log['rsi2_dip'] = 'detected'; return 8
    except: pass
    log['rsi2_dip'] = 'none'; return 0

# === [3] 전문가 로직 (ADL Slope 점수 추가) ===
def analyze_expert_logic(d):
    if d['price'] > d['ma50'] and d['price'] > d['ma200']: season = "SUMMER"
    elif d['price'] < d['ma50'] and d['price'] > d['ma200']: season = "AUTUMN"
    elif d['price'] < d['ma50'] and d['price'] < d['ma200']: season = "WINTER"
    else: season = "SPRING"
    
    score = 0; log = {}
    
    # VIX Ratio
    vix_ratio = d['vix'] / d['vix3m'] if d['vix3m'] else 1.0
    if vix_ratio > 1.0: score -= 10; log['term'] = 'backwardation'
    elif vix_ratio < 0.9: score += 3; log['term'] = 'contango'
    else: log['term'] = 'normal'
    log['vix_ratio'] = vix_ratio

    # RSI Logic
    curr_rsi = d['rsi']
    if curr_rsi < 30: score += 5 if season=="SUMMER" else 4; log['rsi']='under'
    elif curr_rsi >= 70: score -= 1 if season=="SUMMER" else 3; log['rsi']='over'
    else: log['rsi']='neutral'

    # VIX Level
    if d['vix'] > 35: score -= 5; log['vix']='panic'
    elif d['vix'] < 20: score += 2 if season=="SUMMER" else 0; log['vix']='stable'
    else: score -= 3; log['vix']='fear'

    # Bollinger Z-Score
    num = d['price'] - d['ma20']
    den = (d['bb_upper'] - d['ma20']) / 2.0
    z = 0 if den==0 else num/den
    log['z_score'] = z
    if z > 1.8: score -= 3; log['bb']='overbought'
    elif z < -1.8: score += 2; log['bb']='oversold'
    elif 0.5 < z <= 1.8: score += 1; log['bb']='uptrend'
    else: log['bb']='neutral'

    # Trend
    if d['price'] > d['ma20']: score += 2; log['trend']='up'
    else: log['trend']='down'

    # Volume
    if d['volume'] > d['vol_ma20']*1.5: score += 2; log['vol']='explode'
    else: log['vol']='normal'

    # MACD
    if d['macd'] > d['signal']: score += 3 if d['macd'] >=0 else 0; log['macd']='golden'
    else: score -= 3 if d['macd'] >=0 else 5; log['macd']='dead'

    # [신규 추가] ADL 5MA Slope Logic
    # 최근 ADL Slope가 양수면 가산점, 음수면 감점
    try:
        adl_slope = d['hist']['ADL_Slope'].iloc[-1]
        log['adl_slope'] = adl_slope
        if adl_slope > 0:
            score += 2 # 자금 유입 가속
            log['adl_status'] = 'accumulation'
        else:
            score -= 2 # 자금 이탈 가속
            log['adl_status'] = 'distribution'
    except:
        log['adl_status'] = 'none'

    # Others
    score += detect_capitulation(d, log)
    score += detect_vvix_trap(d, log)
    score += detect_rsi2_dip(d, log)

    return season, score, log

# === [4] 행동 결정 ===
def determine_action(score, season, data, log):
    vix_pct_change = ((data['vix'] - data['vix_prev'])/data['vix_prev'])*100
    
    if log.get('term') == 'backwardation': return None, "⛔ 매매 중단 (System Collapse)", "-", "-", "panic", "-", "-"
    if vix_pct_change > 15.0: return None, "⛔ 매매 중단 (VIX 급등)", "-", "-", "panic", "-", "-"
    if log.get('vvix_trap') == 'detected': return None, "⛔ 매매 중단 (VVIX Trap)", "-", "-", "panic", "-", "-"
    
    verdict = ""; target=""; stop=""; mid=""; strat_type=""; strat_basis=""
    
    if score >= 20: verdict="💎💎 극강 추세"; mid="super_strong"; target="100%"; stop="-300%"
    elif score >= 12: verdict="💎 추세 추종"; mid="strong"; target="75%"; stop="-300%"
    elif score >= 8: verdict="✅ 표준 대응"; mid="standard"; target="50%"; stop="-200%"
    elif score >= 5: verdict="⚠️ 속전 속결"; mid="weak"; target="30%"; stop="-150%"
    else: return None, "🛡️ 진입 보류", "-", "-", "no_entry", "-", "-"

    # CDS vs PCS
    if data['vix'] < 18.0 and score >= 12:
        strat_type = "CDS"
        strat_basis = f"VIX {data['vix']:.1f} (안정) + 점수 {score} (강세) 👉 방향성 추구"
        t_delta = 0.55
    else:
        strat_type = "PCS"
        strat_basis = f"VIX {data['vix']:.1f} / 점수 {score} 👉 시간가치 확보"
        t_delta = -0.10

    return t_delta, verdict, target, stop, mid, strat_type, strat_basis

# === [5] 옵션 찾기 (생략 - 기존과 동일) ===
def calculate_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return -0.5
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) - 1

def calculate_call_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def find_best_option(price, iv, target_delta, strategy_type):
    if target_delta is None: return None
    qqq = yf.Ticker("QQQ")
    try:
        options = qqq.options
        valid = []
        now = datetime.now()
        for d in options:
            days = (datetime.strptime(d, "%Y-%m-%d") - now).days
            if days >= 45: valid.append((d, days))
        if not valid: return None
        expiry, dte = min(valid, key=lambda x: x[1])
        T = dte/365.0; r=0.045
        
        best_k = 0; min_diff = 1.0; found_d = 0
        
        if strategy_type == "CDS":
            for k in range(int(price*0.9), int(price*1.1)):
                d = calculate_call_delta(price, k, T, r, iv)
                if abs(d - target_delta) < min_diff:
                    min_diff = abs(d - target_delta); best_k = k; found_d = d
            return {'type':'CDS', 'expiry':expiry, 'dte':dte, 'long':best_k, 'short':best_k+5, 'delta':found_d, 'width':5}
        else:
            for k in range(int(price*0.5), int(price)):
                d = calculate_put_delta(price, k, T, r, iv)
                if abs(d - target_delta) < min_diff:
                    min_diff = abs(d - target_delta); best_k = k; found_d = d
            return {'type':'PCS', 'expiry':expiry, 'dte':dte, 'short':best_k, 'long':best_k-5, 'delta':found_d, 'width':5}
    except: return None

# === [6] 차트 생성 (수정됨: ADL Momentum Chart 추가) ===
def create_charts(data):
    hist = data['hist'].copy()
    
    # Season Color logic
    conds = [
        (hist['Close']>hist['MA50'])&(hist['Close']>hist['MA200']),
        (hist['Close']<hist['MA50'])&(hist['Close']>hist['MA200']),
        (hist['Close']<hist['MA50'])&(hist['Close']<hist['MA200'])
    ]
    choices = ['SUMMER', 'AUTUMN', 'WINTER']
    hist['Season'] = np.select(conds, choices, default='SPRING')
    s_colors = {'SUMMER':'#FFEBEE', 'AUTUMN':'#FFF3E0', 'WINTER':'#E3F2FD', 'SPRING':'#E8F5E9'}
    
    # Figure Layout: 12 Rows now (Added ADL Slope Chart)
    fig = plt.figure(figsize=(10, 36))
    gs = fig.add_gridspec(12, 1, height_ratios=[2, 0.6, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1.5])
    
    # 1. Price
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(hist.index, hist['Close'], color='black', alpha=0.9, label='QQQ')
    ax1.plot(hist.index, hist['MA20'], color='green', ls='--', lw=1)
    ax1.plot(hist.index, hist['MA50'], color='blue', lw=1.5)
    ax1.plot(hist.index, hist['MA200'], color='red', lw=2)
    ax1.fill_between(hist.index, hist['BB_Upper'], hist['BB_Lower'], color='gray', alpha=0.1)
    ax1.set_title('QQQ Price Trend', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # 2. Volume
    ax_vol = fig.add_subplot(gs[1], sharex=ax1)
    c = ['red' if c<o else 'green' for c,o in zip(hist['Close'], hist['Open'])]
    ax_vol.bar(hist.index, hist['Volume'], color=c, alpha=0.5)
    ax_vol.plot(hist.index, hist['Vol_MA20'], color='black', lw=1)
    plt.setp(ax_vol.get_xticklabels(), visible=False)

    # 3. Trend Check
    ax_tr = fig.add_subplot(gs[2], sharex=ax1)
    ax_tr.plot(hist.index, hist['Close'], color='black', alpha=0.8)
    ax_tr.plot(hist.index, hist['MA50'], color='blue')
    # MACD Dead Zone
    mask = hist['MACD'] < hist['Signal']
    hist['dc_grp'] = (mask != mask.shift()).cumsum()
    for _, g in hist[mask].groupby('dc_grp'):
        ax_tr.axvspan(g.index[0], g.index[-1], color='#FFCDD2', alpha=0.4)
    ax_tr.set_title('Trend Check (Red Zone: MACD Dead)', fontsize=10, fontweight='bold')
    plt.setp(ax_tr.get_xticklabels(), visible=False)

    # 4. VIX Absolute
    ax_vx = fig.add_subplot(gs[3], sharex=ax1)
    ax_vx.plot(data['vix_hist'].index, data['vix_hist']['Close'], color='purple', label='VIX')
    if data['vix3m_hist'] is not None:
        ax_vx.plot(data['vix3m_hist'].index, data['vix3m_hist']['Close'], color='gray', ls=':', label='VIX3M')
    ax_vx.axhline(20, color='green', ls='--'); ax_vx.axhline(35, color='red', ls='--')
    ax_vx.legend(loc='upper right')
    plt.setp(ax_vx.get_xticklabels(), visible=False)

    # 5. VIX Ratio
    ax_rt = fig.add_subplot(gs[4], sharex=ax1)
    if data['vix_term_df'] is not None:
        df_t = data['vix_term_df']
        ax_rt.plot(df_t.index, df_t['Ratio'], color='black')
        ax_rt.axhline(1.0, color='red', ls='--')
        ax_rt.fill_between(df_t.index, df_t['Ratio'], 1.0, where=(df_t['Ratio']>1), color='red', alpha=0.2)
        ax_rt.fill_between(df_t.index, df_t['Ratio'], 1.0, where=(df_t['Ratio']<=1), color='green', alpha=0.2)
    plt.setp(ax_rt.get_xticklabels(), visible=False)

    # 6. RSI
    ax_rs = fig.add_subplot(gs[5], sharex=ax1)
    ax_rs.plot(hist.index, hist['RSI'], color='purple')
    ax_rs.axhline(70, color='red', ls='--'); ax_rs.axhline(30, color='green', ls='--')
    plt.setp(ax_rs.get_xticklabels(), visible=False)

    # 7. MACD
    ax_mc = fig.add_subplot(gs[6], sharex=ax1)
    ax_mc.plot(hist.index, hist['MACD'], color='blue')
    ax_mc.plot(hist.index, hist['Signal'], color='orange')
    ax_mc.bar(hist.index, hist['MACD']-hist['Signal'], color='gray', alpha=0.3)
    ax_mc.axhline(0, color='black', lw=0.8)
    plt.setp(ax_mc.get_xticklabels(), visible=False)

    # 8. VVIX/VIX Ratio
    ax_vr = fig.add_subplot(gs[7], sharex=ax1)
    try:
        df_v = data['vix_hist'][['Close']]; df_vv = data['vvix_hist'][['Close']]
        df_v.index = df_v.index.tz_localize(None).normalize()
        df_vv.index = df_vv.index.tz_localize(None).normalize()
        m = pd.merge(df_v, df_vv, left_index=True, right_index=True, suffixes=('_V', '_VV'))
        m['R'] = m['Close_VV']/m['Close_V']
        ax_vr.plot(m.index, m['R'], color='#333'); ax_vr.axhline(7, color='red', ls=':'); ax_vr.axhline(4, color='green', ls=':')
    except: pass
    plt.setp(ax_vr.get_xticklabels(), visible=False)

    # 9. RSI(2)
    ax_r2 = fig.add_subplot(gs[8], sharex=ax1)
    ax_r2.plot(hist.index, hist['RSI_2'], color='gray')
    ax_r2.axhline(10, color='green', ls='--'); ax_r2.axhline(90, color='red', ls='--')
    plt.setp(ax_r2.get_xticklabels(), visible=False)

    # 10. ADL Raw
    ax_ad = fig.add_subplot(gs[9], sharex=ax1)
    if 'ADL' in hist.columns:
        ax_ad.plot(hist.index, hist['ADL'], color='black', lw=1.5, label='ADL')
        ax_ad.plot(hist.index, hist['ADL_MA5'], color='orange', ls='--', lw=1, label='5MA')
        ax_ad.set_title('ADL (Accumulation/Distribution)', fontsize=10, fontweight='bold')
        ax_ad.legend(loc='upper left')
    plt.setp(ax_ad.get_xticklabels(), visible=False)

    # [신규] 11. ADL 5MA Slope (Momentum) - Index 10
    ax_sl = fig.add_subplot(gs[10], sharex=ax1)
    if 'ADL_Slope' in hist.columns:
        # 0보다 크면 Green, 작으면 Red
        cols = ['green' if x > 0 else 'red' for x in hist['ADL_Slope']]
        ax_sl.bar(hist.index, hist['ADL_Slope'], color=cols, alpha=0.7)
        ax_sl.axhline(0, color='black', lw=0.8)
        ax_sl.set_title('ADL 5MA Slope (Money Flow Velocity)', fontsize=10, fontweight='bold')
    else:
        ax_sl.text(0.5, 0.5, "No ADL Slope Data", transform=ax_sl.transAxes, ha='center')
    plt.setp(ax_sl.get_xticklabels(), visible=False)

    # 12. Jaws - Index 11
    ax_jw = fig.add_subplot(gs[11], sharex=ax1)
    if 'ADL' in hist.columns:
        def norm(s): return (s-s.min())/(s.max()-s.min())
        np_p = norm(hist['Close']); np_a = norm(hist['ADL'])
        ax_jw.plot(hist.index, np_p, color='black', lw=1.5, label='Price')
        ax_jw.plot(hist.index, np_a, color='blue', alpha=0.6, label='ADL')
        ax_jw.fill_between(hist.index, np_p, np_a, where=(np_p>np_a), color='red', alpha=0.3, label='Div')
        
        # Bearish Div Logic
        pks = argrelextrema(hist['Close'].values, np.greater, order=5)[0]
        dx=[]; dy=[]; dlines=[]
        for i in range(1, len(pks)):
            p = pks[i-1]; c = pks[i]
            if hist['Close'].iloc[c] > hist['Close'].iloc[p] and hist['ADL'].iloc[c] < hist['ADL'].iloc[p]:
                dx.append(hist.index[c]); dy.append(np_p.iloc[c]); dlines.append((p, c))
        if dx: ax_jw.scatter(dx, dy, color='red', marker='v', s=100, zorder=5)
        for p, c in dlines:
            ax_jw.plot([hist.index[p], hist.index[c]], [np_p.iloc[p], np_p.iloc[c]], color='green', ls='--')
            ax_jw.plot([hist.index[p], hist.index[c]], [np_a.iloc[p], np_a.iloc[c]], color='red', ls='--')
            
        ax_jw.set_title('Jaws Divergence', fontsize=10, fontweight='bold')
    
    # Background coloring
    hist['grp'] = (hist['Season']!=hist['Season'].shift()).cumsum()
    axes_bg = [ax1, ax_vol, ax_vx, ax_rt, ax_rs, ax_mc, ax_vr, ax_r2, ax_ad, ax_sl, ax_jw]
    for ax in axes_bg:
        for _, g in hist.groupby('grp'):
            ax.axvspan(g.index[0], g.index[-1], color=s_colors[g['Season'].iloc[0]], alpha=0.4, zorder=0)

    plt.tight_layout()
    return fig

# === [Main] ===
def main():
    st.title("🦅 HK Advisory (Grand Master v22.6 - ADL Momentum)")
    st.caption("Logic: MACD 4-Zone & Jaws Divergence & ADL 5MA Slope")

    with st.spinner('Calculating...'):
        try:
            data = get_market_data()
            season, score, log = analyze_expert_logic(data)
            delta, verdict, tgt, stop, mid, st_type, st_basis = determine_action(score, season, data, log)
            opt = find_best_option(data['price'], data['iv'], delta, st_type)
        except Exception as e:
            st.error(f"Error: {e}"); return

    # Sidebar
    st.sidebar.title("System Status")
    st.sidebar.metric("Total Score", f"{score}", delta=verdict)
    
    # [신규] ADL Slope Display
    slope_val = log.get('adl_slope', 0)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌊 Money Flow")
    if slope_val > 0:
        st.sidebar.success(f"ADL Slope: +{slope_val:.0f} (Inflow 🚀)")
    else:
        st.sidebar.error(f"ADL Slope: {slope_val:.0f} (Outflow 🩸)")

    st.pyplot(create_charts(data))

if __name__ == "__main__":
    main()
