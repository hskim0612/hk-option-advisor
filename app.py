import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import concurrent.futures

# === [App Security] ===
APP_PASSWORD = "1979"

# === [Page Configuration] ===
st.set_page_config(
    page_title="HK Options Advisory (Grand Master v24.0 - ADL Band)",
    page_icon="🦅",
    layout="wide"
)

# Chart Style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'

# === [0] Login Screen ===
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    st.title("🔒 HK Advisory Secure Access")
    password = st.text_input("Enter Password", type="password")
    
    if st.button("Login"):
        if password == APP_PASSWORD:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Incorrect Password")
    return False

if not check_password():
    st.stop()

# === [1] Data Collection & Processing ===
def fetch_ticker_data(ticker, period="2y"):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        return ticker, t, hist
    except Exception as e:
        return ticker, None, pd.DataFrame()

@st.cache_data(ttl=1800)
def get_market_data():
    tickers_to_fetch = [
        ("QQQ", "2y"), ("^ADD", "2y"), ("^VIX", "1y"), 
        ("^VVIX", "1y"), ("^SKEW", "1y"), ("^VIX3M", "1y")
    ]
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_ticker = {executor.submit(fetch_ticker_data, t, p): t for t, p in tickers_to_fetch}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker, t_obj, hist = future.result()
            results[ticker] = {'obj': t_obj, 'hist': hist}

    # 1. Process QQQ Data
    qqq = results["QQQ"]['obj']
    hist = results["QQQ"]['hist'].copy()
    
    # Moving Averages & Indicators
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    
    # Bollinger Bands (Price)
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

    # Process ADL Data & Bollinger Bands
    add_hist = results["^ADD"]['hist']
    if not add_hist.empty and len(add_hist) > 10:
        hist.index = hist.index.tz_localize(None).normalize()
        add_hist.index = add_hist.index.tz_localize(None).normalize()
        
        hist = hist.join(add_hist['Close'].rename('Net_Issues'), how='left')
        hist['Net_Issues'] = hist['Net_Issues'].ffill().fillna(0)
        hist['ADL'] = hist['Net_Issues'].cumsum()
    else:
        # Fallback Logic
        hist['Net_Issues'] = np.where(hist['Close'] > hist['Close'].shift(1), 1, -1)
        hist['Net_Issues'].iloc[0] = 0
        hist['ADL'] = hist['Net_Issues'].cumsum() * 100
        
    # [NEW] ADL Bollinger Bands Logic
    hist['ADL_MA20'] = hist['ADL'].rolling(window=20).mean()
    hist['ADL_Std'] = hist['ADL'].rolling(window=20).std()
    hist['ADL_Upper'] = hist['ADL_MA20'] + (hist['ADL_Std'] * 2)
    hist['ADL_Lower'] = hist['ADL_MA20'] - (hist['ADL_Std'] * 2)
    
    # ADL Z-Score (Avoid division by zero)
    numerator = hist['ADL'] - hist['ADL_MA20']
    denominator = hist['ADL_Std']
    hist['ADL_Z'] = np.where(denominator == 0, 0, numerator / denominator)

    # 2. Process VIX, VIX3M, VVIX, SKEW
    vix_hist = results["^VIX"]['hist']
    vvix_hist = results["^VVIX"]['hist']
    vix3m_hist = results["^VIX3M"]['hist']
    skew_hist = results["^SKEW"]['hist']
    
    vix3m_val = None
    vix_term_df = None

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
            
    # Process SKEW & VVIX timezones
    if not skew_hist.empty: skew_hist.index = skew_hist.index.tz_localize(None).normalize()
    if not vvix_hist.empty: vvix_hist.index = vvix_hist.index.tz_localize(None).normalize()

    curr = hist.iloc[-1]
    prev = hist.iloc[-2]
    curr_vix = vix_hist['Close'].iloc[-1]
    prev_vix = vix_hist['Close'].iloc[-2]
    
    # [NEW] IV Rank Calculation (52-week)
    try:
        vix_52w = vix_hist['Close'].tail(252)
        vix_min = vix_52w.min()
        vix_max = vix_52w.max()
        if vix_max == vix_min: iv_rank = 50.0
        else: iv_rank = ((curr_vix - vix_min) / (vix_max - vix_min)) * 100
    except:
        iv_rank = 50.0

    vol_pct = (curr['Volume'] / curr['Vol_MA20']) * 100

    # IV Calculation
    try:
        dates = qqq.options
        chain = qqq.option_chain(dates[1])
        current_iv = chain.calls['impliedVolatility'].mean()
    except:
        current_iv = curr_vix / 100.0

    return {
        'price': curr['Close'], 'price_prev': prev['Close'], 'open': curr['Open'],
        'ma20': curr['MA20'], 'ma50': curr['MA50'], 'ma200': curr['MA200'],
        'rsi': curr['RSI'], 'rsi_prev': prev['RSI'], 'rsi2': curr['RSI_2'],
        'bb_upper': curr['BB_Upper'], 'bb_lower': curr['BB_Lower'], 
        'macd': curr['MACD'], 'signal': curr['Signal'],
        'volume': curr['Volume'], 'vol_ma20': curr['Vol_MA20'], 'vol_pct': vol_pct,
        'vix': curr_vix, 'vix_prev': prev_vix, 'vix3m': vix3m_val,
        'iv': current_iv, 'iv_rank': iv_rank,
        'adl_z': curr['ADL_Z'], # ADL Z-Score
        'hist': hist, 'vix_hist': vix_hist, 'vix3m_hist': vix3m_hist, 'vvix_hist': vvix_hist,
        'skew_hist': skew_hist, 'vix_term_df': vix_term_df
    }

# === [2] Advanced Logic Functions (Capitulation / Trap / Dip) ===
def detect_capitulation(data, log):
    if data['vix_term_df'] is None: return 0
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
    except: pass
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
    except: pass
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
    except: pass
    log['rsi2_dip'] = 'none'
    return 0

# === [3] Expert Logic (MACD 4-Zone + SKEW) ===
def analyze_expert_logic(d):
    if d['price'] > d['ma50'] and d['price'] > d['ma200']: season = "SUMMER"
    elif d['price'] < d['ma50'] and d['price'] > d['ma200']: season = "AUTUMN"
    elif d['price'] < d['ma50'] and d['price'] < d['ma200']: season = "WINTER"
    else: season = "SPRING"
    
    score = 0
    log = {}
    
    # 1. VIX Term Structure
    vix_ratio = d['vix'] / d['vix3m'] if d['vix3m'] else 1.0
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
            if abs(-1-i) > len(hist_rsi): break
            if hist_rsi.iloc[-1-i] < 30:
                days_since_escape = i
                is_escape_mode = True
                break
    
    if curr_rsi < 30:
        pts = 5 if season == "SUMMER" else 4 if season in ["AUTUMN", "SPRING"] else 0
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
        pts = 1 if season in ["SUMMER", "SPRING"] else 0 if season == "AUTUMN" else -1
        score += pts
        log['rsi'] = 'neutral'

    # 3. VIX Level
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
    else: log['vix'] = 'none'

    # 4. Bollinger Z-Score
    numerator = d['price'] - d['ma20']
    denominator = (d['bb_upper'] - d['ma20']) / 2.0
    z_score = 0 if denominator == 0 else numerator / denominator
    log['z_score'] = z_score

    if z_score > 1.8:
        score += -3
        log['bb'] = 'overbought_danger'
    elif 0.5 < z_score <= 1.8:
        score += 1
        log['bb'] = 'uptrend'
    elif -0.5 <= z_score <= 0.5:
        score += 0
        log['bb'] = 'neutral'
    elif -1.8 < z_score < -0.5:
        score += 2
        log['bb'] = 'dip_buying'
    else: 
        score += 1
        log['bb'] = 'oversold_guard'

    # 5. Trend
    if d['price'] > d['ma20']:
        score += 3 if season in ["WINTER", "SPRING"] else 2
        log['trend'] = 'up'
    else: log['trend'] = 'down'

    # 6. Volume
    if d['volume'] > d['vol_ma20'] * 1.5:
        score += 3 if season in ["WINTER", "AUTUMN"] else 2
        log['vol'] = 'explode'
    else: log['vol'] = 'normal'

    # 7. MACD
    macd_val = d['macd']
    signal_val = d['signal']
    if macd_val > signal_val:
        if macd_val >= 0:
            score += 3
            log['macd'] = 'zero_up_golden' 
        else:
            score += 0
            log['macd'] = 'zero_down_golden'
    else:
        if macd_val >= 0:
            score += -3
            log['macd'] = 'zero_up_dead'
        else:
            score += -5
            log['macd'] = 'zero_down_dead'

    # 8. SKEW Logic
    if d['skew_hist'] is not None and not d['skew_hist'].empty:
        curr_skew = d['skew_hist']['Close'].iloc[-1]
        log['curr_skew'] = curr_skew
        if curr_skew >= 155:
            score += -15
            log['skew'] = 'black_swan'
        elif 145 <= curr_skew < 155:
            score += -3
            log['skew'] = 'high_risk'
        elif 115 <= curr_skew < 145:
            score += 0
            log['skew'] = 'normal'
        else:
            score += -1
            log['skew'] = 'complacency'
    else:
        log['skew'] = 'none'
        log['curr_skew'] = 0

    score += detect_capitulation(d, log)
    score += detect_vvix_trap(d, log)
    score += detect_rsi2_dip(d, log)

    return season, score, log

# === [4] Action Decision (Final Optimized Logic) ===
def determine_action(score, season, data, log):
    # --- Data Extraction ---
    vix_pct_change = ((data['vix'] - data['vix_prev']) / data['vix_prev']) * 100
    iv_rank = data.get('iv_rank', 0)
    adl_z = data.get('adl_z', 0)
    curr_skew = log.get('curr_skew', 0)
    vix_ratio = log.get('vix_ratio', 1.1)
    
    # ---------------------------------------------------------
    # [Step 0] Kill Switch (System Risk Control)
    # ---------------------------------------------------------
    if curr_skew >= 155:
        return None, f"⛔ Trading Halted (Black Swan SKEW: {curr_skew:.1f})", "-", "-", "panic", "-", "-"
    if vix_ratio >= 1.2:
        return None, f"⛔ Trading Halted (Term Structure Panic: {vix_ratio:.2f})", "-", "-", "panic", "-", "-"
    if vix_pct_change > 20.0:
        return None, "⛔ Trading Halted (VIX Explosion)", "-", "-", "panic", "-", "-"
    
    # ---------------------------------------------------------
    # [Step 1] Score Check (Go / No-Go)
    # ---------------------------------------------------------
    verdict_text = ""
    profit_target = ""
    stop_loss = ""
    matrix_id = ""

    if score >= 20:
        verdict_text = "💎💎 Super Strong"
        matrix_id = "super_strong"
        profit_target = "100%+"
        stop_loss = "-300%"
    elif score >= 12:
        verdict_text = "💎 Strong"
        matrix_id = "strong"
        profit_target = "75%"
        stop_loss = "-300%"
    elif 8 <= score < 12:
        verdict_text = "✅ Standard"
        matrix_id = "standard"
        profit_target = "50%"
        stop_loss = "-200%"
    elif 5 <= score < 8:
        verdict_text = "⚠️ Hit & Run (Weak)"
        matrix_id = "weak"
        profit_target = "30%"
        stop_loss = "-150%"
    else:
        return None, "🛡️ No Entry (Low Score)", "-", "-", "no_entry", "-", "-"

    # ---------------------------------------------------------
    # [Step 2] Strategy Selection Logic (The "11 vs 12" Rule)
    # ---------------------------------------------------------
    strategy_type = ""
    strategy_basis = ""
    target_delta = None

    # [Rule 1] Score 12 이상 (공격 가능 구간)
    if score >= 12:
        # 단, "가격(IV)"이 50% 미만으로 싸고 & "시장(ADL)"이 과열되지 않았을 때만 CDS
        is_cheap_iv = (iv_rank < 50)
        is_adl_healthy = (adl_z <= 2.0)
        
        if is_cheap_iv and is_adl_healthy:
            strategy_type = "CDS"
            strategy_basis = (f"Score {score} (Strong) + IV Rank {iv_rank:.0f}% (Cheap) + ADL Z {adl_z:.2f} (OK) "
                              f"👉 Aggressive Long")
            target_delta = 0.55
        else:
            # 추세는 강하지만, 비싸거나 과열됨 -> 안전하게 PCS
            strategy_type = "PCS"
            reason = "IV Expensive" if not is_cheap_iv else "ADL Overbought"
            strategy_basis = f"Score {score} (Strong) but {reason} 👉 Safety First (PCS)"
            target_delta = -0.10

    # [Rule 2] Score 11 이하 (방어/수익쌓기 구간)
    else:
        strategy_type = "PCS"
        
        # ADL이 과매도(바닥) 상태라면 PCS 승률이 더 높아짐을 명시
        if adl_z < -2.0:
            strategy_basis = f"Score {score} (Standard) + ADL Oversold (Z:{adl_z:.2f}) 👉 High Probability PCS"
        else:
            strategy_basis = f"Score {score} (Standard/Weak) 👉 Defensive Strategy (Harvest Theta)"
            
        target_delta = -0.10

    return target_delta, verdict_text, profit_target, stop_loss, matrix_id, strategy_type, strategy_basis

# === [5] Option Finder ===
def calculate_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

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
        
        if strategy_type == "CDS":
            start_k = int(price * 0.9)
            end_k = int(price * 1.1)
        else: # PCS
            start_k = int(price * 0.5)
            end_k = int(price)

        strikes = np.arange(start_k, end_k)
        d1 = calculate_d1(price, strikes, T, r, iv)
        
        if strategy_type == "CDS": deltas = norm.cdf(d1)
        else: deltas = norm.cdf(d1) - 1
            
        diffs = np.abs(deltas - target_delta)
        best_idx = np.argmin(diffs)
        best_strike = strikes[best_idx]
        found_delta = deltas[best_idx]
        
        if strategy_type == "CDS":
            long_strike = float(best_strike)
            short_strike = float(best_strike + SPREAD_WIDTH)
            return {'type': 'CDS', 'expiry': expiry, 'dte': dte, 'long': long_strike, 'short': short_strike, 'delta': float(found_delta), 'width': SPREAD_WIDTH}
        else:
            short_strike = float(best_strike)
            long_strike = float(best_strike - SPREAD_WIDTH)
            return {'type': 'PCS', 'expiry': expiry, 'dte': dte, 'short': short_strike, 'long': long_strike, 'delta': float(found_delta), 'width': SPREAD_WIDTH}
    except: return None

# === [6] Charts (Added ADL Band) ===
def create_charts(data):
    hist = data['hist'].copy()
    
    # Season Calc
    cond_summer = (hist['Close'] > hist['MA50']) & (hist['Close'] > hist['MA200'])
    cond_autumn = (hist['Close'] < hist['MA50']) & (hist['Close'] > hist['MA200'])
    cond_winter = (hist['Close'] < hist['MA50']) & (hist['Close'] < hist['MA200'])
    conditions = [cond_summer, cond_autumn, cond_winter]
    choices = ['SUMMER', 'AUTUMN', 'WINTER']
    hist['Season'] = np.select(conditions, choices, default='SPRING')
    season_colors = {'SUMMER': '#FFEBEE', 'AUTUMN': '#FFF3E0', 'WINTER': '#E3F2FD', 'SPRING': '#E8F5E9'}
    
    # Increase Figure Size for new chart
    fig = plt.figure(figsize=(10, 36))
    gs = fig.add_gridspec(12, 1, height_ratios=[2, 0.6, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1.5])
    
    ax1 = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax1)
    ax_trend = fig.add_subplot(gs[2], sharex=ax1)
    ax_skew = fig.add_subplot(gs[3], sharex=ax1) 
    ax_vix_abs = fig.add_subplot(gs[4], sharex=ax1)
    ax_ratio = fig.add_subplot(gs[5], sharex=ax1)
    ax_rsi = fig.add_subplot(gs[6], sharex=ax1)
    ax2 = fig.add_subplot(gs[7], sharex=ax1)
    ax_ratio_vvix = fig.add_subplot(gs[8], sharex=ax1)
    ax_rsi2 = fig.add_subplot(gs[9], sharex=ax1)
    ax_adl = fig.add_subplot(gs[10], sharex=ax1)
    ax_adl_band = fig.add_subplot(gs[11], sharex=ax1) # [NEW]

    # ... (Axes 1~10 동일 생략) ...
    # 1. Price
    ax1.plot(hist.index, hist['Close'], color='black', alpha=0.9)
    ax1.plot(hist.index, hist['MA20'], color='green', ls='--')
    ax1.plot(hist.index, hist['MA50'], color='blue')
    ax1.plot(hist.index, hist['MA200'], color='red')
    ax1.fill_between(hist.index, hist['BB_Upper'], hist['BB_Lower'], color='gray', alpha=0.1)
    ax1.set_title('QQQ Price Trend', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # 2. Volume
    colors = ['red' if c < o else 'green' for c, o in zip(hist['Close'], hist['Open'])]
    ax_vol.bar(hist.index, hist['Volume'], color=colors, alpha=0.5)
    ax_vol.plot(hist.index, hist['Vol_MA20'], color='black', lw=1)
    ax_vol.grid(True, alpha=0.3)
    plt.setp(ax_vol.get_xticklabels(), visible=False)

    # 3. Trend
    ax_trend.plot(hist.index, hist['Close'], color='black', alpha=0.8)
    ax_trend.plot(hist.index, hist['MA20'], color='green', ls='--')
    dead_cross = hist['MACD'] < hist['Signal']
    ax_trend.fill_between(hist.index, ax_trend.get_ylim()[0], ax_trend.get_ylim()[1], where=dead_cross, color='#FFCDD2', alpha=0.4)
    ax_trend.set_title('Trend Check (Red: MACD Dead)', fontsize=10, fontweight='bold')
    ax_trend.grid(True, alpha=0.3)
    plt.setp(ax_trend.get_xticklabels(), visible=False)

    # 4. SKEW
    if data['skew_hist'] is not None:
        sk = data['skew_hist']
        ax_skew.plot(sk.index, sk['Close'], color='purple')
        ax_skew.axhline(155, color='red', ls='-')
        ax_skew.axhline(145, color='orange', ls='--')
        ax_skew.axhline(115, color='green', ls=':')
    ax_skew.set_title('SKEW Index', fontsize=12, fontweight='bold')
    ax_skew.grid(True, alpha=0.3)
    plt.setp(ax_skew.get_xticklabels(), visible=False)

    # 5. VIX Abs
    ax_vix_abs.plot(data['vix_hist'].index, data['vix_hist']['Close'], color='purple')
    if data['vix3m_hist'] is not None:
        ax_vix_abs.plot(data['vix3m_hist'].index, data['vix3m_hist']['Close'], color='gray', ls=':')
    ax_vix_abs.set_title('VIX (Spot) vs VIX3M', fontsize=12, fontweight='bold')
    ax_vix_abs.grid(True, alpha=0.3)
    plt.setp(ax_vix_abs.get_xticklabels(), visible=False)

    # 6. Term Structure
    if data['vix_term_df'] is not None:
        tdf = data['vix_term_df']
        ax_ratio.plot(tdf.index, tdf['Ratio'], color='black')
        ax_ratio.axhline(1.0, color='red', ls='--')
        ax_ratio.fill_between(tdf.index, tdf['Ratio'], 1.0, where=(tdf['Ratio']>1), color='red', alpha=0.2)
        ax_ratio.fill_between(tdf.index, tdf['Ratio'], 1.0, where=(tdf['Ratio']<=1), color='green', alpha=0.15)
    ax_ratio.set_title('VIX Term Structure (Ratio)', fontsize=12, fontweight='bold')
    ax_ratio.grid(True, alpha=0.3)
    plt.setp(ax_ratio.get_xticklabels(), visible=False)

    # 7. RSI
    ax_rsi.plot(hist.index, hist['RSI'], color='purple')
    ax_rsi.axhline(70, color='red', ls='--')
    ax_rsi.axhline(30, color='green', ls='--')
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI(14)', fontsize=12, fontweight='bold')
    ax_rsi.grid(True, alpha=0.3)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # 8. MACD
    ax2.plot(hist.index, hist['MACD'], label='MACD', color='blue')
    ax2.plot(hist.index, hist['Signal'], label='Signal', color='orange')
    ax2.bar(hist.index, hist['MACD']-hist['Signal'], color='gray', alpha=0.3)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title('MACD', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # 9. VVIX/VIX Ratio
    try:
        v = data['vix_hist']['Close']
        vv = data['vvix_hist']['Close']
        ratio = vv / v
        ax_ratio_vvix.plot(ratio.index, ratio, color='#333333')
        ax_ratio_vvix.axhline(7.0, color='red', ls=':')
        ax_ratio_vvix.axhline(4.0, color='green', ls=':')
    except: pass
    ax_ratio_vvix.set_title('VVIX / VIX Ratio', fontsize=12, fontweight='bold')
    ax_ratio_vvix.grid(True, alpha=0.3)
    plt.setp(ax_ratio_vvix.get_xticklabels(), visible=False)

    # 10. RSI(2)
    ax_rsi2.plot(hist.index, hist['RSI_2'], color='gray')
    ax_rsi2.axhline(10, color='green', ls='--')
    ax_rsi2.axhline(90, color='red', ls='--')
    ax_rsi2.scatter(hist.index[-1], hist['RSI_2'].iloc[-1], color='red', s=50)
    ax_rsi2.set_title('RSI(2)', fontsize=12, fontweight='bold')
    ax_rsi2.grid(True, alpha=0.3)
    plt.setp(ax_rsi2.get_xticklabels(), visible=False)

    # 11. ADL (Raw)
    ax_adl.plot(hist.index, hist['ADL'], color='black', label='ADL')
    ax_adl.plot(hist.index, hist['ADL_MA20'], color='orange', ls='--', label='20MA')
    ax_adl.set_title('Advance-Decline Line (Trend)', fontsize=12, fontweight='bold')
    ax_adl.grid(True, alpha=0.3)
    plt.setp(ax_adl.get_xticklabels(), visible=False)

    # 12. [NEW] ADL Bollinger Band
    if 'ADL_Upper' in hist.columns:
        ax_adl_band.plot(hist.index, hist['ADL'], color='black', lw=1.5, zorder=3, label='ADL')
        ax_adl_band.plot(hist.index, hist['ADL_Upper'], color='red', ls='--', lw=1, zorder=2, label='Upper')
        ax_adl_band.plot(hist.index, hist['ADL_Lower'], color='green', ls='--', lw=1, zorder=2, label='Lower')
        ax_adl_band.fill_between(hist.index, hist['ADL_Upper'], hist['ADL_Lower'], color='gray', alpha=0.1, zorder=1)
        
        curr_z = hist['ADL_Z'].iloc[-1]
        t_color = 'red' if curr_z > 2.0 else 'green' if curr_z < -2.0 else 'black'
        ax_adl_band.text(hist.index[-1], hist['ADL'].iloc[-1], f" Z:{curr_z:.2f}", color=t_color, fontweight='bold', ha='left')
        ax_adl_band.set_title('ADL Bollinger Bands (Market Breadth)', fontsize=12, fontweight='bold')
        ax_adl_band.legend(loc='upper left', fontsize=8)
    else:
        ax_adl_band.text(0.5, 0.5, "No ADL Data", transform=ax_adl_band.transAxes, ha='center')

    ax_adl_band.grid(True, alpha=0.3)
    ax_adl_band.set_xlabel('Date')

    # Background Colors
    for ax in [ax1, ax_vol, ax_skew, ax_vix_abs, ax_ratio, ax_rsi, ax2, ax_ratio_vvix, ax_rsi2, ax_adl, ax_adl_band]:
        trans = ax.get_xaxis_transform()
        for season_name, color in season_colors.items():
            mask = (hist['Season'] == season_name)
            ax.fill_between(hist.index, 0, 1, where=mask, color=color, alpha=0.4, transform=trans, zorder=0)

    plt.tight_layout()
    return fig

# === [Main] ===
def main():
    st.title("🦅 HK Options Advisory (Grand Master v24.0 - ADL Band)")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Logic: Score 11/12 Split + ADL/IV Safety")

    with st.spinner('Analyzing Market Structure...'):
        try:
            data = get_market_data()
            season, score, log = analyze_expert_logic(data)
            target_delta, verdict_text, profit_target, stop_loss, matrix_id, strat_type, strat_basis = determine_action(score, season, data, log)
            strategy = find_best_option(data['price'], data['iv'], target_delta, strat_type)
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.text(traceback.format_exc())
            return

    # [Sidebar]
    st.sidebar.title("🛠️ System Status")
    
    # IV Rank
    iv_rank = data.get('iv_rank', 0)
    st.sidebar.metric("IV Rank (1Y)", f"{iv_rank:.1f}%")
    if iv_rank < 50: st.sidebar.success("Option Price: Cheap ✅")
    else: st.sidebar.warning("Option Price: Expensive ⚠️")

    # ADL Status
    adl_z = data.get('adl_z', 0)
    st.sidebar.metric("ADL Z-Score", f"{adl_z:.2f}")
    if adl_z > 2.0: st.sidebar.error("Breadth: Overbought 🚨")
    elif adl_z < -2.0: st.sidebar.success("Breadth: Oversold (Buy Dip) ✅")
    else: st.sidebar.info("Breadth: Neutral")

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📊 Total Score: {score}")
    st.sidebar.markdown(f"**Verdict:** {verdict_text}")
    if strat_type: st.sidebar.info(f"Strategy: {strat_type}")
    
    # [Scorecard & Matrix Visualization] - (Simplified for brevity, similar to previous version)
    st.markdown(f"### 🔔 Recommendation: **{strat_type}**")
    st.markdown(f"💡 **Basis:** {strat_basis}")
    
    st.markdown("---")
    st.subheader("📈 Technical Charts (including ADL Band)")
    st.pyplot(create_charts(data))

if __name__ == "__main__":
    main()
