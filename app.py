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
    page_title="HK Options Advisory (Grand Master v24.0 - Forest & Tree)",
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
        ("^VVIX", "1y"), ("^SKEW", "1y"), ("^VIX3M", "1y"),
        ("HYG", "2y"), ("IEI", "2y")
    ]
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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
    
    # Bollinger Bands
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

    # Process ADL Data
    add_hist = results["^ADD"]['hist']
    if not add_hist.empty and len(add_hist) > 10:
        hist.index = hist.index.tz_localize(None).normalize()
        add_hist.index = add_hist.index.tz_localize(None).normalize()
        
        hist = hist.join(add_hist['Close'].rename('Net_Issues'), how='left')
        hist['Net_Issues'] = hist['Net_Issues'].ffill().fillna(0)
        hist['ADL'] = hist['Net_Issues'].cumsum()
        hist['ADL_MA20'] = hist['ADL'].rolling(window=20).mean()
    else:
        # Fallback
        hist['Net_Issues'] = np.where(hist['Close'] > hist['Close'].shift(1), 1, -1)
        hist['Net_Issues'].iloc[0] = 0
        hist['ADL'] = hist['Net_Issues'].cumsum() * 100
        hist['ADL_MA20'] = hist['ADL'].rolling(window=20).mean()
    
    # ADL Bollinger & Z-Score
    hist['ADL_Std'] = hist['ADL'].rolling(window=20).std()
    hist['ADL_Upper'] = hist['ADL_MA20'] + (hist['ADL_Std'] * 2)
    hist['ADL_Lower'] = hist['ADL_MA20'] - (hist['ADL_Std'] * 2)
    
    numerator = hist['ADL'] - hist['ADL_MA20']
    denominator = hist['ADL_Std']
    hist['ADL_Z'] = np.where(denominator == 0, 0, numerator / denominator)

    # Process HYG & IEI (Ratio)
    hyg_hist = results["HYG"]['hist'].copy()
    iei_hist = results["IEI"]['hist'].copy()
    hyg_iei_ratio = pd.DataFrame()

    if not hyg_hist.empty and not iei_hist.empty:
        hyg_hist.index = hyg_hist.index.tz_localize(None).normalize()
        iei_hist.index = iei_hist.index.tz_localize(None).normalize()
        combined = pd.merge(hyg_hist[['Close']], iei_hist[['Close']], left_index=True, right_index=True, suffixes=('_HYG', '_IEI'))
        combined['Ratio'] = combined['Close_HYG'] / combined['Close_IEI']
        
        # [NEW] 18 MA for Ratio
        combined['Ratio_MA18'] = combined['Ratio'].rolling(window=18).mean()
        hyg_iei_ratio = combined

    # Process VIX Data
    vix_hist = results["^VIX"]['hist']
    vvix_hist = results["^VVIX"]['hist']
    vix3m_hist = results["^VIX3M"]['hist']
    skew_hist = results["^SKEW"]['hist']
    
    vix3m_val = None
    vix_term_df = None

    try:
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
    except: pass
    
    # Timezone cleanups
    try:
        if not vvix_hist.empty: vvix_hist.index = vvix_hist.index.tz_localize(None).normalize()
        if not skew_hist.empty: skew_hist.index = skew_hist.index.tz_localize(None).normalize()
    except: pass

    curr = hist.iloc[-1]
    prev = hist.iloc[-2]
    curr_vix = vix_hist['Close'].iloc[-1]
    prev_vix = vix_hist['Close'].iloc[-2]
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
        'rsi': curr['RSI'], 'rsi_prev': prev['RSI'],
        'rsi2': curr['RSI_2'], 
        'bb_upper': curr['BB_Upper'], 'bb_lower': curr['BB_Lower'],
        'macd': curr['MACD'], 'signal': curr['Signal'],
        'volume': curr['Volume'], 'vol_ma20': curr['Vol_MA20'], 'vol_pct': vol_pct,
        'vix': curr_vix, 'vix_prev': prev_vix,
        'vix3m': vix3m_val,
        'iv': current_iv,
        'adl_z': hist['ADL_Z'].iloc[-1],
        'hist': hist, 'vix_hist': vix_hist, 'vix3m_hist': vix3m_hist, 'vvix_hist': vvix_hist,
        'skew_hist': skew_hist,
        'vix_term_df': vix_term_df,
        'hyg_iei_ratio': hyg_iei_ratio
    }

# === [2] Logic Functions ===
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
        
        if (ratio > 1.0 and vol_ratio > 1.5) and (ratio_prev > 1.0 and vol_ratio_prev > 1.5):
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

# === [3] Expert Logic ===
def analyze_expert_logic(d):
    # Season Logic
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
        log['term'] = 'normal'
    log['vix_ratio'] = vix_ratio

    # 2. RSI(14) Logic
    curr_rsi = d['rsi']
    hist_rsi = d['hist']['RSI']
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
        pts = 5 if season == "SUMMER" else 4 if season in ["AUTUMN", "SPRING"] else 0
        score += pts
        log['rsi'] = 'under'
    elif is_escape_mode and days_since_escape <= 7:
        score_map = {1: 3, 2: 4, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
        score += score_map.get(days_since_escape, 0)
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
            score += 7 if season == "WINTER" else 0
            log['vix'] = 'peak_out'
        else:
            score += -5 if season == "WINTER" else -6 if season == "AUTUMN" else -5
            log['vix'] = 'panic_rise'
    elif d['vix'] < 20:
        score += 2 if season == "SUMMER" else 1 if season == "SPRING" else -2 if season == "WINTER" else 0
        log['vix'] = 'stable'
    elif 20 <= d['vix'] <= 35:
        score += 2 if season == "WINTER" else -1 if season == "SPRING" else -3 if season == "SUMMER" else -4
        log['vix'] = 'fear'

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
    else:
        log['trend'] = 'down'

    # 6. Volume
    if d['volume'] > d['vol_ma20'] * 1.5:
        score += 3 if season in ["WINTER", "AUTUMN"] else 2
        log['vol'] = 'explode'
    else:
        log['vol'] = 'normal'

    # 7. MACD
    if d['macd'] > d['signal']:
        if d['macd'] >= 0:
            score += 3
            log['macd'] = 'zero_up_golden' 
        else:
            log['macd'] = 'zero_down_golden'
    else:
        if d['macd'] >= 0:
            score += -3
            log['macd'] = 'zero_up_dead'
        else:
            score += -5
            log['macd'] = 'zero_down_dead'

    # 8. SKEW
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
            log['skew'] = 'normal'
        else:
            score += -1
            log['skew'] = 'complacency'
    else:
        log['curr_skew'] = 0

    # === [NEW] 9. ADL Z-Score Penalty ===
    adl_z = d['adl_z']
    if adl_z < 0:
        score += -5
        log['adl_status'] = 'negative'
    else:
        log['adl_status'] = 'positive'

    # === [NEW] 10. Forest (ADL) & Tree (RSI2) Logic ===
    # "숲이 자랄때 나무가 쓰러지면 줍는다"
    curr_rsi2 = d['rsi2']
    
    if adl_z >= 0.9: # 숲이 기울기가 좋을 때
        if curr_rsi2 < 10:
            score += 5
            log['forest_tree'] = 'strong_buy'
        elif curr_rsi2 < 20:
            score += 2
            log['forest_tree'] = 'buy'
        else:
            log['forest_tree'] = 'neutral_good_forest'
            
    elif adl_z < 0: # 숲이 썩어갈 때
        if curr_rsi2 > 90:
            score += -5
            log['forest_tree'] = 'strong_sell'
        elif curr_rsi2 > 80:
            score += -2
            log['forest_tree'] = 'sell'
        else:
            log['forest_tree'] = 'neutral_bad_forest'
    else:
        log['forest_tree'] = 'none'

    # Special Scores
    score += detect_capitulation(d, log)
    score += detect_vvix_trap(d, log)
    # Note: detect_rsi2_dip removed as per request

    return season, score, log

# === [4] Action Decision ===
def determine_action(score, season, data, log):
    vix_pct_change = ((data['vix'] - data['vix_prev']) / data['vix_prev']) * 100
    current_vix = data['vix']
    
    # Panic Checks
    if log.get('skew') == 'black_swan': return None, f"⛔ Trading Halted (Black Swan: {log.get('curr_skew'):.1f})", "-", "-", "panic", "-", "-"
    if log.get('term') == 'backwardation': return None, "⛔ Trading Halted (System Collapse)", "-", "-", "panic", "-", "-"
    if vix_pct_change > 15.0: return None, "⛔ Trading Halted (VIX Surge)", "-", "-", "panic", "-", "-"
    if log.get('vvix_trap') == 'detected': return None, "⛔ Trading Halted (VVIX Trap)", "-", "-", "panic", "-", "-"
    
    verdict_text = ""
    profit_target = ""
    stop_loss = ""
    matrix_id = ""
    target_delta = None
    
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
        verdict_text = "🛡️ No Entry"
        matrix_id = "no_entry"
        return None, verdict_text, "-", "-", matrix_id, "-", "-"

    strategy_type = ""
    strategy_basis = ""

    if current_vix < 18.0 and score >= 12:
        strategy_type = "CDS"
        strategy_basis = f"VIX {current_vix:.1f} (Low) + Score {score} (Bull) 👉 Directional Bet"
        target_delta = 0.55 
    else:
        strategy_type = "PCS"
        if current_vix >= 18.0:
            strategy_basis = f"VIX {current_vix:.1f} (High) 👉 Sell Premium"
        else:
            strategy_basis = f"Score {score} (Neutral) 👉 Harvest Theta"
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
        else:
            start_k = int(price * 0.5)
            end_k = int(price)

        strikes = np.arange(start_k, end_k)
        d1 = calculate_d1(price, strikes, T, r, iv)
        
        if strategy_type == "CDS":
            deltas = norm.cdf(d1)
        else:
            deltas = norm.cdf(d1) - 1
            
        diffs = np.abs(deltas - target_delta)
        best_idx = np.argmin(diffs)
        best_strike = strikes[best_idx]
        found_delta = deltas[best_idx]
        
        if strategy_type == "CDS":
            return {
                'type': 'CDS', 'expiry': expiry, 'dte': dte,
                'long': float(best_strike), 'short': float(best_strike + SPREAD_WIDTH),
                'delta': float(found_delta), 'width': SPREAD_WIDTH
            }
        else:
            return {
                'type': 'PCS', 'expiry': expiry, 'dte': dte,
                'short': float(best_strike), 'long': float(best_strike - SPREAD_WIDTH),
                'delta': float(found_delta), 'width': SPREAD_WIDTH
            }
    except: return None

# === [6] Charts (Updated) ===
def create_charts(data):
    hist = data['hist'].copy()
    
    # 4 Seasons Logic
    cond_summer = (hist['Close'] > hist['MA50']) & (hist['Close'] > hist['MA200'])
    cond_autumn = (hist['Close'] < hist['MA50']) & (hist['Close'] > hist['MA200'])
    cond_winter = (hist['Close'] < hist['MA50']) & (hist['Close'] < hist['MA200'])
    
    conditions = [cond_summer, cond_autumn, cond_winter]
    choices = ['SUMMER', 'AUTUMN', 'WINTER']
    hist['Season'] = np.select(conditions, choices, default='SPRING')
    
    # [UPDATE] Colors (High Saturation for Spring/Autumn, Distinct Winter)
    season_colors = {
        'SUMMER': '#FFEBEE',  # Light Red
        'AUTUMN': '#FFE0B2',  # Vivid Orange (High Saturation)
        'WINTER': '#BBDEFB',  # Distinct Blue
        'SPRING': '#C8E6C9'   # Vivid Green (High Saturation)
    }
    
    # Figure Setup (Removed HYG Price, ADL Raw) - Total 12 subplots
    fig = plt.figure(figsize=(10, 36))
    gs = fig.add_gridspec(12, 1, height_ratios=[2, 0.6, 2.0, 1.2, 1, 1, 1, 1, 1, 1, 1, 1.5])
    
    ax1 = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax1)
    ax_trend = fig.add_subplot(gs[2], sharex=ax1)
    ax_hyg_ratio = fig.add_subplot(gs[3], sharex=ax1) # Ratio Only
    ax_skew = fig.add_subplot(gs[4], sharex=ax1) 
    ax_vix_abs = fig.add_subplot(gs[5], sharex=ax1)
    ax_ratio = fig.add_subplot(gs[6], sharex=ax1)
    ax_rsi = fig.add_subplot(gs[7], sharex=ax1)
    ax2 = fig.add_subplot(gs[8], sharex=ax1)
    ax_ratio_vvix = fig.add_subplot(gs[9], sharex=ax1)
    ax_rsi2 = fig.add_subplot(gs[10], sharex=ax1)
    ax_adl_band = fig.add_subplot(gs[11], sharex=ax1) # ADL Band Only

    # 1. Price
    ax1.plot(hist.index, hist['Close'], label='QQQ', color='black', alpha=0.9, zorder=2)
    ax1.plot(hist.index, hist['MA20'], label='20MA', color='green', ls='--', lw=1, zorder=2)
    ax1.plot(hist.index, hist['MA50'], label='50MA', color='blue', ls='-', lw=1.5, zorder=2)
    ax1.plot(hist.index, hist['MA200'], label='200MA', color='red', ls='-', lw=2, zorder=2)
    ax1.fill_between(hist.index, hist['BB_Upper'], hist['BB_Lower'], color='gray', alpha=0.1, zorder=1)
    ax1.set_title('QQQ Price Trend with Market Seasons', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # 2. Volume
    colors = ['red' if c < o else 'green' for c, o in zip(hist['Close'], hist['Open'])]
    ax_vol.bar(hist.index, hist['Volume'], color=colors, alpha=0.5, zorder=2)
    ax_vol.plot(hist.index, hist['Vol_MA20'], color='black', lw=1, zorder=2)
    ax_vol.set_title(f"Volume ({data['vol_pct']:.1f}%)", fontsize=10, fontweight='bold')
    ax_vol.grid(True, alpha=0.3)
    plt.setp(ax_vol.get_xticklabels(), visible=False)
    
    # 3. Trend Graph (Updated: MACD Markers + Season Background)
    ax_trend.plot(hist.index, hist['Close'], label='QQQ', color='black', alpha=0.8, zorder=2)
    ax_trend.plot(hist.index, hist['MA20'], label='20MA', color='green', ls='--', lw=1, zorder=2)
    ax_trend.plot(hist.index, hist['MA50'], label='50MA', color='blue', ls='-', lw=1, zorder=2)
    
    # MACD Crossover Markers
    # Dead Cross: MACD < Signal and prev(MACD) >= prev(Signal)
    # Golden Cross: MACD > Signal and prev(MACD) <= prev(Signal)
    cross_dead = (hist['MACD'] < hist['Signal']) & (hist['MACD'].shift(1) >= hist['Signal'].shift(1))
    cross_gold = (hist['MACD'] > hist['Signal']) & (hist['MACD'].shift(1) <= hist['Signal'].shift(1))
    
    ax_trend.scatter(hist.index[cross_dead], hist.loc[cross_dead, 'Close'] + 5, color='red', marker='v', s=80, label='MACD Dead', zorder=5)
    ax_trend.scatter(hist.index[cross_gold], hist.loc[cross_gold, 'Close'] - 5, color='green', marker='^', s=80, label='MACD Gold', zorder=5)

    ax_trend.legend(loc='upper left')
    ax_trend.set_title('QQQ Trend Check (Markers: MACD Cross)', fontsize=10, fontweight='bold')
    ax_trend.grid(True, alpha=0.3)
    plt.setp(ax_trend.get_xticklabels(), visible=False)

    # === [UPDATE] 4. HYG / IEI Ratio (18MA + Crosses) ===
    if 'hyg_iei_ratio' in data and not data['hyg_iei_ratio'].empty:
        ratio_df = data['hyg_iei_ratio']
        ax_hyg_ratio.plot(ratio_df.index, ratio_df['Ratio'], label='HYG/IEI Ratio', color='#8E44AD', lw=1.5, zorder=2)
        ax_hyg_ratio.plot(ratio_df.index, ratio_df['Ratio_MA18'], label='Ratio 18MA', color='orange', ls='--', lw=1, zorder=2)
        
        # Calculate Crosses relative to 18MA
        # Up Cross (Golden): Ratio > MA18 & Prev Ratio <= Prev MA18
        r_up = (ratio_df['Ratio'] > ratio_df['Ratio_MA18']) & (ratio_df['Ratio'].shift(1) <= ratio_df['Ratio_MA18'].shift(1))
        # Down Cross (Dead): Ratio < MA18 & Prev Ratio >= Prev MA18
        r_down = (ratio_df['Ratio'] < ratio_df['Ratio_MA18']) & (ratio_df['Ratio'].shift(1) >= ratio_df['Ratio_MA18'].shift(1))
        
        ax_hyg_ratio.scatter(ratio_df.index[r_up], ratio_df.loc[r_up, 'Ratio'], color='green', marker='^', s=60, label='Up Cross', zorder=4)
        ax_hyg_ratio.scatter(ratio_df.index[r_down], ratio_df.loc[r_down, 'Ratio'], color='red', marker='v', s=60, label='Down Cross', zorder=4)

        ax_hyg_ratio.legend(loc='upper left', fontsize=9)
    else:
        ax_hyg_ratio.text(0.5, 0.5, "Ratio Data Unavailable", transform=ax_hyg_ratio.transAxes, ha='center')

    ax_hyg_ratio.set_title('HYG / IEI Ratio (Risk On/Off) with 18MA Crosses', fontsize=12, fontweight='bold', color='#8E44AD')
    ax_hyg_ratio.grid(True, alpha=0.3)
    plt.setp(ax_hyg_ratio.get_xticklabels(), visible=False)

    # 5. SKEW
    if 'skew_hist' in data and not data['skew_hist'].empty:
        skew_data = data['skew_hist']
        ax_skew.plot(skew_data.index, skew_data['Close'], color='purple', label='SKEW', lw=1.2, zorder=2)
        ax_skew.axhline(155, color='red', ls='-', lw=2, zorder=1)
        ax_skew.axhline(145, color='orange', ls='--', lw=1.5, zorder=1)
        ax_skew.text(skew_data.index[-1], skew_data['Close'].iloc[-1], f"{skew_data['Close'].iloc[-1]:.1f}", color='purple', fontweight='bold')
    ax_skew.set_title('CBOE SKEW Index', fontsize=12, fontweight='bold')
    ax_skew.grid(True, alpha=0.3)
    plt.setp(ax_skew.get_xticklabels(), visible=False)

    # 6. VIX Absolute
    ax_vix_abs.plot(data['vix_hist'].index, data['vix_hist']['Close'], color='purple', label='VIX', zorder=2)
    if data['vix3m_hist'] is not None:
         ax_vix_abs.plot(data['vix3m_hist'].index, data['vix3m_hist']['Close'], color='gray', ls=':', label='VIX3M', zorder=2)
    ax_vix_abs.axhline(35, color='red', ls='--')
    ax_vix_abs.axhline(20, color='green', ls='--')
    ax_vix_abs.set_title('VIX vs VIX3M (Absolute)', fontsize=12, fontweight='bold')
    ax_vix_abs.grid(True, alpha=0.3)
    plt.setp(ax_vix_abs.get_xticklabels(), visible=False)

    # 7. VIX Term Structure
    term_data = data.get('vix_term_df')
    if term_data is not None:
        ax_ratio.plot(term_data.index, term_data['Ratio'], color='black', label='VIX/VIX3M', zorder=2)
        ax_ratio.axhline(1.0, color='red', ls='--')
        ax_ratio.fill_between(term_data.index, term_data['Ratio'], 1.0, where=(term_data['Ratio'] > 1.0), color='red', alpha=0.2)
        ax_ratio.fill_between(term_data.index, term_data['Ratio'], 1.0, where=(term_data['Ratio'] <= 1.0), color='green', alpha=0.15)
    ax_ratio.set_title('VIX Term Structure (Ratio)', fontsize=12, fontweight='bold')
    ax_ratio.grid(True, alpha=0.3)
    plt.setp(ax_ratio.get_xticklabels(), visible=False)

    # 8. RSI(14)
    ax_rsi.plot(hist.index, hist['RSI'], color='purple', label='RSI(14)', zorder=2)
    ax_rsi.axhline(70, color='red', ls='--')
    ax_rsi.axhline(30, color='green', ls='--')
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI (14)', fontsize=12, fontweight='bold')
    ax_rsi.grid(True, alpha=0.3)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # 9. MACD
    ax2.plot(hist.index, hist['MACD'], color='blue', zorder=2)
    ax2.plot(hist.index, hist['Signal'], color='orange', zorder=2)
    ax2.bar(hist.index, hist['MACD']-hist['Signal'], color='gray', alpha=0.3, zorder=2)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title('MACD', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # 10. VVIX/VIX Ratio
    try:
        df_v = data['vix_hist'][['Close']]
        df_vv = data['vvix_hist'][['Close']]
        merged_ratio = pd.merge(df_v, df_vv, left_index=True, right_index=True, suffixes=('_VIX', '_VVIX'))
        merged_ratio['Ratio'] = merged_ratio['Close_VVIX'] / merged_ratio['Close_VIX']
        ax_ratio_vvix.plot(merged_ratio.index, merged_ratio['Ratio'], color='#333333', label='VVIX/VIX', zorder=2)
        ax_ratio_vvix.axhline(7.0, color='red', ls=':')
        ax_ratio_vvix.axhline(4.0, color='green', ls=':')
    except: pass
    ax_ratio_vvix.set_title('VVIX / VIX Ratio', fontsize=12, fontweight='bold')
    ax_ratio_vvix.grid(True, alpha=0.3)
    plt.setp(ax_ratio_vvix.get_xticklabels(), visible=False)

    # 11. RSI(2)
    ax_rsi2.plot(hist.index, hist['RSI_2'], color='gray', label='RSI(2)', zorder=2)
    ax_rsi2.axhline(10, color='green', ls='--')
    ax_rsi2.axhline(90, color='red', ls='--')
    ax_rsi2.set_ylim(0, 100)
    ax_rsi2.set_title('RSI(2)', fontsize=12, fontweight='bold')
    ax_rsi2.grid(True, alpha=0.3)
    plt.setp(ax_rsi2.get_xticklabels(), visible=False)

    # 12. ADL Bollinger Band (Updated Title)
    if 'ADL_Upper' in hist.columns:
        ax_adl_band.plot(hist.index, hist['ADL'], color='black', lw=1.5, zorder=3, label='ADL')
        ax_adl_band.plot(hist.index, hist['ADL_Upper'], color='red', ls='--', lw=1, zorder=2)
        ax_adl_band.plot(hist.index, hist['ADL_Lower'], color='green', ls='--', lw=1, zorder=2)
        ax_adl_band.fill_between(hist.index, hist['ADL_Upper'], hist['ADL_Lower'], color='gray', alpha=0.1, zorder=1)
        
        curr_z = hist['ADL_Z'].iloc[-1]
        t_color = 'red' if curr_z > 2.0 else 'green' if curr_z < -2.0 else 'black'
        ax_adl_band.text(hist.index[-1], hist['ADL'].iloc[-1], f" Z:{curr_z:.2f}", color=t_color, fontweight='bold', ha='left')
    
    ax_adl_band.set_title('ADL Bollinger Bands (상승/하락 종목수의 차이)', fontsize=12, fontweight='bold')
    ax_adl_band.grid(True, alpha=0.3)
    ax_adl_band.set_xlabel('Date')

    # === [Background Coloring for ALL plots] ===
    all_axes = [ax1, ax_vol, ax_trend, ax_hyg_ratio, ax_skew, ax_vix_abs, ax_ratio, ax_rsi, ax2, ax_ratio_vvix, ax_rsi2, ax_adl_band]
    
    for ax in all_axes:
        trans = ax.get_xaxis_transform()
        for season_name, color in season_colors.items():
            mask = (hist['Season'] == season_name)
            # Use zorder=0 to ensure background stays behind everything
            ax.fill_between(hist.index, 0, 1, where=mask, color=color, alpha=0.4, transform=trans, zorder=0)

    plt.tight_layout()
    return fig

# === [Main] ===
def main():
    st.title("🦅 HK Options Advisory (Grand Master v24.0 - Forest & Tree)")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Logic: MACD 4-Zone + ADL/RSI2 Strategy")

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
    st.sidebar.markdown("---")
    
    # ADL Z-Score
    adl_z = data.get('adl_z', 0)
    st.sidebar.metric("ADL Z-Score", f"{adl_z:.2f}")
    if adl_z < 0:
        st.sidebar.error("ADL Negative: -5 Pts Penalty")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📊 Total Score: {score}")
    st.sidebar.markdown(f"**Verdict:** {verdict_text}")

    # Style Helpers
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

    td_style = "style='border: 1px solid #ddd; padding: 4px; color: black; background-color: white;'"
    th_style = "style='border: 1px solid #ddd; padding: 4px; color: black; background-color: #f2f2f2;'"
    
    # 2. Scorecard
    html_score_list = [
        "<h3>2. Expert Matrix (Forest & Tree)</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; text-align: center;'>",
        "<tr>",
        f"<th {th_style}>Indicator</th><th {th_style}>State</th>",
        f"<th {th_style}>Score</th>",
        "</tr>",
        
        # 1. ADL Z-Score Penalty
        f"<tr><td {td_style}><b>ADL Z-Score</b><br>(Forest Health)</td>",
        f"<td {td_style}>Negative (<0)</td>",
        f"<td {hl_score('adl_status', 'negative', 'ALL')}><b style='color:red;'>-5</b></td></tr>",

        # 2. Forest & Tree (Combined)
        f"<tr><td rowspan='4' {td_style}><b>Forest & Tree</b><br>(ADL + RSI2)</td>",
        f"<td {td_style}>Forest Grow (Z&ge;0.9) + Tree Fall (RSI2&lt;10)</td>",
        f"<td {hl_score('forest_tree', 'strong_buy', 'ALL')}><b style='color:green;'>+5</b></td></tr>",
        f"<tr><td {td_style}>Forest Grow (Z&ge;0.9) + Tree Fall (RSI2&lt;20)</td>",
        f"<td {hl_score('forest_tree', 'buy', 'ALL')}>+2</td></tr>",
        f"<tr><td {td_style}>Forest Rot (Z&lt;0) + Tree High (RSI2&gt;80)</td>",
        f"<td {hl_score('forest_tree', 'sell', 'ALL')}>-2</td></tr>",
        f"<tr><td {td_style}>Forest Rot (Z&lt;0) + Tree High (RSI2&gt;90)</td>",
        f"<td {hl_score('forest_tree', 'strong_sell', 'ALL')}><b style='color:red;'>-5</b></td></tr>",

        # 3. SKEW
        f"<tr><td {td_style}><b>SKEW</b></td><td {td_style}>Black Swan (&ge;155)</td><td {hl_score('skew', 'black_swan', 'ALL')}><b style='color:red;'>-15</b></td></tr>",

        # 4. VIX Term
        f"<tr><td {td_style}><b>VIX Term</b></td><td {td_style}>Collapse (&gt;1.0)</td><td {hl_score('term', 'backwardation', 'ALL')}><b style='color:red;'>-10</b></td></tr>",

        # 5. Capitulation
        f"<tr><td {td_style}>Capitulation</td><td {td_style}>Volume + Ratio Spike</td><td {hl_score('capitulation', 'detected', 'ALL')}><b style='color:green;'>+15</b></td></tr>",
        
        # 6. VVIX Trap
        f"<tr><td {td_style}>VVIX Trap</td><td {td_style}>VIX↓ VVIX↑</td><td {hl_score('vvix_trap', 'detected', 'ALL')}><b style='color:red;'>-10</b></td></tr>",
        
        "</table>"
    ]
    st.markdown("".join(html_score_list), unsafe_allow_html=True)
    
    st.info("※ Note: Standard Indicators (RSI14, MACD, Bollinger, etc.) are calculated in the background but displayed compactly in charts.")

    # 3. Final Verdict
    def get_matrix_style(current_id, row_id, bg_color):
        if current_id == row_id:
            return f"style='background-color: {bg_color}; border: 3px solid #666; font-weight: bold; color: #333; height: 50px;'"
        else:
            return "style='background-color: white; border: 1px solid #eee; color: #999;'"
            
    strat_display = f"""
    <div style='background-color:#f1f8e9; padding:15px; border-left:5px solid #4caf50; margin-bottom:15px;'>
        <div style='font-size:18px; font-weight:bold; color:#2e7d32;'>🔔 Recommended Strategy: {strat_type if strat_type else '-'}</div>
        <div style='font-size:14px; color:#555; margin-top:5px;'>💡 <b>Basis:</b> {strat_basis if strat_basis else '-'}</div>
    </div>
    """

    html_verdict_list = [
        f"<h3>3. Final Verdict: <span style='color:white;'>{score} Pts</span></h3>",
        strat_display,
        "<div style='border: 2px solid #ccc; border-radius: 10px; overflow: hidden;'>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; text-align: center;'>",
        f"<tr style='background-color: #333; color: white;'><th {th_style} style='color:white;'>Score</th><th {th_style} style='color:white;'>Verdict</th></tr>",
        f"<tr {get_matrix_style(matrix_id, 'super_strong', '#c8e6c9')}><td>20+</td><td>💎💎 Super Strong</td></tr>",
        f"<tr {get_matrix_style(matrix_id, 'strong', '#dff0d8')}><td>12 ~ 19</td><td>💎 Strong</td></tr>",
        f"<tr {get_matrix_style(matrix_id, 'standard', '#ffffff')}><td>8 ~ 11</td><td>✅ Standard</td></tr>",
        f"<tr {get_matrix_style(matrix_id, 'weak', '#fff9c4')}><td>5 ~ 7</td><td>⚠️ Hit & Run</td></tr>",
        f"<tr {get_matrix_style(matrix_id, 'no_entry', '#f2dede')}><td>< 5</td><td>🛡️ No Entry</td></tr>",
        "</table></div>"
    ]
    st.markdown("".join(html_verdict_list), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Technical Charts")
    st.pyplot(create_charts(data))

if __name__ == "__main__":
    main()
