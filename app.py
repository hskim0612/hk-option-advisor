import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import concurrent.futures # [Optimization] Parallel processing module

# === [App Security] ===
APP_PASSWORD = "1979"

# === [Page Configuration] ===
st.set_page_config(
    page_title="HK Options Advisory (Grand Master v23.0 - SKEW Logic)",
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

# === [1] Data Collection & Processing (Optimized) ===
def fetch_ticker_data(ticker, period="2y"):
    """[Optimization] Helper function for individual ticker fetch"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        return ticker, t, hist
    except Exception as e:
        return ticker, None, pd.DataFrame()

@st.cache_data(ttl=1800)
def get_market_data():
    # [Optimization] Fetch all tickers including HYG/IEI in parallel
    tickers_to_fetch = [
        ("QQQ", "2y"), ("^ADD", "2y"), ("^VIX", "1y"), 
        ("^VVIX", "1y"), ("^SKEW", "1y"), ("^VIX3M", "1y"),
        ("HYG", "2y"), ("IEI", "2y") # [NEW] Added HYG and IEI
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
    
    # Moving Averages & Indicators (Vectorized)
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
        # Fallback Logic
        print("⚠️ ADL Fetch Failed (^ADD) - Using Fallback")
        hist['Net_Issues'] = np.where(hist['Close'] > hist['Close'].shift(1), 1, -1)
        hist['Net_Issues'].iloc[0] = 0
        hist['ADL'] = hist['Net_Issues'].cumsum() * 100
        hist['ADL_MA20'] = hist['ADL'].rolling(window=20).mean()
    
    # ADL Bollinger Bands Logic
    hist['ADL_Std'] = hist['ADL'].rolling(window=20).std()
    hist['ADL_Upper'] = hist['ADL_MA20'] + (hist['ADL_Std'] * 2)
    hist['ADL_Lower'] = hist['ADL_MA20'] - (hist['ADL_Std'] * 2)
    
    # ADL Z-Score
    numerator = hist['ADL'] - hist['ADL_MA20']
    denominator = hist['ADL_Std']
    hist['ADL_Z'] = np.where(denominator == 0, 0, numerator / denominator)

    # [NEW] Process HYG & IEI Data
    hyg_hist = results["HYG"]['hist'].copy()
    iei_hist = results["IEI"]['hist'].copy()
    hyg_iei_ratio = pd.DataFrame()

    if not hyg_hist.empty:
        hyg_hist.index = hyg_hist.index.tz_localize(None).normalize()
        # HYG Moving Averages
        hyg_hist['MA20'] = hyg_hist['Close'].rolling(window=20).mean()
        hyg_hist['MA50'] = hyg_hist['Close'].rolling(window=50).mean()
        hyg_hist['MA200'] = hyg_hist['Close'].rolling(window=200).mean()
    
    if not iei_hist.empty:
        iei_hist.index = iei_hist.index.tz_localize(None).normalize()

    # Calculate HYG/IEI Ratio
    if not hyg_hist.empty and not iei_hist.empty:
        # Inner join on index to align dates
        combined = pd.merge(hyg_hist[['Close']], iei_hist[['Close']], left_index=True, right_index=True, suffixes=('_HYG', '_IEI'))
        combined['Ratio'] = combined['Close_HYG'] / combined['Close_IEI']
        hyg_iei_ratio = combined

    # 2. Process VIX, VIX3M, VVIX, SKEW
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
    except Exception as e:
        print(f"Error fetching VIX/VIX3M: {e}")
    
    try:
        if not vvix_hist.empty:
            vvix_hist.index = vvix_hist.index.tz_localize(None).normalize()
    except Exception as e:
        print(f"Error processing VVIX: {e}")
        
    # Process SKEW (Ensure Timezone compatibility)
    try:
        if not skew_hist.empty:
            skew_hist.index = skew_hist.index.tz_localize(None).normalize()
    except Exception as e:
        print(f"Error processing SKEW: {e}")

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
        'bb_upper': curr['BB_Upper'], 'bb_lower': curr['BB_Lower'], 'bb_lower_prev': prev['BB_Lower'],
        'macd': curr['MACD'], 'signal': curr['Signal'],
        'macd_prev': prev['MACD'], 'signal_prev': prev['Signal'],
        'volume': curr['Volume'], 'vol_ma20': curr['Vol_MA20'], 'vol_pct': vol_pct,
        'vix': curr_vix, 'vix_prev': prev_vix,
        'vix3m': vix3m_val,
        'iv': current_iv,
        'adl_z': hist['ADL_Z'].iloc[-1],
        'hist': hist, 'vix_hist': vix_hist, 'vix3m_hist': vix3m_hist, 'vvix_hist': vvix_hist,
        'skew_hist': skew_hist,
        'vix_term_df': vix_term_df,
        'hyg_hist': hyg_hist,          # [NEW] Return HYG
        'hyg_iei_ratio': hyg_iei_ratio # [NEW] Return Ratio
    }

# === [2] Advanced Logic Functions ===
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

# === [3] Expert Logic (MACD 4-Zone + SKEW) ===
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
        pts = 3 if season in ["WINTER", "SPRING"] else 2
        score += pts
        log['trend'] = 'up'
    else:
        log['trend'] = 'down'

    # 6. Volume Logic
    if d['volume'] > d['vol_ma20'] * 1.5:
        pts = 3 if season in ["WINTER", "AUTUMN"] else 2
        score += pts
        log['vol'] = 'explode'
    else:
        log['vol'] = 'normal'

    # 7. MACD Logic (4-Zone Strategy)
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

    # 8. SKEW Logic (Black Swan Filter)
    if d['skew_hist'] is not None and not d['skew_hist'].empty:
        curr_skew = d['skew_hist']['Close'].iloc[-1]
        log['curr_skew'] = curr_skew
        
        if curr_skew >= 155:
            score += -15
            log['skew'] = 'black_swan'  # Kill Switch Trigger
        elif 145 <= curr_skew < 155:
            score += -3
            log['skew'] = 'high_risk'
        elif 115 <= curr_skew < 145:
            score += 0
            log['skew'] = 'normal'
        else: # < 115
            score += -1
            log['skew'] = 'complacency'
    else:
        log['skew'] = 'none'
        log['curr_skew'] = 0

    # === [Accumulate Special Scores] ===
    pts_cap = detect_capitulation(d, log)
    score += pts_cap
    
    pts_vvix = detect_vvix_trap(d, log)
    score += pts_vvix
    
    pts_rsi2 = detect_rsi2_dip(d, log)
    score += pts_rsi2

    return season, score, log

# === [4] Action Decision ===
def determine_action(score, season, data, log):
    vix_pct_change = ((data['vix'] - data['vix_prev']) / data['vix_prev']) * 100
    current_vix = data['vix']
    
    # 1. Panic Check (Kill Switch)
    if log.get('skew') == 'black_swan':
        return None, f"⛔ Trading Halted (Black Swan Risk: {log.get('curr_skew'):.1f})", "-", "-", "panic", "-", "-"
        
    if log.get('term') == 'backwardation':
        return None, "⛔ Trading Halted (System Collapse)", "-", "-", "panic", "-", "-"
    if vix_pct_change > 15.0:
        return None, "⛔ Trading Halted (VIX Surge)", "-", "-", "panic", "-", "-"
    if log.get('vvix_trap') == 'detected':
        return None, "⛔ Trading Halted (VVIX Trap)", "-", "-", "panic", "-", "-"
    
    # 2. Score Grade & Strategy Selection
    verdict_text = ""
    profit_target = ""
    stop_loss = ""
    matrix_id = ""
    target_delta = None
    
    # Grading
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

    # 3. Strategy Logic (PCS vs CDS)
    strategy_type = ""
    strategy_basis = ""

    if current_vix < 18.0 and score >= 12:
        strategy_type = "CDS"
        strategy_basis = f"VIX {current_vix:.1f} (Low) + Score {score} (Bull) 👉 Directional Bet"
        target_delta = 0.55 
    else:
        strategy_type = "PCS"
        if current_vix >= 18.0:
            strategy_basis = f"VIX {current_vix:.1f} (High) 👉 Sell Premium (Credit)"
        else:
            strategy_basis = f"Score {score} (Neutral) 👉 Harvest Theta"
        target_delta = -0.10 

    return target_delta, verdict_text, profit_target, stop_loss, matrix_id, strategy_type, strategy_basis

# === [5] Option Finder (Vectorized) ===
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
        
        # Search Range
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
        else: # PCS
            deltas = norm.cdf(d1) - 1
            
        diffs = np.abs(deltas - target_delta)
        best_idx = np.argmin(diffs)
        
        best_strike = strikes[best_idx]
        found_delta = deltas[best_idx]
        
        if strategy_type == "CDS":
            long_strike = float(best_strike)
            short_strike = float(best_strike + SPREAD_WIDTH)
            return {
                'type': 'CDS', 'expiry': expiry, 'dte': dte,
                'long': long_strike, 'short': short_strike,
                'delta': float(found_delta), 'width': SPREAD_WIDTH
            }
        else: # PCS
            short_strike = float(best_strike)
            long_strike = float(best_strike - SPREAD_WIDTH)
            return {
                'type': 'PCS', 'expiry': expiry, 'dte': dte,
                'short': short_strike, 'long': long_strike,
                'delta': float(found_delta), 'width': SPREAD_WIDTH
            }

    except Exception as e:
        print(f"Option Search Error: {e}")
        return None

# === [6] Charts (Optimized Rendering) ===
def create_charts(data):
    hist = data['hist'].copy()
    
    # 4 Seasons
    cond_summer = (hist['Close'] > hist['MA50']) & (hist['Close'] > hist['MA200'])
    cond_autumn = (hist['Close'] < hist['MA50']) & (hist['Close'] > hist['MA200'])
    cond_winter = (hist['Close'] < hist['MA50']) & (hist['Close'] < hist['MA200'])
    
    conditions = [cond_summer, cond_autumn, cond_winter]
    choices = ['SUMMER', 'AUTUMN', 'WINTER']
    hist['Season'] = np.select(conditions, choices, default='SPRING')
    
    season_colors = {
        'SUMMER': '#FFEBEE', 'AUTUMN': '#FFF3E0', 'WINTER': '#E3F2FD', 'SPRING': '#E8F5E9'
    }
    
    # [UPDATE] Figure Size Increased for new plots (HYG, HYG/IEI)
    # Total rows: 14
    fig = plt.figure(figsize=(10, 42))
    
    # [NEW] Inserted HYG and Ratio charts after Trend chart
    gs = fig.add_gridspec(14, 1, height_ratios=[2, 0.6, 1.5, 1.2, 1.2, 1, 1, 1, 1, 1, 1, 1, 1.5, 1.5])
    
    ax1 = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax1)
    ax_trend = fig.add_subplot(gs[2], sharex=ax1)
    
    # [NEW Axes]
    ax_hyg = fig.add_subplot(gs[3], sharex=ax1)
    ax_hyg_ratio = fig.add_subplot(gs[4], sharex=ax1)
    
    ax_skew = fig.add_subplot(gs[5], sharex=ax1) 
    ax_vix_abs = fig.add_subplot(gs[6], sharex=ax1)
    ax_ratio = fig.add_subplot(gs[7], sharex=ax1)
    ax_rsi = fig.add_subplot(gs[8], sharex=ax1)
    ax2 = fig.add_subplot(gs[9], sharex=ax1)
    ax_ratio_vvix = fig.add_subplot(gs[10], sharex=ax1)
    ax_rsi2 = fig.add_subplot(gs[11], sharex=ax1)
    ax_adl = fig.add_subplot(gs[12], sharex=ax1)
    ax_adl_band = fig.add_subplot(gs[13], sharex=ax1)

    # 1. Price Chart
    ax1.plot(hist.index, hist['Close'], label='QQQ', color='black', alpha=0.9, zorder=2)
    ax1.plot(hist.index, hist['MA20'], label='20MA', color='green', ls='--', lw=1, zorder=2)
    ax1.plot(hist.index, hist['MA50'], label='50MA', color='blue', ls='-', lw=1.5, zorder=2)
    ax1.plot(hist.index, hist['MA200'], label='200MA', color='red', ls='-', lw=2, zorder=2)
    ax1.fill_between(hist.index, hist['BB_Upper'], hist['BB_Lower'], color='gray', alpha=0.1, label='Bollinger', zorder=1)
    ax1.set_title('QQQ Price Trend with Market Seasons', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # 2. Volume
    colors = ['red' if c < o else 'green' for c, o in zip(hist['Close'], hist['Open'])]
    ax_vol.bar(hist.index, hist['Volume'], color=colors, alpha=0.5, zorder=2)
    ax_vol.plot(hist.index, hist['Vol_MA20'], color='black', lw=1, zorder=2)
    ax_vol.set_title(f"Volume ({data['vol_pct']:.1f}%)", fontsize=10, fontweight='bold')
    ax_vol.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_vol.get_xticklabels(), visible=False)
    
    # 3. Trend Graph
    ax_trend.plot(hist.index, hist['Close'], label='QQQ', color='black', alpha=0.8, zorder=2)
    ax_trend.plot(hist.index, hist['MA20'], label='20MA', color='green', ls='--', lw=1, zorder=2)
    ax_trend.plot(hist.index, hist['MA50'], label='50MA', color='blue', ls='-', lw=1, zorder=2)
    
    dead_cross_mask = hist['MACD'] < hist['Signal']
    ax_trend.fill_between(hist.index, ax_trend.get_ylim()[0], ax_trend.get_ylim()[1], 
                          where=dead_cross_mask, color='#FFCDD2', alpha=0.4, 
                          transform=ax_trend.get_xaxis_transform(), zorder=0, label='MACD < Signal (Dead)')

    handles, labels = ax_trend.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_trend.legend(by_label.values(), by_label.keys(), loc='upper left')
    ax_trend.set_title('QQQ Trend Check (Background: MACD Dead Cross)', fontsize=10, fontweight='bold')
    ax_trend.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_trend.get_xticklabels(), visible=False)

    # === [NEW] 4. HYG Price & MAs ===
    if 'hyg_hist' in data and not data['hyg_hist'].empty:
        hyg = data['hyg_hist']
        ax_hyg.plot(hyg.index, hyg['Close'], label='HYG (Junk Bond)', color='black', alpha=0.9, zorder=2)
        ax_hyg.plot(hyg.index, hyg['MA20'], label='20MA', color='green', ls='--', lw=1, zorder=2)
        ax_hyg.plot(hyg.index, hyg['MA200'], label='200MA', color='red', ls='-', lw=2, zorder=2)
        
        # Check for Bearish Divergence (Simple visual aid)
        curr_price = hyg['Close'].iloc[-1]
        ma200_val = hyg['MA200'].iloc[-1]
        status = "HEALTHY" if curr_price > ma200_val else "DANGER (Below 200MA)"
        color_status = "green" if curr_price > ma200_val else "red"
        
        ax_hyg.text(hyg.index[-1], curr_price, f" {status}", color=color_status, fontweight='bold', va='center')
        ax_hyg.legend(loc='upper left', fontsize=9)
    else:
        ax_hyg.text(0.5, 0.5, "HYG Data Unavailable", transform=ax_hyg.transAxes, ha='center')

    ax_hyg.set_title('HYG (Canary in Coal Mine) - Watch 200MA Break', fontsize=12, fontweight='bold', color='#D35400')
    ax_hyg.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_hyg.get_xticklabels(), visible=False)

    # === [NEW] 5. HYG / IEI Ratio ===
    if 'hyg_iei_ratio' in data and not data['hyg_iei_ratio'].empty:
        ratio_df = data['hyg_iei_ratio']
        ax_hyg_ratio.plot(ratio_df.index, ratio_df['Ratio'], label='HYG/IEI Ratio', color='#8E44AD', lw=1.5, zorder=2)
        
        # Add simple MA for trend
        ratio_ma20 = ratio_df['Ratio'].rolling(20).mean()
        ax_hyg_ratio.plot(ratio_df.index, ratio_ma20, label='Ratio 20MA', color='orange', ls='--', lw=1, zorder=2)
        
        curr_ratio = ratio_df['Ratio'].iloc[-1]
        prev_ratio = ratio_df['Ratio'].iloc[-2]
        
        # Arrow logic
        arrow = "↗️" if curr_ratio > prev_ratio else "↘️"
        ax_hyg_ratio.text(ratio_df.index[-1], curr_ratio, f" {curr_ratio:.3f} {arrow}", color='purple', fontweight='bold')
        
        ax_hyg_ratio.legend(loc='upper left', fontsize=9)
    else:
        ax_hyg_ratio.text(0.5, 0.5, "Ratio Data Unavailable", transform=ax_hyg_ratio.transAxes, ha='center')

    ax_hyg_ratio.set_title('HYG / IEI Ratio (Risk Appetite: Rising=Risk On, Falling=Risk Off)', fontsize=12, fontweight='bold', color='#8E44AD')
    ax_hyg_ratio.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_hyg_ratio.get_xticklabels(), visible=False)

    # 6. SKEW Index Chart (Modified Visual)
    if 'skew_hist' in data and not data['skew_hist'].empty:
        skew_data = data['skew_hist']
        skew_data.index = skew_data.index.tz_localize(None).normalize()
        ax_skew.plot(skew_data.index, skew_data['Close'], color='purple', label='SKEW Index', lw=1.2, zorder=2)
        
        # Risk Lines
        ax_skew.axhline(155, color='red', ls='-', lw=2, label='Black Swan (155)', zorder=2)
        ax_skew.axhline(145, color='orange', ls='--', lw=1.5, label='High Risk (145)', zorder=2)
        ax_skew.axhline(115, color='green', ls=':', lw=1.5, label='Complacency (115)', zorder=2)
        
        curr_skew = skew_data['Close'].iloc[-1]
        ax_skew.text(skew_data.index[-1], curr_skew, f"{curr_skew:.1f}", 
                     color='purple', fontsize=9, fontweight='bold', ha='left', va='bottom')
        
        ax_skew.legend(loc='upper left', fontsize=8)
    else:
        ax_skew.text(0.5, 0.5, "No SKEW Data", transform=ax_skew.transAxes, ha='center', color='red')
        
    ax_skew.set_title('CBOE SKEW Index (Black Swan Risk)', fontsize=12, fontweight='bold')
    ax_skew.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_skew.get_xticklabels(), visible=False)

    # 7. VIX Absolute
    ax_vix_abs.plot(data['vix_hist'].index, data['vix_hist']['Close'], color='purple', label='VIX (Spot)', zorder=2)
    if data['vix3m_hist'] is not None and not data['vix3m_hist'].empty:
          ax_vix_abs.plot(data['vix3m_hist'].index, data['vix3m_hist']['Close'], color='gray', ls=':', label='VIX3M', zorder=2)
    
    ax_vix_abs.axhline(35, color='red', ls='--', zorder=2)
    ax_vix_abs.axhline(20, color='green', ls='--', zorder=2)
    ax_vix_abs.set_title('VIX vs VIX3M (Absolute Level)', fontsize=12, fontweight='bold')
    ax_vix_abs.legend(loc='upper right')
    ax_vix_abs.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_vix_abs.get_xticklabels(), visible=False)

    # 8. VIX Term Structure
    term_data = data.get('vix_term_df')
    if term_data is not None and not term_data.empty:
        ax_ratio.plot(term_data.index, term_data['Ratio'], color='black', lw=1.2, label='Ratio (VIX/VIX3M)', zorder=2)
        ax_ratio.axhline(1.0, color='red', ls='--', alpha=0.8, lw=1.5, label='Threshold (1.0)', zorder=2)
        
        ax_ratio.fill_between(term_data.index, term_data['Ratio'], 1.0, 
                             where=(term_data['Ratio'] > 1.0), color='red', alpha=0.2, interpolate=True, zorder=1)
        ax_ratio.fill_between(term_data.index, term_data['Ratio'], 1.0, 
                             where=(term_data['Ratio'] <= 1.0), color='green', alpha=0.15, interpolate=True, zorder=1)
        ax_ratio.legend(loc='upper right', fontsize=8)
    else:
        ax_ratio.text(0.5, 0.5, "Data Insufficient", transform=ax_ratio.transAxes, ha='center', color='red', zorder=2)
        
    ax_ratio.set_title('VIX Term Structure (Ratio = VIX / VIX3M)', fontsize=12, fontweight='bold')
    ax_ratio.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_ratio.get_xticklabels(), visible=False)

    # 9. RSI(14)
    ax_rsi.plot(hist.index, hist['RSI'], color='purple', label='RSI(14)', zorder=2)
    ax_rsi.axhline(70, color='red', ls='--', alpha=0.7, zorder=2)
    ax_rsi.axhline(30, color='green', ls='--', alpha=0.7, zorder=2)
    ax_rsi.fill_between(hist.index, hist['RSI'], 70, where=(hist['RSI'] >= 70), color='red', alpha=0.3, zorder=1)
    ax_rsi.fill_between(hist.index, hist['RSI'], 30, where=(hist['RSI'] <= 30), color='green', alpha=0.3, zorder=1)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI (14)', fontsize=12, fontweight='bold')
    ax_rsi.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # 10. MACD
    ax2.plot(hist.index, hist['MACD'], label='MACD', color='blue', zorder=2)
    ax2.plot(hist.index, hist['Signal'], label='Signal', color='orange', zorder=2)
    ax2.bar(hist.index, hist['MACD']-hist['Signal'], color='gray', alpha=0.3, zorder=2)
    ax2.axhline(0, color='black', lw=0.8, zorder=2)
    ax2.set_title('MACD', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # 11. VVIX / VIX Ratio
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
            ax_ratio_vvix.fill_between(merged_ratio.index, merged_ratio['Ratio'], 7.0, 
                                     where=(merged_ratio['Ratio'] > 7.0), color='red', alpha=0.2, zorder=1)
            ax_ratio_vvix.fill_between(merged_ratio.index, merged_ratio['Ratio'], 4.0, 
                                     where=(merged_ratio['Ratio'] < 4.0), color='green', alpha=0.2, zorder=1)
            ax_ratio_vvix.legend(loc='upper left', fontsize=8)
        else:
            ax_ratio_vvix.text(0.5, 0.5, "No Data", transform=ax_ratio_vvix.transAxes, ha='center', zorder=2)
    except Exception as e:
        ax_ratio_vvix.text(0.5, 0.5, f"Error: {e}", transform=ax_ratio_vvix.transAxes, ha='center', color='red', zorder=2)

    ax_ratio_vvix.set_title('VVIX / VIX Ratio', fontsize=12, fontweight='bold')
    ax_ratio_vvix.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_ratio_vvix.get_xticklabels(), visible=False)

    # 12. RSI(2)
    ax_rsi2.plot(hist.index, hist['RSI_2'], color='gray', label='RSI(2)', linewidth=1.2, zorder=2)
    ax_rsi2.axhline(10, color='green', linestyle='--', alpha=0.7, zorder=2)
    ax_rsi2.axhline(90, color='red', linestyle='--', alpha=0.7, zorder=2)
    ax_rsi2.fill_between(hist.index, hist['RSI_2'], 10, where=(hist['RSI_2'] < 10), color='green', alpha=0.3, zorder=1)
    ax_rsi2.fill_between(hist.index, hist['RSI_2'], 90, where=(hist['RSI_2'] > 90), color='red', alpha=0.3, zorder=1)
    ax_rsi2.scatter(hist.index[-1], hist['RSI_2'].iloc[-1], color='red', s=50, zorder=5)
    ax_rsi2.set_ylim(0, 100)
    ax_rsi2.set_title('RSI(2) - Short-term Pullback', fontsize=12, fontweight='bold')
    ax_rsi2.grid(True, alpha=0.3, zorder=1)
    plt.setp(ax_rsi2.get_xticklabels(), visible=False)

    # 13. ADL (Raw)
    ax_adl.plot(hist.index, hist['ADL'], color='black', label='ADL', linewidth=1.5, zorder=2)
    ax_adl.plot(hist.index, hist['ADL_MA20'], color='orange', ls='--', label='ADL 20MA', linewidth=1, zorder=2)
    
    if not hist['ADL'].empty:
        last_adl = hist['ADL'].iloc[-1]
        ax_adl.text(hist.index[-1], last_adl, f"{last_adl:.0f}", 
                    color='black', fontsize=9, fontweight='bold', ha='left', va='center')
    
    ax_adl.axhline(0, color='gray', ls=':', alpha=0.5, zorder=1)
    ax_adl.set_title('Advance-Decline Line (Raw)', fontsize=12, fontweight='bold')
    ax_adl.legend(loc='upper left')
    ax_adl.grid(True, alpha=0.3, zorder=1)
    ax_adl.set_xlabel('Date', fontsize=10)
    plt.setp(ax_adl.get_xticklabels(), visible=False) # Hide X labels to match others

    # 14. ADL Bollinger Band
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

    # === [Background Coloring] ===
    # Updated list to include new axes
    all_axes_except_trend = [ax1, ax_vol, ax_hyg, ax_hyg_ratio, ax_skew, ax_vix_abs, ax_ratio, ax_rsi, ax2, ax_ratio_vvix, ax_rsi2, ax_adl, ax_adl_band]
    
    for ax in all_axes_except_trend:
        trans = ax.get_xaxis_transform()
        for season_name, color in season_colors.items():
            mask = (hist['Season'] == season_name)
            ax.fill_between(hist.index, 0, 1, where=mask, 
                            color=color, alpha=0.4, transform=trans, zorder=0)

    plt.tight_layout()
    return fig

# === [Main] ===
def main():
    st.title("🦅 HK Options Advisory (Grand Master v23.0 - SKEW Logic)")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Logic: MACD 4-Zone + SKEW (Safety) | Optimized")

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
    
    term_df = data.get('vix_term_df')
    if term_df is not None:
        curr_ratio = term_df['Ratio'].iloc[-1]
        st.sidebar.metric("Current Ratio", f"{curr_ratio:.4f}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Risk Indicators")

    # Ratio
    ratio_val = data['vix'] / data['vix3m'] if data['vix3m'] else 1.0
    if ratio_val > 1.0: st.sidebar.error(f"Ratio: {ratio_val:.4f} ⚠️")
    elif ratio_val < 0.9: st.sidebar.success(f"Ratio: {ratio_val:.4f} ✅")
    else: st.sidebar.info(f"Ratio: {ratio_val:.4f}")

    # SKEW Status (New)
    skew_status = log.get('skew', 'none')
    curr_skew = log.get('curr_skew', 0)
    if skew_status == 'black_swan': st.sidebar.error(f"SKEW: {curr_skew:.1f} (BLACK SWAN 🚨)")
    elif skew_status == 'high_risk': st.sidebar.warning(f"SKEW: {curr_skew:.1f} (High Risk)")
    elif skew_status == 'complacency': st.sidebar.info(f"SKEW: {curr_skew:.1f} (Complacency)")
    else: st.sidebar.success(f"SKEW: {curr_skew:.1f} (Normal)")

    # VVIX Change
    vvix_hist = data['vvix_hist']['Close']
    if len(vvix_hist) > 1:
        vvix_change = ((vvix_hist.iloc[-1] - vvix_hist.iloc[-2]) / vvix_hist.iloc[-2]) * 100
        if vvix_change > 5.0: st.sidebar.error(f"VVIX Change: +{vvix_change:.1f}% ⚠️")
        else: st.sidebar.success(f"VVIX Change: {vvix_change:.1f}%")

    # RSI(2)
    rsi2_val = data['rsi2']
    if rsi2_val < 10: st.sidebar.success(f"RSI(2): {rsi2_val:.1f} (Dip) ✅")
    else: st.sidebar.info(f"RSI(2): {rsi2_val:.1f}")

    # Signals
    if log.get('capitulation') == 'detected': st.sidebar.success("Capitulation: ✅ DETECTED")
    else: st.sidebar.info("Capitulation: ❌ None")
    
    if log.get('vvix_trap') == 'detected': st.sidebar.error("VVIX Trap: ⚠️ DETECTED")
    else: st.sidebar.success("VVIX Trap: ✅ None")
    
    # [NEW SIDEBAR] ADL Z-Score
    st.sidebar.markdown("---")
    adl_z = data.get('adl_z', 0)
    st.sidebar.metric("ADL Z-Score", f"{adl_z:.2f}")
    if adl_z > 2.0: st.sidebar.error("Breadth: Overbought 🚨")
    elif adl_z < -2.0: st.sidebar.success("Breadth: Oversold (Buy Dip) ✅")

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📊 Total Score: {score}")
    st.sidebar.markdown(f"**Verdict:** {verdict_text}")
    if strat_type:
        st.sidebar.info(f"Strategy: {strat_type}")

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

    def hl_season(row_season):
        if season == row_season:
            return "style='border: 3px solid #2196F3; background-color: #E3F2FD; font-weight: bold; color: black; padding: 4px;'"
        return "style='border: 1px solid #ddd; padding: 4px; color: black; background-color: white;'"

    td_style = "style='border: 1px solid #ddd; padding: 4px; color: black; background-color: white;'"
    th_style = "style='border: 1px solid #ddd; padding: 4px; color: black; background-color: #f2f2f2;'"
    vix_ratio_disp = f"{log.get('vix_ratio', 0):.2f}"
    z_disp = f"{log.get('z_score', 0):.2f}"
    skew_disp = f"{log.get('curr_skew', 0):.1f}"

    # 1. Season Matrix
    html_season_list = [
        "<h3>1. Market Season Matrix</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; text-align: center;'>",
        "<tr>",
        f"<th {th_style}>Season</th><th {th_style}>Condition</th><th {th_style}>Character</th>",
        "</tr>",
        f"<tr><td {hl_season('SUMMER')}>☀️ SUMMER</td><td {hl_season('SUMMER')}>Price > 50MA & 200MA</td><td {hl_season('SUMMER')}>Bull Market</td></tr>",
        f"<tr><td {hl_season('AUTUMN')}>🍂 AUTUMN</td><td {hl_season('AUTUMN')}>Price < 50MA but > 200MA</td><td {hl_season('AUTUMN')}>Correction</td></tr>",
        f"<tr><td {hl_season('WINTER')}>❄️ WINTER</td><td {hl_season('WINTER')}>Price < 50MA & 200MA</td><td {hl_season('WINTER')}>Bear Market (-5 pts)</td></tr>",
        f"<tr><td {hl_season('SPRING')}>🌱 SPRING</td><td {hl_season('SPRING')}>Price > 50MA but < 200MA</td><td {hl_season('SPRING')}>Recovery</td></tr>",
        "</table>",
        f"<p>※ QQQ: <b>${data['price']:.2f}</b> (Vol: {data['vol_pct']:.1f}% of 20MA)</p>"
    ]
    st.markdown("".join(html_season_list), unsafe_allow_html=True)

    # 2. Scorecard
    html_score_list = [
        "<h3>2. Expert Matrix (Mobile Ver.)</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; text-align: center;'>",
        "<tr>",
        f"<th {th_style}>Indicator</th><th {th_style}>State</th>",
        f"<th {th_style}>☀️</th><th {th_style}>🍂</th><th {th_style}>❄️</th><th {th_style}>🌱</th>",
        "</tr>",
        
        # 0. SKEW (New)
        f"<tr><td rowspan='4' {td_style}><b>SKEW</b><br><span style='font-size:10px; color:purple;'>Current:{skew_disp}</span></td>",
        f"<td {td_style}><b>Black Swan</b><br>(&ge;155)</td>",
        f"<td colspan='4' {hl_score('skew', 'black_swan', 'ALL')}><b style='color:red;'>-15</b></td></tr>",
        f"<tr><td {td_style}>High Risk<br>(145~155)</td>",
        f"<td colspan='4' {hl_score('skew', 'high_risk', 'ALL')}>-3</td></tr>",
        f"<tr><td {td_style}>Normal<br>(115~145)</td>",
        f"<td colspan='4' {hl_score('skew', 'normal', 'ALL')}>0</td></tr>",
        f"<tr><td {td_style}>Complacency<br>(&lt;115)</td>",
        f"<td colspan='4' {hl_score('skew', 'complacency', 'ALL')}>-1</td></tr>",
        
        # 1. VIX Term
        f"<tr><td rowspan='3' {td_style}><b>VIX Term</b><br><span style='font-size:10px; color:blue;'>Ratio:{vix_ratio_disp}</span></td>",
        f"<td {td_style}><b>Easy</b><br>(&lt;0.9)</td>",
        f"<td colspan='4' {hl_score('term', 'contango', 'ALL')}>+3</td></tr>",
        
        f"<tr><td {td_style}>Normal<br>(0.9~1)</td>",
        f"<td colspan='4' {hl_score('term', 'normal', 'ALL')}>0</td></tr>",
        
        f"<tr><td {td_style}><b>Collapse</b><br>(&gt;1.0)</td>",
        f"<td colspan='4' {hl_score('term', 'backwardation', 'ALL')}><b>-10</b></td></tr>",
        
        # 2. Capitulation
        f"<tr><td {td_style}><b>Capitulation</b></td>",
        f"<td {td_style}><b>2-Day</b><br>R&gt;1,V&gt;1.5</td>",
        f"<td colspan='4' {hl_score('capitulation', 'detected', 'ALL')}><b style='color:green;'>+15</b></td></tr>",
        
        # 3. VVIX Trap
        f"<tr><td {td_style}><b>VVIX Trap</b></td>",
        f"<td {td_style}><b>Danger</b><br>VIX↓VVIX↑</td>",
        f"<td colspan='4' {hl_score('vvix_trap', 'detected', 'ALL')}><b style='color:red;'>-10</b></td></tr>",

        # 4. RSI(14)
        f"<tr><td rowspan='4' {td_style}>RSI(14)</td>",
        f"<td {td_style}>Over (>70)</td>",
        f"<td {hl_score('rsi', 'over', 'SUMMER')}>-1</td><td {hl_score('rsi', 'over', 'AUTUMN')}>-3</td><td {hl_score('rsi', 'over', 'WINTER')}><b style='color:red;'>-10</b></td><td {hl_score('rsi', 'over', 'SPRING')}>-2</td></tr>",
        
        f"<tr><td {td_style}>Neutral</td>",
        f"<td {hl_score('rsi', 'neutral', 'SUMMER')}>+1</td><td {hl_score('rsi', 'neutral', 'AUTUMN')}>0</td><td {hl_score('rsi', 'neutral', 'WINTER')}>-1</td><td {hl_score('rsi', 'neutral', 'SPRING')}>+1</td></tr>",
        
        f"<tr><td {td_style}>Under (<30)</td>",
        f"<td {hl_score('rsi', 'under', 'SUMMER')}>+5</td><td {hl_score('rsi', 'under', 'AUTUMN')}>+4</td><td {hl_score('rsi', 'under', 'WINTER')}>0</td><td {hl_score('rsi', 'under', 'SPRING')}>+4</td></tr>",
        
        f"<tr><td {td_style}>🚀 Escape</td>",
        f"<td {hl_score('rsi', 'escape', 'SUMMER')}>3~5</td><td {hl_score('rsi', 'escape', 'AUTUMN')}>3~5</td><td {hl_score('rsi', 'escape', 'WINTER')}>3~5</td><td {hl_score('rsi', 'escape', 'SPRING')}>3~5</td></tr>",
        
        # 5. RSI(2)
        f"<tr><td {td_style}><b>RSI(2)</b></td>",
        f"<td {td_style}><b>Dip</b><br>(&lt;10)</td>",
        f"<td colspan='4' {hl_score('rsi2_dip', 'detected', 'ALL')}><b style='color:green;'>+8</b></td></tr>",

        # 6. VIX Level
        f"<tr><td rowspan='4' {td_style}>VIX</td>",
        f"<td {td_style}>Stable (<20)</td>",
        f"<td {hl_score('vix', 'stable', 'SUMMER')}>+2</td><td {hl_score('vix', 'stable', 'AUTUMN')}>0</td><td {hl_score('vix', 'stable', 'WINTER')}>-2</td><td {hl_score('vix', 'stable', 'SPRING')}>+1</td></tr>",
        
        f"<tr><td {td_style}>Fear (20-35)</td>",
        f"<td {hl_score('vix', 'fear', 'SUMMER')}>-3</td><td {hl_score('vix', 'fear', 'AUTUMN')}>-4</td><td {hl_score('vix', 'fear', 'WINTER')}>+2</td><td {hl_score('vix', 'fear', 'SPRING')}>-1</td></tr>",
        
        f"<tr><td {td_style}>Panic Rise</td>",
        f"<td {hl_score('vix', 'panic_rise', 'SUMMER')}>-5</td><td {hl_score('vix', 'panic_rise', 'AUTUMN')}>-6</td><td {hl_score('vix', 'panic_rise', 'WINTER')}>-5</td><td {hl_score('vix', 'panic_rise', 'SPRING')}>-4</td></tr>",
        
        f"<tr><td {td_style}>📉 Peak Out</td>",
        f"<td {hl_score('vix', 'peak_out', 'SUMMER')}>-</td><td {hl_score('vix', 'peak_out', 'AUTUMN')}>-</td><td {hl_score('vix', 'peak_out', 'WINTER')}>+7</td><td {hl_score('vix', 'peak_out', 'SPRING')}>-</td></tr>",
        
        # 7. Bollinger
        f"<tr><td rowspan='5' {td_style}>BB Z-Score<br><span style='font-size:10px; color:blue;'>{z_disp}</span></td>",
        f"<td {td_style} style='color:red;'><b>Overbought</b><br>(&gt;1.8)</td>",
        f"<td colspan='4' {hl_score('bb', 'overbought_danger', 'ALL')}><b style='color:red;'>-3</b></td></tr>",
        
        f"<tr><td {td_style}><b>Uptrend</b><br>(0.5~1.8)</td>",
        f"<td colspan='4' {hl_score('bb', 'uptrend', 'ALL')}>+1</td></tr>",
        
        f"<tr><td {td_style}>Neutral</td>",
        f"<td colspan='4' {hl_score('bb', 'neutral', 'ALL')}>0</td></tr>",
        
        f"<tr><td {td_style}><b>Value</b><br>(-1.8~-0.5)</td>",
        f"<td colspan='4' {hl_score('bb', 'dip_buying', 'ALL')}>+2</td></tr>",
        
        f"<tr><td {td_style}><b>Bottom</b><br>(Z&le;-1.8)</td>",
        f"<td colspan='4' {hl_score('bb', 'oversold_guard', 'ALL')}><b>+1</b></td></tr>",
        
        # 8. Trend & Vol
        f"<tr><td {td_style}>Trend</td><td {td_style}>Price > 20MA</td>",
        f"<td {hl_score('trend', 'up', 'SUMMER')}>+2</td><td {hl_score('trend', 'up', 'AUTUMN')}>+2</td><td {hl_score('trend', 'up', 'WINTER')}>+3</td><td {hl_score('trend', 'up', 'SPRING')}>+3</td></tr>",
        
        f"<tr><td {td_style}>Volume</td><td {td_style}>Explode</td>",
        f"<td {hl_score('vol', 'explode', 'SUMMER')}>+2</td><td {hl_score('vol', 'explode', 'AUTUMN')}>+3</td><td {hl_score('vol', 'explode', 'WINTER')}>+3</td><td {hl_score('vol', 'explode', 'SPRING')}>+2</td></tr>",
        
        # 9. MACD (4-Zone)
        f"<tr><td rowspan='4' {td_style}>MACD</td>",
        
        # Case 1
        f"<td {td_style}>📈 <b>Accel</b><br><span style='font-size:10px;'>(Up+Gold)</span></td>",
        f"<td colspan='4' {hl_score('macd', 'zero_up_golden', 'ALL')}><b style='color:green;'>+3</b></td></tr>",
        
        # Case 2
        f"<td {td_style}>📉 <b>Corr</b><br><span style='font-size:10px;'>(Up+Dead)</span></td>",
        f"<td colspan='4' {hl_score('macd', 'zero_up_dead', 'ALL')}><b style='color:orange;'>-3</b></td></tr>",
        
        # Case 3
        f"<td {td_style}>🎣 <b>Trap</b><br><span style='font-size:10px;'>(Down+Gold)</span></td>",
        f"<td colspan='4' {hl_score('macd', 'zero_down_golden', 'ALL')}><b style='color:gray;'>0</b></td></tr>",
        
        # Case 4
        f"<td {td_style}>☔ <b>Crash</b><br><span style='font-size:10px;'>(Down+Dead)</span></td>",
        f"<td colspan='4' {hl_score('macd', 'zero_down_dead', 'ALL')}><b style='color:red;'>-5</b></td></tr>",
        
        "</table>"
    ]
    st.markdown("".join(html_score_list), unsafe_allow_html=True)

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
        f"<h3>3. Final Verdict: <span style='color:white;'>{score} Pts</span> - Dynamic Exit Matrix</h3>",
        strat_display,
        "<div style='border: 2px solid #ccc; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>",
        "<table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; text-align: center;'>",
        f"<tr style='background-color: #333; color: white;'>",
        f"<th {th_style} style='color:white;'>Score Range</th>",
        f"<th {th_style} style='color:white;'>Verdict</th>",
        f"<th {th_style} style='color:white;'>🎯 Target</th>",
        f"<th {th_style} style='color:white;'>🛑 Stop Loss</th>",
        "</tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'panic', '#ffebee')}>",
        "<td>System Collapse / Black Swan</td><td>⛔ Trading Halted (Panic)</td><td>-</td><td>-</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'super_strong', '#c8e6c9')}>",
        "<td>20+ (with Signals)</td><td>💎💎 Super Strong</td><td style='color:green;'>+100%</td><td style='color:red;'>-300%</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'strong', '#dff0d8')}>",
        "<td>12 ~ 19</td><td>💎 Strong</td><td style='color:green;'>+75%</td><td style='color:red;'>-300%</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'standard', '#ffffff')}>",
        "<td>8 ~ 11</td><td>✅ Standard</td><td style='color:green;'>+50%</td><td style='color:red;'>-200%</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'weak', '#fff9c4')}>",
        "<td>5 ~ 7</td><td>⚠️ Hit & Run</td><td style='color:green;'>+30%</td><td style='color:red;'>-150%</td></tr>",
        
        f"<tr {get_matrix_style(matrix_id, 'no_entry', '#f2dede')}>",
        "<td>< 5</td><td>🛡️ No Entry</td><td>-</td><td>-</td></tr>",
        
        "</table>",
        "</div>"
    ]
    st.markdown("".join(html_verdict_list), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Technical Charts")
    st.pyplot(create_charts(data))

if __name__ == "__main__":
    main()
