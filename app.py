import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === [앱 보안 설정] ===
APP_PASSWORD = "1979"

# === [페이지 기본 설정] ===
st.set_page_config(
    page_title="HK 옵션투자자문 (Grand Master v20.0)",
    page_icon="🦅",
    layout="wide"
)

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
    
    # 2. VIX & VIX3M 데이터 처리
    vix_ticker = yf.Ticker("^VIX")
    vix_hist = vix_ticker.history(period="1y")
    
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
            
            # pd.merge 사용 (Inner Join)
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

# === [4] 차트 (수정: Plotly + Click-to-Lock) ===
def create_charts(data, locked_date=None):
    hist = data['hist']
    vix_hist = data['vix_hist']
    vix3m_hist = data['vix3m_hist']
    term_df = data.get('vix_term_df')
    
    # 1. Subplots 생성 (비율: Price 30%, Vol 10%, 나머지 15%씩)
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.30, 0.10, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=(
            "QQQ Price Trend", 
            f"Volume ({data['vol_pct']:.1f}%)", 
            "RSI (14)", 
            "MACD", 
            "VIX Level (Absolute)", 
            "Structure of Volatility (Ratio = VIX / VIX3M)"
        )
    )

    # === 1. Price Chart (Row 1) ===
    # Bollinger Band (Upper & Lower - Area Fill)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], line=dict(width=0), 
                             showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], line=dict(width=0), 
                             fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)', 
                             name='Bollinger', hoverinfo='skip'), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='red', width=1.5), name='200MA', hovertemplate='200MA: $%{y:.2f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], line=dict(color='blue', width=1.5), name='50MA', hovertemplate='50MA: $%{y:.2f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='green', width=1, dash='dot'), name='20MA', hovertemplate='20MA: $%{y:.2f}<extra></extra>'), row=1, col=1)
    
    # Price
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], line=dict(color='black', width=1.5), name='Close', hovertemplate='Close: $%{y:.2f}<extra></extra>'), row=1, col=1)

    # === 2. Volume Chart (Row 2) ===
    # Color Logic: Close >= Open (Green), Close < Open (Red)
    colors = ['green' if c >= o else 'red' for c, o in zip(hist['Close'], hist['Open'])]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume', opacity=0.5, hovertemplate='Vol: %{y}<extra></extra>'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Vol_MA20'], line=dict(color='black', width=1), name='Vol MA20', hovertemplate='VolMA: %{y}<extra></extra>'), row=2, col=1)

    # === 3. RSI Chart (Row 3) ===
    fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple', width=1.5), name='RSI', hovertemplate='RSI: %{y:.1f}<extra></extra>'), row=3, col=1)
    
    # RSI Reference Lines & Zones
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_width=0.5, line_color="black", row=3, col=1)
    
    # RSI Background coloring (Over/Under)
    rsi_upper = hist['RSI'].clip(lower=70)
    fig.add_trace(go.Scatter(x=hist.index, y=rsi_upper, line=dict(width=0), 
                             fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.0)', showlegend=False, hoverinfo='skip'), row=3, col=1)
    
    # === 4. MACD Chart (Row 4) ===
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='blue', width=1), name='MACD', hovertemplate='MACD: %{y:.2f}<extra></extra>'), row=4, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], line=dict(color='orange', width=1), name='Signal', hovertemplate='Sig: %{y:.2f}<extra></extra>'), row=4, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist['MACD']-hist['Signal'], marker_color='gray', opacity=0.3, name='Hist', hovertemplate='Hist: %{y:.2f}<extra></extra>'), row=4, col=1)
    fig.add_hline(y=0, line_width=0.8, line_color="black", row=4, col=1)

    # === 5. VIX Level Chart (Row 5) ===
    fig.add_trace(go.Scatter(x=vix_hist.index, y=vix_hist['Close'], line=dict(color='purple', width=1.5), name='VIX', hovertemplate='VIX: %{y:.2f}<extra></extra>'), row=5, col=1)
    if vix3m_hist is not None and not vix3m_hist.empty:
        fig.add_trace(go.Scatter(x=vix3m_hist.index, y=vix3m_hist['Close'], line=dict(color='gray', width=1, dash='dot'), name='VIX3M', hovertemplate='VIX3M: %{y:.2f}<extra></extra>'), row=5, col=1)
    
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Panic", row=5, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Stable", row=5, col=1)

    # === 6. VIX Ratio Chart (Row 6) ===
    if term_df is not None and not term_df.empty:
        # Ratio Line
        fig.add_trace(go.Scatter(x=term_df.index, y=term_df['Ratio'], line=dict(color='black', width=1.2), name='Ratio', hovertemplate='Ratio: %{y:.3f}<extra></extra>'), row=6, col=1)
        
        # Guidelines
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=6, col=1)
        fig.add_hline(y=0.9, line_dash="dash", line_color="green", row=6, col=1)

        # Conditional Fills Logic
        # 1. Backwardation (> 1.0)
        ratio_high = term_df['Ratio'].apply(lambda x: max(x, 1.0))
        fig.add_trace(go.Scatter(x=term_df.index, y=[1.0]*len(term_df), line=dict(width=0), showlegend=False, hoverinfo='skip'), row=6, col=1)
        fig.add_trace(go.Scatter(x=term_df.index, y=ratio_high, fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', 
                                 line=dict(width=0), name='Backwardation', hoverinfo='skip'), row=6, col=1)

        # 2. Contango (< 0.9)
        ratio_low = term_df['Ratio'].apply(lambda x: min(x, 0.9))
        fig.add_trace(go.Scatter(x=term_df.index, y=[0.9]*len(term_df), line=dict(width=0), showlegend=False, hoverinfo='skip'), row=6, col=1)
        fig.add_trace(go.Scatter(x=term_df.index, y=ratio_low, fill='tonexty', fillcolor='rgba(0, 128, 0, 0.2)', 
                                 line=dict(width=0), name='Contango', hoverinfo='skip'), row=6, col=1)
    else:
        fig.add_annotation(text="데이터 부족: VIX/VIX3M Ratio 표시 불가", 
                           xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, font=dict(color="red"), row=6, col=1)

    # [새 기능] 고정 수직선 추가
    if locked_date is not None:
        fig.add_vline(
            x=locked_date, 
            line=dict(color='black', width=2, dash='solid'),
            opacity=0.9,
            annotation_text=f"🔒 {locked_date.strftime('%Y-%m-%d')}", 
            annotation_position="top"
        )

    # === Global Layout Settings ===
    fig.update_layout(
        height=1500,  # 전체 높이 설정
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        hovermode='x unified',  # [핵심] 모든 데이터 동시 표시
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_color="black",
            font_family="Arial"
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # X축 설정 (Spike Line 포함)
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='#f0f0f0', 
        rangeslider_visible=False,
        showspikes=True,
        spikemode='across',      # [핵심] 차트 전체 관통
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='rgba(150, 150, 150, 0.4)',  # 연한 회색 가이드
        spikedash='dot'
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', showspikes=False)

    # RSI Y축 고정
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig

# === [메인 화면] ===
def main():
    # [1] Session State 초기화 (날짜 고정용)
    if 'locked_date' not in st.session_state:
        st.session_state.locked_date = None

    st.title("🦅 HK Advisory (Grand Master v20.0)")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System: Institutional Grade")

    # [2] 고정 해제 버튼 (고정 상태일 때만 표시)
    if st.session_state.locked_date:
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🔓 고정 해제"):
                st.session_state.locked_date = None
                st.session_state["main_chart"] = None  # [중요] 차트 선택 상태도 강제 초기화
                st.rerun()
        with col1:
             st.info(f"🔒 고정된 시점: {st.session_state.locked_date.strftime('%Y년 %m월 %d일')} | 차트의 다른 지점을 클릭하면 이동합니다.")

    with st.spinner('시장 구조 및 변동성 정밀 분석 중...'):
        try:
            data = get_market_data()
            season, score, log = analyze_expert_logic(data)
            target_delta, verdict_text, profit_target, stop_loss, matrix_id = determine_action(score, season, data, log)
            strategy = find_best_option(data['price'], data['iv'], target_delta)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            return

    # [NEW] Sidebar Debugging Panel
    st.sidebar.title("🛠️ 시스템 상태")
    st.sidebar.markdown("---")
    
    # 1. 데이터 카운트
    vix_count = len(data['vix_hist']) if not data['vix_hist'].empty else 0
    vix3m_count = len(data['vix3m_hist']) if data['vix3m_hist'] is not None and not data['vix3m_hist'].empty else 0
    
    term_df = data.get('vix_term_df')
    ratio_count = len(term_df) if term_df is not None else 0
    
    st.sidebar.metric("VIX Raw Data", f"{vix_count} rows")
    st.sidebar.metric("VIX3M Raw Data", f"{vix3m_count} rows")
    
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
        
        # VIX Term Structure
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
        
        # VIX Level
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
        
        # Trend & Volume & MACD
        f"<tr><td {td_style}>추세 (20MA)</td><td {td_style}>20일선 위</td>",
        f"<td {hl_score('trend', 'up', 'SUMMER')}>+2</td><td {hl_score('trend', 'up', 'AUTUMN')}>+2</td><td {hl_score('trend', 'up', 'WINTER')}>+3</td><td {hl_score('trend', 'up', 'SPRING')}>+3</td>",
        f"<td align='left' {td_style}>회복</td></tr>",
        
        f"<tr><td {td_style}>거래량</td><td {td_style}>폭증 (>150%)</td>",
        f"<td {hl_score('vol', 'explode', 'SUMMER')}>+2</td><td {hl_score('vol', 'explode', 'AUTUMN')}>+3</td><td {hl_score('vol', 'explode', 'WINTER')}>+3</td><td {hl_score('vol', 'explode', 'SPRING')}>+2</td>",
        f"<td align='left' {td_style}><b>손바뀜</b></td></tr>",
        
        f"<tr><td {td_style}>거래량</td><td {td_style}>일반</td>",
        f"<td {hl_score('vol', 'normal', 'SUMMER')}>0</td><td {hl_score('vol', 'normal', 'AUTUMN')}>0</td><td {hl_score('vol', 'normal', 'WINTER')}>0</td><td {hl_score('vol', 'normal', 'SPRING')}>0</td>",
        f"<td align='left' {td_style}>-</td></tr>",
        
        f"<tr><td rowspan='4' {td_style}>MACD</td>",
        f"<td {td_style}>📈 상승 전환</td>",
        f"<td {hl_score('macd', 'break_up', 'SUMMER')}>+3</td><td {hl_score('macd', 'break_up', 'AUTUMN')}>+3</td><td {hl_score('macd', 'break_up', 'WINTER')}>+3</td><td {hl_score('macd', 'break_up', 'SPRING')}>+3</td>",
        f"<td align='left' {td_style}><b>강력 매수</b></td></tr>",
        
        f"<tr><td {td_style}>☁️ 상승 추세</td>",
        f"<td {hl_score('macd', 'above', 'SUMMER')}>+1</td><td {hl_score('macd', 'above', 'AUTUMN')}>+1</td><td {hl_score('macd', 'above', 'WINTER')}>+1</td><td {hl_score('macd', 'above', 'SPRING')}>+1</td>",
        f"<td align='left' {td_style}>순풍</td></tr>",
        
        f"<tr><td {td_style}>📉 하락 전환</td>",
        f"<td {hl_score('macd', 'break_down', 'SUMMER')}>-3</td><td {hl_score('macd', 'break_down', 'AUTUMN')}>-3</td><td {hl_score('macd', 'break_down', 'WINTER')}>-3</td><td {hl_score('macd', 'break_down', 'SPRING')}>-3</td>",
        f"<td align='left' {td_style}><b>강력 매도</b></td></tr>",
        
        f"<tr><td {td_style}>☔ 하락 추세</td>",
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
    st.subheader("📈 기술적 분석 차트 (Interactive)")
    
    # [3] 차트 생성 (Session State의 locked_date 전달)
    chart_fig = create_charts(data, locked_date=st.session_state.locked_date)
    
    # [4] 차트 출력 (Key를 지정하여 Session State에 이벤트 저장)
    st.plotly_chart(
        chart_fig, 
        use_container_width=True, 
        on_select="rerun",
        selection_mode="points",
        key="main_chart"
    )

    # [5] 이벤트 처리: Session State에서 직접 데이터 추출 (안전한 방식)
    if "main_chart" in st.session_state and st.session_state["main_chart"]:
        selection_data = st.session_state["main_chart"]
        
        # selection 데이터 구조 검증
        if "selection" in selection_data and "points" in selection_data["selection"]:
            points = selection_data["selection"]["points"]
            
            # 클릭된 포인트가 존재할 경우
            if len(points) > 0:
                # 1. 날짜 추출
                clicked_x = points[0]["x"]
                
                # 2. 날짜 형식 정규화 (Timezone 제거하여 비교 오류 방지)
                if isinstance(clicked_x, str):
                    clicked_date = pd.to_datetime(clicked_x).tz_localize(None)
                else:
                    clicked_date = pd.to_datetime(clicked_x).tz_localize(None)
                
                # 3. 상태 업데이트 (기존과 다를 때만 실행 -> 무한 루프 방지)
                # 주의: locked_date가 None이거나, 클릭한 날짜가 현재 고정된 날짜와 다를 때만 갱신
                current_locked = st.session_state.locked_date
                if current_locked is None or current_locked != clicked_date:
                    st.session_state.locked_date = clicked_date
                    
                    # 모바일 터치 피드백 (선택 사항)
                    st.toast(f"📅 {clicked_date.strftime('%Y-%m-%d')} 시점이 고정되었습니다.", icon="🔒")
                    
                    # 이벤트 상태 초기화 (중복 처리 방지)
                    st.session_state["main_chart"] = None 
                    st.rerun()

    # --- [Debug Info] ---
    # st.sidebar.markdown("---")
    # if "main_chart" in st.session_state:
    #     st.sidebar.write("Last Selection:", st.session_state["main_chart"])

if __name__ == "__main__":
    main()
