import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === [페이지 기본 설정] ===
st.set_page_config(
    page_title="HK 옵션투자자문 (Expert)",
    page_icon="💎",
    layout="wide"
)

# 차트 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'

# === [1] 데이터 수집 및 가공 (캐싱 적용) ===
# 30분(1800초) 동안은 데이터를 저장해두고, 그 이후엔 새로 가져옵니다.
@st.cache_data(ttl=1800)
def get_market_data():
    qqq = yf.Ticker("QQQ")
    # 2년치 데이터
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
    
    # 거래량 이동평균
    hist['Vol_MA20'] = hist['Volume'].rolling(window=20).mean()
    
    # VIX
    vix_hist = yf.Ticker("^VIX").history(period="1y")
    
    curr = hist.iloc[-1]
    prev = hist.iloc[-2]
    curr_vix = vix_hist['Close'].iloc[-1]
    prev_vix = vix_hist['Close'].iloc[-2]
    
    # IV 추출 (실패시 VIX 대용)
    try:
        dates = qqq.options
        chain = qqq.option_chain(dates[1])
        current_iv = chain.calls['impliedVolatility'].mean()
    except:
        current_iv = curr_vix / 100.0

    return {
        'price': curr['Close'], 'price_prev': prev['Close'],
        'ma20': curr['MA20'], 'ma50': curr['MA50'], 'ma200': curr['MA200'],
        'rsi': curr['RSI'], 'rsi_prev': prev['RSI'],
        'bb_upper': curr['BB_Upper'], 'bb_lower': curr['BB_Lower'], 'bb_lower_prev': prev['BB_Lower'],
        'macd': curr['MACD'], 'signal': curr['Signal'],
        'macd_prev': prev['MACD'], 'signal_prev': prev['Signal'],
        'volume': curr['Volume'], 'vol_ma20': curr['Vol_MA20'],
        'vix': curr_vix, 'vix_prev': prev_vix,
        'iv': current_iv,
        'hist': hist, 'vix_hist': vix_hist
    }

# === [2] 전문가 스코어링 로직 ===
def analyze_expert_logic(d):
    # 계절 판단
    if d['price'] > d['ma50'] and d['price'] > d['ma200']: season = "SUMMER ☀️"
    elif d['price'] < d['ma50'] and d['price'] > d['ma200']: season = "AUTUMN 🍂"
    elif d['price'] < d['ma50'] and d['price'] < d['ma200']: season = "WINTER ❄️"
    else: season = "SPRING 🌱"
    
    score = 0
    reasons = [] # 점수 근거 기록
    
    # A. RSI
    if d['rsi'] > 70:
        pts = -1 if "SUMMER" in season else -3 if "AUTUMN" in season else -5
        score += pts
        reasons.append(f"RSI 과열({d['rsi']:.1f}): {pts}점")
    elif d['rsi'] < 30:
        pts = 5 if "SUMMER" in season else 4 if "AUTUMN" in season else 0
        score += pts
        reasons.append(f"RSI 과매도({d['rsi']:.1f}): {pts}점")
    
    # Expert: RSI 탈출
    if d['rsi_prev'] < 30 and d['rsi'] >= 30:
        pts = 6 if "WINTER" in season else 5
        score += pts
        reasons.append(f"🔥 RSI 30 상향 돌파: +{pts}점")

    # B. VIX
    if d['vix'] > 35:
        if d['vix'] > d['vix_prev']:
            pts = -5
            reasons.append("VIX 패닉 상승중: -5점")
        else:
            pts = 7
            reasons.append("🎯 VIX 피크아웃(꺾임): +7점")
        score += pts
    elif 25 <= d['vix'] <= 35:
        pts = 2 if "WINTER" in season else -3
        score += pts
        reasons.append(f"VIX 공포구간: {pts}점")

    # C. Bollinger
    if d['price_prev'] < d['bb_lower_prev'] and d['price'] >= d['bb_lower']:
        pts = 5 if "WINTER" in season else 4
        score += pts
        reasons.append(f"↩️ 볼린저밴드 내부 복귀: +{pts}점")

    # D. 추세
    if d['price'] > d['ma20']:
        pts = 3
        score += pts
        reasons.append("20일선 회복: +3점")

    return season, score, reasons

def determine_action(score, season):
    if score >= 10:
        return -0.20, "💎 강력 매수 (Strong Buy)", "success"
    elif 5 <= score < 10:
        return -0.20, "⚖️ 매수 우위 (Buy)", "info"
    elif 0 <= score < 5:
        return -0.15, "🛡️ 중립/관망 (Neutral)", "warning"
    elif -5 <= score < 0:
        return -0.10, "⚠️ 위험 관리 (Warning)", "error"
    else:
        return None, "⛔ 진입 금지 (No Entry)", "error"

# === [3] 전략 계산 (블랙숄즈) ===
def calculate_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return -0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

def find_best_option(price, iv, target_delta, target_dte, width):
    if target_delta is None: return None
    
    T = target_dte / 365.0
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
        'short': best_strike,
        'long': best_strike - width,
        'delta': found_delta
    }

# === [4] 차트 그리기 ===
def plot_charts(data):
    hist = data['hist']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Price Chart
    ax1.plot(hist.index, hist['Close'], label='QQQ', color='black', alpha=0.7)
    ax1.plot(hist.index, hist['MA20'], label='20MA', color='green', ls='--', lw=1)
    ax1.plot(hist.index, hist['MA200'], label='200MA', color='red', lw=2)
    ax1.fill_between(hist.index, hist['BB_Upper'], hist['BB_Lower'], color='gray', alpha=0.1)
    ax1.set_title('QQQ Price & Trend', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    
    # MACD Chart
    ax2.plot(hist.index, hist['MACD'], label='MACD', color='blue')
    ax2.plot(hist.index, hist['Signal'], label='Signal', color='orange')
    ax2.bar(hist.index, hist['MACD']-hist['Signal'], color='gray', alpha=0.3)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.legend(loc='upper left')
    
    return fig

# === [메인 화면 구성] ===
def main():
    st.title("📊 HK 옵션투자자문 대시보드")
    st.markdown(f"Last Updated: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 전략 설정")
        target_dte = st.slider("목표 만기일 (DTE)", 30, 60, 45)
        spread_width = st.selectbox("스프레드 폭 ($)", [5, 10, 20], index=1)
        
        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear()
            st.rerun()

    # 데이터 로딩
    with st.spinner('미국 시장 데이터를 분석 중입니다...'):
        try:
            data = get_market_data()
        except Exception as e:
            st.error(f"데이터 수집 실패: {e}")
            return

    # 분석 실행
    season, score, reasons = analyze_expert_logic(data)
    target_delta, verdict_text, verdict_color = determine_action(score, season)
    
    # 1. 핵심 지표 대시보드
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("QQQ 현재가", f"${data['price']:.2f}", f"{data['price']-data['price_prev']:.2f}")
    col2.metric("시장 계절", season)
    col3.metric("HK 점수", f"{score}점")
    col4.metric("VIX 지수", f"{data['vix']:.2f}", f"{data['vix']-data['vix_prev']:.2f}", delta_color="inverse")

    # 2. 최종 판정 박스
    st.markdown("---")
    if verdict_color == "success":
        st.success(f"## 📢 최종 판정: {verdict_text}")
    elif verdict_color == "warning":
        st.warning(f"## 📢 최종 판정: {verdict_text}")
    else:
        st.error(f"## 📢 최종 판정: {verdict_text}")

    # 3. 추천 전략 및 근거
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("📝 점수 산정 근거")
        if reasons:
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.write("- 특이 사항 없음 (중립)")
            
    with c2:
        st.subheader("🎯 추천 전략 (Put Credit Spread)")
        strategy = find_best_option(data['price'], data['iv'], target_delta, target_dte, spread_width)
        
        if strategy:
            st.write(f"**만기 (DTE):** 약 {target_dte}일 후")
            st.write(f"🔴 **Sell Put:** ${strategy['short']} (Delta {strategy['delta']:.2f})")
            st.write(f"🟢 **Buy Put:** ${strategy['long']}")
            st.info("반드시 **Net Credit**(돈을 받는 상태)인지 확인하세요.")
        else:
            st.write("현재 진입 가능한 적절한 옵션이 없습니다.")

    # 4. 차트
    st.markdown("---")
    st.subheader("📈 기술적 분석 차트")
    st.pyplot(plot_charts(data))

if __name__ == "__main__":
    main()