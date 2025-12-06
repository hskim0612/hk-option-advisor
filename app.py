import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 페이지 설정 및 데이터 가져오기
# ---------------------------------------------------------
st.set_page_config(page_title="Market Technical Analysis Report", layout="wide")

st.title("📊 Market Technical Analysis & Volatility Report")

# 데이터 캐싱 (속도 향상)
@st.cache_data
def get_market_data():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*2)  # 2년치 데이터

    tickers = {
        "SPX": "^GSPC",
        "VIX": "^VIX",
        "VIX3M": "^VIX3M",  # VIX 3개월물 (Term Structure용)
        "VIX6M": "^VIX6M",  # VIX 6개월물 (Term Structure용)
        "SKEW": "^SKEW"
    }
    
    data = yf.download(list(tickers.values()), start=start_date, end=end_date)['Close']
    
    # 컬럼 이름 변경 (티커 -> 읽기 쉬운 이름)
    data = data.rename(columns={v: k for k, v in tickers.items()})
    
    # 결측치 제거 (최근 데이터가 없는 경우 방지)
    data = data.fillna(method='ffill')
    
    return data

try:
    df = get_market_data()
    
    # 최신 데이터 날짜 확인
    last_date = df.index[-1].strftime('%Y-%m-%d')
    st.write(f"Last Updated: **{last_date}**")

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------
# 2. 메인 차트: S&P 500 & Moving Averages
# ---------------------------------------------------------
st.header("1. S&P 500 (SPX) Trend")

# 이동평균선 계산
df['MA20'] = df['SPX'].rolling(window=20).mean()
df['MA60'] = df['SPX'].rolling(window=60).mean()
df['MA200'] = df['SPX'].rolling(window=200).mean()

fig_spx = go.Figure()

# 캔들스틱 대신 라인 차트로 간소화 (전체 흐름 파악용)
fig_spx.add_trace(go.Scatter(x=df.index, y=df['SPX'], mode='lines', name='SPX Price', line=dict(color='white', width=1.5)))
fig_spx.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA 20', line=dict(color='yellow', width=1)))
fig_spx.add_trace(go.Scatter(x=df.index, y=df['MA60'], mode='lines', name='MA 60', line=dict(color='orange', width=1)))
fig_spx.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='MA 200', line=dict(color='red', width=1.5)))

fig_spx.update_layout(
    title='S&P 500 Price & Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_dark',
    height=500,
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig_spx, use_container_width=True)


# ---------------------------------------------------------
# 3. 보조 지표 (RSI & MACD)
# ---------------------------------------------------------
col1, col2 = st.columns(2)

# RSI 계산 함수
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['SPX'])

with col1:
    st.subheader("RSI (Relative Strength Index)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='cyan')))
    
    # 과매수/과매도 기준선
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig_rsi.update_layout(template='plotly_dark', height=350, yaxis_range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)

# MACD 계산
exp12 = df['SPX'].ewm(span=12, adjust=False).mean()
exp26 = df['SPX'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp12 - exp26
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Hist'] = df['MACD'] - df['Signal']

with col2:
    st.subheader("MACD (Moving Average Convergence Divergence)")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='yellow')))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red')))
    fig_macd.add_bar(x=df.index, y=df['Hist'], name='Histogram')
    
    fig_macd.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig_macd, use_container_width=True)


st.markdown("---")
st.header("2. Volatility & Tail Risk Analysis")

# ---------------------------------------------------------
# [수정됨] 4. Tail Risk: SKEW Index (위로 이동됨)
# ---------------------------------------------------------
st.subheader("Tail Risk: SKEW Index (Black Swan Risk)")

# 최근 SKEW 값 표시
current_skew = df['SKEW'].iloc[-1]
skew_delta = current_skew - df['SKEW'].iloc[-2]

st.metric(label="Current SKEW Index", value=f"{current_skew:.2f}", delta=f"{skew_delta:.2f}")

fig_skew = go.Figure()
fig_skew.add_trace(go.Scatter(x=df.index, y=df['SKEW'], mode='lines', name='SKEW', line=dict(color='magenta', width=1.5)))

# [수정됨] 기준선 145 -> 150으로 변경
fig_skew.add_hline(y=150, line_dash="dash", line_color="red", 
                   annotation_text="Extreme Fear / Tail Risk Warning (150)", 
                   annotation_position="top left")

# 일반적인 공포 구간 (135)
fig_skew.add_hline(y=135, line_dash="dot", line_color="orange", 
                   annotation_text="Elevated Risk (135)", 
                   annotation_position="bottom left")

# 보통 구간 (100)
fig_skew.add_hline(y=100, line_dash="dash", line_color="gray")

fig_skew.update_layout(
    title='CBOE SKEW Index History',
    yaxis_title='SKEW Index',
    template='plotly_dark',
    height=400
)

st.plotly_chart(fig_skew, use_container_width=True)

# ---------------------------------------------------------
# [수정됨] 5. VIX Term Structure (아래로 이동됨)
# ---------------------------------------------------------
st.subheader("VIX Term Structure (Spot vs 3M vs 6M)")

# 가장 최근 데이터 추출
latest_vix = df.iloc[-1][['VIX', 'VIX3M', 'VIX6M']]
latest_vix_dates = ['Spot VIX', '3-Month VIX', '6-Month VIX']
latest_vix_values = [latest_vix['VIX'], latest_vix['VIX3M'], latest_vix['VIX6M']]

# Term Structure Line Chart
fig_term = go.Figure()

# Spot VIX Trend
fig_term.add_trace(go.Scatter(x=df.index, y=df['VIX'], name='Spot VIX', line=dict(color='green', width=1)))
fig_term.add_trace(go.Scatter(x=df.index, y=df['VIX3M'], name='VIX 3M', line=dict(color='cyan', width=1, dash='dot')))

# Contango/Backwardation 확인
vix_spread = latest_vix['VIX3M'] - latest_vix['VIX']
structure_status = "Contango (Normal)" if vix_spread > 0 else "Backwardation (Fear)"
st.info(f"Current Structure Status: **{structure_status}** (Spread: {vix_spread:.2f})")

fig_term.update_layout(
    title='VIX vs VIX3M Trend',
    yaxis_title='Volatility Points',
    template='plotly_dark',
    height=400
)

st.plotly_chart(fig_term, use_container_width=True)

# 현재 Term Structure 스냅샷 (Bar Chart)
fig_snapshot = go.Figure(data=[go.Bar(
    x=latest_vix_dates, 
    y=latest_vix_values,
    text=[f"{v:.2f}" for v in latest_vix_values],
    textposition='auto',
    marker_color=['green', 'cyan', 'blue']
)])

fig_snapshot.update_layout(
    title=f"Term Structure Snapshot ({last_date})",
    yaxis_title="VIX Value",
    template="plotly_dark",
    height=300
)

st.plotly_chart(fig_snapshot, use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Data Source: Yahoo Finance | Disclaimer: This is for informational purposes only.")
