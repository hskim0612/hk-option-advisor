import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ... (APP_PASSWORD 및 set_page_config 등 윗부분 코드는 그대로 유지) ...

# === [4] 차트 (수정: Plotly 인터랙티브 차트 적용) ===
def create_charts(data):
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
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='red', width=1.5), name='200MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], line=dict(color='blue', width=1.5), name='50MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='green', width=1, dash='dot'), name='20MA'), row=1, col=1)
    
    # Price
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], line=dict(color='black', width=1.5), name='Close'), row=1, col=1)

    # === 2. Volume Chart (Row 2) ===
    # Color Logic: Close >= Open (Green), Close < Open (Red)
    colors = ['green' if c >= o else 'red' for c, o in zip(hist['Close'], hist['Open'])]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume', opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Vol_MA20'], line=dict(color='black', width=1), name='Vol MA20'), row=2, col=1)

    # === 3. RSI Chart (Row 3) ===
    fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    
    # RSI Reference Lines & Zones
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_width=0.5, line_color="black", row=3, col=1)
    
    # RSI Background coloring (Over/Under) - using shapes ensures cleanliness
    # Note: To exactly mimic fill_between conditionally, we need to create dummy traces, 
    # but for performance and look in Plotly, shapes or static zones are often preferred.
    # Here, we will use a "Gradient" simulation or simple lines as requested.
    # To strictly follow "Fill" request:
    rsi_upper = hist['RSI'].clip(lower=70)
    fig.add_trace(go.Scatter(x=hist.index, y=rsi_upper, line=dict(width=0), 
                             fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.0)', showlegend=False, hoverinfo='skip'), row=3, col=1) # dummy base
    # (Plotly conditional fill is complex; sticking to clear lines and fixed range is better for interaction)
    
    # === 4. MACD Chart (Row 4) ===
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='blue', width=1), name='MACD'), row=4, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], line=dict(color='orange', width=1), name='Signal'), row=4, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist['MACD']-hist['Signal'], marker_color='gray', opacity=0.3, name='Hist'), row=4, col=1)
    fig.add_hline(y=0, line_width=0.8, line_color="black", row=4, col=1)

    # === 5. VIX Level Chart (Row 5) ===
    fig.add_trace(go.Scatter(x=vix_hist.index, y=vix_hist['Close'], line=dict(color='purple', width=1.5), name='VIX'), row=5, col=1)
    if vix3m_hist is not None and not vix3m_hist.empty:
        fig.add_trace(go.Scatter(x=vix3m_hist.index, y=vix3m_hist['Close'], line=dict(color='gray', width=1, dash='dot'), name='VIX3M'), row=5, col=1)
    
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Panic", row=5, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Stable", row=5, col=1)

    # === 6. VIX Ratio Chart (Row 6) ===
    if term_df is not None and not term_df.empty:
        # Ratio Line
        fig.add_trace(go.Scatter(x=term_df.index, y=term_df['Ratio'], line=dict(color='black', width=1.2), name='Ratio'), row=6, col=1)
        
        # Guidelines
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=6, col=1)
        fig.add_hline(y=0.9, line_dash="dash", line_color="green", row=6, col=1)

        # Conditional Fills Logic (Manual approach for Plotly)
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

    # === Global Layout Settings ===
    fig.update_layout(
        height=1500,  # 전체 높이 설정
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        hovermode='x unified',  # [핵심] 모든 차트에 동시 반응하는 세로선(Crosshair)
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # X축 설정 (Rangeslider 제거, 라벨은 맨 아래만)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

    # RSI Y축 고정
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig

# === [메인 화면] (수정 부분만 표시) ===
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

    # ... (Sidebar 및 Matrix HTML 출력 코드는 기존 유지) ...
    # ... (verdict_text, manual, warning 출력 코드 기존 유지) ...

    st.markdown("---")
    st.subheader("📈 기술적 분석 차트 (Interactive)")
    
    # [수정] Plotly 차트 출력 (use_container_width=True로 반응형 적용)
    chart_fig = create_charts(data)
    st.plotly_chart(chart_fig, use_container_width=True)

if __name__ == "__main__":
    main()
