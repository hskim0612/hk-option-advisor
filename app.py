import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd

def get_macro_score_v18():
    print("🌐 거시경제 데이터(FRED) 분석 중...")
    
    start = datetime.now() - timedelta(days=400) # 1년 이상 데이터 필요
    end = datetime.now()
    
    # FRED 코드 매핑
    # DGS10: 10년물 국채, UNRATE: 실업률, CPIAUCSL: 소비자물가, FEDFUNDS: 기준금리
    indicators = {
        'US10Y': 'DGS10',
        'Unemployment': 'UNRATE',
        'CPI': 'CPIAUCSL',
        'FedRate': 'FEDFUNDS'
    }
    
    try:
        macro_data = web.DataReader(list(indicators.values()), 'fred', start, end)
        macro_data.columns = list(indicators.keys())
        
        # 데이터 전처리 (결측치 채움)
        df = macro_data.ffill().dropna()
        current = df.iloc[-1]
        
        score = 0
        reasons = []
        
        # 1. 10년물 금리 (상대적 평가: 200일 이동평균 대비)
        # 금리가 평소(200MA)보다 급격히 높으면 기술주 악재
        ma200_yield = df['US10Y'].rolling(window=200).mean().iloc[-1]
        
        if current['US10Y'] > ma200_yield * 1.1: # 200일선보다 10% 이상 높음
            score -= 3
            reasons.append(f"📉 금리 부담 (현재 {current['US10Y']:.2f}% > 200MA {ma200_yield:.2f}%)")
        elif current['US10Y'] < ma200_yield:
            score += 2
            reasons.append("📈 금리 안정 (200MA 하회)")
            
        # 2. 실업률 (Sahm Rule 로직: 급격한 악화 감지)
        # 최근 3개월 평균 실업률
        curr_unemp_ma3 = df['Unemployment'].iloc[-3:].mean()
        # 지난 12개월 최저 실업률
        min_unemp_12m = df['Unemployment'].iloc[-12:].min()
        
        if curr_unemp_ma3 >= min_unemp_12m + 0.5:
            score -= 5 # 경기 침체 경고 (강력한 매도/헷지 신호)
            reasons.append(f"🚨 침체 경고 (Sahm Rule 발동: 실업률 급등)")
        else:
            score += 1
            reasons.append("✅ 고용 안정")

        # 3. CPI (인플레이션 추세) - 전년 동기 대비(YoY) 변화율 하락 여부
        # 데이터가 월간이므로 12개월 전 데이터와 비교
        cpi_yoy_now = (df['CPI'].iloc[-1] / df['CPI'].iloc[-13] - 1) * 100
        cpi_yoy_prev = (df['CPI'].iloc[-2] / df['CPI'].iloc[-14] - 1) * 100
        
        if cpi_yoy_now < cpi_yoy_prev:
            score += 2
            reasons.append("✅ 디스인플레이션 (물가 상승률 둔화)")
        else:
            score -= 2
            reasons.append("⚠️ 물가 재반등 우려")

        # 4. 연준 스탠스 (3개월 전 금리와 비교)
        fed_now = current['FedRate']
        fed_3m_ago = df['FedRate'].iloc[-90] if len(df) > 90 else df['FedRate'].iloc[0]
        
        if fed_now < fed_3m_ago - 0.1:
            score += 3
            reasons.append("🕊️ 금리 인하 사이클 (유동성 공급)")
        elif fed_now > fed_3m_ago + 0.1:
            score -= 3
            reasons.append("🦅 금리 인상 사이클 (유동성 축소)")
        else:
            reasons.append("⚖️ 금리 동결/중립")
            
        return score, reasons

    except Exception as e:
        print(f"⚠️ 매크로 데이터 수집 실패: {e}")
        return 0, ["데이터 수집 오류로 0점 처리"]

# 사용 예시
# tech_score = 7 (기존 v17 로직)
# macro_score, macro_reasons = get_macro_score_v18()
# final_score = tech_score + macro_score
