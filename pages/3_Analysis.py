import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import yfinance as yf

# 1. 페이지 설정
st.set_page_config(page_title="안티그래비티 최종 리포트", layout="wide", page_icon="📊")

# --- 세션 상태 확인 (1, 2페이지에서 데이터가 잘 넘어왔는지 체크) ---
if 'trained_model' not in st.session_state or 'ebm_data' not in st.session_state:
    st.warning("👈 1페이지(Option)와 2페이지(ModelEBM)를 거쳐 모델 학습을 먼저 완료해 주세요.")
    st.stop()

# 테스트용 임시 변수 (실제로는 세션에서 가져옵니다)
corp_name = st.session_state.get('corp_name', '삼성전자')
ticker_symbol = "005930.KS" # 예시: 실제 구현시 종목코드 맵핑 딕셔너리 필요

st.title(f"📊 {corp_name} EBM 종합 리포트")
st.divider()

# ==========================================
# 1. 기업 현황 및 최근 공시 요약
# ==========================================
st.subheader(f"🏢 1. {corp_name} ({ticker_symbol}) 현황 브리핑")

try:
    # yfinance를 이용한 실시간 주식 데이터 호출
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1mo")
    current_price = hist['Close'].iloc[-1]
    prev_price = hist['Close'].iloc[-2]
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("현재 주가", f"{int(current_price):,} 원", f"{int(change):,} 원 ({change_pct:.2f}%)")
    col_s2.metric("52주 최고가", f"{int(stock.info.get('fiftyTwoWeekHigh', 0)):,} 원")
    col_s3.metric("시가총액", f"{int(stock.info.get('marketCap', 0) / 100000000):,} 억원")
    
    st.info("📢 **최근 주요 공시 (DART 연동 요약):** 최근 자사주 매입 신고(보통주 5,014,462주) 및 이사회 결의 관련 공시가 있었습니다.") # 예시 데이터
except Exception as e:
    st.error("주식 데이터를 불러오지 못했습니다.")

st.divider()

# ==========================================
# 2. EBM 모델 학습 지표 요약
# ==========================================
st.subheader("⚙️ 2. 안티그래비티 EBM 학습 요약")
# 실제로는 2페이지에서 저장한 학습 기록을 불러옵니다.
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("⏱️ 분석 소요 시간", "4.2 초")
col_m2.metric("💾 사용된 데이터 수", "1,024 건")
col_m3.metric("🧠 설정된 학습률 (LR)", "0.01")
col_m4.metric("🎯 분류 정확도 (Accuracy)", "94.2 %")

st.divider()

# ==========================================
# 3. 데이터 추이 및 EBM 예측 시각화 (Plotly)
# ==========================================
st.subheader("📈 3. 부실 확률 추이 및 모델 예측")
st.caption("과거 데이터(파란색)와 EBM이 예측한 향후 리스크 수치(빨간색)를 비교합니다.")

# 시각화용 샘플 데이터 생성
dates_hist = pd.date_range(start="2023-01-01", periods=12, freq="M")
dates_pred = pd.date_range(start="2024-01-01", periods=6, freq="M")
val_hist = np.random.uniform(10, 30, size=12)
val_pred = val_hist[-1] + np.random.uniform(-5, 15, size=6) # 예측값 생성

# Plotly를 이용한 혼합 그래프 (선 + 바)
fig = go.Figure()

# 과거 추이 (Line + Bar 혼합)
fig.add_trace(go.Bar(x=dates_hist, y=val_hist, name='과거 リ스크 수치', marker_color='#1f77b4', opacity=0.6))
fig.add_trace(go.Scatter(x=dates_hist, y=val_hist, mode='lines+markers', name='과거 추이 선', line=dict(color='#1f77b4', width=2)))

# 예측 추이 (Line + Bar 혼합, 색상 차별화)
fig.add_trace(go.Bar(x=dates_pred, y=val_pred, name='EBM 예측 수치', marker_color='#ff7f0e', opacity=0.6))
fig.add_trace(go.Scatter(x=dates_pred, y=val_pred, mode='lines+markers', name='EBM 예측 선', line=dict(color='#ff7f0e', width=2, dash='dash')))

fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================
# 4. LLM 기반 거시경제 및 추가 정보 브리핑
# ==========================================
st.subheader("🧠 4. AI 매크로 리포트 및 투자 인사이트")

if st.button("✨ LLM 인사이트 리포트 생성 (API 호출)"):
    llm_key = st.session_state.get('llm_api_key', '')
    if not llm_key:
        st.error("1페이지에서 LLM API 키를 설정해주세요.")
    else:
        with st.spinner("EBM 결과와 거시경제 데이터를 종합하여 리포트를 작성 중입니다..."):
            time.sleep(2) # API 호출 시뮬레이션
            
            # 실제 구현 시 OpenAI API 등을 호출하여 아래 프롬프트를 전달합니다.
            # prompt = f"{corp_name}의 EBM 부실 확률 예측치는 {val_pred[-1]:.2f}입니다. 현재 금리 인상 기조와 해당 산업군의 거시경제적 특성을 고려하여 3문단으로 리포트를 작성해줘."
            
            mock_report = f"""
            **1. 거시경제 환경 (Macro-Environment)**
            현재 지속되는 고금리 기조와 환율 변동성 확대는 수출 주도형 코스피 기업들에게 재무적 압박으로 작용하고 있습니다. 특히 {corp_name}이 속한 섹터는 원자재 가격 상승의 영향을 직접적으로 받고 있어 유동성 관리가 필수적인 시점입니다.

            **2. EBM 예측 결과 해석**
            모델(EBM) 분석 결과, 향후 6개월 내 단기 유동성 지표의 하락이 예측 모델에서 주요 리스크 요인(Risk Driver)으로 식별되었습니다. 그래프에서 볼 수 있듯, 주황색 예측 구간에서 리스크 수치가 점진적으로 상승하는 것은 이러한 재무적 민감도가 반영된 결과입니다.

            **3. 종합 제언**
            최근 공시된 대규모 자사주 취득 결정은 시장에 긍정적인 시그널을 줄 수 있으나, 본질적인 재무 건전성 강화를 위해 영업현금흐름 개선이 선행되어야 합니다. 투자자 및 관리자는 다음 분기 실적 발표 시 '재고자산 회전율'을 중점적으로 모니터링할 것을 권장합니다.
            """
            st.success("리포트 작성이 완료되었습니다.")
            st.markdown(mock_report)