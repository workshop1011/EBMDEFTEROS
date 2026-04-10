import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import OpenDartReader

st.set_page_config(page_title="단기 시계열 예측", layout="wide", page_icon="⏳")

st.title("⏳ 건설/연계업종 분기별 단기 예측 모델 (Specialization)")
st.write("모델의 학습 성능 지표를 먼저 확인한 후, 기업의 미래 리스크 및 수익성을 예측합니다.")

# 1페이지에서 API 키가 설정되었는지 확인
if 'dart_api_key' not in st.session_state or not st.session_state['dart_api_key']:
    st.warning("👈 1페이지(Option)에서 DART 및 LLM API 키를 설정해 주세요.")
    st.stop()

# ==========================================
# 1. 분기별 데이터 호출 및 기업 확인
# ==========================================
st.subheader("1. 분기별 시계열 데이터 호출")
corp_name = st.text_input("분석할 기업명 (예: 현대건설)", value=st.session_state.get('corp_name', '현대건설'))

@st.cache_data
def fetch_quarterly_data(corp):
    time.sleep(1.5) 
    if corp == "에러테스트":
        raise ValueError("ERR_MISSING_Q3: 2023년 3분기 재무 데이터 누락")
        
    quarters = [f"2021 Q{i}" for i in range(1, 5)] + [f"2022 Q{i}" for i in range(1, 5)] + [f"2023 Q{i}" for i in range(1, 5)]
    np.random.seed(42)
    
    df = pd.DataFrame({
        '분기': quarters,
        '매출액': np.random.uniform(30000, 50000, 12) + np.arange(12) * 1000,
        '영업이익': np.random.uniform(1000, 3000, 12) + np.arange(12) * 50,
        '미청구공사(위험자산)': np.random.uniform(5000, 10000, 12),
        '부채비율(%)': np.random.uniform(150, 200, 12)
    })
    return df

if st.button("분기 데이터 수집 시작"):
    with st.spinner(f"'{corp_name}'의 최근 3년 치 분기 보고서를 DART에서 수집 중입니다..."):
        try:
            q_data = fetch_quarterly_data(corp_name)
            st.session_state['q_data'] = q_data
            st.session_state['q_corp'] = corp_name
            st.success("데이터 수집 완료!")
        except Exception as e:
            st.error("❌ 데이터 수집 및 분석 준비 실패")
            st.warning(f"시스템 오류 코드: {str(e)}")
            with st.expander("🤖 LLM 분석 피드백 보기", expanded=True):
                st.write("**AI 진단 결과:**")
                st.info(f"해당 기업({corp_name})의 특정 분기 공시 자료가 누락되었습니다. 결측치 보간 후 재시도가 필요합니다.")
            st.stop()

# ==========================================
# 2. 데이터 기반 분석 개시
# ==========================================
if 'q_data' in st.session_state:
    df = st.session_state['q_data']
    st.markdown(f"**호출된 기업:** `[ {st.session_state['q_corp']} ]`")
    
    st.subheader("2. 시계열 모델 분석")
    task_type = st.radio("분석 태스크 선택", ["단기 주가 및 리스크 예측 (회귀)", "흑자/적자 전환 예측 (분류)"])
    
    if st.button("🚀 시계열 예측 모델 가동", type="primary"):
        with st.spinner("안티그래비티 엔진이 모델 성능을 검증하고 미래를 예측 중입니다..."):
            time.sleep(2) 

            # ==========================================
            # 💡 [신규 기능] 4종류의 주요 모델 지표 출력
            # ==========================================
            st.subheader("📈 3. 모델 성능 검증 지표 (Model Performance)")
            st.write("예측 결과 도출 전, 학습된 모델의 기술적 신뢰도를 먼저 표시합니다.")
            
            # 성능 지표 샘플 데이터 (실제 모델 학습 결과값 연동 영역)
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("정확도 (Accuracy)", "94.8%", help="전체 예측 중 정답을 맞춘 비율")
            m_col2.metric("정밀도 (Precision)", "91.2%", help="부실이라고 예측한 것 중 실제 부실인 비율")
            m_col3.metric("재현율 (Recall)", "88.5%", help="실제 부실 중 모델이 찾아낸 비율")
            m_col4.metric("F1-Score", "89.8%", help="정밀도와 재현율의 조화 평균값")
            
            st.divider()

            # ==========================================
            # 4. 분석 성공 시: 5대 핵심 지표 산출
            # ==========================================
            st.subheader("📊 4. 5대 핵심 예측 지표 (향후 1년 기준)")
            
            prob_default = np.random.uniform(2, 15)
            prob_profit = np.random.uniform(60, 95)
            op_margin = np.random.uniform(4, 12)
            health_score = 100 - prob_default
            momentum = np.random.uniform(-10, 20)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("핵심 재무 건전성", f"{health_score:.1f} 점")
            col2.metric("단기 부도 확률", f"{prob_default:.1f} %", delta_color="inverse")
            col3.metric("수익 달성 확률", f"{prob_profit:.1f} %")
            col4.metric("예상 영업이익률", f"{op_margin:.1f} %")
            col5.metric("주가 모멘텀 동력", f"{momentum:+.1f} %")
            
            st.divider()
            
            # ==========================================
            # 5. 주식 미래 예측 선 그래프 시각화
            # ==========================================
            st.subheader(f"📉 5. {st.session_state['q_corp']} 향후 4분기 주가 추이 예측")
            
            past_quarters = df['분기'].tolist()
            future_quarters = [f"2024 Q{i}" for i in range(1, 5)]
            past_stock = np.random.uniform(40000, 60000, 12).tolist()
            future_stock = [past_stock[-1] * (1 + (momentum/100) * (i/4)) + np.random.normal(0, 1000) for i in range(1, 5)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=past_quarters, y=past_stock, mode='lines+markers', name='과거 실제 주가', line=dict(color='#1f77b4', width=3)))
            fig.add_trace(go.Scatter(x=[past_quarters[-1]] + future_quarters, y=[past_stock[-1]] + future_stock, mode='lines+markers', name='시계열 모델 예측치', line=dict(color='#d62728', width=3, dash='dash')))
            
            fig.update_layout(xaxis_title="분기", yaxis_title="주가 (KRW)", hovermode="x unified", height=500)
            fig.add_vrect(x0=past_quarters[-1], x1=future_quarters[-1], fillcolor="rgba(214, 39, 40, 0.1)", layer="below", line_width=0, annotation_text="예측 구간")
            
            st.plotly_chart(fig, use_container_width=True)
            st.success("모델 검증 및 시계열 분석이 완료되었습니다.")