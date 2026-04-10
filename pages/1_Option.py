import streamlit as st
import pandas as pd
import numpy as np
import OpenDartReader

# 1. 페이지 설정 (가장 먼저 실행)
st.set_page_config(page_title="안티그래비티 EBM 시스템", layout="wide", page_icon="🚀")

# 2. 세션 상태 초기화 (API 키 저장용)
if 'dart_api_key' not in st.session_state:
    st.session_state['dart_api_key'] = ""
if 'llm_api_key' not in st.session_state:
    st.session_state['llm_api_key'] = ""

# 3. 파이프라인 로직 클래스
class EbmPipelineAssistant:
    def __init__(self, dart_key, llm_key):
        self.dart = OpenDartReader(dart_key)
        self.llm_key = llm_key # 향후 OpenAI 등 LLM 호출에 사용

    def fetch_data(self, corp_name, year):
        """1단계: DART 데이터 수집 (에러 방지 강화 버전)"""
        try:
            df = self.dart.finstate_all(corp_name, year, reprt_code='11011')
            if df is None or df.empty:
                return None
            
            # 기업이나 금융업 특성에 따라 '매출액' 대신 '영업수익'을 쓰는 경우가 있어 방어 로직 추가
            target_accounts = ['자산총계', '부채총계', '자본총계', '매출액', '영업이익', '당기순이익', '영업수익']
            df_filtered = df[df['account_nm'].isin(target_accounts)].copy()
            
            if df_filtered.empty:
                st.warning("표준 계정과목을 찾을 수 없습니다. (금융업 등은 계정명이 다를 수 있습니다)")
                return None

            # 쉼표(,)가 포함된 문자열 금액을 안전하게 숫자로 변환
            if df_filtered['thstrm_amount'].dtype == 'object':
                df_filtered['thstrm_amount'] = df_filtered['thstrm_amount'].str.replace(',', '')
            df_filtered['thstrm_amount'] = pd.to_numeric(df_filtered['thstrm_amount'], errors='coerce')
            
            # 핵심 해결책: 에러가 나지 않도록 유저가 입력한 기업명을 강제로 컬럼에 주입
            df_filtered['corp_name'] = corp_name 
            
            # 피벗 테이블 생성
            df_pivot = df_filtered.pivot_table(
                index='corp_name', columns='account_nm', values='thstrm_amount', aggfunc='first'
            ).reset_index()

            # '매출액'이 없고 '영업수익'만 있을 경우 EBM 변환을 위해 이름을 맞춰줌
            if '영업수익' in df_pivot.columns and '매출액' not in df_pivot.columns:
                df_pivot['매출액'] = df_pivot['영업수익']

            # 세션에 검색한 기업명 저장 (3페이지 분석 리포트에서 재사용하기 위함)
            st.session_state['corp_name'] = corp_name

            return df_pivot
            
        except Exception as e:
            st.error(f"DART 호출 오류: {e}")
            return None

    def interpret_columns(self, df):
        """2단계: 컬럼 해석 브리핑"""
        info_list = []
        for col in df.columns:
            if col == 'corp_name': continue
            dtype = "수치형(Numeric)" if pd.api.types.is_numeric_dtype(df[col]) else "범주형(Categorical)"
            missing = df[col].isnull().sum()
            status = "✅ 정상" if missing == 0 else f"⚠️ 결측치 {missing}건"
            info_list.append({"항목(Feature)": col, "데이터 타입": dtype, "상태": status})
        return pd.DataFrame(info_list)

    def transform_for_ebm(self, df):
        """3단계: EBM 파생 변수(비율) 생성"""
        df_ebm = df.copy()
        try:
            if '부채총계' in df_ebm.columns and '자본총계' in df_ebm.columns:
                df_ebm['[파생] 부채비율(%)'] = (df_ebm['부채총계'] / df_ebm['자본총계']) * 100
            if '영업이익' in df_ebm.columns and '매출액' in df_ebm.columns:
                df_ebm['[파생] 영업이익률(%)'] = (df_ebm['영업이익'] / df_ebm['매출액']) * 100
            return df_ebm
        except Exception as e:
            st.warning("변환 중 일부 계정이 누락되어 원본을 유지합니다.")
            return df

    def recommend_mode(self, df):
        """4단계: 타겟 추천 (LLM 연동 가능 영역)"""
        recommendation = """
        **💡 AI 파이프라인 분석 모드 추천 결과**
        현재 수집된 데이터는 '절대적 수치(매출액 등)'와 '비율적 수치(부채비율 등)'가 혼재되어 있습니다. 
        목적에 따라 다음 분석을 추천합니다:
        
        * 📈 **추천 1: 회귀(Regression) 모드 (정밀 신용평가용)**
            * **이유:** 재무 지표의 연속적 변화에 따른 정밀한 '신용 점수(0~1000점)'를 산출해야 할 때 적합합니다. EBM은 특정 비율이 점수에 미치는 영향을 +15점, -5점 형태로 명확히 보여줍니다.
        * 🎯 **추천 2: 분류(Classification) 모드 (부실 징후 조기 탐지)**
            * **이유:** 기업의 '흑자/적자', '부도/생존' 등 명확한 이분법적 리스크를 감지할 때 유리합니다.
        """
        return recommendation


# ==========================================
# 🖥️ UI 영역 (사이드바 및 메인 화면)
# ==========================================

# --- 사이드바: 인증 및 설정 ---
with st.sidebar:
    st.title("🔐 API 인증 설정")
    st.caption("시스템 사용을 위해 키를 입력해 주세요.")
    
    dart_input = st.text_input("1. DART API Key", type="password", value=st.session_state['dart_api_key'])
    llm_input = st.text_input("2. LLM API Key (OpenAI 등)", type="password", value=st.session_state['llm_api_key'], help="분석 결과 해석 및 리포팅에 사용됩니다.")
    
    if st.button("🔑 키 저장 및 적용"):
        st.session_state['dart_api_key'] = dart_input
        st.session_state['llm_api_key'] = llm_input
        st.success("API 키가 안전하게 저장되었습니다!")

# --- 메인 화면 ---
st.title("🚀 코스피 상장기업 EBM 분석 파이프라인")
st.write("다트(DART) 재무 데이터를 EBM 모델에 맞게 변환하고 최적의 분석 방향을 추천합니다.")

if not st.session_state['dart_api_key'] or not st.session_state['llm_api_key']:
    st.warning("👈 좌측 사이드바에서 DART 및 LLM API 키를 모두 입력해야 도구가 활성화됩니다.")
else:
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        corp_name = st.text_input("🏢 분석 대상 기업명 (예: 삼성전자, 현대자동차)", value="삼성전자")
    with col2:
        year = st.selectbox("📅 분석 연도", ["2023", "2022", "2021"])

    if st.button("데이터 수집 및 파이프라인 실행", type="primary"):
        assistant = EbmPipelineAssistant(st.session_state['dart_api_key'], st.session_state['llm_api_key'])
        
        with st.status(f"'{corp_name}' 데이터 처리 중...", expanded=True) as status:
            st.write("📥 DART에서 회계 데이터를 수집하고 있습니다...")
            raw_data = assistant.fetch_data(corp_name, year)
            
            if raw_data is not None:
                st.write("✅ 데이터 수집 완료")
                
                st.write("🔍 컬럼 구조를 해석합니다...")
                col_info = assistant.interpret_columns(raw_data)
                
                st.write("⚙️ EBM 모델용 피처 엔지니어링을 수행합니다...")
                ebm_data = assistant.transform_for_ebm(raw_data)
                
                # ⭐️ 2페이지로 넘기기 위해 세션에 데이터 저장
                st.session_state['ebm_data'] = ebm_data
                st.session_state['recommendation'] = assistant.recommend_mode(ebm_data)
                
                status.update(label="파이프라인 실행 완료!", state="complete", expanded=False)
                
                # --- 결과 출력 영역 ---
                st.subheader("1. 데이터 컬럼 해석 브리핑")
                st.dataframe(col_info, use_container_width=True)
                
                st.subheader("2. EBM 변환 데이터 (비율 지표 추가)")
                st.dataframe(ebm_data, use_container_width=True)
                
                st.subheader("3. 시스템 추천 결과")
                st.info(st.session_state['recommendation'])
                
            else:
                status.update(label="데이터 수집 실패", state="error")
                st.error("해당 기업의 공시 자료를 찾을 수 없습니다. 기업명(정식 명칭)을 확인해 주세요.")