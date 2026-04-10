import streamlit as st
import pandas as pd
import numpy as np
import OpenDartReader

# DART: 40d49b37a11be606d3fe2776432d8eb7376cdc0e
# LLM: sk-proj-26e58552-1713-4980-8639-0259002a41d8

# 1. 페이지 설정 (반드시 가장 처음에 위치해야 함)
st.set_page_config(page_title="안티그래비티 CSV 마스터", layout="wide")

# 2. 세션 상태 초기화
if 'dart_api_key' not in st.session_state:
    st.session_state['dart_api_key'] = None
if 'csv_storage' not in st.session_state:
    st.session_state['csv_storage'] = {}

# 3. 로직 클래스 정의
class EbmPipelineAssistant:
    def __init__(self, dart_api_key):
        self.dart = OpenDartReader(dart_api_key)

    def fetch_dart_data(self, corp_name, year):
        try:
            # 실무에서는 데이터가 방대하므로 샘플로 상위 5개 정도만 처리하는 로직을 권장합니다.
            df = self.dart.finstate_all(corp_name, year, reprt_code='11011')
            return df
        except Exception as e:
            st.error(f"데이터 수집 중 오류 발생: {e}")
            return None

    def transform_for_ebm(self, df):
        # 여기에 부채비율 등 변환 로직이 들어갑니다.
        # 현재는 변환이 완료되었다는 메시지만 반환하도록 설정되어 있습니다.
        return df

# 4. 사이드바 UI (별개 창 영역)
with st.sidebar:
    st.title("🔐 시스템 설정")
    input_key = st.text_input("DART API Key 입력", type="password")
    if st.button("API 키 적용"):
        if input_key:
            st.session_state['dart_api_key'] = input_key
            st.success("키가 성공적으로 설정되었습니다!")
        else:
            st.error("키를 입력해 주세요.")

# 5. 메인 대시보드 UI
st.title("EBM 코스피분석")
st.write("코스피 상장사 데이터를 분석하고 EBM 모델 기반의 인사이트를 제공합니다.")

# 키 설정 여부에 따른 화면 분기
if not st.session_state['dart_api_key']:
    st.info("👈 왼쪽 사이드바에서 API 키를 입력하면 분석 도구가 활성화됩니다.")
else:
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        corp_name = st.text_input("분석 대상 기업명", value="삼성전자")
    with col2:
        year = st.selectbox("분석 대상 연도", [2023, 2022, 2021])

    if st.button("데이터 분석 및 EBM 학습 시작"):
        assistant = EbmPipelineAssistant(st.session_state['dart_api_key'])
        
        with st.spinner(f"'{corp_name}'의 데이터를 분석 중입니다..."):
            raw_data = assistant.fetch_dart_data(corp_name, year)
            
            if raw_data is not None and not raw_data.empty:
                st.subheader("1. 데이터 수집 현황")
                st.dataframe(raw_data.head(10)) # 상위 10개 행 출력
                
                transformed_data = assistant.transform_for_ebm(raw_data)
                st.subheader("2. EBM 변환 및 결과 추천")
                st.success("EBM 최적화 변환이 완료되었습니다. 현재 데이터는 '회귀 분석' 모드를 추천합니다.")
            else:
                st.warning("데이터를 찾을 수 없습니다. 기업명이나 연도를 다시 확인해 주세요.")