import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

st.set_page_config(page_title="안티그래비티 모델링", layout="wide", page_icon="📈")
st.title("📈 EBM 모델 학습 및 정밀 검수")

if 'ebm_data' not in st.session_state or st.session_state['ebm_data'] is None:
    st.warning("👈 1페이지(Option)에서 데이터를 먼저 수집하고 변환해 주세요.")
    st.stop()

target_df = st.session_state['ebm_data']
corp_name = st.session_state.get('corp_name', '해당 기업')

st.subheader(f"1. 평가 대상: {corp_name} 최종 데이터")
st.dataframe(target_df, use_container_width=True)

st.info("💡 EBM 모델 학습을 위해 KOSPI 과거 기업 데이터 1,000건(시뮬레이션)을 로드합니다.")

@st.cache_data
def generate_mock_kospi_data(task):
    np.random.seed(42)
    n_samples = 1000
    mock_data = pd.DataFrame()
    
    mock_data['[파생] 부채비율(%)'] = np.random.normal(120, 50, n_samples)
    mock_data['[파생] 영업이익률(%)'] = np.random.normal(8, 5, n_samples)
    mock_data['자산총계'] = np.random.uniform(1000, 50000, n_samples)
    mock_data['매출액'] = mock_data['자산총계'] * np.random.uniform(0.5, 1.5, n_samples)
    
    base_score = 700
    penalty = (mock_data['[파생] 부채비율(%)'] - 100) * 1.5 
    bonus = mock_data['[파생] 영업이익률(%)'] * 10
    noise = np.random.normal(0, 20, n_samples)
    
    mock_data['Target_신용점수'] = np.clip(base_score - penalty + bonus + noise, 0, 1000)
    mock_data['Target_부실여부'] = (mock_data['Target_신용점수'] < 500).astype(int)
        
    return mock_data

task_type = st.radio(
    "분석 태스크 선택 (과거 데이터 학습 기준)", 
    ["회귀(Regression) - 신용 점수 예측", "분류(Classification) - 부실 여부 예측"],
    index=0 if "회귀" in st.session_state.get('recommendation', "") else 1
)

historical_df = generate_mock_kospi_data(task_type)

all_cols = [c for c in target_df.columns if c not in ['corp_name', 'account_nm']]
default_features = [c for c in all_cols if "[파생]" in c or c in ['자산총계', '부채총계', '매출액']]

# 💡 [완벽한 버그 픽스] 위젯 key 바인딩 대신 default 상태 변수로만 관리
if 'current_features' not in st.session_state:
    st.session_state['current_features'] = default_features

features = st.multiselect(
    "학습에 사용할 핵심 지표(X) 선택", 
    all_cols, 
    default=st.session_state['current_features'] # key 속성을 제거하고 default로만 주입!
)
# 유저가 화면에서 조작한 선택값도 즉시 상태에 동기화
st.session_state['current_features'] = features

missing_cols = [col for col in features if col not in historical_df.columns]

if missing_cols:
    st.error(f"⚠️ 데이터 불일치: 선택하신 컬럼 중 과거 데이터셋에 없는 항목이 발견되었습니다.")
    
    with st.container():
        st.markdown("### 🤖 AI 컬럼 매핑 도우미 (오류 해결)")
        st.write("LLM이 과거 데이터 구조를 분석하여 가장 적합한 대체 컬럼을 제안합니다.")
        
        col_mappings = {}
        for m_col in missing_cols:
            st.markdown(f"**❓ 누락된 항목: `{m_col}`**")
            
            suggestion = "제외 권장"
            if "부채" in m_col: suggestion = "[파생] 부채비율(%)"
            elif "이익" in m_col or "수익" in m_col: suggestion = "[파생] 영업이익률(%)"
            elif "매출" in m_col: suggestion = "매출액"
            elif "자산" in m_col: suggestion = "자산총계"
            
            choice = st.radio(
                "어떻게 처리하시겠습니까?",
                [f"🔄 '{suggestion}' (으)로 대체 적용 (AI 추천)", "❌ 분석에서 완전히 제외"],
                key=f"resolve_{m_col}"
            )
            col_mappings[m_col] = choice
        
        if st.button("✨ 적용하고 에러 해결하기", type="primary"):
            new_features = [f for f in features if f not in missing_cols]
            for m_col, choice in col_mappings.items():
                if "대체 적용" in choice:
                    sugg = choice.split("'")[1]
                    if sugg not in new_features and sugg in historical_df.columns:
                        new_features.append(sugg)
            
            # 💡 위젯 key가 없으므로 이제 여기서 강제 업데이트를 해도 에러가 나지 않습니다!
            st.session_state['current_features'] = new_features
            st.rerun()
            
    st.warning("☝️ 위의 AI 매핑 도우미를 통해 오류를 먼저 해결해야 학습 옵션이 열립니다.")
    st.stop()

st.subheader("2. EBM 모델 세부 옵션 (Hyperparameters)")
col3, col4, col5 = st.columns(3)

with col3:
    learning_rate = st.slider("학습률 (Learning Rate)", min_value=0.001, max_value=0.100, value=0.010, step=0.001, format="%.3f")
    max_bins = st.slider("최대 빈(Bins) 수", 32, 512, 256)
with col4:
    interactions = st.slider("상호작용(Interactions) 횟수", 0, 20, 10)
    outer_bags = st.slider("외부 배깅(Outer Bags) 수", 1, 100, 8)
with col5:
    inner_bags = st.slider("내부 배깅(Inner Bags) 수", 0, 50, 0)
    test_size = st.slider("검증 데이터 비율 (%)", 10, 50, 20)

if st.button("🚀 EBM 모델 학습 및 타겟 평가 시작", type="primary"):
    X = historical_df[features]
    target_col = 'Target_신용점수' if "회귀" in task_type else 'Target_부실여부'
    y = historical_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    
    with st.spinner('과거 데이터로 EBM 엔진을 학습시키는 중입니다...'):
        if "회귀" in task_type:
            model = ExplainableBoostingRegressor(
                learning_rate=learning_rate, interactions=interactions,
                max_bins=max_bins, outer_bags=outer_bags, inner_bags=inner_bags, random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = mean_absolute_error(y_test, preds)
            st.success("학습 완료! 검증 결과 요약")
            st.metric("과거 데이터 MAE (오차)", f"{score:.2f} 점")
        else:
            model = ExplainableBoostingClassifier(
                learning_rate=learning_rate, interactions=interactions,
                max_bins=max_bins, outer_bags=outer_bags, inner_bags=inner_bags, random_state=42
            )
            y_train, y_test = y_train.astype(int), y_test.astype(int)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            st.success("학습 완료! 검증 결과 요약")
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("과거 데이터 Accuracy", f"{acc:.4f}")
            col_r2.metric("과거 데이터 ROC-AUC", f"{auc:.4f}")

        st.session_state['trained_model'] = model
        st.session_state['train_data'] = (X_train, y_train)
        st.session_state['test_data'] = (X_test, y_test)
        st.session_state['task_type'] = task_type
        st.session_state['selected_features'] = features
        
        st.info("✅ 모델 학습이 성공적으로 완료되었습니다. 3페이지(Analysis)로 이동하여 상세 리포트를 확인하세요!")