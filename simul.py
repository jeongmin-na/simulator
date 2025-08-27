import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json

# ------------------ 기본 설정 ------------------
st.set_page_config(
    page_title="🚀 고급 Cursor AI 요금제 시뮬레이터", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ 스타일링 ------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 데이터 설정 ------------------
@st.cache_data
def load_pricing_data():
    # 가격 단위: 이미지 기준은 $/Mtoken 이므로 1k 기준으로 환산 (÷1000)
    # 범위가 있는 항목은 min/max 제공. 범위가 없는 항목은 min=max로 설정
    return {
        # Anthropic
        "Claude Opus 4.1": {
            "price_input_min": 14.0/1000, "price_input_max": 14.0/1000,
            "price_output_min": 75.0/1000, "price_output_max": 75.0/1000,
            "avg_input": 900, "avg_output": 1800
        },
        "Claude Sonnet 4": {
            "price_input_min": 3.0/1000, "price_input_max": 6.0/1000,
            "price_output_min": 15.0/1000, "price_output_max": 22.5/1000,
            "avg_input": 750, "avg_output": 1500
        },
        # OpenAI
        "OpenAI GPT-5": {
            "price_input_min": 1.25/1000, "price_input_max": 1.25/1000,
            "price_output_min": 10.0/1000, "price_output_max": 10.0/1000,
            "avg_input": 1000, "avg_output": 2000
        },
        "OpenAI GPT-4.1/o3": {
            "price_input_min": 2.0/1000, "price_input_max": 2.0/1000,
            "price_output_min": 8.0/1000, "price_output_max": 8.0/1000,
            "avg_input": 900, "avg_output": 1800
        },
        "OpenAI 2.5 Pro": {
            "price_input_min": 1.25/1000, "price_input_max": 2.5/1000,
            "price_output_min": 10.0/1000, "price_output_max": 15.0/1000,
            "avg_input": 900, "avg_output": 1800
        },
        # Google Gemini
        "Gemini 2.5 Flash": {
            "price_input_min": 0.3/1000, "price_input_max": 0.3/1000,
            "price_output_min": 2.5/1000, "price_output_max": 2.5/1000,
            "avg_input": 800, "avg_output": 1600
        },
    }

pricing_data = load_pricing_data()

# ------------------ 비용 상수 ------------------
# 월 기본요금과 포함 요청 수, 수수료 (백만 토큰당 $0.25)
BASE_MONTHLY_FEE = 60.0
INCLUDED_REQUESTS = {
    "Claude Sonnet 4": 225,
    "OpenAI GPT-5": 500,
    "Gemini 2.5 Flash": 550,
}
SURCHARGE_PER_MTOKEN = 0.25

def get_price_by_basis(config: dict, basis: str) -> tuple:
    """가격 기준에 따른 1k당 입력/출력 단가를 반환합니다.
    basis: 'min' | 'avg' | 'max'"""
    if basis == 'min':
        return config["price_input_min"], config["price_output_min"]
    if basis == 'max':
        return config["price_input_max"], config["price_output_max"]
    # avg
    input_price = (config["price_input_min"] + config["price_input_max"]) / 2
    output_price = (config["price_output_min"] + config["price_output_max"]) / 2
    return input_price, output_price

# ------------------ 계산 함수 ------------------
def calculate_costs(
    monthly_usage,
    model_split,
    pricing_data,
    input_multiplier=1.0,
    output_multiplier=1.0,
    price_basis: str = 'avg',
    avg_tokens_override: dict | None = None,
    included_requests: dict | None = None,
    surcharge_per_mtoken: float = 0.0
):
    results = []
    for model, config in pricing_data.items():
        if model in model_split:
            usage_share = int(monthly_usage * (model_split[model] / 100.0))

            # 포함 요청 처리 (무료 구간)
            included = (included_requests or {}).get(model, 0)
            free_requests = min(usage_share, included)
            billable_requests = max(0, usage_share - included)

            # 평균 토큰 오버라이드
            if avg_tokens_override and model in avg_tokens_override:
                avg_input = avg_tokens_override[model]["avg_input"]
                avg_output = avg_tokens_override[model]["avg_output"]
            else:
                avg_input = config["avg_input"]
                avg_output = config["avg_output"]

            # 청구 대상 요청에 대해서만 토큰 계산
            input_tokens = billable_requests * avg_input * input_multiplier
            output_tokens = billable_requests * avg_output * output_multiplier
            price_in, price_out = get_price_by_basis(config, price_basis)
            
            input_cost = (input_tokens / 1000) * price_in
            output_cost = (output_tokens / 1000) * price_out
            # 커서 수수료: (입력+출력) 총 토큰 기준, $/1M tokens
            surcharge_fee = ((input_tokens + output_tokens) / 1_000_000) * (surcharge_per_mtoken or 0.0)
            total_cost = input_cost + output_cost + surcharge_fee
            
            results.append({
                "모델": model,
                "사용량": int(usage_share),
                "무료 포함 요청": int(free_requests),
                "청구 요청": int(billable_requests),
                "Input Tokens": int(input_tokens),
                "Output Tokens": int(output_tokens),
                "Input 비용($)": round(input_cost, 4),
                "Output 비용($)": round(output_cost, 4),
                "수수료($)": round(surcharge_fee, 4),
                "총 비용($)": round(total_cost, 4),
                "비용 비율(%)": 0  # 나중에 계산
            })
    
    df = pd.DataFrame(results)
    if not df.empty:
        total_cost = df["총 비용($)"].sum()
        df["비용 비율(%)"] = (df["총 비용($)"] / total_cost * 100).round(1)
    
    return df

# ------------------ 메인 UI ------------------
st.markdown('<div class="main-header"><h1>🚀 고급 Cursor AI 요금제 시뮬레이터</h1><p>다양한 관점에서 AI 모델 사용 비용을 분석하고 최적화하세요</p></div>', unsafe_allow_html=True)

# ------------------ 최소 사이드바 ------------------
st.sidebar.button("simulation")

# ------------------ 우측 컨트롤 패널 ------------------
st.markdown("### 🎛️ 시뮬레이션 설정")

col_a, col_b, col_c = st.columns(3)

# 기본 사용자 설정
with col_a:
    st.subheader("👤 사용자 프로필")
    user_type = st.selectbox("사용자 유형", ["개발자 (Heavy)", "연구자 (Medium)", "일반 사용자 (Light)", "기업팀 (Enterprise)", "사용자 정의"], key="user_type_main")

    usage_presets = {
        "개발자 (Heavy)": 1500,
        "연구자 (Medium)": 800,
        "일반 사용자 (Light)": 300,
        "기업팀 (Enterprise)": 3000,
        "사용자 정의": 500
    }
    default_usage = usage_presets[user_type]
    monthly_usage = st.slider("월간 요청 수", 50, 5000, default_usage, step=50, key="monthly_usage_main")

# 고급 설정
with col_b:
    st.subheader("⚙️ 고급 설정")
    input_multiplier = st.slider("Input 토큰 배수", 0.5, 3.0, 1.0, 0.1, help="평균 대비 입력 길이 조정", key="input_mult_main")
    output_multiplier = st.slider("Output 토큰 배수", 0.5, 3.0, 1.0, 0.1, help="평균 대비 출력 길이 조정", key="output_mult_main")

# 가격 기준 및 기타
with col_c:
    st.subheader("💵 가격 기준")
    price_basis = st.selectbox("가격 기준", ["avg", "min", "max"], index=0, help="범위가 있는 가격의 기준 선택", key="price_basis_main")

# 모델별 사용 비율
st.markdown("### 🤖 모델 사용 비율")
model_split = {}

if user_type == "개발자 (Heavy)":
    defaults = {"Claude Opus 4.1": 10, "Claude Sonnet 4": 40, "OpenAI GPT-5": 20, "OpenAI GPT-4.1/o3": 15, "OpenAI 2.5 Pro": 10, "Gemini 2.5 Flash": 5}
elif user_type == "연구자 (Medium)":
    defaults = {"Claude Opus 4.1": 5, "Claude Sonnet 4": 30, "OpenAI GPT-5": 20, "OpenAI GPT-4.1/o3": 25, "OpenAI 2.5 Pro": 10, "Gemini 2.5 Flash": 10}
elif user_type == "일반 사용자 (Light)":
    defaults = {"Claude Opus 4.1": 0, "Claude Sonnet 4": 25, "OpenAI GPT-5": 10, "OpenAI GPT-4.1/o3": 20, "OpenAI 2.5 Pro": 15, "Gemini 2.5 Flash": 30}
elif user_type == "기업팀 (Enterprise)":
    defaults = {"Claude Opus 4.1": 15, "Claude Sonnet 4": 35, "OpenAI GPT-5": 15, "OpenAI GPT-4.1/o3": 20, "OpenAI 2.5 Pro": 10, "Gemini 2.5 Flash": 5}
else:
    defaults = {m: int(100/len(pricing_data)) for m in pricing_data}

cols = st.columns(3)
idx = 0
total_percent = 0
for model in pricing_data.keys():
    with cols[idx % 3]:
        val = st.slider(f"{model}", 0, 100, defaults.get(model, 0), key=f"model_{model}")
    model_split[model] = val
    total_percent += val
    idx += 1

if total_percent != 100:
    st.error(f"⚠️ 비율 합계: {total_percent}% (100%가 되어야 함)")
    st.stop()
else:
    st.success("✅ 비율 합계: 100%")

# ------------------ 평균 토큰 사용자 입력 ------------------
st.markdown("### 🧮 모델별 평균 토큰(요청당)")
avg_tokens_override = {}
with st.expander("평균 토큰 직접 입력", expanded=False):
    for model, cfg in pricing_data.items():
        c1, c2 = st.columns(2)
        with c1:
            avg_in = st.number_input(
                f"{model} 평균 Input tokens", min_value=1, max_value=20000,
                value=int(cfg["avg_input"]), step=50, key=f"avg_in_{model}"
            )
        with c2:
            avg_out = st.number_input(
                f"{model} 평균 Output tokens", min_value=1, max_value=20000,
                value=int(cfg["avg_output"]), step=50, key=f"avg_out_{model}"
            )
        avg_tokens_override[model] = {"avg_input": avg_in, "avg_output": avg_out}

# ------------------ 메인 콘텐츠 ------------------
# 기본 계산 (우측 컨트롤 패널 값을 사용)
results_df = calculate_costs(
    monthly_usage,
    model_split,
    pricing_data,
    input_multiplier,
    output_multiplier,
    price_basis,
    avg_tokens_override=avg_tokens_override,
    included_requests=INCLUDED_REQUESTS,
    surcharge_per_mtoken=SURCHARGE_PER_MTOKEN
)

# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 기본 분석", "📈 시각화 대시보드", "⚖️ 모델 비교", "📅 시간별 분석", "💡 최적화 제안"])

# ------------------ 탭 1: 기본 분석 ------------------
with tab1:
    st.header("📊 기본 비용 분석")
    
    # 주요 지표 표시
    col1, col2, col3, col4 = st.columns(4)
    
    surcharge_total = results_df["수수료($)"].sum() if "수수료($)" in results_df.columns else 0.0
    total_cost = results_df["총 비용($)"].sum() + BASE_MONTHLY_FEE
    total_requests = results_df["사용량"].sum()
    cost_per_request = total_cost / total_requests if total_requests > 0 else 0
    most_expensive_model = results_df.loc[results_df["총 비용($)"].idxmax(), "모델"] if not results_df.empty else "N/A"
    
    with col1:
        st.metric("💰 월간 총 비용", f"${total_cost:.2f}")
    with col2:
        st.metric("🔢 총 요청 수", f"{total_requests:,}")
    with col3:
        st.metric("📊 요청당 평균 비용", f"${cost_per_request:.4f}")
    with col4:
        st.metric("🏆 최고 비용 모델", most_expensive_model)

    # 기본요금 및 수수료 별도 지표
    col5, col6 = st.columns(2)
    with col5:
        st.metric("📦 기본요금", f"${BASE_MONTHLY_FEE:.2f}")
    with col6:
        st.metric("💸 수수료 합계", f"${surcharge_total:.2f}")
    
    # 상세 테이블
    st.subheader("📋 모델별 상세 분석")
    
    # 컬럼 선택
    display_columns = st.multiselect(
        "표시할 컬럼 선택:",
        options=results_df.columns.tolist(),
        default=["모델", "사용량", "총 비용($)", "비용 비율(%)"]
    )
    
    if display_columns:
        formatted_df = results_df[display_columns].copy()
        
        # 숫자 포맷팅
        if "총 비용($)" in formatted_df.columns:
            formatted_df["총 비용($)"] = formatted_df["총 비용($)"].apply(lambda x: f"${x:.4f}")
        if "Input 비용($)" in formatted_df.columns:
            formatted_df["Input 비용($)"] = formatted_df["Input 비용($)"].apply(lambda x: f"${x:.4f}")
        if "Output 비용($)" in formatted_df.columns:
            formatted_df["Output 비용($)"] = formatted_df["Output 비용($)"].apply(lambda x: f"${x:.4f}")
        
        st.dataframe(formatted_df, use_container_width=True)
    
    # 비용 절약 팁
    st.subheader("💡 빠른 절약 팁")
    cheapest_model = results_df.loc[results_df["총 비용($)"].idxmin(), "모델"] if not results_df.empty else "N/A"
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"💰 **가장 경제적인 모델**: {cheapest_model}")
        if not results_df.empty:
            savings = (results_df["총 비용($)"].max() - results_df["총 비용($)"].min()) * monthly_usage / 100
            st.success(f"모델 최적화로 월 최대 ${savings:.2f} 절약 가능")
    
    with col2:
        st.warning("🔧 **최적화 포인트**:")
        st.write("• Input/Output 토큰 길이 조정")
        st.write("• 모델별 사용 비율 재조정")
        st.write("• 작업 유형별 모델 선택")

# ------------------ 탭 2: 시각화 대시보드 ------------------
with tab2:
    st.header("📈 시각화 대시보드")
    
    if not results_df.empty:
        # 그래프 레이아웃 선택
        chart_layout = st.radio(
            "차트 레이아웃:",
            ["2x2 그리드", "세로 배열", "가로 배열"],
            horizontal=True
        )
        
        if chart_layout == "2x2 그리드":
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
        elif chart_layout == "세로 배열":
            col1 = col2 = col3 = col4 = st.container()
        else:  # 가로 배열
            col1, col2, col3, col4 = st.columns(4)
        
        # 1. 비용 분포 파이 차트
        with col1:
            st.subheader("🥧 모델별 비용 분포")
            fig_pie = px.pie(
                results_df, 
                values="총 비용($)", 
                names="모델",
                title="비용 분포",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # 2. Input vs Output 비용 비교
        with col2:
            st.subheader("📊 Input vs Output 비용")
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name='Input 비용',
                x=results_df['모델'],
                y=results_df['Input 비용($)'],
                marker_color='lightblue'
            ))
            fig_bar.add_trace(go.Bar(
                name='Output 비용',
                x=results_df['모델'],
                y=results_df['Output 비용($)'],
                marker_color='darkblue'
            ))
            fig_bar.update_layout(
                title="Input vs Output 비용 비교",
                barmode='stack',
                xaxis_title="모델",
                yaxis_title="비용 ($)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # 3. 토큰 사용량 히트맵
        with col3:
            st.subheader("🔥 토큰 사용량 히트맵")
            heatmap_data = results_df[['모델', 'Input Tokens', 'Output Tokens']].set_index('모델')
            
            fig_heatmap = px.imshow(
                heatmap_data.T,
                labels=dict(x="모델", y="토큰 타입", color="토큰 수"),
                title="토큰 사용량 히트맵",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 4. 효율성 분석 (요청당 비용)
        with col4:
            st.subheader("⚡ 모델 효율성 분석")
            results_df['요청당 비용'] = results_df['총 비용($)'] / results_df['사용량']
            
            fig_efficiency = px.scatter(
                results_df,
                x='사용량',
                y='요청당 비용',
                size='총 비용($)',
                color='모델',
                title="효율성 분석 (요청당 비용)",
                labels={'사용량': '월간 사용량', '요청당 비용': '요청당 비용 ($)'}
            )
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # 추가 인터랙티브 차트
        st.subheader("🎯 커스텀 분석")
        
        analysis_type = st.selectbox(
            "분석 유형 선택:",
            ["비용 트렌드", "토큰 효율성", "모델 성능 대비 비용", "사용량 분포"]
        )
        
        if analysis_type == "비용 트렌드":
            # 사용량 변화에 따른 비용 트렌드
            usage_range = np.arange(100, 3000, 100)
            trend_data = []
            
            for usage in usage_range:
                temp_df = calculate_costs(
                    usage, model_split, pricing_data,
                    input_multiplier, output_multiplier, price_basis,
                    avg_tokens_override=avg_tokens_override,
                    included_requests=INCLUDED_REQUESTS,
                    surcharge_per_mtoken=SURCHARGE_PER_MTOKEN
                )
                for _, row in temp_df.iterrows():
                    trend_data.append({
                        'Usage': usage,
                        'Model': row['모델'],
                        'Cost': row['총 비용($)']
                    })
            
            trend_df = pd.DataFrame(trend_data)
            fig_trend = px.line(
                trend_df,
                x='Usage',
                y='Cost',
                color='Model',
                title="사용량에 따른 비용 트렌드",
                labels={'Usage': '월간 사용량', 'Cost': '비용 ($)'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        elif analysis_type == "토큰 효율성":
            results_df['토큰당 비용'] = results_df['총 비용($)'] / (results_df['Input Tokens'] + results_df['Output Tokens'])
            fig_token_eff = px.bar(
                results_df,
                x='모델',
                y='토큰당 비용',
                title="토큰당 비용 효율성",
                color='토큰당 비용',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_token_eff, use_container_width=True)

# ------------------ 탭 3: 모델 비교 ------------------
with tab3:
    st.header("⚖️ 모델 심화 비교")
    
    if not results_df.empty:
        # 비교할 모델 선택
        selected_models = st.multiselect(
            "비교할 모델 선택:",
            options=results_df['모델'].tolist(),
            default=results_df['모델'].tolist()[:3]
        )
        
        if len(selected_models) >= 2:
            comparison_df = results_df[results_df['모델'].isin(selected_models)].copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 비교 테이블")
                
                # 순위 계산
                comparison_df['비용 순위'] = comparison_df['총 비용($)'].rank()
                comparison_df['효율성 순위'] = comparison_df['요청당 비용'].rank()
                
                display_cols = ['모델', '총 비용($)', '요청당 비용', '비용 순위', '효율성 순위']
                st.dataframe(comparison_df[display_cols], use_container_width=True)
                
            with col2:
                st.subheader("🏆 승자 분석")
                
                cheapest = comparison_df.loc[comparison_df['총 비용($)'].idxmin()]
                most_efficient = comparison_df.loc[comparison_df['요청당 비용'].idxmin()]
                
                st.success(f"💰 **최저 비용**: {cheapest['모델']} (${cheapest['총 비용($)']:.4f})")
                st.info(f"⚡ **최고 효율성**: {most_efficient['모델']} (${most_efficient['요청당 비용']:.6f}/요청)")
                
                # 절약 계산
                if len(comparison_df) > 1:
                    max_cost = comparison_df['총 비용($)'].max()
                    min_cost = comparison_df['총 비용($)'].min()
                    savings_percent = ((max_cost - min_cost) / max_cost) * 100
                    
                    st.metric(
                        "💡 최대 절약 가능 비율",
                        f"{savings_percent:.1f}%",
                        f"${max_cost - min_cost:.4f} 절약"
                    )
            
            # 상세 비교 차트
            st.subheader("📈 상세 비교 분석")
            
            # 레이더 차트를 위한 데이터 준비
            metrics = ['총 비용($)', 'Input 비용($)', 'Output 비용($)', '요청당 비용']
            
            fig_radar = go.Figure()
            
            for model in selected_models:
                model_data = comparison_df[comparison_df['모델'] == model].iloc[0]
                
                # 정규화 (0-1 스케일로 변환, 비용은 역순으로)
                values = []
                for metric in metrics:
                    val = model_data[metric]
                    max_val = comparison_df[metric].max()
                    min_val = comparison_df[metric].min()
                    # 비용 지표는 낮을수록 좋으므로 역순 정규화
                    normalized = 1 - (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    values.append(normalized)
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=model
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="모델 성능 레이더 차트 (높을수록 좋음)"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)

# ------------------ 탭 4: 시간별 분석 ------------------
with tab4:
    st.header("📅 시간별 사용량 및 비용 분석")
    
    # 시간별 분석 설정
    time_analysis_type = st.selectbox(
        "분석 기간:",
        ["일별 (한 달)", "주별 (한 분기)", "월별 (1년)"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 시간대별 사용량 패턴
        st.subheader("⏰ 시간대별 사용 패턴")
        
        if 'usage_pattern' not in st.session_state:
            # 가상의 시간대별 사용 패턴 생성
            hours = list(range(24))
            # 일반적인 업무 시간에 높은 사용량
            base_pattern = [0.1, 0.05, 0.03, 0.02, 0.02, 0.03, 0.05, 0.1, 
                          0.15, 0.18, 0.2, 0.15, 0.1, 0.12, 0.18, 0.2, 
                          0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
            
            pattern_df = pd.DataFrame({
                '시간': hours,
                '사용 비율': base_pattern,
                '예상 비용($)': [monthly_usage * p * total_cost / monthly_usage for p in base_pattern]
            })
            st.session_state.usage_pattern = pattern_df
        
        pattern_df = st.session_state.usage_pattern
        
        fig_hourly = px.bar(
            pattern_df,
            x='시간',
            y='사용 비율',
            title="시간대별 사용 패턴",
            labels={'시간': '시간 (24시간)', '사용 비율': '전체 사용량 대비 비율'}
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        
    with col2:
        # 요일별 분석
        st.subheader("📊 요일별 사용 분석")
        
        if 'weekly_pattern' not in st.session_state:
            days = ['월', '화', '수', '목', '금', '토', '일']
            # 평일에 높은 사용량
            weekly_usage = [0.2, 0.22, 0.2, 0.18, 0.15, 0.03, 0.02]
            
            weekly_df = pd.DataFrame({
                '요일': days,
                '사용 비율': weekly_usage,
                '예상 비용($)': [total_cost * u for u in weekly_usage]
            })
            st.session_state.weekly_pattern = weekly_df
        
        weekly_df = st.session_state.weekly_pattern
        
        fig_weekly = px.pie(
            weekly_df,
            values='사용 비율',
            names='요일',
            title="요일별 사용 분포"
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    # 예측 분석
    st.subheader("🔮 사용량 예측 및 시나리오")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        growth_rate = st.slider("월간 성장률 (%)", -20, 50, 10)
    with col2:
        seasonal_factor = st.slider("계절 요인", 0.5, 2.0, 1.0, 0.1)
    with col3:
        months_to_predict = st.slider("예측 기간 (개월)", 3, 12, 6)
    
    # 예측 데이터 생성
    future_months = []
    future_costs = []
    
    current_cost = total_cost
    for month in range(1, months_to_predict + 1):
        # 성장률과 계절 요인 적용
        growth_factor = (1 + growth_rate/100) ** month
        seasonal = seasonal_factor * (1 + 0.1 * np.sin(month * np.pi / 6))  # 계절 변동
        
        predicted_cost = current_cost * growth_factor * seasonal
        future_months.append(f"{month}개월 후")
        future_costs.append(predicted_cost)
    
    prediction_df = pd.DataFrame({
        '기간': future_months,
        '예상 비용($)': future_costs
    })
    
    fig_prediction = px.line(
        prediction_df,
        x='기간',
        y='예상 비용($)',
        title=f"향후 {months_to_predict}개월 비용 예측",
        markers=True
    )
    fig_prediction.add_hline(y=total_cost, line_dash="dash", line_color="red", 
                           annotation_text="현재 비용")
    
    st.plotly_chart(fig_prediction, use_container_width=True)

# ------------------ 탭 5: 최적화 제안 ------------------
with tab5:
    st.header("💡 AI 기반 최적화 제안")
    
    if not results_df.empty:
        # 최적화 분석
        st.subheader("🎯 맞춤형 최적화 제안")
        
        # 현재 상황 분석
        total_monthly_cost = total_cost
        highest_cost_model = results_df.loc[results_df['총 비용($)'].idxmax()]
        lowest_cost_model = results_df.loc[results_df['총 비용($)'].idxmin()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 현재 상황 진단")
            
            # 비용 레벨 판정
            if total_monthly_cost < 10:
                cost_level = "경제적"
                level_color = "green"
            elif total_monthly_cost < 50:
                cost_level = "적정"
                level_color = "blue"
            elif total_monthly_cost < 100:
                cost_level = "높음"
                level_color = "orange"
            else:
                cost_level = "매우 높음"
                level_color = "red"
            
            st.markdown(f"**비용 수준**: :{level_color}[{cost_level}] (${total_monthly_cost:.2f}/월)")
            
            # 사용 패턴 분석
            usage_distribution = results_df['사용량'].std() / results_df['사용량'].mean()
            if usage_distribution > 0.5:
                pattern_analysis = "불균등한 모델 사용"
            else:
                pattern_analysis = "균등한 모델 사용"
            
            st.write(f"**사용 패턴**: {pattern_analysis}")
            st.write(f"**주요 비용 모델**: {highest_cost_model['모델']} ({highest_cost_model['비용 비율(%)']:.1f}%)")
            
        with col2:
            st.subheader("🚀 최적화 기회")
            
            # 잠재 절약 계산
            potential_savings = []
            
            # 1. 가장 비싼 모델을 가장 저렴한 모델로 대체
            if len(results_df) > 1:
                current_expensive_cost = highest_cost_model['총 비용($)']
                if_replaced_cost = highest_cost_model['사용량'] * (lowest_cost_model['총 비용($)'] / lowest_cost_model['사용량'])
                savings1 = current_expensive_cost - if_replaced_cost
                potential_savings.append(("모델 교체", savings1))
            
            # 2. 토큰 길이 최적화
            if input_multiplier > 1.0 or output_multiplier > 1.0:
                optimized_cost = calculate_costs(
                    monthly_usage, model_split, pricing_data,
                    0.8, 0.8, price_basis,
                    avg_tokens_override=avg_tokens_override,
                    included_requests=INCLUDED_REQUESTS,
                    surcharge_per_mtoken=SURCHARGE_PER_MTOKEN
                )['총 비용($)'].sum() + BASE_MONTHLY_FEE
                savings2 = total_cost - optimized_cost
                potential_savings.append(("토큰 최적화", savings2))
            
            # 3. 사용량 재분배
            # 가장 효율적인 모델의 비율을 높이는 시나리오
            optimized_split = model_split.copy()
            most_efficient_model = results_df.loc[results_df['요청당 비용'].idxmin(), '모델']
            
            for model in optimized_split:
                if model == most_efficient_model:
                    optimized_split[model] = min(100, optimized_split[model] + 20)
                else:
                    optimized_split[model] = max(0, optimized_split[model] - 5)
            
            # 비율 정규화
            total_optimized = sum(optimized_split.values())
            if total_optimized > 0:
                optimized_split = {k: (v / total_optimized) * 100 for k, v in optimized_split.items()}
                optimized_cost = calculate_costs(
                    monthly_usage, optimized_split, pricing_data,
                    input_multiplier, output_multiplier, price_basis,
                    avg_tokens_override=avg_tokens_override,
                    included_requests=INCLUDED_REQUESTS,
                    surcharge_per_mtoken=SURCHARGE_PER_MTOKEN
                )['총 비용($)'].sum() + BASE_MONTHLY_FEE
                savings3 = total_cost - optimized_cost
                potential_savings.append(("사용량 재분배", savings3))
            
            # 최적화 제안 표시
            for optimization, savings in potential_savings:
                if savings > 0:
                    savings_percent = (savings / total_cost) * 100
                    st.success(f"**{optimization}**: ${savings:.4f} 절약 ({savings_percent:.1f}%)")
        
        # 세부 최적화 시나리오
        st.subheader("🔧 최적화 시나리오 비교")
        
        scenarios = {
            "현재": {"usage": monthly_usage, "split": model_split, "input_mult": input_multiplier, "output_mult": output_multiplier},
            "경제적": {"usage": monthly_usage, "split": {most_efficient_model: 70, **{k: 30/(len(model_split)-1) for k in model_split if k != most_efficient_model}}, "input_mult": 0.8, "output_mult": 0.8},
            "균형적": {"usage": monthly_usage, "split": {k: 100/len(model_split) for k in model_split}, "input_mult": 1.0, "output_mult": 1.0},
            "고성능": {"usage": monthly_usage, "split": {highest_cost_model['모델']: 60, **{k: 40/(len(model_split)-1) for k in model_split if k != highest_cost_model['모델']}}, "input_mult": 1.2, "output_mult": 1.2}
        }
        
        scenario_results = []
        for scenario_name, params in scenarios.items():
            # 비율 정규화
            total_split = sum(params["split"].values())
            normalized_split = {k: (v / total_split) * 100 for k, v in params["split"].items()}
            
            scenario_df = calculate_costs(
                params["usage"], 
                normalized_split, 
                pricing_data, 
                params["input_mult"], 
                params["output_mult"],
                price_basis,
                avg_tokens_override=avg_tokens_override,
                included_requests=INCLUDED_REQUESTS,
                surcharge_per_mtoken=SURCHARGE_PER_MTOKEN
            )
            total_scenario_cost = scenario_df["총 비용($)"].sum() + BASE_MONTHLY_FEE
            
            scenario_results.append({
                "시나리오": scenario_name,
                "월간 비용($)": total_scenario_cost,
                "vs 현재": total_scenario_cost - total_cost,
                "절약률(%)": ((total_cost - total_scenario_cost) / total_cost * 100) if total_cost > 0 else 0
            })
        
        scenario_comparison_df = pd.DataFrame(scenario_results)
        
        # 시나리오 비교 테이블
        st.dataframe(scenario_comparison_df, use_container_width=True)
        
        # 시나리오 비교 차트
        fig_scenarios = px.bar(
            scenario_comparison_df,
            x='시나리오',
            y='월간 비용($)',
            title="최적화 시나리오별 비용 비교",
            color='절약률(%)',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_scenarios, use_container_width=True)
        
        # 액션 플랜
        st.subheader("📋 실행 가능한 액션 플랜")
        
        best_scenario = scenario_comparison_df.loc[scenario_comparison_df['월간 비용($)'].idxmin()]
        if best_scenario['시나리오'] != '현재':
            st.success(f"**추천 시나리오**: {best_scenario['시나리오']} (월 ${best_scenario['절약률(%)']:.1f}% 절약)")
            
            with st.expander("🎯 단계별 실행 가이드"):
                st.markdown("""
                ### 1단계: 즉시 실행 가능한 최적화
                - ✅ Input 텍스트 길이 줄이기 (불필요한 내용 제거)
                - ✅ Output 길이 제한 설정
                - ✅ 프롬프트 효율성 개선
                
                ### 2단계: 모델 사용 패턴 최적화 (1주일 내)
                - 🔄 간단한 작업은 경제적 모델 사용
                - 🔄 복잡한 작업만 고성능 모델 사용
                - 🔄 작업 유형별 모델 매핑 정의
                
                ### 3단계: 시스템 레벨 최적화 (1개월 내)
                - 🚀 배치 처리로 토큰 효율성 향상
                - 🚀 캐싱 시스템 도입
                - 🚀 자동화된 모델 선택 로직 구현
                """)
        
        # 월별 절약 목표
        st.subheader("🎯 절약 목표 설정")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_savings = st.slider("목표 절약률 (%)", 5, 50, 20)
        with col2:
            target_cost = total_cost * (1 - target_savings/100)
            st.metric("목표 월간 비용", f"${target_cost:.2f}")
        with col3:
            monthly_savings = total_cost - target_cost
            annual_savings = monthly_savings * 12
            st.metric("연간 절약 예상액", f"${annual_savings:.2f}")

# ------------------ 데이터 내보내기 ------------------
st.markdown("---")
st.subheader("📁 데이터 내보내기")

col_e1, col_e2 = st.columns(2)
with col_e1:
    if st.button("📊 CSV 다운로드"):
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="results.csv 다운로드",
            data=csv,
            file_name=f"cursor_ai_cost_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

with col_e2:
    if st.button("💾 설정 저장"):
        settings = {
            "user_type": user_type,
            "monthly_usage": monthly_usage,
            "input_multiplier": input_multiplier,
            "output_multiplier": output_multiplier,
            "model_split": model_split
        }
        settings_json = json.dumps(settings, indent=2, ensure_ascii=False)
        st.download_button(
            label="settings.json 다운로드",
            data=settings_json,
            file_name=f"simulator_settings_{datetime.now().strftime('%Y%m%d')}.json",
            mime='application/json'
        )

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🚀 고급 Cursor AI 요금제 시뮬레이터 | "
    f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    "</div>",
    unsafe_allow_html=True
)
