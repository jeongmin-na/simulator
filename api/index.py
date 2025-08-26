from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Literal
import pandas as pd

# 한국어 주석: Vercel 서버리스에서 동작할 최소 FastAPI 래퍼

app = FastAPI()


# ------------------ 데이터 설정 ------------------
def load_pricing_data() -> Dict[str, Dict[str, float]]:
    # 한국어 주석: Streamlit 의존성을 피하기 위해 이 모듈에 가격 데이터를 복제합니다.
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


PRICING_DATA = load_pricing_data()


def get_price_by_basis(config: Dict[str, float], basis: Literal["min", "avg", "max"]) -> tuple[float, float]:
    # 한국어 주석: 가격 기준에 따라 1k 토큰당 입력/출력 단가 반환
    if basis == "min":
        return config["price_input_min"], config["price_output_min"]
    if basis == "max":
        return config["price_input_max"], config["price_output_max"]
    input_price = (config["price_input_min"] + config["price_input_max"]) / 2
    output_price = (config["price_output_min"] + config["price_output_max"]) / 2
    return input_price, output_price


def calculate_costs(
    monthly_usage: int,
    model_split: Dict[str, float],
    pricing_data: Dict[str, Dict[str, float]],
    input_multiplier: float = 1.0,
    output_multiplier: float = 1.0,
    price_basis: Literal["min", "avg", "max"] = "avg",
):
    # 한국어 주석: 핵심 비용 계산 로직 (Streamlit 미사용)
    results = []
    for model, config in pricing_data.items():
        if model in model_split:
            usage_share = monthly_usage * (model_split[model] / 100.0)
            input_tokens = usage_share * config["avg_input"] * input_multiplier
            output_tokens = usage_share * config["avg_output"] * output_multiplier
            price_in, price_out = get_price_by_basis(config, price_basis)

            input_cost = (input_tokens / 1000) * price_in
            output_cost = (output_tokens / 1000) * price_out
            total_cost = input_cost + output_cost

            results.append({
                "model": model,
                "usage": int(usage_share),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(total_cost, 4),
                "cost_ratio": 0,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        total_cost_sum = df["total_cost"].sum()
        df["cost_ratio"] = (df["total_cost"] / total_cost_sum * 100).round(1)

    return df


# ------------------ 스키마 ------------------
class CalculateRequest(BaseModel):
    monthly_usage: int = Field(..., ge=0, description="월간 요청 수")
    model_split: Dict[str, float] = Field(..., description="모델별 사용 비율(%) 합계 100")
    input_multiplier: float = 1.0
    output_multiplier: float = 1.0
    price_basis: Literal["min", "avg", "max"] = "avg"


@app.get("/")
def health():
    # 한국어 주석: 상태 확인 엔드포인트
    return {"status": "ok", "models": list(PRICING_DATA.keys())}


@app.get("/pricing")
def get_pricing():
    # 한국어 주석: 현재 가격표 반환
    return PRICING_DATA


@app.post("/calculate")
def post_calculate(req: CalculateRequest):
    # 한국어 주석: 비용 계산 수행
    # 비율 합계 검증 (유연 허용: 100±0.01)
    total_percent = sum(req.model_split.values())
    if abs(total_percent - 100.0) > 0.01:
        return {"error": f"model_split 합계는 100이어야 합니다. 현재: {total_percent}"}

    df = calculate_costs(
        monthly_usage=req.monthly_usage,
        model_split=req.model_split,
        pricing_data=PRICING_DATA,
        input_multiplier=req.input_multiplier,
        output_multiplier=req.output_multiplier,
        price_basis=req.price_basis,
    )

    if df.empty:
        return {"results": [], "total_cost": 0.0}

    return {
        "results": df.to_dict(orient="records"),
        "total_cost": float(df["total_cost"].sum()),
    }

