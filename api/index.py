from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Literal, List

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
    # 한국어 주석: 핵심 비용 계산 로직 (순수 Python 리스트/딕셔너리)
    results: List[Dict] = []
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

    # 한국어 주석: 비율 재계산
    total_cost_sum = sum(item["total_cost"] for item in results)
    if total_cost_sum > 0:
        for item in results:
            item["cost_ratio"] = round(item["total_cost"] / total_cost_sum * 100, 1)

    return {
        "results": results,
        "total_cost": float(total_cost_sum),
    }


# ------------------ 스키마 ------------------
class CalculateRequest(BaseModel):
    monthly_usage: int = Field(..., ge=0, description="월간 요청 수")
    model_split: Dict[str, float] = Field(..., description="모델별 사용 비율(%) 합계 100")
    input_multiplier: float = 1.0
    output_multiplier: float = 1.0
    price_basis: Literal["min", "avg", "max"] = "avg"


@app.get("/", response_class=HTMLResponse)
def home():
    # 한국어 주석: 간단한 웹 UI 제공 (스트림릿 대체용 최소 페이지)
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset='utf-8' />
      <meta name='viewport' content='width=device-width, initial-scale=1' />
      <title>AI Cost Simulator (API UI)</title>
      <style>
        body { font-family: system-ui, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }
        h1 { margin-bottom: 8px; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 12px 0; }
        label { display:block; margin: 8px 0 4px; font-weight: 600; }
        input, select, textarea { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }
        button { padding: 10px 16px; border-radius: 6px; border: none; background: #4f46e5; color: #fff; cursor: pointer; }
        pre { background: #0b1020; color: #c7e1ff; padding: 12px; border-radius: 8px; overflow: auto; }
      </style>
    </head>
    <body>
      <h1>AI Cost Simulator</h1>
      <p>이 페이지는 서버리스 API를 사용해 간단히 계산을 실행합니다. 전체 대시보드는 Streamlit에서 실행하세요.</p>
      <div class="card">
        <label>Monthly Usage</label>
        <input id="monthly_usage" type="number" value="1000" min="0" />
        <label>Price Basis</label>
        <select id="price_basis">
          <option value="avg" selected>avg</option>
          <option value="min">min</option>
          <option value="max">max</option>
        </select>
        <label>Model Split (JSON)</label>
        <textarea id="model_split" rows="8">{
  "Claude Opus 4.1": 10,
  "Claude Sonnet 4": 40,
  "OpenAI GPT-5": 20,
  "OpenAI GPT-4.1/o3": 15,
  "OpenAI 2.5 Pro": 10,
  "Gemini 2.5 Flash": 5
}</textarea>
        <div style="display:flex; gap:8px; margin-top:12px;">
          <button id="run">Run Calculate</button>
          <button id="pricing">Get Pricing</button>
          <a href="/health" target="_blank" style="margin-left:auto;">Health</a>
        </div>
      </div>
      <div class="card">
        <pre id="out">결과가 여기에 표시됩니다...</pre>
      </div>
      <script>
        const el = (id) => document.getElementById(id);
        const show = (data) => el('out').textContent = JSON.stringify(data, null, 2);
        document.getElementById('pricing').onclick = async () => {
          const res = await fetch('/pricing');
          show(await res.json());
        };
        document.getElementById('run').onclick = async () => {
          let split = {};
          try { split = JSON.parse(el('model_split').value); } catch (e) { show({ error: 'model_split JSON 파싱 오류' }); return; }
          const body = {
            monthly_usage: Number(el('monthly_usage').value),
            model_split: split,
            input_multiplier: 1.0,
            output_multiplier: 1.0,
            price_basis: el('price_basis').value
          };
          const res = await fetch('/calculate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
          show(await res.json());
        };
      </script>
    </body>
    </html>
    """


@app.get("/health", response_class=JSONResponse)
def health():
    # 한국어 주석: 상태 확인 엔드포인트 (JSON)
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

    calc = calculate_costs(
        monthly_usage=req.monthly_usage,
        model_split=req.model_split,
        pricing_data=PRICING_DATA,
        input_multiplier=req.input_multiplier,
        output_multiplier=req.output_multiplier,
        price_basis=req.price_basis,
    )

    return calc

