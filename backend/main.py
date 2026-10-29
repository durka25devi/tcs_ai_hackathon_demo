from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys
from typing import List, Optional

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.schemas import (
    Product, OverviewSummary, ComboDiscoverRequest, ComboRule,
    PriceScenarioRequest, PriceScenarioResponse,
    ComboEvaluateRequest, ComboEvaluateResponse,
    RecommendationRequest, RecommendationResponse,
    ChatRequest, ChatResponse
)
from backend.data_loader import data_loader
from backend.combo_discovery import discover_combos
from backend.demand_model import demand_estimator
from backend.pricing_engine import evaluate_price_scenarios
from backend.combo_evaluator import evaluate_combo_offer
from backend.recommendation_engine import generate_recommendations
from backend.llm_service import llm_service

app = FastAPI(
    title="AI-Powered Canteen Pricing and Combo Offer Support System",
    description="Decision-support system for canteen pricing, basket discovery, and combo optimization with Groq Qwen reasoning",
    version="1.0.0"
)

# Enable CORS for frontend Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI-Powered Canteen Pricing Support System API is active. Access /docs for Swagger UI."}


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "data_loaded": True,
        "demand_model_trained": demand_estimator.model is not None,
        "products_count": len(data_loader.products),
        "llm_service_configured": llm_service.is_configured(),
        "llm_model": llm_service.model
    }

@app.get("/api/overview", response_model=OverviewSummary)
def get_overview():
    try:
        metrics = data_loader.get_overview_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products", response_model=List[Product])
def get_products():
    try:
        prods = data_loader.products.to_dict(orient="records")
        return prods
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/combos", response_model=List[ComboRule])
def get_default_combos(
    algorithm: str = Query("fpgrowth", description="'fpgrowth' or 'apriori'"),
    min_support: float = Query(0.03, ge=0.001, le=1.0),
    min_confidence: float = Query(0.20, ge=0.01, le=1.0),
    min_lift: float = Query(1.0, ge=0.0)
):
    try:
        rules = discover_combos(
            algorithm=algorithm,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift
        )
        return rules
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/combos/discover", response_model=List[ComboRule])
def post_discover_combos(req: ComboDiscoverRequest):
    try:
        rules = discover_combos(
            algorithm=req.algorithm,
            min_support=req.min_support,
            min_confidence=req.min_confidence,
            min_lift=req.min_lift
        )
        return rules
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/demand-model/metrics")
def get_demand_model_metrics():
    return demand_estimator.train_summary

@app.post("/api/pricing/evaluate", response_model=PriceScenarioResponse)
def post_evaluate_pricing(req: PriceScenarioRequest):
    try:
        res = evaluate_price_scenarios(
            product_id=req.product_id,
            candidate_prices=req.candidate_prices,
            day_of_week=req.day_of_week or "Wednesday",
            promotion_active=req.promotion_active or 0
        )
        return res
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/combos/evaluate", response_model=ComboEvaluateResponse)
def post_evaluate_combo(req: ComboEvaluateRequest):
    try:
        res = evaluate_combo_offer(
            combo_items=req.combo_items,
            proposed_price=req.proposed_price,
            assumed_daily_volume=req.assumed_daily_volume,
            min_contribution_margin=req.min_contribution_margin,
            student_budget_limit=req.student_budget_limit or 80.0
        )
        return res
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/recommendations", response_model=RecommendationResponse)
def post_recommendations(req: RecommendationRequest):
    try:
        res = generate_recommendations(
            max_student_budget=req.max_student_budget,
            min_contribution_margin=req.min_contribution_margin,
            stock_buffer_safety_pct=req.stock_buffer_safety_pct
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Groq Qwen Chat Assistant Endpoints
# ==========================================

@app.get("/api/llm/status")
def get_llm_status():
    """Returns the current LLM configuration and readiness status."""
    return {
        "configured": llm_service.is_configured(),
        "model": llm_service.model,
        "temperature": llm_service.temperature,
        "max_tokens": llm_service.max_tokens,
        "timeout_seconds": llm_service.timeout,
        "has_api_key": bool(llm_service.api_key and llm_service.api_key != "your_groq_api_key_here")
    }

@app.post("/api/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest):
    """
    Conversational pricing assistant endpoint.
    Accepts user question + bounded history, extracts intent, computes structured
    evidence via existing Python ML & financial engines, and asks Groq Qwen
    (or deterministic fallback) to generate grounded natural-language explanations.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    try:
        res = llm_service.explain(
            message=req.message,
            history=req.history,
            context_hint=req.context_hint
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat reasoning error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
