from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Product(BaseModel):
    product_id: str
    product_name: str
    category: str
    current_price: float
    ingredient_cost: float
    packaging_cost: float
    portion_size_g: float
    available_stock: int
    waste_quantity: int
    unit_variable_cost: float
    current_unit_contribution: float
    contribution_margin_pct: float

class OverviewSummary(BaseModel):
    total_transactions: int
    total_units_sold: int
    total_revenue: float
    total_estimated_contribution: float
    total_variable_cost: float
    overall_contribution_margin_pct: float
    date_range_start: str
    date_range_end: str
    best_selling_products: List[Dict[str, Any]]
    slow_moving_products: List[Dict[str, Any]]

class ComboDiscoverRequest(BaseModel):
    algorithm: str = Field(default="fpgrowth", description="'fpgrowth' or 'apriori'")
    min_support: float = Field(default=0.03, ge=0.001, le=1.0)
    min_confidence: float = Field(default=0.20, ge=0.01, le=1.0)
    min_lift: float = Field(default=1.0, ge=0.0)

class ComboRule(BaseModel):
    combo_id: str
    antecedents: List[str]
    consequents: List[str]
    items: List[str]
    item_names: List[str]
    support: float
    confidence: float
    lift: float
    sum_individual_prices: float
    total_variable_cost: float
    default_combo_contribution: float

class PriceScenarioRequest(BaseModel):
    product_id: str
    candidate_prices: Optional[List[float]] = None
    day_of_week: Optional[str] = "Wednesday"
    promotion_active: Optional[int] = 0

class PriceScenarioItem(BaseModel):
    candidate_price: float
    is_baseline: bool
    estimated_daily_quantity: float
    estimated_daily_revenue: float
    unit_variable_cost: float
    estimated_daily_variable_cost: float
    estimated_daily_contribution: float
    contribution_change_vs_baseline: float
    revenue_change_vs_baseline: float

class PriceScenarioResponse(BaseModel):
    product: Product
    day_of_week: str
    promotion_active: int
    scenarios: List[PriceScenarioItem]
    model_notes: str

class ComboEvaluateRequest(BaseModel):
    combo_items: List[str] # List of product_ids
    proposed_price: float
    assumed_daily_volume: int = Field(default=25, ge=1)
    min_contribution_margin: float = Field(default=0.25, ge=0.0, le=1.0)
    student_budget_limit: Optional[float] = 80.0

class ComboEvaluateResponse(BaseModel):
    included_products: List[Dict[str, Any]]
    sum_individual_prices: float
    proposed_combo_price: float
    customer_savings: float
    customer_savings_pct: float
    combo_variable_cost: float
    contribution_per_combo: float
    contribution_margin_pct: float
    assumed_daily_volume: int
    hypothetical_daily_revenue: float
    hypothetical_daily_contribution: float
    max_bottleneck_stock: int
    stock_bottleneck_product: str
    passes_budget: bool
    passes_margin: bool
    passes_stock: bool
    constraint_status: Dict[str, Any]
    warnings: List[str]
    caveat_notes: str

class RecommendationRequest(BaseModel):
    max_student_budget: float = Field(default=75.0, ge=10.0)
    min_contribution_margin: float = Field(default=0.25, ge=0.0, le=1.0)
    stock_buffer_safety_pct: float = Field(default=0.85, ge=0.1, le=1.0)

class PricingRecommendation(BaseModel):
    product_id: str
    product_name: str
    current_price: float
    recommended_price: float
    price_change: float
    baseline_contribution: float
    recommended_contribution: float
    projected_contribution_gain: float
    projected_daily_demand: float
    status: str
    rationale: str

class ComboRecommendation(BaseModel):
    combo_name: str
    product_ids: List[str]
    individual_total_price: float
    recommended_combo_price: float
    customer_savings: float
    customer_savings_pct: float
    contribution_per_combo: float
    contribution_margin_pct: float
    daily_bottleneck_capacity: int
    support: float
    confidence: float
    lift: float
    rationale: str

class RecommendationResponse(BaseModel):
    objective: str
    student_budget_limit: float
    min_margin_threshold: float
    pricing_recommendations: List[PricingRecommendation]
    combo_recommendations: List[ComboRecommendation]
    general_caveats: List[str]

# ==========================================
# Chat & Conversational Assistant Schemas
# ==========================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., max_length=4000)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User question or prompt")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Bounded prior conversation turns")
    context_hint: Optional[Dict[str, Any]] = Field(default=None, description="Optional active screen or product context")

class EvidenceData(BaseModel):
    intent: str
    headline: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    table: Optional[List[Dict[str, Any]]] = None
    notes: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    evidence: Optional[EvidenceData] = None
    intent: str
    model_used: str
    is_llm_generated: bool
    status: str # 'success', 'fallback_deterministic', 'missing_credentials', 'error'
