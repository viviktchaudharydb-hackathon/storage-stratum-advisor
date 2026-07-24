"""
Enterprise Infrastructure Advisor - GCP-Native, Multi-Domain
One app for Storage, Server/Compute, Database, and Middleware decisions.

LangGraph conditional workflow (domain-agnostic) × Vertex AI Gemini ×
Google Search grounding × Procurement RAG × deterministic TCO.
Runs on Cloud Run with service-account auth - zero API keys.
"""

import os
import json
import re
import operator
import logging
from typing import TypedDict, List, Optional, Dict, Any, Annotated, Literal
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types as genai_types

from domains import DOMAINS, DEPLOYMENTS, get_matching_vendors
from advisor_extensions import ProcurementRAG, TCOEngine
from compliance import run_compliance_checks
from arb_report import build_arb_document
from blueprints import BLUEPRINTS, derive_components, analyze_synergy, default_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infra-advisor")

st.set_page_config(page_title="Infrastructure Advisor - Vertex AI", page_icon="🏛️", layout="wide")

# ==========================================================
# Configuration
# ==========================================================

class Config:
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-south1")
    MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    MAX_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "8192"))

    @classmethod
    def validate(cls) -> bool:
        if not cls.PROJECT_ID:
            st.error("❌ GOOGLE_CLOUD_PROJECT is not set.")
            return False
        return True


@st.cache_resource
def get_genai_client() -> genai.Client:
    return genai.Client(vertexai=True, project=Config.PROJECT_ID, location=Config.LOCATION)

# ==========================================================
# Requirements (domain-generic)
# ==========================================================

@dataclass
class Requirements:
    domain: str
    workload: str
    capacity: int          # in domain units (TB, servers, TB data, instances)
    priority: str
    deployment: str
    availability_target: str

    @property
    def unit(self) -> str:
        return DOMAINS[self.domain]["unit"]

    def metrics(self) -> Dict[str, str]:
        return DOMAINS[self.domain]["workloads"].get(self.workload, {}).get("metrics", {})

    def to_search_prompt(self) -> str:
        year = datetime.now().year
        keywords = DOMAINS[self.domain]["workloads"].get(self.workload, {}).get("keywords", "")
        return (
            f"Summarize the current ({year}) enterprise {self.domain.lower()} market for "
            f"{self.workload} workloads ({self.deployment} deployment; {keywords}). "
            f"Cover leading vendors, recent announcements, and pricing trends. Under 250 words."
        )

# ==========================================================
# Market Intelligence - Gemini + Google Search grounding
# ==========================================================

class MarketIntel:
    def __init__(self, client: genai.Client):
        self.client = client

    def search(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.models.generate_content(
                model=Config.MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                    temperature=0.2,
                    max_output_tokens=Config.MAX_TOKENS,
                ),
            )
            answer = response.text or "No data available"
            sources: List[Dict[str, str]] = []
            try:
                gm = getattr(response.candidates[0], "grounding_metadata", None)
                if gm and getattr(gm, "grounding_chunks", None):
                    for chunk in gm.grounding_chunks[:5]:
                        web = getattr(chunk, "web", None)
                        if web:
                            sources.append({"title": getattr(web, "title", "") or "",
                                            "url": getattr(web, "uri", "") or ""})
            except (IndexError, AttributeError):
                pass
            return {"answer": answer, "sources": sources, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Market intel failed: {e}")
            return {"answer": f"Search unavailable: {e}", "sources": [], "error": str(e)}

# ==========================================================
# AI Analysis - Gemini JSON mode (thinking disabled)
# ==========================================================

class AIAnalyzer:
    def __init__(self, client: genai.Client):
        self.client = client

    def _generate_json(self, system: str, prompt: str, temperature: float) -> Optional[dict]:
        """JSON call with one retry. thinking_budget=0 prevents Gemini 2.5's
        default thinking from consuming max_output_tokens and returning
        empty text (the 'Invalid or empty AI response' failure mode)."""
        raw = ""
        for attempt in (1, 2):
            try:
                response = self.client.models.generate_content(
                    model=Config.MODEL,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        temperature=temperature,
                        max_output_tokens=Config.MAX_TOKENS,
                        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                raw = response.text or ""
                if not raw:
                    try:
                        fr = response.candidates[0].finish_reason
                        logger.error(f"Empty Gemini response (attempt {attempt}), finish_reason={fr}")
                    except (IndexError, AttributeError):
                        logger.error(f"Empty Gemini response (attempt {attempt}), no candidates")
                    continue
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse failed (attempt {attempt}): {e} | raw: {raw[:300]}")
                continue
            except Exception as e:
                logger.error(f"Gemini call failed (attempt {attempt}): {e}")
                continue
        return None

    def analyze_architecture(self, req: Requirements, market_data: Dict) -> Dict:
        metrics = ", ".join(f"{k}: {v}" for k, v in req.metrics().items()) or "N/A"
        market_insight = (market_data.get("answer") or "No market data")[:800]

        prompt = f"""As an enterprise {req.domain.lower()} architect, analyze these requirements:

Domain: {req.domain}
Workload: {req.workload}
Scale: {req.capacity} {req.unit}
Deployment: {req.deployment}
Priority: {req.priority}
Availability SLA: {req.availability_target}
Performance Profile: {metrics}

Market Context: {market_insight}

Return JSON with exactly these keys:
- architecture_type (string - the recommended architecture pattern for this domain)
- key_recommendations (array of 3-4 concise strings)
- performance_requirements (object)
- scalability_notes (string)
- redundancy_approach (string)"""

        result = self._generate_json(
            system=f"You are an enterprise {req.domain.lower()} architect. Return only valid JSON.",
            prompt=prompt, temperature=0.2)
        if not result or "architecture_type" not in result:
            return {"architecture_type": "Unknown",
                    "key_recommendations": ["Analysis failed - please retry"],
                    "error": "Invalid or empty AI response"}
        result.setdefault("key_recommendations", [])
        return result

    def evaluate_vendors(self, req: Requirements, vendors: List[str], market_data: Dict,
                         procurement_context: Optional[List[Dict]] = None) -> List[Dict]:
        if not vendors:
            return []
        vendor_db = DOMAINS[req.domain]["vendors"]
        vendor_info = []
        for v in vendors[:8]:
            data = vendor_db.get(v)
            if not data:
                continue
            vendor_info.append(
                f"- {v}: Strengths: {', '.join(data['strengths'])} | Cost: {data.get('cost_profile', 'N/A')}"
                + (f" | Best for: {data['sweet_spot']}" if data.get("sweet_spot") else "")
            )
        if not vendor_info:
            return []

        proc_section = ""
        if procurement_context:
            chunks = [f"[{d['id']}] {d['text'][:400]}" for d in procurement_context[:3]]
            proc_section = ("\nInternal Procurement Context (RAG - retrieved from company documents):\n"
                            + "\n\n".join(chunks) + "\n")

        prompt = f"""Evaluate these {req.domain.lower()} vendors for the following requirements:

Domain: {req.domain}
Workload: {req.workload}
Scale: {req.capacity} {req.unit}
Deployment: {req.deployment}
Priority: {req.priority}
Availability SLA: {req.availability_target}

Available Vendors:
{chr(10).join(vendor_info)}
{proc_section}
Return JSON with an array 'vendors' containing the top 3-4 vendors ranked by fit:
- name (string - must exactly match a name from Available Vendors)
- fit_score (number 1-10; consider workload match, scale fit, priority alignment, AND existing procurement agreements/policy from internal context)
- strengths (array of 2-3 strings specific to this use case; mention existing agreements where relevant)
- considerations (array of 2-3 strings; flag policy hurdles like new-vendor security review where relevant)

Prioritize vendors whose cost profile matches the stated priority, and weigh internal
procurement context into the ranking. Do NOT estimate costs - TCO is computed separately."""

        result = self._generate_json(
            system=f"You are an enterprise {req.domain.lower()} analyst. Return only JSON with a 'vendors' array.",
            prompt=prompt, temperature=0.3)
        if not result or not isinstance(result.get("vendors"), list):
            return [{"name": "Error", "fit_score": 0, "strengths": [],
                     "considerations": ["Vendor evaluation failed - invalid AI response"],
                     "error": "Invalid or empty AI response"}]
        return [v for v in result["vendors"] if isinstance(v, dict) and v.get("name")]

# ==========================================================
# LangGraph Workflow (domain-agnostic)
# ==========================================================

class AgentState(TypedDict):
    requirements: Requirements
    messages: Annotated[List[str], operator.add]
    market_data: Dict[str, Any]
    architecture_analysis: Dict[str, Any]
    vendor_candidates: List[str]
    vendor_recommendations: List[Dict[str, Any]]
    procurement_context: List[Dict[str, Any]]
    tco_estimates: List[Dict[str, Any]]
    compliance_results: List[Dict[str, Any]]
    final_report: Dict[str, Any]
    current_step: str


@st.cache_resource
def create_advisor_graph():
    client = get_genai_client()
    intel = MarketIntel(client)
    ai = AIAnalyzer(client)
    rag = ProcurementRAG(client)

    def gather_market_intelligence(state: AgentState) -> dict:
        req = state["requirements"]
        market_data = intel.search(req.to_search_prompt())
        return {"current_step": "market_intelligence",
                "messages": [f"✅ Market insights gathered for {req.domain} / {req.workload}"],
                "market_data": market_data}

    def analyze_architecture(state: AgentState) -> dict:
        req = state["requirements"]
        analysis = ai.analyze_architecture(req, state.get("market_data", {}))
        return {"current_step": "architecture_analysis",
                "architecture_analysis": analysis,
                "messages": ["✅ Architecture analysis complete (Gemini)"]}

    def find_vendor_candidates(state: AgentState) -> dict:
        req = state["requirements"]
        matching = get_matching_vendors(req.domain, req.workload, req.deployment, req.capacity)
        return {"current_step": "vendor_matching",
                "vendor_candidates": matching,
                "messages": [f"✅ Found {len(matching)} matching {req.domain.lower()} vendors"]}

    def route_after_vendor_match(state: AgentState) -> Literal["vendors_found", "no_vendors"]:
        return "vendors_found" if state.get("vendor_candidates") else "no_vendors"

    def retrieve_procurement(state: AgentState) -> dict:
        req = state["requirements"]
        candidates = state.get("vendor_candidates", [])
        query = (f"{req.domain} {req.workload} procurement agreements pricing discounts "
                 f"policy for vendors: {', '.join(candidates)}")
        context = rag.retrieve(query, top_k=3)
        msg = (f"📄 Retrieved {len(context)} procurement documents (RAG)"
               if context else "📄 No relevant procurement documents found")
        return {"current_step": "procurement_rag", "procurement_context": context, "messages": [msg]}

    def evaluate_vendors(state: AgentState) -> dict:
        req = state["requirements"]
        candidates = state.get("vendor_candidates", [])
        if not candidates:
            return {"current_step": "vendor_evaluation", "vendor_recommendations": [],
                    "tco_estimates": [], "messages": []}
        procurement = state.get("procurement_context", [])
        recs = ai.evaluate_vendors(req, candidates, state.get("market_data", {}), procurement)

        discounts = rag.negotiated_discounts(req.domain)
        vendor_db = DOMAINS[req.domain]["vendors"]
        tco_cfg = DOMAINS[req.domain]["tco"]
        tco_estimates = []
        for rec in recs:
            meta = vendor_db.get(rec.get("name", ""))
            if meta:
                tco_estimates.append(
                    TCOEngine.estimate(rec["name"], meta, req.capacity, discounts, tco_cfg))
        return {"current_step": "vendor_evaluation",
                "vendor_recommendations": recs,
                "tco_estimates": tco_estimates,
                "messages": [f"✅ Evaluated {len(recs)} vendors (Gemini) · 💰 TCO computed for {len(tco_estimates)}"]}

    def explain_no_vendors(state: AgentState) -> dict:
        req = state["requirements"]
        vendor_db = DOMAINS[req.domain]["vendors"]
        workload_vendors = [v for v, d in vendor_db.items() if req.workload in d.get("workloads", [])]
        suggestions = []
        if workload_vendors:
            suggestions.append(
                f"Found {len(workload_vendors)} {req.domain.lower()} vendors supporting {req.workload}, "
                "but they don't match your deployment model or scale requirements")
            suggestions.append(
                f"Consider {'Hybrid' if req.deployment != 'Hybrid' else 'On-Premises or Cloud'} deployment")
        else:
            suggestions.append(f"No vendors in database specialize in {req.workload}")
        cfg = DOMAINS[req.domain]["capacity_slider"]
        if req.capacity <= cfg["min"] * 2:
            suggestions.append(
                f"Scale ({req.capacity} {req.unit}) is small - consider cloud/managed or entry-level options")
        return {"current_step": "no_vendors_found",
                "vendor_recommendations": [{"name": "No Suitable Vendors", "fit_score": 0,
                                            "strengths": [], "considerations": suggestions}],
                "tco_estimates": [],
                "messages": ["⚠️ No suitable vendors found"]}


    def compliance_check(state: AgentState) -> dict:
        """Node: deterministic bank-policy guardrails - no LLM involvement."""
        req = state["requirements"]
        recs = state.get("vendor_recommendations", [])
        if not recs or recs[0].get("fit_score", 0) == 0:
            return {"current_step": "compliance_check", "compliance_results": [], "messages": []}
        results = run_compliance_checks(
            domain=req.domain, workload=req.workload, deployment=req.deployment,
            availability_target=req.availability_target,
            vendor_recommendations=recs,
            vendor_db=DOMAINS[req.domain]["vendors"],
            preferred_vendors=rag.preferred_vendors(),
            tco_estimates=state.get("tco_estimates", []),
        )
        warns = sum(1 for r in results if r["overall"] == "warn")
        fails = sum(1 for r in results if r["overall"] == "fail")
        return {"current_step": "compliance_check",
                "compliance_results": results,
                "messages": [f"🛡️ Compliance checks: {len(results)} vendors - {fails} fail, {warns} need review"]}

    def generate_report(state: AgentState) -> dict:
        req = state["requirements"]
        vendors = state.get("vendor_recommendations", [])
        has_real = bool(vendors) and vendors[0].get("fit_score", 0) > 0
        report = {
            "requirements_summary": {
                "domain": req.domain, "workload": req.workload,
                "scale": f"{req.capacity} {req.unit}", "deployment": req.deployment,
                "priority": req.priority, "availability_sla": req.availability_target,
            },
            "architecture": state.get("architecture_analysis", {}),
            "market_sources": state.get("market_data", {}).get("sources", []),
            "procurement_documents_used": [d["id"] for d in state.get("procurement_context", [])],
            "tco_estimates_3yr": state.get("tco_estimates", []),
            "compliance_results": state.get("compliance_results", []),
            "top_vendors": vendors[:3],
            "next_steps": [
                "Engage with top 2-3 vendors for detailed sizing",
                "Request formal quotations with 3-year TCO breakdown",
                "Plan proof-of-concept with representative workload",
                "Validate performance against requirements",
            ] if has_real else [
                "Reassess deployment model requirements",
                "Consider multi-vendor or hybrid approaches",
                "Consult with vendors for custom solutions",
            ],
        }
        return {"current_step": "report_generation", "final_report": report,
                "messages": ["✅ Analysis complete!"]}

    wf = StateGraph(AgentState)
    wf.add_node("gather_intelligence", gather_market_intelligence)
    wf.add_node("analyze_architecture", analyze_architecture)
    wf.add_node("find_vendors", find_vendor_candidates)
    wf.add_node("retrieve_procurement", retrieve_procurement)
    wf.add_node("evaluate_vendors", evaluate_vendors)
    wf.add_node("explain_no_vendors", explain_no_vendors)
    wf.add_node("compliance_check", compliance_check)
    wf.add_node("generate_report", generate_report)

    wf.set_entry_point("gather_intelligence")
    wf.add_edge("gather_intelligence", "analyze_architecture")
    wf.add_edge("analyze_architecture", "find_vendors")
    wf.add_conditional_edges("find_vendors", route_after_vendor_match,
                             {"vendors_found": "retrieve_procurement", "no_vendors": "explain_no_vendors"})
    wf.add_edge("retrieve_procurement", "evaluate_vendors")
    wf.add_edge("evaluate_vendors", "compliance_check")
    wf.add_edge("compliance_check", "generate_report")
    wf.add_edge("explain_no_vendors", "generate_report")
    wf.add_edge("generate_report", END)
    return wf.compile()

# ==========================================================
# UI
# ==========================================================

def render_sidebar():
    """Returns (mode, payload): ('single', Requirements) or ('solution', dict) or (None, None)."""
    with st.sidebar:
        st.header("📋 Requirements")

        mode = st.radio("Mode", ["Single Domain", "🧩 Solution Blueprint"], horizontal=True,
                        help="Blueprint mode correlates multiple domains with linked sizing")

        if mode == "🧩 Solution Blueprint":
            bp_name = st.selectbox("Use Case", list(BLUEPRINTS.keys()))
            bp = BLUEPRINTS[bp_name]
            st.caption(bp["description"])
            drv = bp["driver"]
            driver_value = st.slider(drv["label"], drv["min"], drv["max"], drv["default"], drv["step"])
            deployment = st.selectbox("Deployment", DEPLOYMENTS)
            priority = st.selectbox("Design Priority", ["Performance", "Cost", "Scalability", "Simplicity"])
            availability = st.selectbox("Availability SLA", ["99.9%", "99.99%", "99.999%"], index=1)

            # Sizing assumptions: transparent AND editable - nothing hardcoded
            params = dict(default_params(bp_name))
            with st.expander("⚙️ Sizing assumptions (editable)", expanded=False):
                st.caption("Deterministic formulas · your inputs. Adjust to your standards; "
                           "the stack re-derives instantly.")
                for key, spec in bp.get("params", {}).items():
                    params[key] = st.number_input(
                        spec["label"], min_value=spec["min"], max_value=spec["max"],
                        value=spec["default"], step=spec["step"],
                        help=spec.get("help", ""), key=f"bp_{bp_name}_{key}")

            comps = derive_components(bp_name, driver_value, params)
            with st.expander("📐 Derived sizing (correlated)", expanded=True):
                for c in comps:
                    unit = DOMAINS[c["domain"]]["unit"]
                    st.write(f"**{c['domain']}** · {c['workload']} → **{c['capacity']} {unit}**")
                    st.caption(c["rationale"])

            st.divider()
            if st.button("🚀 Analyze Full Stack", type="primary", use_container_width=True):
                return "solution", {"blueprint": bp_name, "driver_value": driver_value,
                                    "sizing_params": params,
                                    "components": comps, "deployment": deployment,
                                    "priority": priority, "availability": availability}
            return None, None

        domain = st.selectbox("🏛️ Domain", list(DOMAINS.keys()),
                              help="Which infrastructure decision are you making?")
        cfg = DOMAINS[domain]

        workload = st.selectbox("Workload Type", list(cfg["workloads"].keys()))
        s = cfg["capacity_slider"]
        capacity = st.slider(s["label"], s["min"], s["max"], s["default"], s["step"])
        deployment = st.selectbox("Deployment", DEPLOYMENTS)
        priority = st.selectbox("Design Priority", ["Performance", "Cost", "Scalability", "Simplicity"])
        availability = st.selectbox("Availability SLA", ["99.9%", "99.99%", "99.999%"], index=1)

        st.divider()
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            return "single", Requirements(domain, workload, capacity, priority, deployment, availability)

        with st.expander("ℹ️ About"):
            st.info(f"""
**One advisor, four domains:** {', '.join(DOMAINS.keys())}

**GCP-Native Stack:**
- 🧠 Vertex AI Gemini · 🔍 Google Search grounding
- 📄 Procurement RAG (Vertex embeddings)
- 💰 Deterministic 3-yr TCO engine
- 🔐 Service-account auth - zero API keys
            """)
    return None, None


PROGRESS_STEPS = ["market_intelligence", "architecture_analysis", "vendor_matching",
                  "procurement_rag", "vendor_evaluation", "compliance_check", "report_generation"]
STEP_LABELS = {
    "market_intelligence": "🔍 Market Intel", "architecture_analysis": "🏗️ Architecture",
    "vendor_matching": "🔎 Vendor Match", "procurement_rag": "📄 Procurement RAG",
    "vendor_evaluation": "🏢 Evaluation", "compliance_check": "🛡️ Compliance", "no_vendors_found": "⚠️ No Match",
    "report_generation": "📋 Report",
}

def render_workflow_progress(current: str):
    slot = "vendor_evaluation" if current == "no_vendors_found" else current
    if slot in PROGRESS_STEPS:
        idx = PROGRESS_STEPS.index(slot)
        st.progress((idx + 1) / len(PROGRESS_STEPS),
                    text=f"Current: {STEP_LABELS.get(current, 'Processing...')}")



# ==========================================================
# Provenance badges - show WHERE each result comes from
# ==========================================================

PROV = {
    "search":  ("🔍 LIVE WEB",        "#1a56b8", "Gemini + Google Search grounding - real-time, cited"),
    "llm":     ("🤖 LLM (GEMINI)",    "#b3261e", "Vertex AI Gemini - AI judgment, human review required"),
    "rag":     ("📄 RAG",             "#137333", "Vertex AI embeddings over internal procurement docs"),
    "rules":   ("⚙️ RULE ENGINE",     "#137333", "Deterministic code - auditable, no LLM"),
    "math":    ("🧮 DETERMINISTIC",   "#137333", "Computed cost model - the LLM never invents a number"),
    "hybrid":  ("🤖+📄 LLM × RAG",    "#7627bb", "Gemini ranking grounded by retrieved procurement context"),
}

def provenance(kind: str):
    """Render a small source-of-truth chip under a section header."""
    label, color, tip = PROV[kind]
    st.markdown(
        f"<span style='background:{color}1A;color:{color};border:1px solid {color};"
        f"border-radius:12px;padding:2px 10px;font-size:0.72rem;font-weight:600;'"
        f" title='{tip}'>{label}</span> <span style='color:#5f6368;font-size:0.72rem;'>{tip}</span>",
        unsafe_allow_html=True)


def render_provenance_legend():
    with st.expander("🧭 How to read this analysis - what comes from where", expanded=False):
        st.markdown("""
| Badge | Source | Trust model |
|---|---|---|
| 🔍 **LIVE WEB** | Gemini + Google Search grounding | Real-time market data with cited URLs |
| 🤖 **LLM (Gemini)** | Vertex AI Gemini, JSON mode | AI judgment - ranks & explains, never prices |
| 📄 **RAG** | Vertex AI embeddings over internal docs | Retrieval with relevance scores - inspectable |
| ⚙️ **RULE ENGINE** | Deterministic Python | Policy & matching as code - fully auditable |
| 🧮 **DETERMINISTIC** | TCO cost model | Transparent formula, negotiated discounts applied |

**Design principle:** AI where judgment helps · code where auditability matters · human sign-off where accountability lives.
""")


def render_results(state: AgentState):
    req = state["requirements"]
    report = state.get("final_report", {})
    vendor_db = DOMAINS[req.domain]["vendors"]

    st.header(f"📊 {req.domain} Analysis Summary")
    render_provenance_legend()
    cols = st.columns(4)
    cols[0].metric("Scale", f"{req.capacity} {req.unit}")
    metric_items = list(req.metrics().items())
    for i, (k, v) in enumerate(metric_items[:2], start=1):
        cols[i].metric(k, v)
    cols[3].metric("SLA", req.availability_target)

    if market_data := state.get("market_data"):
        with st.expander("🔍 Market Intelligence"):
            provenance("search")
            st.write(market_data.get("answer") or "No market data available")
            if sources := market_data.get("sources"):
                st.markdown("**Sources:**")
                for src in sources:
                    title = src.get("title") or src.get("url") or "source"
                    st.markdown(f"- [{title}]({src.get('url', '#')})")

    st.subheader("🏗️ Architecture Recommendations")
    provenance("llm")
    if arch := state.get("architecture_analysis"):
        if "error" in arch:
            st.warning(f"⚠️ {arch.get('error')}")
        if arch_type := arch.get("architecture_type"):
            st.info(f"**Recommended:** {arch_type}")
        for rec in arch.get("key_recommendations", []):
            st.write(f"• {rec}")
        if notes := arch.get("scalability_notes"):
            st.caption(f"📈 Scalability: {notes}")
        if redundancy := arch.get("redundancy_approach"):
            st.caption(f"🛡️ Redundancy: {redundancy}")

    if procurement := state.get("procurement_context"):
        with st.expander(f"📄 Internal Procurement Context ({len(procurement)} documents retrieved)"):
            provenance("rag")
            for doc in procurement:
                meta = doc.get("meta", {})
                vendor_tag = f" · vendor: {meta['vendor']}" if meta.get("vendor") not in (None, "", "none") else ""
                discount_tag = f" · discount: {meta['discount_pct']}%" if meta.get("discount_pct") else ""
                st.markdown(f"**{doc['id']}** (relevance: {doc.get('score', 0):.2f}{vendor_tag}{discount_tag})")
                st.caption(doc["text"][:300] + ("…" if len(doc["text"]) > 300 else ""))

    if candidates := state.get("vendor_candidates"):
        st.caption(f"💡 {len(candidates)} vendors matched your workload and deployment criteria")
        provenance("rules")

    if tco_list := state.get("tco_estimates"):
        st.subheader("💰 3-Year TCO Comparison")
        provenance("math")
        chart_df = pd.DataFrame({
            "Vendor": [t["vendor"] for t in tco_list],
            "3-Year TCO ($)": [t["total_3yr"] for t in tco_list],
        }).set_index("Vendor")
        st.bar_chart(chart_df)
        tcols = st.columns(min(len(tco_list), 4))
        for col, t in zip(tcols, tco_list[:4]):
            with col:
                delta = f"-{t['negotiated_discount_pct']}% negotiated" if t["has_agreement"] else None
                st.metric(t["vendor"], TCOEngine.fmt(t["total_3yr"]), delta,
                          delta_color="inverse" if delta else "off")
                st.caption(f"{t['model']} · {TCOEngine.fmt(t['range'][0])}–{TCOEngine.fmt(t['range'][1])}")
        st.caption("Model: domain list rate by cost profile (or cloud tier) − negotiated discounts "
                   "from procurement docs + facilities + migration baseline. Rates in domains.py.")


    if compliance := state.get("compliance_results"):
        st.subheader("🛡️ Compliance Guardrails")
        provenance("rules")
        badge = {"pass": "✅ PASS", "warn": "🟡 REVIEW", "fail": "🔴 FAIL"}
        for block in compliance:
            with st.expander(f"{badge[block['overall']]} - {block['vendor']}",
                             expanded=(block["overall"] != "pass")):
                for c in block["checks"]:
                    icon = {"pass": "✅", "warn": "🟡", "fail": "🔴"}[c["status"]]
                    st.write(f"{icon} **{c['name']}** - {c['detail']}")

    st.subheader("🏆 Vendor Recommendations")
    provenance("hybrid")
    if vendors := state.get("vendor_recommendations"):
        for i, vendor in enumerate(vendors[:4], 1):
            if not vendor:
                continue
            score = vendor.get("fit_score", 0)
            name = vendor.get("name", "Unknown")
            if "error" in vendor or name == "Error":
                with st.expander("⚠️ Evaluation Error", expanded=True):
                    st.error(f"Error: {vendor.get('error', 'Unknown error')}")
                    for c in vendor.get("considerations", []):
                        st.write(f"• {c}")
                continue
            if score == 0:
                with st.expander(f"⚠️ {name}", expanded=True):
                    for c in vendor.get("considerations", []):
                        st.warning(f"• {c}")
            else:
                meta = vendor_db.get(name, {})
                comp = next((c for c in state.get("compliance_results", []) if c["vendor"] == name), None)
                comp_badge = {"pass": " | 🛡️✅", "warn": " | 🛡️🟡", "fail": " | 🛡️🔴"}.get(comp["overall"], "") if comp else ""
                with st.expander(f"{i}. {name} - Score: {score}/10 | Cost: {meta.get('cost_profile', 'N/A').title()}{comp_badge}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Strengths:**")
                        for s in vendor.get("strengths", []):
                            st.write(f"✓ {s}")
                        if sweet := meta.get("sweet_spot"):
                            st.caption(f"📏 Sweet Spot: {sweet}")
                    with c2:
                        st.markdown("**Considerations:**")
                        for c in vendor.get("considerations", []):
                            st.write(f"• {c}")
                    tco = next((t for t in state.get("tco_estimates", []) if t["vendor"] == name), None)
                    if tco:
                        agreement = " (existing agreement applied)" if tco["has_agreement"] else ""
                        st.info(f"💰 **3-Yr TCO:** {TCOEngine.fmt(tco['range'][0])}–{TCOEngine.fmt(tco['range'][1])}{agreement}")
                    if services := meta.get("services"):
                        st.caption(" · ".join(f"{k.title()}: {v}" for k, v in services.items()))
    else:
        st.info("No vendor recommendations available")

    if next_steps := report.get("next_steps"):
        st.subheader("🚀 Next Steps")
        for i, step in enumerate(next_steps, 1):
            st.write(f"{i}. {step}")

    st.divider()
    _, col1, col2 = st.columns([3, 1.4, 1])
    with col1:
        if report and state.get("vendor_recommendations"):
            try:
                arb_bytes = build_arb_document(state, TCOEngine.fmt)
                st.download_button(
                    "📄 ARB Decision Record (.docx)", arb_bytes,
                    f"ARB_{req.domain.split()[0]}_{datetime.now():%Y%m%d_%H%M%S}.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary")
            except Exception as e:
                logger.error(f"ARB generation failed: {e}")
                st.caption("ARB document generation unavailable")
    with col2:
        if report:
            st.download_button("📥 JSON", json.dumps(report, indent=2, default=str),
                               f"{req.domain.split()[0].lower()}_{datetime.now():%Y%m%d_%H%M%S}.json",
                               "application/json")

# ==========================================================
# Main
# ==========================================================


def _blank_state(req: Requirements) -> dict:
    return {"requirements": req, "messages": [], "market_data": {},
            "architecture_analysis": {}, "vendor_candidates": [],
            "vendor_recommendations": [], "procurement_context": [],
            "tco_estimates": [], "compliance_results": [], "final_report": {},
            "current_step": "initializing"}


def run_solution(payload: dict):
    """Run the (unchanged) LangGraph workflow once per correlated component."""
    workflow = create_advisor_graph()
    results = []
    progress = st.progress(0.0, text="Starting stack analysis...")
    comps = payload["components"]
    for i, c in enumerate(comps):
        req = Requirements(c["domain"], c["workload"], c["capacity"],
                           payload["priority"], payload["deployment"], payload["availability"])
        progress.progress(i / len(comps), text=f"Analyzing {c['domain']} · {c['workload']}...")
        final = dict(_blank_state(req))
        for update in workflow.stream(_blank_state(req)):
            if isinstance(update, dict):
                delta = list(update.values())[0]
                for k, v in delta.items():
                    if k == "messages":
                        final["messages"] = final.get("messages", []) + v
                    else:
                        final[k] = v
        results.append({"domain": c["domain"], "workload": c["workload"],
                        "capacity": c["capacity"], "unit": DOMAINS[c["domain"]]["unit"],
                        "rationale": c["rationale"], "state": final})
    progress.progress(1.0, text="Stack analysis complete")
    return results


def render_solution_results(payload: dict, results: list):
    st.header(f"🧩 {payload['blueprint']} - Full-Stack Analysis")
    st.caption(f"Driver: {payload['driver_value']} · Deployment: {payload['deployment']} · "
               f"SLA: {payload['availability']} · Components sized by correlated formulas")

    synergy = analyze_synergy(results)

    # Stack summary metrics
    cols = st.columns(len(results) + 1)
    total = synergy.get("estimated_stack_tco_3yr", 0)
    cols[0].metric("Stack 3-Yr TCO", TCOEngine.fmt(total) if total else "N/A")
    for col, comp in zip(cols[1:], results):
        top = next((r for r in comp["state"].get("vendor_recommendations", [])
                    if r.get("fit_score", 0) > 0), None)
        col.metric(comp["domain"].split(" /")[0], top["name"] if top else "-",
                   f"{comp['capacity']} {comp['unit']}", delta_color="off")

    # Cross-domain synergy
    st.subheader("🔗 Cross-Domain Vendor Correlation")
    provenance("rules")
    if synergy["bundle_opportunities"]:
        for b in synergy["bundle_opportunities"]:
            st.success(f"💼 {b}")
        for note in synergy["concentration_notes"]:
            st.warning(f"⚖️ {note}")
    else:
        st.info("No single vendor spans multiple components - a best-of-breed stack. "
                "Consider integration effort in delivery planning.")

    # Combined TCO chart (top vendor per component)
    rows = []
    for comp in results:
        state = comp["state"]
        top = next((r for r in state.get("vendor_recommendations", []) if r.get("fit_score", 0) > 0), None)
        if top:
            tco = next((t for t in state.get("tco_estimates", []) if t["vendor"] == top["name"]), None)
            if tco:
                rows.append({"Component": f"{comp['domain'].split(' /')[0]}\n{top['name']}",
                             "3-Year TCO ($)": tco["total_3yr"]})
    if rows:
        st.subheader("💰 Stack TCO by Component (top-ranked vendor each)")
        df = pd.DataFrame(rows).set_index("Component")
        st.bar_chart(df)

    # Per-component detail
    st.subheader("📦 Component Analyses")
    for comp in results:
        top = next((r for r in comp["state"].get("vendor_recommendations", [])
                    if r.get("fit_score", 0) > 0), None)
        header = (f"{comp['domain']} · {comp['workload']} · {comp['capacity']} {comp['unit']}"
                  + (f" → {top['name']}" if top else " → no match"))
        with st.expander(header):
            st.caption(f"Sizing rationale: {comp['rationale']}")
            render_results(comp["state"])

    # Export
    st.divider()
    export = {
        "blueprint": payload["blueprint"],
        "driver_value": payload["driver_value"],
        "sizing_assumptions": payload.get("sizing_params", {}),
        "deployment": payload["deployment"],
        "availability": payload["availability"],
        "synergy": synergy,
        "components": [{"domain": c["domain"], "workload": c["workload"],
                        "capacity": c["capacity"], "unit": c["unit"],
                        "rationale": c["rationale"],
                        "report": c["state"].get("final_report", {})} for c in results],
    }
    st.download_button("📥 Stack Report (JSON)", json.dumps(export, indent=2, default=str),
                       f"stack_{datetime.now():%Y%m%d_%H%M%S}.json", "application/json")


def main():
    st.title("🏛️ Enterprise Infrastructure Advisor")
    st.caption("One advisor for Storage · Server · Database · Middleware - "
               "LangGraph × Vertex AI Gemini × Google Search × Procurement RAG")

    if not Config.validate():
        st.stop()

    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None
    if "solution_results" not in st.session_state:
        st.session_state.solution_results = None

    mode, payload = render_sidebar()

    if mode == "solution" and payload:
        st.session_state.agent_state = None
        try:
            with st.spinner("Running correlated multi-domain analysis on Vertex AI..."):
                st.session_state.solution_results = {
                    "payload": payload, "results": run_solution(payload)}
            st.success("✅ Full-stack analysis completed!")
        except Exception as e:
            logger.exception("Solution analysis failed")
            st.error(f"❌ Stack analysis failed: {e}")
            st.stop()

    requirements = payload if mode == "single" else None

    if requirements:
        st.session_state.solution_results = None
        initial_state: AgentState = {
            "requirements": requirements, "messages": [], "market_data": {},
            "architecture_analysis": {}, "vendor_candidates": [],
            "vendor_recommendations": [], "procurement_context": [],
            "tco_estimates": [], "compliance_results": [], "final_report": {}, "current_step": "initializing",
        }
        st.subheader("🔗 LangGraph Workflow")
        progress_placeholder = st.empty()
        try:
            workflow = create_advisor_graph()
            with st.spinner(f"Analyzing {requirements.domain} on Vertex AI..."):
                accumulated: Dict[str, Any] = dict(initial_state)
                for state_update in workflow.stream(initial_state):
                    if isinstance(state_update, dict):
                        node_delta = list(state_update.values())[0]
                        for k, v in node_delta.items():
                            if k == "messages":
                                accumulated["messages"] = accumulated.get("messages", []) + v
                            else:
                                accumulated[k] = v
                        with progress_placeholder.container():
                            render_workflow_progress(accumulated.get("current_step", ""))
                st.session_state.agent_state = accumulated
                st.success("✅ Workflow completed!")
        except Exception as e:
            logger.exception("Workflow failed")
            st.error(f"❌ Workflow failed: {e}")
            st.stop()

    if st.session_state.solution_results:
        st.divider()
        render_solution_results(st.session_state.solution_results["payload"],
                                st.session_state.solution_results["results"])
    elif st.session_state.agent_state:
        st.divider()
        render_results(st.session_state.agent_state)
    else:
        domain_list = "\n".join(
            f"- **{name}** - {len(cfg['vendors'])} vendors, {len(cfg['workloads'])} workload profiles"
            for name, cfg in DOMAINS.items())
        st.markdown(f"""
### One App for Every Infrastructure Decision

{domain_list}

**Same workflow for every domain:**
```
Market Intel (Google Search) → Architecture (Gemini) → Vendor Match
                                                          ├─→ Procurement RAG → Evaluation + TCO Engine
                                                          └─→ No Vendors → Guidance
                                                                    ↓
                                                                 Report
```

Adding a domain (Network, Backup Software, …) is a **data-only change** in `domains.py` -
the LangGraph workflow, RAG, and TCO engine are domain-agnostic.

Pick a domain in the sidebar to begin.
        """)


if __name__ == "__main__":
    main()
