# Enterprise Infrastructure Advisor

**One app for Storage · Server/Compute · Database · Middleware decisions.**

An intelligent, multi-domain infrastructure advisory platform powered by LangGraph, Vertex AI Gemini, Google Search grounding, and Procurement RAG, with a deterministic Total Cost of Ownership (TCO) engine.

Entrypoint: `infra_advisor.py` (multi-domain). `storage_advisor.py` (storage-only v2) is kept for reference.

## 🌟 Key Features

- **Multi-Domain Intelligence**: Evaluates workloads across Storage, Server/Compute, Database, and Middleware using domain-scoped knowledge bases.
- **AI meets Determinism**: Uses Gemini for architecture reasoning and vendor matching, while keeping cost (TCO) and compliance evaluation strictly deterministic.
- **Domain-Scoped Procurement RAG**: Integrates with mock procurement documents (e.g., Oracle ULA 30%, Red Hat EA 20%) to scope vendor discounts and influence rankings based on active agreements.
- **Policy-as-Code Governance**: Built-in compliance guardrails that evaluate Tier-1 SLAs, data residency, and vendor onboarding status—completely independent of the LLM.
- **Solution Blueprints (Cross-Domain Correlation)**: Models complex projects like AI/ML Training Platforms or Core Banking Modernizations across multiple domains with editable sizing assumptions.
- **One-Click ARB Decision Records**: Auto-generates formal Architecture Review Board (ARB) papers as `.docx` with TCO tables, compliance badges, and human sign-off blocks.

## 🏗️ Architecture

The platform uses a **domain registry pattern**. The core workflow, RAG, and TCO engine are domain-agnostic. Everything domain-specific—workload profiles, vendor knowledge bases, TCO rates, and slider configurations—lives in `domains.py`.

Adding a new domain (e.g., Network, Security) requires only data changes; no core workflow code needs to be modified.

```text
Market Intel (Google Search) → Architecture (Gemini) → Vendor Match
                                                          ├─→ Procurement RAG → Evaluation + TCO Engine
                                                          ├─→ Compliance Guardrails (Deterministic)
                                                          └─→ No Vendors → Guidance
                                                                    ↓
                                                           Report & ARB Docx
```

### The AI vs. Deterministic Split
1. **Analysis & Ranking (AI)**: Understands workload needs and market trends.
2. **Cost & Governance (Deterministic Code)**: Calculates TCO using fixed formulas and checks compliance rules.
3. **Decision Record (Human)**: Final output is an ARB doc requiring human sign-off.
This structure mirrors real enterprise processes while mitigating LLM risk.

## 🚀 Deployment

The app is designed to run on Google Cloud Run.

```bash
./deploy.sh YOUR_PROJECT_ID asia-south1
```

## 🎮 Demo Scenarios

Here are a few ways to demonstrate the platform:

1. **Database Evaluation**: 
   - *Scenario*: OLTP/Core Banking · 20TB · On-Prem · 99.999% SLA
   - *Expected Outcome*: Oracle ULA (30% discount) surfaces via RAG; TCO compares Oracle against PostgreSQL's license-free economics.
2. **Middleware Evaluation**: 
   - *Scenario*: Messaging · Hybrid
   - *Expected Outcome*: Red Hat agreement + open-source-first policy influences vendor ranking.
3. **Storage Evaluation**: 
   - *Scenario*: Backup · 300TB · On-Prem
   - *Expected Outcome*: Dell framework agreement (18% discount) surfaces.
4. **Cross-Domain Solution Blueprints**: 
   - *Scenario*: AI/ML Training Platform → 8 GPU servers
   - *Expected Outcome*: Automatically evaluates Server, Storage, and Database in one run. Correlates sizing (e.g., 40TB storage per server) and surfaces bundle opportunities (e.g., Dell across Server + Storage).

## 🧩 Solution Blueprints & Sizing Assumptions

Real projects span domains. Blueprint mode (`blueprints.py`) correlates them using **deterministic sizing formulas** derived from a single driver metric.

- **Shipped Blueprints**: AI/ML Training Platform, Core Banking Modernization, Enterprise Data Lake & Analytics.
- **Cross-Domain Synergy Analysis**: Identifies bundle-negotiation opportunities if vendors overlap across components (e.g., DB and Server from the same vendor).
- **Interactive Assumptions**: Blueprint sizing assumptions (e.g., "Storage TB per GPU server") are first-class, named parameters visible in the sidebar. Adjusting these instantly re-derives the entire stack's requirements and costs.

## 🛡️ Governance & Compliance Layer

The compliance guardrail node (`compliance.py`) is a deterministic policy-as-code node running after vendor evaluation. It checks:
- **Tier-1 SLA floors**: Enforces 99.999% for core banking/payments.
- **Data Residency**: Verifies in-region + CMEK for cloud workloads.
- **Vendor Onboarding**: Flags missing agreements with an 8–12 week security assessment requirement.
- **Concentration Risk**.

Results render as **PASS / REVIEW / FAIL** badges per vendor and feed directly into the ARB document.

## 🗺️ Roadmap & Scaling

- **Identity**: IAP integration for internal SSO.
- **Persistence**: Firestore integration for per-user report history.
- **Real Documents**: Vertex AI RAG Engine over real procurement PDFs in GCS (replacing mock RAG).
- **Live Integrations**: Seed the Domain registry from CMDB/ServiceNow for live vendor and agreement data.
- **Warm Pools**: Use `--min-instances 1` to keep corpus embeddings warm for faster responses.

See `architecture.svg` for a system diagram detailing the AI vs. deterministic split and the GCP service map.
