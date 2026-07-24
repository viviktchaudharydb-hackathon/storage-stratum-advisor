"""
Solution Blueprints - cross-domain correlation layer.

A blueprint maps a business use case to multiple infrastructure domains with
DETERMINISTIC sizing formulas linked to one driver metric. Example: an AI/ML
training platform sized by GPU server count derives its dataset storage and
metadata database capacity from that same driver - the domains are correlated,
not analyzed in isolation.

The existing LangGraph workflow runs once per component (unchanged); this layer
derives the inputs, then aggregates outputs: combined TCO, cross-domain vendor
synergy, and stack-level compliance posture.
"""

from typing import Dict, Any, List, Callable
from dataclasses import dataclass


@dataclass
class Component:
    domain: str
    workload: str
    size_fn: Callable[[int, Dict[str, float]], int]   # (driver, params) -> capacity
    rationale: str                                     # template; {param} placeholders allowed


# Each blueprint exposes its sizing ASSUMPTIONS as named, user-editable
# parameters. The formulas stay deterministic; the inputs are transparent
# and adjustable in the UI - nothing hidden in code.
BLUEPRINTS: Dict[str, Dict[str, Any]] = {

    "AI/ML Training Platform": {
        "description": "GPU compute + high-throughput dataset storage + experiment/metadata store, "
                       "sized from the number of GPU servers.",
        "driver": {"label": "GPU Servers", "min": 2, "max": 64, "default": 8, "step": 2},
        "params": {
            "tb_per_gpu_server": {"label": "Storage TB per GPU server", "default": 40, "min": 10, "max": 120, "step": 5,
                                  "help": "Datasets + checkpoints + staging per GPU node to avoid I/O stall"},
            "metadata_tb_per_2_servers": {"label": "Metadata TB per 2 GPU servers", "default": 1, "min": 1, "max": 10, "step": 1,
                                          "help": "Experiment tracking / feature metadata store sizing"},
        },
        "components": [
            Component("Server / Compute", "AI/GPU Compute",
                      lambda n, p: n,
                      "Driver metric - one entry per GPU server (e.g., 8×GPU HGX class)."),
            Component("Storage", "AI/ML Training",
                      lambda n, p: int(n * p["tb_per_gpu_server"]),
                      "{tb_per_gpu_server}TB high-performance storage per GPU server "
                      "(datasets + checkpoints + staging) to keep GPUs fed."),
            Component("Database", "NoSQL / Document",
                      lambda n, p: max(2, int((n // 2) * p["metadata_tb_per_2_servers"])),
                      "Experiment/metadata store: {metadata_tb_per_2_servers}TB per 2 GPU servers, 2TB floor."),
        ],
    },

    "Core Banking Modernization": {
        "description": "OLTP database + dedicated DB hosts + resilient block storage + "
                       "transactional messaging, sized from primary database size.",
        "driver": {"label": "Primary DB Size (TB)", "min": 2, "max": 100, "default": 10, "step": 2},
        "params": {
            "storage_multiple": {"label": "Storage multiple of DB size", "default": 4, "min": 2, "max": 8, "step": 1,
                                 "help": "Sync replica + backup staging + growth headroom (Tier-1 standard: 4×)"},
            "hosts_per_5tb": {"label": "DB hosts per 5TB (HA pairs)", "default": 2, "min": 1, "max": 4, "step": 1,
                              "help": "High-memory hosts per 5TB; 4-node floor across sites"},
            "msg_instances_per_2tb": {"label": "Messaging instances per 2TB", "default": 1, "min": 1, "max": 4, "step": 1,
                                      "help": "Transactional messaging scaled with throughput proxy; 4-instance floor"},
        },
        "components": [
            Component("Database", "OLTP / Core Banking",
                      lambda d, p: d,
                      "Driver metric - primary transactional data set."),
            Component("Server / Compute", "Database Hosts",
                      lambda d, p: max(4, int((d // 5) * p["hosts_per_5tb"])),
                      "{hosts_per_5tb} high-memory hosts per 5TB for HA pairs, 4-node floor "
                      "(primary + standby across sites)."),
            Component("Storage", "OLTP Database",
                      lambda d, p: int(d * p["storage_multiple"]),
                      "{storage_multiple}× primary size: synchronous replica + backup staging + "
                      "growth headroom, per Tier-1 resilience standards."),
            Component("Middleware", "Messaging / Streaming",
                      lambda d, p: max(4, int((d // 2) * p["msg_instances_per_2tb"])),
                      "{msg_instances_per_2tb} messaging instance(s) per 2TB throughput proxy, "
                      "4-instance floor for quorum + HA."),
        ],
    },

    "Enterprise Data Lake & Analytics": {
        "description": "Object/file storage for the lake + analytics database + ingestion streaming, "
                       "sized from raw data volume.",
        "driver": {"label": "Raw Data Volume (TB)", "min": 50, "max": 2000, "default": 400, "step": 50},
        "params": {
            "lake_multiple_pct": {"label": "Lake storage % of raw volume", "default": 150, "min": 100, "max": 300, "step": 10,
                                  "help": "Landing + curated zones; compression offsets copies (default 150%)"},
            "warehouse_pct": {"label": "Warehouse % of raw volume", "default": 10, "min": 5, "max": 30, "step": 5,
                              "help": "Share of raw volume materialized into the analytics layer"},
            "stream_per_100tb": {"label": "Streaming instances per 100TB", "default": 1, "min": 1, "max": 5, "step": 1,
                                 "help": "Ingestion cluster sizing; 4-instance floor"},
        },
        "components": [
            Component("Storage", "File Services",
                      lambda d, p: int(d * p["lake_multiple_pct"] / 100),
                      "{lake_multiple_pct}% of raw volume: landing + curated zones."),
            Component("Database", "OLAP / Analytics",
                      lambda d, p: max(5, int(d * p["warehouse_pct"] / 100)),
                      "{warehouse_pct}% of raw volume materialized into the analytics/warehouse layer."),
            Component("Middleware", "Messaging / Streaming",
                      lambda d, p: max(4, int((d // 100) * p["stream_per_100tb"])),
                      "{stream_per_100tb} ingestion instance(s) per 100TB raw volume, 4-instance floor."),
        ],
    },
}


def default_params(blueprint_name: str) -> Dict[str, float]:
    return {k: v["default"] for k, v in BLUEPRINTS[blueprint_name].get("params", {}).items()}


def derive_components(blueprint_name: str, driver_value: int,
                      params: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """Resolve a blueprint into concrete per-domain requirements.
    params: user-adjusted sizing assumptions (defaults if None)."""
    bp = BLUEPRINTS[blueprint_name]
    p = {**default_params(blueprint_name), **(params or {})}
    out = []
    for c in bp["components"]:
        out.append({
            "domain": c.domain,
            "workload": c.workload,
            "capacity": c.size_fn(driver_value, p),
            "rationale": c.rationale.format(**p),
        })
    return out


def analyze_synergy(component_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-domain vendor correlation over per-component results.

    component_results: [{domain, workload, capacity, unit, state(final AgentState dict)}]

    Finds vendors recommended in more than one component (single-vendor
    leverage / bundle opportunity) and flags stack-level concentration.
    Deterministic - no LLM.
    """
    vendor_hits: Dict[str, List[str]] = {}
    vendor_spend: Dict[str, int] = {}
    total_spend = 0

    for comp in component_results:
        state = comp["state"]
        tco_by_vendor = {t["vendor"]: t for t in state.get("tco_estimates", [])}
        for rec in state.get("vendor_recommendations", [])[:3]:
            name = rec.get("name", "")
            if not name or rec.get("fit_score", 0) == 0:
                continue
            # Only count vendors validated against this domain's DB - a TCO entry
            # exists iff the name matched the registry (guards against LLM drift)
            tco = tco_by_vendor.get(name)
            if not tco:
                continue
            # Normalize umbrella vendors across domains (Dell == Dell, GCP* == GCP family)
            family = name.split(" (")[0].split(" / ")[0]
            for prefix in ("AWS", "Azure", "GCP", "Google"):
                if family.startswith(prefix):
                    family = "GCP" if prefix == "Google" else prefix
            vendor_hits.setdefault(family, []).append(f"{comp['domain']}: {name}")
            vendor_spend[family] = vendor_spend.get(family, 0) + tco["total_3yr"]
        # Total = top-1 vendor TCO per component (the presumptive selection)
        top = next((r for r in state.get("vendor_recommendations", []) if r.get("fit_score", 0) > 0), None)
        if top:
            t = tco_by_vendor.get(top.get("name", ""))
            if t:
                total_spend += t["total_3yr"]

    multi_domain = {v: hits for v, hits in vendor_hits.items() if len(set(h.split(":")[0] for h in hits)) > 1}

    concentration = []
    if total_spend:
        for v, spend in vendor_spend.items():
            share = spend / max(total_spend, 1)
            if v in multi_domain and share > 0.5:
                concentration.append(
                    f"{v} spans {len(set(h.split(':')[0] for h in multi_domain[v]))} domains at a large "
                    f"spend share - single-vendor leverage available, but review concentration limits")

    return {
        "multi_domain_vendors": multi_domain,
        "bundle_opportunities": [
            f"{v} appears across {', '.join(sorted(set(h.split(':')[0] for h in hits)))} - "
            f"negotiate a bundled/stack agreement instead of per-domain deals"
            for v, hits in multi_domain.items()
        ],
        "concentration_notes": concentration,
        "estimated_stack_tco_3yr": total_spend,
    }
