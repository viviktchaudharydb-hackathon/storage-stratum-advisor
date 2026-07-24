"""Tests for the deterministic TCO engine and the deterministic parts of
ProcurementRAG (frontmatter parsing, discount scoping). No network, no LLM."""

import pytest
from advisor_extensions import (
    TCOEngine, ProcurementRAG, _parse_frontmatter,
)

NETAPP = {"deployment": ["on-prem", "hybrid", "cloud"], "cost_profile": "competitive"}
PURE = {"deployment": ["on-prem", "hybrid"], "cost_profile": "premium"}
AWS = {"deployment": ["cloud"], "cost_profile": "variable"}


# ---------------- TCO: pricing model selection ----------------

def test_onprem_vendor_priced_capex():
    t = TCOEngine.estimate("Pure Storage", PURE, 100, {}, req_deployment="On-Premises")
    assert t["model"].startswith("CapEx")
    assert t["facilities"] == 180 * 100
    assert t["list_base"] == 2600 * 100  # premium rate


def test_cloud_only_vendor_priced_opex():
    t = TCOEngine.estimate("AWS", AWS, 300, {}, req_deployment="On-Premises")
    assert t["model"].startswith("OpEx")
    assert t["facilities"] == 0
    # 300 TB falls in the 100-500 tier at $20/TB-month
    assert t["list_base"] == 20.0 * 300 * 36


def test_hybrid_vendor_follows_user_cloud_choice():
    """A hybrid-capable vendor evaluated for a Cloud deployment is priced
    on cloud consumption tiers, not CapEx + facilities."""
    cloud = TCOEngine.estimate("NetApp", NETAPP, 300, {}, req_deployment="Cloud")
    onprem = TCOEngine.estimate("NetApp", NETAPP, 300, {}, req_deployment="On-Premises")
    assert cloud["model"].startswith("OpEx") and cloud["facilities"] == 0
    assert onprem["model"].startswith("CapEx") and onprem["facilities"] > 0
    assert cloud["total_3yr"] != onprem["total_3yr"]


def test_hybrid_deployment_stays_capex_conservatively():
    t = TCOEngine.estimate("NetApp", NETAPP, 300, {}, req_deployment="Hybrid")
    assert t["model"].startswith("CapEx")


# ---------------- TCO: discounts and arithmetic ----------------

def test_negotiated_discount_applied_to_base_only():
    plain = TCOEngine.estimate("Pure Storage", PURE, 100, {})
    disc = TCOEngine.estimate("Pure Storage", PURE, 100, {"Pure Storage": 0.18})
    assert disc["negotiated_discount_pct"] == 18
    assert disc["has_agreement"] is True
    # facilities + migration are NOT discounted
    expected = plain["list_base"] * 0.82 + plain["facilities"] + plain["migration"]
    assert disc["total_3yr"] == round(expected)


def test_uncertainty_band_is_plus_minus_15pct():
    t = TCOEngine.estimate("Pure Storage", PURE, 100, {})
    lo, hi = t["range"]
    total = t["list_base"] + t["facilities"] + t["migration"]
    assert lo == round(total * 0.85)
    assert hi == round(total * 1.15)


def test_domain_tco_cfg_overrides_defaults():
    cfg = {"list_rates": {"premium": 42000}, "facilities_per_unit": 0, "migration_flat": 60000}
    t = TCOEngine.estimate("Oracle", {"deployment": ["on-prem"], "cost_profile": "premium"},
                           20, {"Oracle": 0.30}, tco_cfg=cfg)
    assert t["list_base"] == 42000 * 20
    assert t["facilities"] == 0
    assert t["total_3yr"] == round(42000 * 20 * 0.70 + 60000)  # the demo's Oracle $648K


def test_cloud_tier_selection_boundaries():
    assert TCOEngine._cloud_rate(99) == 26.0
    assert TCOEngine._cloud_rate(100) == 20.0
    assert TCOEngine._cloud_rate(500) == 15.0


def test_fmt_money():
    assert TCOEngine.fmt(648_000) == "$648K"
    assert TCOEngine.fmt(1_250_000) == "$1.25M"


# ---------------- RAG deterministic parts ----------------

DOC = """---
vendor: Oracle
discount_pct: 30
domain: Database
---
Enterprise ULA covering database licenses.
"""


def test_frontmatter_parsing():
    meta, body = _parse_frontmatter(DOC)
    assert meta == {"vendor": "Oracle", "discount_pct": "30", "domain": "Database"}
    assert body.strip().startswith("Enterprise ULA")


def test_frontmatter_absent_returns_full_body():
    meta, body = _parse_frontmatter("just a plain document")
    assert meta == {} and body == "just a plain document"


def _rag_with(docs):
    rag = ProcurementRAG.__new__(ProcurementRAG)  # skip __init__ (no client)
    rag.docs = docs
    rag.matrix = None
    return rag


CORPUS = [
    {"id": "dell-storage.md", "meta": {"vendor": "Dell", "discount_pct": "18", "domain": "Storage"}, "text": ""},
    {"id": "oracle-ula.md", "meta": {"vendor": "Oracle", "discount_pct": "30", "domain": "Database"}, "text": ""},
    {"id": "redhat-ea.md", "meta": {"vendor": "Red Hat", "discount_pct": "20"}, "text": ""},
    {"id": "policy.md", "meta": {"vendor": "none"}, "text": ""},
]


def test_discounts_scoped_by_domain():
    rag = _rag_with(CORPUS)
    db = rag.negotiated_discounts("Database")
    assert db == {"Oracle": 0.30, "Red Hat": 0.20}  # untagged Red Hat applies everywhere
    assert "Dell" not in db  # storage framework must not leak into Database


def test_preferred_vendors_matches_discount_scoping():
    """Compliance badge and TCO discount must agree - same scoping."""
    rag = _rag_with(CORPUS)
    assert set(rag.preferred_vendors("Database")) == set(rag.negotiated_discounts("Database"))
    assert "Dell" in rag.preferred_vendors("Storage")
    assert "Dell" not in rag.preferred_vendors("Database")


def test_invalid_discount_pct_ignored():
    rag = _rag_with([{"id": "x.md", "meta": {"vendor": "V", "discount_pct": "abc"}, "text": ""}])
    assert rag.negotiated_discounts() == {}
