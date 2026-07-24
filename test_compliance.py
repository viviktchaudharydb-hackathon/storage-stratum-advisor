"""Tests for the deterministic compliance guardrail engine - policy is code,
so policy is tested. No network, no LLM."""

from compliance import run_compliance_checks

VENDOR_DB = {
    "Oracle": {"deployment": ["on-prem", "hybrid", "cloud"]},
    "AWS Aurora / RDS": {"deployment": ["cloud"]},
    "PostgreSQL (EDB)": {"deployment": ["on-prem", "hybrid", "cloud"]},
}


def _run(workload="OLTP / Core Banking", deployment="On-Premises", sla="99.999%",
         recs=None, preferred=None, tco=None):
    recs = recs if recs is not None else [{"name": "Oracle", "fit_score": 9}]
    return run_compliance_checks(
        domain="Database", workload=workload, deployment=deployment,
        availability_target=sla, vendor_recommendations=recs,
        vendor_db=VENDOR_DB, preferred_vendors=preferred or [],
        tco_estimates=tco or [],
    )


def _check(result, name):
    return next(c for c in result["checks"] if c["name"] == name)


# ---------------- Tier-1 SLA floor ----------------

def test_tier1_workload_fails_below_five_nines():
    r = _run(sla="99.99%")[0]
    assert _check(r, "Tier-1 SLA floor")["status"] == "fail"
    assert r["overall"] == "fail"  # any fail => overall fail


def test_tier1_workload_passes_at_five_nines():
    r = _run(sla="99.999%")[0]
    assert _check(r, "Tier-1 SLA floor")["status"] == "pass"


def test_non_tier1_gets_production_floor_warn_at_three_nines():
    r = _run(workload="OLAP / Analytics", sla="99.9%")[0]
    assert _check(r, "Production SLA floor")["status"] == "warn"


# ---------------- Data residency ----------------

def test_onprem_vendor_onprem_deployment_residency_pass():
    r = _run(deployment="On-Premises")[0]
    assert _check(r, "Data residency")["status"] == "pass"


def test_cloud_deployment_triggers_residency_review():
    r = _run(deployment="Cloud")[0]
    assert _check(r, "Data residency")["status"] == "warn"


def test_cloud_only_vendor_triggers_residency_even_onprem_deployment():
    r = _run(recs=[{"name": "AWS Aurora / RDS", "fit_score": 8}], deployment="On-Premises")[0]
    assert _check(r, "Data residency")["status"] == "warn"


# ---------------- Vendor onboarding ----------------

def test_agreement_on_file_passes_onboarding():
    r = _run(preferred=["Oracle"])[0]
    assert _check(r, "Vendor onboarding")["status"] == "pass"


def test_no_agreement_warns_with_security_assessment():
    r = _run(preferred=[])[0]
    c = _check(r, "Vendor onboarding")
    assert c["status"] == "warn" and "security" in c["detail"].lower()


# ---------------- Concentration + structural guards ----------------

def test_preferred_vendor_with_dominant_share_warns_concentration():
    tco = [{"vendor": "Oracle", "total_3yr": 700_000},
           {"vendor": "PostgreSQL (EDB)", "total_3yr": 200_000}]
    r = _run(preferred=["Oracle"], tco=tco)[0]
    assert _check(r, "Concentration risk")["status"] == "warn"


def test_unknown_vendor_names_are_skipped_entirely():
    """A hallucinated vendor gets no compliance verdict at all - the
    deterministic layer refuses to evaluate what it can't look up."""
    out = _run(recs=[{"name": "MadeUp Storage Inc", "fit_score": 9},
                     {"name": "Oracle", "fit_score": 8}])
    assert [r["vendor"] for r in out] == ["Oracle"]


def test_zero_fit_score_entries_are_skipped():
    out = _run(recs=[{"name": "Oracle", "fit_score": 0}])
    assert out == []


def test_overall_is_warn_when_any_check_warns():
    r = _run(deployment="Cloud", sla="99.999%", preferred=["Oracle"])[0]
    assert r["overall"] == "warn"
