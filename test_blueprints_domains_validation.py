"""Tests for blueprint sizing formulas, cross-domain synergy, the domain
registry matcher, and the LLM-output validator. No network, no LLM."""

from blueprints import derive_components, analyze_synergy, default_params, BLUEPRINTS
from domains import get_matching_vendors, DOMAINS
from advisor_extensions import validate_recommendations


# ---------------- Blueprint sizing ----------------

def test_core_banking_default_sizing_matches_demo():
    comps = {c["domain"]: c["capacity"] for c in derive_components("Core Banking Modernization", 10)}
    assert comps == {"Database": 10, "Server / Compute": 4, "Storage": 40, "Middleware": 5}


def test_sizing_floors_hold_at_minimum_driver():
    comps = {c["domain"]: c["capacity"] for c in derive_components("Core Banking Modernization", 2)}
    assert comps["Server / Compute"] == 4   # 4-node floor
    assert comps["Middleware"] == 4         # 4-instance quorum floor


def test_param_override_rederives_stack():
    comps = {c["domain"]: c["capacity"]
             for c in derive_components("Core Banking Modernization", 10, {"storage_multiple": 6})}
    assert comps["Storage"] == 60  # the live 4x -> 6x demo moment


def test_rationale_templates_resolve_all_placeholders():
    for name in BLUEPRINTS:
        for c in derive_components(name, BLUEPRINTS[name]["driver"]["default"]):
            assert "{" not in c["rationale"], f"unresolved placeholder in {name}/{c['domain']}"


def test_blueprint_workloads_exist_in_registry():
    """Every blueprint component must reference a real domain + workload,
    or the workflow silently gets an empty candidate list."""
    for name, bp in BLUEPRINTS.items():
        for c in bp["components"]:
            assert c.domain in DOMAINS, f"{name}: unknown domain {c.domain}"
            assert c.workload in DOMAINS[c.domain]["workloads"], \
                f"{name}: {c.workload!r} not in {c.domain} workloads"


# ---------------- Synergy ----------------

def _state(recs, tco):
    return {"vendor_recommendations": recs, "tco_estimates": tco}


def test_synergy_counts_only_registry_validated_vendors():
    """A recommended name with no TCO entry (i.e. not in the registry)
    must not appear in synergy - the anti-drift guard."""
    results = [
        {"domain": "Storage", "state": _state(
            [{"name": "Dell", "fit_score": 9}, {"name": "Hallucinated Inc", "fit_score": 8}],
            [{"vendor": "Dell", "total_3yr": 100}])},
        {"domain": "Server / Compute", "state": _state(
            [{"name": "Dell", "fit_score": 9}],
            [{"vendor": "Dell", "total_3yr": 200}])},
    ]
    syn = analyze_synergy(results)
    assert set(syn["multi_domain_vendors"]) == {"Dell"}
    assert syn["estimated_stack_tco_3yr"] == 300


def test_synergy_normalizes_hyperscaler_families():
    results = [
        {"domain": "Database", "state": _state(
            [{"name": "AWS Aurora / RDS", "fit_score": 9}],
            [{"vendor": "AWS Aurora / RDS", "total_3yr": 100}])},
        {"domain": "Server / Compute", "state": _state(
            [{"name": "AWS EC2", "fit_score": 9}],
            [{"vendor": "AWS EC2", "total_3yr": 100}])},
    ]
    syn = analyze_synergy(results)
    assert "AWS" in syn["multi_domain_vendors"]
    assert len(syn["bundle_opportunities"]) == 1


def test_single_domain_vendor_is_not_a_bundle():
    results = [{"domain": "Storage", "state": _state(
        [{"name": "Dell", "fit_score": 9}], [{"vendor": "Dell", "total_3yr": 100}])}]
    assert analyze_synergy(results)["multi_domain_vendors"] == {}


# ---------------- Domain registry matcher ----------------

def test_cloud_deployment_excludes_onprem_only_vendors():
    got = get_matching_vendors("Server / Compute", "Database Hosts", "Cloud")
    assert "Dell" not in got and "HPE" not in got
    assert all("cloud" in DOMAINS["Server / Compute"]["vendors"][v]["deployment"] for v in got)


def test_hybrid_deployment_matches_everything_suitable():
    got = get_matching_vendors("Database", "OLTP / Core Banking", "Hybrid")
    assert "Oracle" in got and "PostgreSQL (EDB)" in got and "AWS Aurora / RDS" in got


def test_min_capacity_filter():
    # DDN requires 500 TB minimum
    assert "DDN" not in get_matching_vendors("Storage", "HPC", "On-Premises", capacity=100)
    assert "DDN" in get_matching_vendors("Storage", "HPC", "On-Premises", capacity=1000)


def test_workload_filter():
    got = get_matching_vendors("Storage", "AI/ML Training", "On-Premises")
    assert "Cohesity" not in got  # backup vendor, not AI/ML


# ---------------- LLM-output validator ----------------

CANDIDATES = ["Oracle", "PostgreSQL (EDB)", "Microsoft SQL Server"]


def test_validator_drops_non_candidate_vendors():
    recs, dropped = validate_recommendations(
        [{"name": "Oracle", "fit_score": 9},
         {"name": "MongoDB", "fit_score": 8}],  # real vendor, but NOT a candidate this run
        CANDIDATES)
    assert [r["name"] for r in recs] == ["Oracle"]
    assert dropped == ["MongoDB"]


def test_validator_clamps_fit_score():
    recs, _ = validate_recommendations(
        [{"name": "Oracle", "fit_score": 97},
         {"name": "PostgreSQL (EDB)", "fit_score": -3},
         {"name": "Microsoft SQL Server", "fit_score": "8.5"}],
        CANDIDATES)
    scores = {r["name"]: r["fit_score"] for r in recs}
    assert scores["Oracle"] == 10.0
    assert scores["PostgreSQL (EDB)"] == 0.0
    assert scores["Microsoft SQL Server"] == 8.5


def test_validator_handles_garbage_shapes():
    recs, dropped = validate_recommendations(
        ["not a dict", {"fit_score": 9}, {"name": "Oracle", "fit_score": None,
                                          "strengths": "one string", "considerations": [" ok ", ""]}],
        CANDIDATES)
    assert len(recs) == 1 and recs[0]["name"] == "Oracle"
    assert recs[0]["fit_score"] == 0.0
    assert recs[0]["strengths"] == []          # non-list coerced to empty
    assert recs[0]["considerations"] == ["ok"]  # stripped, empties removed
    assert len(dropped) == 2


def test_validator_dedupes_and_resorts():
    recs, _ = validate_recommendations(
        [{"name": "PostgreSQL (EDB)", "fit_score": 7},
         {"name": "Oracle", "fit_score": 9},
         {"name": "Oracle", "fit_score": 5}],
        CANDIDATES)
    assert [r["name"] for r in recs] == ["Oracle", "PostgreSQL (EDB)"]
    assert recs[0]["fit_score"] == 9  # first occurrence wins on dupes


def test_validator_empty_input():
    assert validate_recommendations([], CANDIDATES) == ([], [])
    assert validate_recommendations(None, CANDIDATES) == ([], [])
