"""
Compliance guardrail engine - deterministic bank-policy checks applied to
every recommended vendor. No LLM involvement by design: policy is code.

Each check returns status: "pass" | "warn" | "fail" with a detail string.
Rules are intentionally simple and transparent - auditable by an ARB.
"""

from typing import List, Dict, Any

# Workload tiers requiring the highest availability (bank policy §2)
TIER1_WORKLOADS = {"OLTP / Core Banking", "OLTP Database", "Messaging / Streaming"}

# Regions considered in-country for regulated data (bank policy §4)
APPROVED_REGIONS_NOTE = "asia-south1 / asia-south2 with CMEK"

# Cloud-only vendors imply regulated-data residency review
CLOUD_ONLY = lambda meta: meta.get("deployment", []) == ["cloud"]


def run_compliance_checks(
    domain: str,
    workload: str,
    deployment: str,
    availability_target: str,
    vendor_recommendations: List[Dict[str, Any]],
    vendor_db: Dict[str, Any],
    preferred_vendors: List[str],
    tco_estimates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Returns one result block per recommended vendor."""
    results = []
    tco_by_vendor = {t["vendor"]: t for t in tco_estimates}

    for rec in vendor_recommendations:
        name = rec.get("name", "")
        meta = vendor_db.get(name)
        if not meta or rec.get("fit_score", 0) == 0:
            continue

        checks = []

        # --- 1. Availability SLA floor (policy §2) ---
        if workload in TIER1_WORKLOADS:
            if availability_target == "99.999%":
                checks.append({"name": "Tier-1 SLA floor", "status": "pass",
                               "detail": "99.999% meets Tier-1 requirement for this workload"})
            else:
                checks.append({"name": "Tier-1 SLA floor", "status": "fail",
                               "detail": f"Tier-1 workload requires 99.999% with synchronous replication; "
                                         f"selected SLA is {availability_target}"})
        else:
            if availability_target in ("99.99%", "99.999%"):
                checks.append({"name": "Production SLA floor", "status": "pass",
                               "detail": f"{availability_target} meets the 99.99% production minimum"})
            else:
                checks.append({"name": "Production SLA floor", "status": "warn",
                               "detail": "99.9% is below the 99.99% production minimum - "
                                         "acceptable only for non-production tiers"})

        # --- 2. Data residency (policy §4) ---
        if CLOUD_ONLY(meta) or deployment in ("Cloud", "Hybrid"):
            checks.append({"name": "Data residency", "status": "warn",
                           "detail": f"Cloud deployment: regulated data must remain in-region "
                                     f"({APPROVED_REGIONS_NOTE}). Confirm region pinning and CMEK "
                                     f"before design approval"})
        else:
            checks.append({"name": "Data residency", "status": "pass",
                           "detail": "On-premises deployment - residency requirement inherently met"})

        # --- 3. Vendor onboarding status (policy §1, §3) ---
        if name in preferred_vendors:
            checks.append({"name": "Vendor onboarding", "status": "pass",
                           "detail": "Active agreement on file - preferred vendor, no new "
                                     "security assessment required"})
        else:
            checks.append({"name": "Vendor onboarding", "status": "warn",
                           "detail": "No active agreement found. New vendors require security "
                                     "assessment (8–12 weeks) and RBI outsourcing compliance "
                                     "review before PO issuance"})

        # --- 4. Concentration risk ---
        # Warn if this vendor already holds agreements in other domains too
        # (simple proxy: preferred in this run AND appears with large TCO share)
        tco = tco_by_vendor.get(name)
        if tco and tco_estimates:
            total = sum(t["total_3yr"] for t in tco_estimates) or 1
            share = tco["total_3yr"] / total
            if name in preferred_vendors and share > 0.6:
                checks.append({"name": "Concentration risk", "status": "warn",
                               "detail": f"Vendor represents {share:.0%} of evaluated spend and "
                                         f"already holds agreements - consider dual-vendor strategy "
                                         f"per concentration guidelines"})
            else:
                checks.append({"name": "Concentration risk", "status": "pass",
                               "detail": "No concentration concern at this spend share"})

        overall = "fail" if any(c["status"] == "fail" for c in checks) else (
            "warn" if any(c["status"] == "warn" for c in checks) else "pass")

        results.append({"vendor": name, "overall": overall, "checks": checks})

    return results
