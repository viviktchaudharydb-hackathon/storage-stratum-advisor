"""
Extensions for the Storage Advisor:
- ProcurementRAG: embeds a local procurement corpus with Vertex AI embeddings
  and retrieves relevant context per analysis (in-memory cosine search).
- TCOEngine: deterministic 3-year TCO model - never delegated to the LLM -
  with negotiated-discount awareness parsed from the procurement corpus.
"""

import os
import re
import glob
import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger("storage-advisor.ext")

DATA_DIR = os.environ.get("PROCUREMENT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data", "procurement"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-005")


# ==========================================================
# Procurement corpus + RAG
# ==========================================================

def _parse_frontmatter(text: str) -> (Dict[str, str], str):
    """Parse a simple '---' frontmatter block. Returns (metadata, body)."""
    meta: Dict[str, str] = {}
    body = text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    return meta, body


def load_corpus(data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """Load all .md docs with frontmatter metadata. One chunk per doc
    (docs are intentionally small for the hackathon demo)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            meta, body = _parse_frontmatter(f.read())
        docs.append({
            "id": os.path.basename(path),
            "meta": meta,
            "text": body.strip(),
        })
    logger.info(f"Loaded {len(docs)} procurement docs from {data_dir}")
    return docs


class ProcurementRAG:
    """Minimal in-memory RAG over the procurement corpus.

    Embeddings come from Vertex AI (text-embedding-005) via the shared
    genai client; retrieval is cosine similarity over a numpy matrix.
    For production scale, swap this for Vertex AI RAG Engine or
    Vertex AI Search without changing the node interface.
    """

    def __init__(self, client, docs: Optional[List[Dict[str, Any]]] = None):
        self.client = client
        self.docs = docs if docs is not None else load_corpus()
        self.matrix: Optional[np.ndarray] = None
        if self.docs:
            self.matrix = self._embed([d["text"] for d in self.docs])

    def _embed(self, texts: List[str]) -> Optional[np.ndarray]:
        try:
            result = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
            )
            vecs = np.array([e.values for e in result.embeddings], dtype=np.float32)
            # Normalize once so retrieval is a plain dot product
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return top_k docs with similarity scores. Empty list on failure -
        the workflow degrades gracefully to no procurement context."""
        if self.matrix is None or not self.docs:
            return []
        q = self._embed([query])
        if q is None:
            return []
        scores = self.matrix @ q[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.docs[i], "score": float(scores[i])}
            for i in order
            if scores[i] > 0.3  # relevance floor
        ]

    def negotiated_discounts(self, domain: Optional[str] = None) -> Dict[str, float]:
        """Deterministic: parse discount_pct from doc frontmatter.
        If domain is given, only agreements tagged for that domain (or
        untagged) apply - so Dell's storage framework (18%) and server
        addendum (15%) don't collide."""
        discounts: Dict[str, float] = {}
        for d in self.docs:
            meta = d["meta"]
            vendor = meta.get("vendor", "")
            pct = meta.get("discount_pct", "")
            doc_domain = meta.get("domain", "")
            if domain and doc_domain and doc_domain != domain:
                continue
            if vendor and vendor.lower() not in ("none", "multiple") and pct:
                try:
                    discounts[vendor] = float(pct) / 100.0
                except ValueError:
                    continue
        return discounts

    def preferred_vendors(self) -> List[str]:
        """Vendors with any active agreement doc - used to flag policy-preferred options."""
        return list(self.negotiated_discounts().keys())


# ==========================================================
# Deterministic TCO Engine
# ==========================================================

class TCOEngine:
    """3-year TCO model. Deterministic by design - pricing is a business
    decision, not something to delegate to an LLM. Rates are indicative
    list-price approximations ($/TB effective, 3-year, incl. support);
    tune LIST_RATES to your procurement reality.
    """

    # $/TB over 3 years by cost profile (on-prem/hybrid vendors)
    LIST_RATES = {
        "premium": 2600,
        "competitive": 1800,
        "budget": 1100,
    }
    # Cloud: $/TB-month by capacity tier → converted to 3yr below
    CLOUD_MONTHLY_TIERS = [
        (100, 26.0),      # <100 TB
        (500, 20.0),      # 100–500 TB
        (float("inf"), 15.0),
    ]
    ONPREM_FACILITIES_PER_TB_3YR = 180   # power/cooling/rack per bank policy
    MIGRATION_FLAT = 25000               # baseline migration/services cost

    @classmethod
    def _cloud_rate(cls, capacity_tb: int, tiers=None) -> float:
        tiers = tiers or cls.CLOUD_MONTHLY_TIERS
        for limit, rate in tiers:
            if capacity_tb < limit:
                return rate
        return tiers[-1][1]

    @classmethod
    def estimate(
        cls,
        vendor_name: str,
        vendor_meta: Dict[str, Any],
        capacity_tb: int,
        discounts: Dict[str, float],
        tco_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Returns a structured 3-year TCO estimate for one vendor.

        tco_cfg (optional, from the domain registry) overrides the storage
        defaults: {list_rates, cloud_monthly_tiers, facilities_per_unit,
        migration_flat}. This keeps the engine domain-generic.
        """
        cfg = tco_cfg or {}
        list_rates = cfg.get("list_rates", cls.LIST_RATES)
        cloud_tiers = cfg.get("cloud_monthly_tiers", cls.CLOUD_MONTHLY_TIERS)
        facilities_per_unit = cfg.get("facilities_per_unit", cls.ONPREM_FACILITIES_PER_TB_3YR)
        migration_flat = cfg.get("migration_flat", cls.MIGRATION_FLAT)

        deployments = vendor_meta.get("deployment", [])
        is_cloud_only = deployments == ["cloud"]
        discount = discounts.get(vendor_name, 0.0)

        if is_cloud_only:
            monthly_rate = cls._cloud_rate(capacity_tb, cloud_tiers)
            base = monthly_rate * capacity_tb * 36
            facilities = 0
            migration = migration_flat
            model = "OpEx (pay-as-you-go, 3-yr run rate)"
        else:
            profile = vendor_meta.get("cost_profile", "competitive")
            rate = list_rates.get(profile, list_rates.get("competitive", 1800))
            base = rate * capacity_tb
            facilities = facilities_per_unit * capacity_tb
            migration = migration_flat
            model = "CapEx + support (3-yr)"

        discounted_base = base * (1 - discount)
        total = discounted_base + facilities + migration
        low, high = total * 0.85, total * 1.15  # sizing/config uncertainty band

        return {
            "vendor": vendor_name,
            "model": model,
            "capacity_tb": capacity_tb,
            "list_base": round(base),
            "negotiated_discount_pct": round(discount * 100),
            "facilities": round(facilities),
            "migration": round(migration),
            "total_3yr": round(total),
            "range": (round(low), round(high)),
            "per_tb_3yr": round(total / capacity_tb) if capacity_tb else 0,
            "has_agreement": discount > 0,
        }

    @staticmethod
    def fmt(n: float) -> str:
        if n >= 1_000_000:
            return f"${n/1_000_000:.2f}M"
        return f"${n/1_000:.0f}K"
