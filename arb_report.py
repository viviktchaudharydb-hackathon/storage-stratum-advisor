"""
ARB Decision Record generator - produces a formal Architecture Review Board
paper as a Word document (bytes, for st.download_button).

Uses python-docx at runtime (Cloud Run). Sections follow a typical bank ARB
template: context, options considered, cost comparison (with the deterministic
model disclosed), procurement context, compliance results, recommendation,
and sign-off placeholders.
"""

import io
from datetime import datetime
from typing import Dict, Any, List

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

ACCENT = RGBColor(0x1A, 0x47, 0x8F)
STATUS_COLORS = {"pass": RGBColor(0x1E, 0x7A, 0x33),
                 "warn": RGBColor(0xB0, 0x6A, 0x00),
                 "fail": RGBColor(0xB0, 0x1E, 0x1E)}
STATUS_LABEL = {"pass": "PASS", "warn": "REVIEW", "fail": "FAIL"}


def _h(doc, text, level=1):
    doc.add_heading(text, level=level)


def _kv_table(doc, rows: List[tuple]):
    table = doc.add_table(rows=len(rows), cols=2)
    table.style = "Light Grid Accent 1"
    for i, (k, v) in enumerate(rows):
        table.rows[i].cells[0].text = k
        table.rows[i].cells[1].text = str(v)
        for run in table.rows[i].cells[0].paragraphs[0].runs:
            run.bold = True
    return table


def build_arb_document(state: Dict[str, Any], fmt_money) -> bytes:
    """state: the final AgentState dict. fmt_money: TCOEngine.fmt."""
    req = state["requirements"]
    report = state.get("final_report", {})
    arch = state.get("architecture_analysis", {})
    vendors = [v for v in state.get("vendor_recommendations", []) if v.get("fit_score", 0) > 0]
    tco_list = state.get("tco_estimates", [])
    tco_by_vendor = {t["vendor"]: t for t in tco_list}
    compliance = state.get("compliance_results", [])
    procurement = state.get("procurement_context", [])
    sources = state.get("market_data", {}).get("sources", [])

    doc = Document()

    # Title
    title = doc.add_heading("Architecture Review Board - Decision Record", level=0)
    sub = doc.add_paragraph(
        f"{req.domain} · {req.workload} · Generated {datetime.now():%d %b %Y %H:%M} "
        f"· Enterprise Infrastructure Advisor (Vertex AI)")
    sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in sub.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # 1. Context
    _h(doc, "1. Context and Requirements")
    _kv_table(doc, [
        ("Domain", req.domain),
        ("Workload", req.workload),
        ("Scale", f"{req.capacity} {req.unit}"),
        ("Deployment model", req.deployment),
        ("Design priority", req.priority),
        ("Availability SLA", req.availability_target),
    ])

    # 2. Recommended architecture
    _h(doc, "2. Recommended Architecture")
    if arch.get("architecture_type"):
        p = doc.add_paragraph()
        p.add_run(f"Architecture pattern: {arch['architecture_type']}").bold = True
    for rec in arch.get("key_recommendations", []):
        doc.add_paragraph(rec, style="List Bullet")
    if arch.get("redundancy_approach"):
        doc.add_paragraph(f"Redundancy approach: {arch['redundancy_approach']}")
    if arch.get("scalability_notes"):
        doc.add_paragraph(f"Scalability: {arch['scalability_notes']}")

    # 3. Options considered
    _h(doc, "3. Options Considered")
    for i, v in enumerate(vendors[:4], 1):
        _h(doc, f"3.{i} {v['name']} - fit score {v.get('fit_score', 'N/A')}/10", level=2)
        if v.get("strengths"):
            doc.add_paragraph("Strengths:", style="Intense Quote")
            for s in v["strengths"]:
                doc.add_paragraph(s, style="List Bullet")
        if v.get("considerations"):
            doc.add_paragraph("Considerations:", style="Intense Quote")
            for c in v["considerations"]:
                doc.add_paragraph(c, style="List Bullet")

    # 4. Cost comparison
    _h(doc, "4. Three-Year TCO Comparison")
    if tco_list:
        table = doc.add_table(rows=1, cols=5)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        for j, h in enumerate(["Vendor", "Model", "Negotiated Discount", "3-Yr TCO (est.)", "Range"]):
            hdr[j].text = h
            for run in hdr[j].paragraphs[0].runs:
                run.bold = True
        for t in tco_list:
            row = table.add_row().cells
            row[0].text = t["vendor"]
            row[1].text = t["model"]
            row[2].text = f"{t['negotiated_discount_pct']}%" if t["has_agreement"] else "-"
            row[3].text = fmt_money(t["total_3yr"])
            row[4].text = f"{fmt_money(t['range'][0])} – {fmt_money(t['range'][1])}"
        note = doc.add_paragraph(
            "Methodology: deterministic model - domain list rate by cost profile (or cloud "
            "capacity tier), less negotiated discounts sourced from procurement agreements, "
            "plus facilities and migration baseline; ±15% sizing uncertainty band. "
            "Figures are pre-quotation estimates for comparison, not commercial offers.")
        for run in note.runs:
            run.font.size = Pt(8)
            run.italic = True

    # 5. Procurement context
    _h(doc, "5. Procurement Context (retrieved)")
    if procurement:
        for d in procurement:
            meta = d.get("meta", {})
            p = doc.add_paragraph()
            p.add_run(d["id"]).bold = True
            tags = []
            if meta.get("vendor") not in (None, "", "none", "multiple"):
                tags.append(f"vendor: {meta['vendor']}")
            if meta.get("discount_pct"):
                tags.append(f"discount: {meta['discount_pct']}%")
            if meta.get("valid_until"):
                tags.append(f"valid until: {meta['valid_until']}")
            if tags:
                p.add_run("  (" + " · ".join(tags) + ")").italic = True
            excerpt = d["text"][:350] + ("…" if len(d["text"]) > 350 else "")
            doc.add_paragraph(excerpt)
    else:
        doc.add_paragraph("No relevant internal procurement documents were retrieved for this analysis.")

    # 6. Compliance assessment
    _h(doc, "6. Compliance Assessment")
    if compliance:
        for block in compliance:
            p = doc.add_paragraph()
            p.add_run(f"{block['vendor']} - overall: ").bold = True
            status_run = p.add_run(STATUS_LABEL[block["overall"]])
            status_run.bold = True
            status_run.font.color.rgb = STATUS_COLORS[block["overall"]]
            for c in block["checks"]:
                cp = doc.add_paragraph(style="List Bullet")
                sr = cp.add_run(f"[{STATUS_LABEL[c['status']]}] ")
                sr.bold = True
                sr.font.color.rgb = STATUS_COLORS[c["status"]]
                cp.add_run(f"{c['name']}: {c['detail']}")
    else:
        doc.add_paragraph("Compliance checks were not executed for this analysis.")

    # 7. Recommendation & next steps
    _h(doc, "7. Recommendation and Next Steps")
    if vendors:
        top = vendors[0]
        p = doc.add_paragraph()
        p.add_run(f"Recommended option: {top['name']}").bold = True
        tco = tco_by_vendor.get(top["name"])
        if tco:
            doc.add_paragraph(
                f"Estimated 3-year TCO {fmt_money(tco['range'][0])}–{fmt_money(tco['range'][1])}"
                + (" under the existing negotiated agreement." if tco["has_agreement"]
                   else "; no existing agreement - commercial negotiation required."))
    for step in report.get("next_steps", []):
        doc.add_paragraph(step, style="List Number")

    # 8. Market sources
    if sources:
        _h(doc, "8. Market Intelligence Sources")
        for s in sources:
            doc.add_paragraph(f"{s.get('title') or s.get('url')} - {s.get('url')}", style="List Bullet")

    # Sign-off
    _h(doc, "Approvals")
    table = doc.add_table(rows=4, cols=3)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for j, h in enumerate(["Role", "Name / Signature", "Date"]):
        hdr[j].text = h
        for run in hdr[j].paragraphs[0].runs:
            run.bold = True
    for i, role in enumerate(["Solution Architect", "Architecture Review Board Chair",
                              "Procurement / Vendor Management"], start=1):
        table.rows[i].cells[0].text = role

    disclaimer = doc.add_paragraph(
        "\nGenerated by the Enterprise Infrastructure Advisor. AI-assisted content "
        "(architecture analysis, vendor fit narrative) is marked by origin in the tool; "
        "cost figures and compliance results are computed deterministically. This record "
        "requires human review and approval before any procurement action.")
    for run in disclaimer.runs:
        run.font.size = Pt(8)
        run.italic = True

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
