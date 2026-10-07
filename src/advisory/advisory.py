"""One-page maintenance advisory for the maintenance head: health band, top risk,
recommended action + urgency, and a simple expected-cost comparison (all costs in config)."""
from __future__ import annotations

from pathlib import Path

from src.config import load_config

FAULT_TEXT = {
    "PD": ("Partial discharge in oil/paper", "Check for gas bubbles / moisture; acoustic PD survey; keep DGA trending weekly."),
    "D1": ("Low-energy discharge (sparking)", "Check floating metal parts, OLTC diverter leakage; lab DGA within 7 days; plan internal inspection."),
    "D2": ("High-energy arcing", "Arcing can escalate to failure: consider immediate de-energisation after lab DGA confirmation; electrical tests (winding resistance, SFRA)."),
    "T1": ("Thermal fault < 300 C", "Check overloading and cooling; paper overheating likely - track CO/CO2 ratio."),
    "T2": ("Thermal fault 300-700 C", "Typical of bad joints, circulating currents in core/clamps or OLTC contacts: confirm with lab DGA, "
           "IR thermography of bushings/OLTC, winding resistance + core-insulation (megger) test at the next planned outage; "
           "cap loading at 0.8 pu until resolved."),
    "T3": ("Thermal fault > 700 C", "Hot metal (core/connection) - urgent: lab DGA confirmation, de-rate to 0.7 pu, plan outage for internal inspection."),
    "DT": ("Mixed thermal + electrical fault", "Lab DGA + electrical tests; plan internal inspection."),
}

URGENCY = {"Very Poor": "IMMEDIATE (24-72 h)", "Poor": "HIGH - within 2 weeks",
           "Fair": "MEDIUM - within 3 months", "Good": "LOW - next routine maintenance",
           "Very Good": "ROUTINE"}


def build(hi_result: dict, life: dict, events_30d: list[str], as_of: str) -> dict:
    c = load_config()["costs_inr"]
    band = hi_result["band"]
    dga = hi_result["dga_diagnosis"]
    fault = dga.get("ml_class") if dga["status"] == "Abnormal" else None

    if fault:
        risk_title, action = FAULT_TEXT.get(fault, ("Abnormal gassing", "Lab DGA confirmation and inspection."))
        top_risk = f"{risk_title} (ML: {fault}, Duval Triangle 1: {dga['duval_triangle_1']})"
    else:
        d = hi_result["top_drivers"][0] if hi_result["top_drivers"] else {"parameter": "none"}
        top_risk = f"Largest deficit: {d['parameter']}"
        action = "Follow band action: " + hi_result["band_action"]
    if any("BUCHHOLZ" in e for e in events_30d):
        action += " Buchholz gas alarm in last 30 days: sample Buchholz gas and analyse with oil DGA before any re-energisation after trip."

    failure_cost = c["repair_major_failure"] + c["outage_cost_per_hour"] * c["outage_hours_failure"]
    pof = c["pof_12m_by_band"][band]
    action_cost = c["inspection_and_oil_treatment"] + c["outage_cost_per_hour"] * c["outage_hours_planned"] + 6 * c["extra_dga_sample"]
    risk_no_action = pof * failure_cost
    risk_with_action = pof * (1 - c["action_effectiveness"]) * failure_cost
    benefit = risk_no_action - risk_with_action
    cb = {
        "failure_cost_inr": failure_cost,
        "probability_of_failure_12m": pof,
        "expected_loss_no_action_inr": risk_no_action,
        "action_cost_inr": action_cost,
        "expected_loss_with_action_inr": risk_with_action,
        "net_benefit_inr": benefit - action_cost,
        "benefit_cost_ratio": benefit / action_cost,
    }
    return {"as_of": as_of, "health_index": hi_result["health_index"], "band": band,
            "components": hi_result["components"], "top_drivers": hi_result["top_drivers"],
            "top_risk": top_risk, "recommended_action": action, "urgency": URGENCY[band],
            "events_30d": events_30d, "life": life, "cost_benefit": cb}


def _inr(x: float) -> str:
    return f"INR {x / 1e5:,.1f} lakh" if abs(x) < 1e7 else f"INR {x / 1e7:,.2f} crore"


def to_markdown(a: dict) -> str:
    cb = a["cost_benefit"]
    drv = "\n".join(f"{i + 1}. {d['parameter']} = {d['value']} (-{d['points_lost']} pts)" for i, d in enumerate(a["top_drivers"]))
    comp = ", ".join(f"{k} {v}" for k, v in a["components"].items())
    return f"""# Maintenance Advisory - 10 MVA 33/11 kV Transformer
**As of:** {a['as_of']}   |   **Health index:** {a['health_index']} / 100   |   **Band:** {a['band']}   |   **Urgency:** {a['urgency']}

**Component scores:** {comp}

## Top risk
{a['top_risk']}

**Top drivers of the score**
{drv}

## Recommended action
{a['recommended_action']}

## Cost-benefit (12-month expected cost, assumptions in config.yaml)
| | Value |
|---|---|
| Cost if the transformer fails (repair + {load_config()['costs_inr']['outage_hours_failure']} h outage) | {_inr(cb['failure_cost_inr'])} |
| Probability of failure in 12 months at this band (assumed) | {cb['probability_of_failure_12m']:.0%} |
| Expected loss - no action | {_inr(cb['expected_loss_no_action_inr'])} |
| Cost of recommended action | {_inr(cb['action_cost_inr'])} |
| Expected loss - after action | {_inr(cb['expected_loss_with_action_inr'])} |
| **Net benefit of acting now** | **{_inr(cb['net_benefit_inr'])}** (benefit/cost = {cb['benefit_cost_ratio']:.1f}) |

Insulation life used over the monitored period: {a['life']['life_consumed_pct']:.2f} % of {a['life']['normal_life_hours']:,} h (IEC 60076-7); max hot-spot {a['life']['max_hot_spot_c']:.0f} C.

*Prototype output on public + synthetic data; confirm with lab DGA and site inspection before acting.*
"""


def to_pdf(a: dict, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    st = getSampleStyleSheet()
    cb = a["cost_benefit"]
    band_col = {"Very Good": "#1b7f3b", "Good": "#5aa02c", "Fair": "#d4a017", "Poor": "#e06c00", "Very Poor": "#c0392b"}[a["band"]]
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    el = [Paragraph("Maintenance Advisory - 10 MVA, 33/11 kV Power Transformer", st["Title"]),
          Paragraph(f"As of {a['as_of']} &nbsp;|&nbsp; To: Head of Maintenance &nbsp;|&nbsp; From: Asset Analytics", st["Normal"]),
          Spacer(1, 8)]
    head = Table([["Health index", "Band", "Urgency"],
                  [f"{a['health_index']} / 100", a["band"], a["urgency"]]], colWidths=[5 * cm, 5 * cm, 7.4 * cm])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2d3d")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                              ("BACKGROUND", (1, 1), (1, 1), colors.HexColor(band_col)), ("TEXTCOLOR", (1, 1), (1, 1), colors.white),
                              ("FONTSIZE", (0, 1), (-1, 1), 13), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                              ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("TOPPADDING", (0, 1), (-1, 1), 6), ("BOTTOMPADDING", (0, 1), (-1, 1), 6)]))
    el += [head, Spacer(1, 6),
           Paragraph("Component scores: " + ", ".join(f"<b>{k}</b> {v}" for k, v in a["components"].items()), st["Normal"]),
           Spacer(1, 6), Paragraph("Top risk", st["Heading2"]), Paragraph(a["top_risk"], st["Normal"]),
           Paragraph("Drivers: " + "; ".join(f"{d['parameter']} = {d['value']} (-{d['points_lost']} pts)" for d in a["top_drivers"]), st["Normal"]),
           Paragraph("Recommended action", st["Heading2"]), Paragraph(a["recommended_action"], st["Normal"]),
           Paragraph("Cost-benefit (12-month expected cost)", st["Heading2"])]
    t = Table([["Item", "INR"],
               ["Cost if failure occurs (repair + outage)", _inr(cb["failure_cost_inr"])],
               ["Probability of failure, 12 months (assumed for band)", f"{cb['probability_of_failure_12m']:.0%}"],
               ["Expected loss - no action", _inr(cb["expected_loss_no_action_inr"])],
               ["Cost of recommended action", _inr(cb["action_cost_inr"])],
               ["Expected loss - after action", _inr(cb["expected_loss_with_action_inr"])],
               ["Net benefit of acting now", f"{_inr(cb['net_benefit_inr'])}  (B/C {cb['benefit_cost_ratio']:.1f})"]],
              colWidths=[10.4 * cm, 7 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6ee")), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]))
    el += [t, Spacer(1, 6),
           Paragraph(f"Insulation ageing (IEC 60076-7): {a['life']['life_consumed_pct']:.2f} % of normal life used in the monitored "
                     f"period; max hot-spot {a['life']['max_hot_spot_c']:.0f} C. Events last 30 days: {', '.join(a['events_30d']) or 'none'}.", st["Normal"]),
           Spacer(1, 6),
           Paragraph("<i>Prototype built on public (ETT, IEC TC 10) and synthetic data. Costs and failure probabilities are stated "
                     "assumptions (config.yaml). Confirm with laboratory DGA and site inspection before acting.</i>", st["Normal"])]
    doc.build(el)
