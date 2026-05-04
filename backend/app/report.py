"""
PDF report generation using ReportLab.

Produces a professionally formatted one-page-plus PDF containing:
  - AI narrative (LLM-generated explanation)
  - Performance metrics table
  - Portfolio allocation table
  - SHAP feature importance tables (up to 3 tickers)
  - Benchmark comparison callout
  - Boilerplate disclaimer footer
"""
import datetime
import io
from typing import Any, Dict, List, Optional


def generate_report(
    tickers: List[str],
    allocation: List[Dict[str, Any]],
    metrics: Dict[str, float],
    narrative: str = "",
    shap_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    benchmark_label: str = "NIFTY 50 (^NSEI)",
) -> bytes:
    """
    Build a PDF report and return the raw bytes.

    Parameters
    ----------
    tickers        : list of ticker symbols in the portfolio
    allocation     : list of {ticker, weight} dicts
    metrics        : performance metrics dict from /optimize-portfolio
    narrative      : LLM-generated text explanation (may be empty)
    shap_data      : optional {ticker: [feature dicts]} from /explain-views
    benchmark_label: human-readable name for the benchmark
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    # ── Colour constants ────────────────────────────────────────────────────
    BRAND_BLUE   = colors.Color(0.18, 0.38, 0.90)
    LIGHT_BLUE   = colors.Color(0.88, 0.92, 1.00)
    DARK_BG      = colors.Color(0.07, 0.09, 0.14)
    GREY_TEXT    = colors.Color(0.45, 0.50, 0.58)
    GREEN        = colors.Color(0.13, 0.77, 0.37)
    RED          = colors.Color(0.91, 0.30, 0.30)

    # ── Document setup ──────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=2.2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    base_styles = getSampleStyleSheet()
    title_s = ParagraphStyle(
        "ReportTitle", parent=base_styles["Title"],
        fontSize=22, spaceAfter=4, textColor=DARK_BG,
    )
    sub_s = ParagraphStyle(
        "Sub", parent=base_styles["Normal"],
        fontSize=9, textColor=GREY_TEXT, spaceAfter=2,
    )
    h2_s = ParagraphStyle(
        "H2", parent=base_styles["Heading2"],
        fontSize=13, spaceBefore=14, spaceAfter=4, textColor=DARK_BG,
    )
    body_s = ParagraphStyle(
        "Body", parent=base_styles["Normal"],
        fontSize=10, leading=15, textColor=DARK_BG,
    )
    bold_s = ParagraphStyle(
        "Bold", parent=body_s, fontName="Helvetica-Bold",
    )
    small_s = ParagraphStyle(
        "Small", parent=base_styles["Normal"],
        fontSize=8, textColor=GREY_TEXT,
    )

    def _table_style(header_color=BRAND_BLUE) -> TableStyle:
        return TableStyle([
            ("BACKGROUND",   (0, 0), (-1,  0), header_color),
            ("TEXTCOLOR",    (0, 0), (-1,  0), colors.white),
            ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BLUE, colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.Color(0.80, 0.83, 0.90)),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ])

    def _fmt_metric(k: str, v: float) -> str:
        k_l = k.lower()
        if any(x in k_l for x in ("return", "volatility", "drawdown")):
            return f"{v:+.2%}"
        return f"{v:.4f}"

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Black-Litterman Portfolio Report", title_s))
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(
        f"Generated: {ts}   |   Benchmark: {benchmark_label}", sub_s,
    ))
    story.append(Paragraph(
        f"Portfolio: {', '.join(tickers)}", sub_s,
    ))
    story.append(HRFlowable(
        width="100%", thickness=1.5, color=BRAND_BLUE, spaceAfter=8,
    ))

    # ── AI Narrative ────────────────────────────────────────────────────────
    if narrative:
        story.append(Paragraph("AI Portfolio Analysis", h2_s))
        story.append(Paragraph(narrative, body_s))
        story.append(Spacer(1, 6))

    # ── Performance Metrics ─────────────────────────────────────────────────
    story.append(Paragraph("Performance Metrics", h2_s))
    metric_rows = [["Metric", "Value"]]
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        val = _fmt_metric(k, v)
        metric_rows.append([label, val])
    t = Table(metric_rows, colWidths=[10 * cm, 6 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 8))

    # ── Portfolio Allocation ─────────────────────────────────────────────────
    story.append(Paragraph("Portfolio Allocation", h2_s))
    alloc_sorted = sorted(allocation, key=lambda x: -x.get("weight", 0.0))
    alloc_rows = [["Ticker", "Weight"]]
    for item in alloc_sorted:
        w = item.get("weight", 0.0)
        alloc_rows.append([item["ticker"], f"{w:.2%}"])
    t2 = Table(alloc_rows, colWidths=[10 * cm, 6 * cm])
    t2.setStyle(_table_style())
    story.append(t2)
    story.append(Spacer(1, 8))

    # ── SHAP Feature Importances ─────────────────────────────────────────────
    if shap_data:
        story.append(Paragraph("XGBoost Feature Importances (SHAP)", h2_s))
        story.append(Paragraph(
            "Mean absolute SHAP value per feature — larger means stronger influence "
            "on the predicted return view for that stock.", body_s,
        ))
        story.append(Spacer(1, 4))
        for ticker, feats in list(shap_data.items())[:3]:
            if not feats:
                continue
            story.append(Paragraph(ticker, bold_s))
            shap_rows = [["Feature", "SHAP Value", "Feature Value"]]
            for feat in feats[:6]:
                shap_rows.append([
                    feat.get("feature", ""),
                    f"{feat.get('shap_value', 0.0):+.6f}",
                    f"{feat.get('feature_value', 0.0):.4f}",
                ])
            t3 = Table(shap_rows, colWidths=[7 * cm, 4.5 * cm, 4.5 * cm])
            t3.setStyle(_table_style(header_color=colors.Color(0.40, 0.45, 0.55)))
            story.append(t3)
            story.append(Spacer(1, 6))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(
        width="100%", thickness=0.5, color=colors.lightgrey, spaceBefore=12,
    ))
    story.append(Paragraph(
        "This report was automatically generated by the Black-Litterman RAG Portfolio Optimizer. "
        "Past performance is not indicative of future results. "
        "This is for educational and research purposes only and does not constitute financial advice.",
        small_s,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
