"""
Section 15 (Save Report) / 16 (Report Formats): a real PDF, not a chat-log
export or a mocked file.

Deliberately scoped to the fields directly traceable to a single
FinalResponse/Evidence -- Section 15's own report list also wants
maps/overlays, charts, and cross-file/temporal-findings sections. Those
need infrastructure this pass doesn't build (real map rendering, tracking
findings across a project's files over time); a placeholder version of
those sections would be exactly the "partially implemented prototype"
the upgrade brief asked to avoid, so they're left out rather than faked
-- the report says so explicitly in its own closing note, rather than
silently having a gap a reader might not notice.

generate_pdf_report() takes a real, in-memory FinalResponse (not a JSON
round-trip of one), so response.evidence.task and .modality_used are
actual TaskType/Modality enum members, not strings -- _enum_val() below
reads their .value rather than relying on str(), which is worth being
explicit about: str() on a `class X(str, Enum)` member gives "X.MEMBER",
not "MEMBER", unlike the newer enum.StrEnum.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_pdf_report(
    query: str,
    response,
    output_path: str,
    project_name: Optional[str] = None,
    image_assets: Optional[list[dict]] = None,
    change_map_path: Optional[str] = None,
    boxes_path: Optional[str] = None,
) -> str:
    """Writes a PDF to output_path and returns that same path. Raises on a
    reportlab failure rather than swallowing it -- callers (api/main.py)
    decide what an HTTP-layer failure response looks like."""
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=9, textColor=colors.grey)

    story = [
        Paragraph("SatQuery AI \u2014 Analysis Report", styles["Title"]),
    ]
    if project_name:
        story.append(Paragraph(f"Project: {_esc(project_name)}", small))
    story.append(Paragraph(
        "Generated " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), small
    ))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Research question", styles["Heading2"]))
    story.append(Paragraph(_esc(query), styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Answer", styles["Heading2"]))
    if response.abstained:
        story.append(Paragraph(
            "<i>The system abstained: " + _esc(response.abstain_reason or "insufficient evidence") + "</i>",
            styles["BodyText"],
        ))
    else:
        story.append(Paragraph(_esc(response.answer_text), styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    conf = response.confidence
    story.append(Paragraph("Confidence", styles["Heading2"]))
    conf_line = conf.band + (f" ({conf.value:.2f})" if conf.value is not None else "")
    story.append(Paragraph(_esc(conf_line) + " \u2014 " + _esc(conf.basis), styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    ev = response.evidence

    assets = image_assets or []
    if assets or change_map_path or boxes_path:
        story.append(Paragraph("Imagery and evidence", styles["Heading2"]))
        comparison = []
        for asset in assets[:2]:
            comparison.append(_comparison_cell(
                asset.get("label", "Image"),
                asset["path"],
                asset.get("width"),
                asset.get("height"),
            ))
        if change_map_path:
            comparison.append(_comparison_cell("Change map", change_map_path))
        if boxes_path:
            comparison.append(_comparison_cell("Boxes", boxes_path))
        if comparison:
            story.append(Table([comparison], colWidths=[1.55 * inch] * len(comparison), hAlign="LEFT"))
            story.append(Spacer(1, 0.12 * inch))
        if ev.detections:
            story.append(Paragraph("Detected boxes", styles["Heading3"]))
            story.append(_table([
                ["Label", "Confidence", "Box (pixels)"],
                *[
                    [d.label, f"{d.score:.3f}", ", ".join(f"{v:.1f}" for v in d.box_px)]
                    for d in ev.detections
                ],
            ]))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Methodology", styles["Heading2"]))
    story.append(_table([
        ["Task type", _enum_val(ev.task)],
        ["Model used", ev.model_used],
        ["Modalities", ", ".join(_enum_val(m) for m in ev.modality_used) or "\u2014"],
    ]))
    story.append(Spacer(1, 0.15 * inch))

    if ev.stats:
        story.append(Paragraph("Key findings", styles["Heading2"]))
        story.append(_table([
            [k.replace("_", " "), (f"{v:.3f}" if isinstance(v, float) else str(v))]
            for k, v in ev.stats.items()
        ]))
        story.append(Spacer(1, 0.15 * inch))

    if ev.web_sources:
        story.append(Paragraph("Web sources", styles["Heading2"]))
        for src in ev.web_sources:
            story.append(Paragraph(
                f'<link href="{_esc(src.url)}">{_esc(src.title or src.url)}</link>', styles["BodyText"]
            ))
            story.append(Paragraph(_esc(src.snippet), small))
            story.append(Spacer(1, 0.08 * inch))
        story.append(Spacer(1, 0.1 * inch))

    if ev.warnings:
        story.append(Paragraph("Limitations / warnings", styles["Heading2"]))
        for w in ev.warnings:
            story.append(Paragraph("\u2022 " + _esc(w), styles["BodyText"]))
        story.append(Spacer(1, 0.15 * inch))

    story.append(HRFlowable(width="100%", color=colors.lightgrey))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "This report includes the source imagery available to the analysis, visual "
        "evidence, methodology, key numeric findings, and any web citations for "
        "this single analysis.",
        small,
    ))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(output_path, pagesize=LETTER).build(story)
    return output_path


def _table(rows: list[list[str]]) -> Table:
    t = Table(rows, colWidths=[1.7 * inch, 4.3 * inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.whitesmoke),
    ]))
    return t


def _report_image(path: str, width: int | None = None, height: int | None = None) -> Image:
    """Fit an evidence image to the printable page while preserving detail."""
    max_width = 6.4 * inch
    max_height = 4.7 * inch
    if width and height:
        scale = min(max_width / width, max_height / height)
        return Image(path, width=max(1, width * scale), height=max(1, height * scale))
    return Image(path, width=max_width, height=max_height)


def _comparison_cell(label: str, path: str, width: int | None = None, height: int | None = None) -> list:
    """Build a compact report column for side-by-side visual comparison."""
    image_width = 1.42 * inch
    if width and height:
        image_height = image_width * height / max(1, width)
    else:
        image_height = 1.05 * inch
    return [
        Paragraph(_esc(label), ParagraphStyle("comparison-label", fontSize=8, textColor=colors.grey)),
        Image(path, width=image_width, height=min(1.15 * inch, max(0.35 * inch, image_height))),
    ]


def _enum_val(x) -> str:
    return x.value if hasattr(x, "value") else str(x)


def _esc(s) -> str:
    # reportlab's Paragraph markup is a tiny HTML-like subset -- any
    # query/answer/model text has to be escaped before it's handed in, or
    # a literal "<" in an answer would corrupt the layout.
    return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
