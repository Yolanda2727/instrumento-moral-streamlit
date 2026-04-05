from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, Sequence
from xml.sax.saxutils import escape

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 16 * mm
RIGHT_MARGIN = 16 * mm
TOP_MARGIN = 18 * mm
BOTTOM_MARGIN = 16 * mm
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN


def _safe_text(value: Any) -> str:
    if value is None:
        return "No disponible"
    if isinstance(value, float):
        if pd.isna(value):
            return "No disponible"
        return f"{value:.2f}"
    if pd.isna(value):
        return "No disponible"
    text = str(value).strip()
    return text or "No disponible"


def _truncate_text(value: Any, limit: int = 320) -> str:
    text = _safe_text(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "EthoscopeTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#0B1F3A"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "EthoscopeSubtitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#C08A2B"),
            spaceAfter=12,
        ),
        "section": ParagraphStyle(
            "EthoscopeSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#0B1F3A"),
            spaceBefore=6,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "EthoscopeBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#24384A"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "EthoscopeSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#5A6B7C"),
        ),
        "table_header": ParagraphStyle(
            "EthoscopeTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
        "table_body": ParagraphStyle(
            "EthoscopeTableBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=9.2,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#24384A"),
        ),
        "bullet": ParagraphStyle(
            "EthoscopeBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#24384A"),
            leftIndent=10,
            firstLineIndent=-7,
            spaceAfter=3,
        ),
    }


def _key_value_table(rows: Sequence[tuple[str, Any]], styles: Dict[str, ParagraphStyle], value_width: float) -> Table:
    data = []
    for label, value in rows:
        data.append([
            Paragraph(escape(_safe_text(label)), styles["table_body"]),
            Paragraph(escape(_safe_text(value)), styles["table_body"]),
        ])
    table = Table(data, colWidths=[CONTENT_WIDTH - value_width, value_width], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F7FAFC"), colors.HexColor("#EEF4F9")]),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D7E2EE")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7E2EE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _dataframe_table(
    title: str,
    dataframe: pd.DataFrame,
    styles: Dict[str, ParagraphStyle],
    *,
    col_widths: Sequence[float] | None = None,
    truncate_columns: Iterable[str] | None = None,
    max_rows: int | None = None,
) -> list:
    story: list = [Paragraph(escape(title), styles["section"])]
    if dataframe is None or dataframe.empty:
        story.append(Paragraph("Sin datos disponibles para esta sección.", styles["body"]))
        return story

    display_df = dataframe.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows).copy()

    truncate_targets = set(truncate_columns or [])
    headers = [Paragraph(escape(str(col)), styles["table_header"]) for col in display_df.columns]
    rows = [headers]
    for _, row in display_df.iterrows():
        rendered_row = []
        for column in display_df.columns:
            cell_value = row[column]
            text = _truncate_text(cell_value) if column in truncate_targets else _safe_text(cell_value)
            rendered_row.append(Paragraph(escape(text), styles["table_body"]))
        rows.append(rendered_row)

    width_map = list(col_widths) if col_widths else [CONTENT_WIDTH / max(len(display_df.columns), 1)] * len(display_df.columns)
    table = Table(rows, colWidths=width_map, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F3A")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D7E2EE")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7E2EE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    return story


def _plotly_figure_to_image(fig: go.Figure, width: int = 1200, height: int = 700) -> BytesIO | None:
    try:
        image_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=2)
    except Exception:
        return None
    return BytesIO(image_bytes)


def _figure_story(title: str, caption: str, fig: go.Figure, styles: Dict[str, ParagraphStyle]) -> list:
    story: list = [Paragraph(escape(title), styles["section"])]
    buffer = _plotly_figure_to_image(fig)
    if buffer is None:
        story.append(Paragraph(
            "No fue posible incrustar esta gráfica en el PDF. Verifica la disponibilidad de Kaleido en el entorno.",
            styles["body"],
        ))
        if caption:
            story.append(Paragraph(escape(caption), styles["small"]))
        return story

    image = Image(buffer, width=CONTENT_WIDTH, height=CONTENT_WIDTH * 0.55)
    story.append(image)
    if caption:
        story.append(Spacer(1, 4))
        story.append(Paragraph(escape(caption), styles["small"]))
    return story


def _list_story(title: str, items: Sequence[Any], styles: Dict[str, ParagraphStyle]) -> list:
    story: list = [Paragraph(escape(title), styles["section"])]
    clean_items = [item for item in items if _safe_text(item) != "No disponible"]
    if not clean_items:
        story.append(Paragraph("Sin elementos registrados.", styles["body"]))
        return story
    for item in clean_items:
        story.append(Paragraph(f"- {escape(_safe_text(item))}", styles["bullet"]))
    return story


def _draw_page_frame(canvas, doc, app_title: str, app_brand_line: str) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E2EE"))
    canvas.setLineWidth(0.6)
    canvas.line(LEFT_MARGIN, PAGE_HEIGHT - 12 * mm, PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 12 * mm)
    canvas.line(LEFT_MARGIN, 10 * mm, PAGE_WIDTH - RIGHT_MARGIN, 10 * mm)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(colors.HexColor("#0B1F3A"))
    canvas.drawString(LEFT_MARGIN, PAGE_HEIGHT - 9 * mm, app_title)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5A6B7C"))
    canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 9 * mm, app_brand_line)
    canvas.drawString(LEFT_MARGIN, 6.5 * mm, "Reporte exportado desde el software ETHOSCOPE")
    canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, 6.5 * mm, f"Página {doc.page}")
    canvas.restoreState()


def build_individual_report_pdf(
    *,
    app_title: str,
    app_brand_line: str,
    author_name: str,
    author_credentials: Sequence[str],
    main_function: str,
    generated_at: str,
    participant_rows: Sequence[tuple[str, Any]],
    metric_rows: Sequence[tuple[str, Any]],
    narrative_summary: str,
    recommendations: Sequence[str],
    framework_scores_df: pd.DataFrame,
    stage_scores_df: pd.DataFrame,
    choice_detail_df: pd.DataFrame,
    interpretation_result: Dict[str, Any] | None,
    interpretation_note: str | None,
    figures: Sequence[tuple[str, str, go.Figure]],
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=f"{app_title} - Reporte individual",
        author=author_name,
    )
    styles = _build_styles()
    story: list = []

    story.append(Paragraph(escape(app_title), styles["title"]))
    story.append(Paragraph(escape(app_brand_line), styles["subtitle"]))
    story.append(Paragraph(escape(main_function), styles["body"]))
    story.append(Paragraph(
        escape(f"Autor: {author_name} | {' | '.join(author_credentials)}"),
        styles["small"],
    ))
    story.append(Paragraph(escape(f"Fecha de generación: {generated_at}"), styles["small"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Ficha del participante", styles["section"]))
    story.append(_key_value_table(participant_rows, styles, value_width=CONTENT_WIDTH * 0.46))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Indicadores principales", styles["section"]))
    story.append(_key_value_table(metric_rows, styles, value_width=CONTENT_WIDTH * 0.34))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Síntesis narrativa del reporte", styles["section"]))
    story.append(Paragraph(escape(narrative_summary), styles["body"]))

    story.extend(_list_story("Recomendaciones de mejora argumentativa", recommendations, styles))
    story.append(PageBreak())

    story.extend(_dataframe_table(
        "Tabla de marcos éticos",
        framework_scores_df,
        styles,
        col_widths=[CONTENT_WIDTH * 0.62, CONTENT_WIDTH * 0.38],
    ))
    story.append(Spacer(1, 10))
    story.extend(_dataframe_table(
        "Tabla de estadios morales",
        stage_scores_df,
        styles,
        col_widths=[CONTENT_WIDTH * 0.4, CONTENT_WIDTH * 0.6],
    ))
    story.append(Spacer(1, 10))

    for title, caption, fig in figures:
        story.extend(_figure_story(title, caption, fig, styles))
        story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("Interpretación IA integrada", styles["section"]))
    if interpretation_result:
        model_name = interpretation_result.get("_model", "No informado")
        story.append(Paragraph(escape(f"Modelo utilizado: {model_name}"), styles["small"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(escape(_safe_text(interpretation_result.get("resumen_ejecutivo"))), styles["body"]))
        story.extend(_dataframe_table(
            "Tabla analítica de interpretación",
            pd.DataFrame(interpretation_result.get("tabla_analitica", [])),
            styles,
            col_widths=[60, 85, 92, 78, 65, 143],
            truncate_columns=[
                "dimension",
                "hallazgo_principal",
                "interpretacion",
                "riesgo_asociado",
                "nivel_atencion",
                "recomendacion",
            ],
            max_rows=12,
        ))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Lecturas especializadas", styles["section"]))
        story.append(Paragraph(escape(_safe_text(interpretation_result.get("interpretacion_etica"))), styles["body"]))
        story.append(Paragraph(escape(_safe_text(interpretation_result.get("interpretacion_legal"))), styles["body"]))
        story.append(Paragraph(escape(_safe_text(interpretation_result.get("interpretacion_bioetica"))), styles["body"]))
        story.extend(_list_story("Consideraciones clave", interpretation_result.get("consideraciones_clave", []), styles))
        story.extend(_list_story("Fortalezas argumentativas", interpretation_result.get("fortalezas_argumentativas", []), styles))
        story.extend(_list_story("Debilidades argumentativas", interpretation_result.get("debilidades_argumentativas", []), styles))
        story.extend(_list_story("Recomendaciones formativas IA", interpretation_result.get("recomendaciones_formativas", []), styles))
        risks_df = pd.DataFrame(interpretation_result.get("riesgos", []))
        story.extend(_dataframe_table(
            "Matriz de riesgos y alertas",
            risks_df,
            styles,
            col_widths=[CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.56, CONTENT_WIDTH * 0.22],
            truncate_columns=["riesgo", "descripcion", "nivel"],
            max_rows=12,
        ))
    else:
        story.append(Paragraph(
            escape(_safe_text(interpretation_note or "No fue posible integrar una interpretación IA en este reporte.")),
            styles["body"],
        ))

    story.append(PageBreak())
    story.extend(_dataframe_table(
        "Detalle de respuestas del participante",
        choice_detail_df,
        styles,
        col_widths=[40, 92, 126, 64, 70, 131],
        truncate_columns=["dilema", "opcion", "nivel_moral", "marco_etico", "justificacion"],
        max_rows=20,
    ))

    doc.build(
        story,
        onFirstPage=lambda canvas, document: _draw_page_frame(canvas, document, app_title, app_brand_line),
        onLaterPages=lambda canvas, document: _draw_page_frame(canvas, document, app_title, app_brand_line),
    )
    buffer.seek(0)
    return buffer.getvalue()
