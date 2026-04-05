from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _render_text_card(title: str, content: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{content or 'Sin contenido.'}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_list_block(title: str, values: List[str]) -> None:
    st.markdown(f"#### {title}")
    if not values:
        st.info("Sin elementos disponibles para esta sección.")
        return
    for value in values:
        st.markdown(
            f"""
            <div class="section-card" style="padding:0.75rem 1rem; margin-bottom:0.55rem;">
                <p style="margin:0;">{value}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _interpretation_to_markdown(result: Dict[str, Any]) -> str:
    lines = [
        "# Interpretación IA",
        "",
        "## Resumen ejecutivo",
        str(result.get("resumen_ejecutivo", "")),
        "",
        "## Interpretación ética",
        str(result.get("interpretacion_etica", "")),
        "",
        "## Interpretación legal",
        str(result.get("interpretacion_legal", "")),
        "",
        "## Interpretación bioética",
        str(result.get("interpretacion_bioetica", "")),
        "",
        "## Riesgos y alertas",
    ]
    for risk in result.get("riesgos", []):
        lines.append(
            f"- {risk.get('riesgo', '')} | Nivel: {risk.get('nivel', '')} | {risk.get('descripcion', '')}"
        )
    lines.extend(["", "## Consideraciones clave"])
    for item in result.get("consideraciones_clave", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Fortalezas argumentativas"])
    for item in result.get("fortalezas_argumentativas", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Debilidades argumentativas"])
    for item in result.get("debilidades_argumentativas", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Recomendaciones formativas"])
    for item in result.get("recomendaciones_formativas", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Tabla analítica final"])
    for row in result.get("tabla_analitica", []):
        lines.append(
            "- "
            f"Dimensión: {row.get('dimension', '')} | "
            f"Hallazgo: {row.get('hallazgo_principal', '')} | "
            f"Interpretación: {row.get('interpretacion', '')} | "
            f"Riesgo asociado: {row.get('riesgo_asociado', '')} | "
            f"Nivel de atención: {row.get('nivel_atencion', '')} | "
            f"Recomendación: {row.get('recomendacion', '')}"
        )
    return "\n".join(lines)


def _interpretation_to_csv_bytes(result: Dict[str, Any]) -> bytes:
    rows = [
        {"seccion": "resumen_ejecutivo", "contenido": result.get("resumen_ejecutivo", "")},
        {"seccion": "interpretacion_etica", "contenido": result.get("interpretacion_etica", "")},
        {"seccion": "interpretacion_legal", "contenido": result.get("interpretacion_legal", "")},
        {"seccion": "interpretacion_bioetica", "contenido": result.get("interpretacion_bioetica", "")},
        {"seccion": "consideraciones_clave", "contenido": json.dumps(result.get("consideraciones_clave", []), ensure_ascii=False)},
        {"seccion": "fortalezas_argumentativas", "contenido": json.dumps(result.get("fortalezas_argumentativas", []), ensure_ascii=False)},
        {"seccion": "debilidades_argumentativas", "contenido": json.dumps(result.get("debilidades_argumentativas", []), ensure_ascii=False)},
        {"seccion": "recomendaciones_formativas", "contenido": json.dumps(result.get("recomendaciones_formativas", []), ensure_ascii=False)},
        {"seccion": "riesgos", "contenido": json.dumps(result.get("riesgos", []), ensure_ascii=False)},
        {"seccion": "tabla_analitica", "contenido": json.dumps(result.get("tabla_analitica", []), ensure_ascii=False)},
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _render_download_buttons(result: Dict[str, Any], title: str) -> None:
    export_slug = title.strip().lower().replace(" ", "_")
    col1, col2 = st.columns(2)
    col1.download_button(
        "Descargar interpretación CSV",
        data=_interpretation_to_csv_bytes(result),
        file_name=f"{export_slug}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        "Descargar interpretación Markdown",
        data=_interpretation_to_markdown(result).encode("utf-8"),
        file_name=f"{export_slug}.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _render_risk_cards(risks: List[Dict[str, Any]]) -> None:
    st.markdown("#### Riesgos y alertas")
    if not risks:
        st.info("Sin riesgos identificados para esta sección.")
        return
    for risk in risks:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>{risk.get('riesgo', 'Riesgo')}</h3>
                <p><strong>Nivel:</strong> {risk.get('nivel', 'No definido')}</p>
                <p>{risk.get('descripcion', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_interpretation_report(result: Dict[str, Any], title: str) -> None:
    st.subheader(title)
    st.caption(f"Modelo OpenAI utilizado: {result.get('_model', 'No informado')}")
    _render_download_buttons(result, title)

    _render_text_card("Resumen ejecutivo", result.get("resumen_ejecutivo", "Sin contenido."))
    _render_text_card("Interpretación ética", result.get("interpretacion_etica", "Sin contenido."))
    _render_text_card("Interpretación legal", result.get("interpretacion_legal", "Sin contenido."))
    _render_text_card("Interpretación bioética", result.get("interpretacion_bioetica", "Sin contenido."))

    _render_risk_cards(result.get("riesgos", []))
    _render_list_block("Consideraciones clave", result.get("consideraciones_clave", []))
    _render_list_block("Fortalezas argumentativas", result.get("fortalezas_argumentativas", []))
    _render_list_block("Debilidades argumentativas", result.get("debilidades_argumentativas", []))
    _render_list_block("Recomendaciones formativas", result.get("recomendaciones_formativas", []))

    st.markdown("#### Tabla analítica final")
    analytic_rows = result.get("tabla_analitica", [])
    if analytic_rows:
        st.dataframe(pd.DataFrame(analytic_rows), use_container_width=True)
    else:
        st.info("Sin tabla analítica disponible.")