from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _render_list_block(title: str, values: List[str]) -> None:
    st.markdown(f"#### {title}")
    if not values:
        st.info("Sin elementos disponibles para esta sección.")
        return
    for value in values:
        st.write(f"- {value}")


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

    st.markdown("#### Resumen ejecutivo")
    st.write(result.get("resumen_ejecutivo", "Sin contenido."))

    st.markdown("#### Interpretación ética")
    st.write(result.get("interpretacion_etica", "Sin contenido."))

    st.markdown("#### Interpretación legal")
    st.write(result.get("interpretacion_legal", "Sin contenido."))

    st.markdown("#### Interpretación bioética")
    st.write(result.get("interpretacion_bioetica", "Sin contenido."))

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