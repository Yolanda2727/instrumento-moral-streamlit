from __future__ import annotations

import hashlib
import hmac
import os
import re
import unicodedata
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from admin_reports import load_admin_report_store
from analysis_module import (
    argumentative_pattern_table,
    automatic_interpretive_synthesis,
    build_quantitative_report,
    clean_text_series,
    cluster_thematic_justifications,
    extract_keywords_by_group,
    internal_consistency_estimate,
    profession_interpretive_trends,
)
from persistence import COLUMNS, load_persistence_store
from services.openai_interpreter import OpenAIInterpreterError, interpret_payload
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.pdf_export import build_individual_report_pdf
from utils.render_interpretation import render_interpretation_report

# =========================
# Configuración general
# =========================
st.set_page_config(
    page_title="ETHOSCOPE — Analítica de Razonamiento Moral",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
APP_BRAND_NAME = "ETHOSCOPE"
APP_BRAND_LINE = "Analítica de razonamiento moral y deliberación ética aplicada"
APP_TITLE = "ETHOSCOPE"
APP_SUBTITLE = (
    "Plataforma académica para explorar dilemas profesionales, marcos éticos y patrones de argumentación "
    "en estudiantes, docentes e investigadores."
)
DEFAULT_ROUTE_SIZE = 8
MIN_JUSTIFICATION_CHARS = 25
ADMIN_SESSION_KEY = "admin_authenticated"
INDIVIDUAL_REPORT_SESSION_KEY = "latest_individual_report"
MIN_AGE = 16
MAX_AGE = 85
MAX_CHILDREN = 20
MAX_WORK_HOURS = 80
AUTHOR_NAME = "Profesor Anderson Díaz Pérez"
AUTHOR_CREDENTIALS = [
    "Doctor en Bioética",
    "Doctor en Salud Pública",
    "Magíster en Ciencias Básicas Biomédicas",
    "Especialista en Inteligencia Artificial",
    "Profesional en Instrumentación Quirúrgica",
]
MAIN_FUNCTION = (
    "Valorar de manera formativa cómo estudiantes y profesionales argumentan frente a dilemas éticos, "
    "integrando razonamiento moral, marcos éticos y calidad básica de la justificación escrita."
)
PROGRAM_OBJECTIVES = [
    "Fortalecer la deliberación ética mediante dilemas contextualizados por área profesional.",
    "Identificar patrones argumentativos asociados a estadios morales y marcos éticos dominantes.",
    "Generar retroalimentación individual inmediata para docencia, reflexión y mejora argumentativa.",
    "Consolidar visualizaciones colectivas útiles para análisis académico, formación ética e investigación educativa exploratoria.",
]

LEGACY_CSV_PATH = Path(os.getenv("MORAL_TEST_LEGACY_CSV_PATH", os.getenv("MORAL_TEST_DATA_PATH", "data/responses.csv")))
SQLITE_PATH = Path(os.getenv("MORAL_TEST_SQLITE_PATH", "data/responses.db"))
EXPORT_DIR = Path("data/exports")
PERSISTENCE_STORE = load_persistence_store(sqlite_path=SQLITE_PATH, legacy_csv_path=LEGACY_CSV_PATH)
ADMIN_REPORT_STORE = load_admin_report_store()

BASE_STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "para", "por", "con",
    "del", "se", "que", "al", "lo", "como", "su", "sus", "si", "no", "es", "son", "o",
    "u", "mi", "tu", "yo", "me", "te", "le", "les", "ha", "han", "ser", "hacer", "porque",
    "más", "mas", "muy", "ya", "cuando", "donde", "qué", "este", "esta", "estos", "estas",
    "ese", "esa", "eso", "desde", "entre", "sin", "sobre", "ante", "bajo", "durante", "hasta",
    "contra", "todo", "toda", "todos", "todas", "uno", "dos", "tres", "pero", "también",
}

FRAMEWORKS = ["utilitarismo", "deontologia", "regla_de_oro", "virtudes", "contrato_social", "cuidado"]
FRAMEWORK_LABELS = {
    "utilitarismo": "Utilitarismo",
    "deontologia": "Deontología",
    "regla_de_oro": "Regla de oro",
    "virtudes": "Ética de virtudes",
    "contrato_social": "Contrato social",
    "cuidado": "Ética del cuidado",
}
ACADEMIC_COLOR_SEQUENCE = ["#0B1F3A", "#1C5F8D", "#2E8A99", "#C08A2B", "#7C5430", "#4F6D7A", "#8F3B3B"]
PLOTLY_EXPORT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "ethoscope",
        "height": 720,
        "width": 1280,
        "scale": 2,
    },
}

FRAMEWORK_INVENTORY = [
    ("F_U1", "utilitarismo", "Lo correcto es maximizar el bienestar total, aunque alguien salga perjudicado."),
    ("F_U2", "utilitarismo", "El mejor balance beneficios/daños suele ser moralmente preferible."),
    ("F_D1", "deontologia", "Hay actos incorrectos aunque produzcan buenos resultados; reglas no deben romperse."),
    ("F_D2", "deontologia", "Nunca usar personas solo como medio, sin importar beneficios."),
    ("F_G1", "regla_de_oro", "¿Aceptaría yo ser tratado así?"),
    ("F_G2", "regla_de_oro", "Busco reglas razonables si todos las aplicaran."),
    ("F_V1", "virtudes", "Importa qué persona me vuelvo: prudencia, justicia, compasión y honestidad."),
    ("F_V2", "virtudes", "Las virtudes pueden orientar mejor que reglas rígidas en casos complejos."),
    ("F_C1", "contrato_social", "Las reglas deben poder justificarse públicamente y revisarse por consenso."),
    ("F_C2", "contrato_social", "Debe priorizarse la transparencia y la rendición de cuentas."),
    ("F_CA1", "cuidado", "La vulnerabilidad y las relaciones importan: proteger y cuidar puede ser lo correcto."),
    ("F_CA2", "cuidado", "Debe minimizarse el sufrimiento de quienes tienen menos poder o mayor fragilidad."),
]

STAGE_TEMPLATE = {
    1: "Me preocupa principalmente el castigo o la consecuencia personal.",
    2: "Lo correcto es lo que trae el mejor beneficio práctico total.",
    3: "Lo correcto es lo que haría una buena persona según su entorno cercano.",
    4: "Seguir reglas y protocolos sostiene el orden y la confianza social.",
    5: "La decisión debe poder justificarse públicamente; importan la proporcionalidad y la revisión.",
    6: "Los principios universales de dignidad y justicia deben guiar incluso si implican costo personal.",
}


def opt(key: str, text: str, stage: int, framework: str) -> dict:
    level = "preconvencional" if stage in [1, 2] else ("convencional" if stage in [3, 4] else "postconvencional")
    return {"key": key, "text": text, "stage": stage, "level": level, "framework": framework}


def make_dilemma(
    item_id: str,
    title: str,
    prompt: str,
    options: List[dict],
    stage_statements: Dict[int, str],
    pedagogical_justification: str = "",
) -> dict:
    return {
        "id": item_id,
        "title": title,
        "prompt": prompt,
        "options": options,
        "stage_statements": stage_statements,
        "pedagogical_justification": pedagogical_justification,
    }


BANK: List[dict] = []

# Núcleo Kohlberg
BANK.append(make_dilemma(
    "K1", "Heinz (Kohlberg)",
    "Heinz no puede pagar un medicamento para salvar a su esposa. Considera robarlo. ¿Qué debería hacer?",
    [
        opt("A", "No: castigo o cárcel.", 1, "deontologia"),
        opt("B", "Sí: maximiza el beneficio familiar.", 2, "utilitarismo"),
        opt("C", "Sí: un buen esposo haría eso.", 3, "regla_de_oro"),
        opt("D", "No: romper la ley daña el orden.", 4, "deontologia"),
        opt("E", "Sí: la vida puede prevalecer sobre una ley injusta si hay justificación pública.", 5, "contrato_social"),
        opt("F", "Buscar una solución íntegra que preserve dignidad y proporcionalidad.", 6, "virtudes"),
    ],
    STAGE_TEMPLATE,
))

BANK.append(make_dilemma(
    "K2", "Bote atestado",
    "Un bote salvavidas está sobrecargado. Si nadie sale, todos mueren. ¿Qué criterio debería usarse?",
    [
        opt("A", "Seguir una regla o autoridad clara.", 4, "deontologia"),
        opt("B", "Realizar un sorteo transparente.", 5, "contrato_social"),
        opt("C", "Maximizar la supervivencia total.", 2, "utilitarismo"),
        opt("D", "Proteger primero a los más vulnerables.", 3, "cuidado"),
        opt("E", "No forzar la salida: dignidad y consentimiento son decisivos.", 6, "deontologia"),
        opt("F", "Usar criterios revisables según capacidad y rol.", 5, "utilitarismo"),
    ],
    STAGE_TEMPLATE,
))

health = [
    ("H1", "Triage: un ventilador", "Dos pacientes graves y un ventilador. Uno tiene mayor probabilidad de recuperación. ¿Qué haces?", [
        opt("A", "Asigno el ventilador donde el beneficio clínico esperado sea mayor.", 2, "utilitarismo"),
        opt("B", "Respeto el orden de llegada.", 4, "deontologia"),
        opt("C", "Priorizo a quien esté en mayor vulnerabilidad.", 3, "cuidado"),
        opt("D", "La decisión debe ser colegiada y basada en protocolo transparente.", 5, "contrato_social"),
        opt("E", "No usaría a nadie como medio; buscaría alternativas antes de excluir.", 6, "deontologia"),
        opt("F", "Equilibraría justicia, compasión y proporcionalidad.", 6, "virtudes"),
    ]),
    ("H2", "Error de medicación detectado tarde", "Detectas un error de medicación ya administrada. El daño podría ser moderado. ¿Cómo actúas?", [
        opt("A", "No informo para evitar sanción.", 1, "deontologia"),
        opt("B", "Lo manejo internamente para minimizar consecuencias.", 2, "utilitarismo"),
        opt("C", "Lo cuento al equipo porque una buena relación exige honestidad.", 3, "regla_de_oro"),
        opt("D", "Lo reporto formalmente según el protocolo de seguridad del paciente.", 4, "deontologia"),
        opt("E", "Informo con transparencia al paciente o familia y activo mitigación.", 5, "contrato_social"),
        opt("F", "Pongo primero dignidad, verdad, reparación y aprendizaje sistémico.", 6, "virtudes"),
    ]),
    ("H3", "Consentimiento: paciente confuso", "Paciente con confusión fluctuante requiere un procedimiento urgente, pero no inmediato. ¿Qué haces?", [
        opt("A", "Procedo para evitar problemas después.", 1, "utilitarismo"),
        opt("B", "Procedo si el beneficio supera el riesgo aunque no exista consentimiento pleno.", 2, "utilitarismo"),
        opt("C", "Busco familia o cuidador por una lógica de cuidado.", 3, "cuidado"),
        opt("D", "Sigo reglas legales sobre capacidad y consentimiento sustituto.", 4, "deontologia"),
        opt("E", "Solicito segunda opinión y documento la mínima intervención necesaria.", 5, "contrato_social"),
        opt("F", "Respeto autonomía y dignidad con estricta proporcionalidad.", 6, "deontologia"),
    ]),
    ("H4", "Escasez de insumos estériles", "Faltan insumos estériles críticos y se propone improvisar. ¿Qué decides?", [
        opt("A", "Improviso para que no me culpen por retrasar.", 1, "utilitarismo"),
        opt("B", "Improviso si con eso aumenta la probabilidad de éxito hoy.", 2, "utilitarismo"),
        opt("C", "Pienso en el paciente como si fuera yo.", 3, "regla_de_oro"),
        opt("D", "No improviso: los estándares de esterilidad son una obligación.", 4, "deontologia"),
        opt("E", "Escalo la decisión, dejo trazabilidad y busco alternativa segura.", 5, "contrato_social"),
        opt("F", "Primero no maleficencia y dignidad del paciente; si es inseguro, se detiene.", 6, "virtudes"),
    ]),
    ("H5", "Confidencialidad clínica", "Un familiar exige información sensible sin autorización del paciente. ¿Qué haces?", [
        opt("A", "Entrego la información para evitar conflicto.", 1, "utilitarismo"),
        opt("B", "La entrego si creo que eso ayudará al paciente.", 2, "cuidado"),
        opt("C", "Dialogo con empatía y busco acuerdo con el paciente.", 3, "cuidado"),
        opt("D", "Respeto la confidencialidad salvo excepciones legales.", 4, "deontologia"),
        opt("E", "Revelo solo lo mínimo necesario, con registro y justificación.", 5, "contrato_social"),
        opt("F", "Pongo primero autonomía y dignidad, con manejo humano y proporcional.", 6, "virtudes"),
    ]),
    ("H6", "Reporte de evento adverso", "Ocurre un evento adverso serio. ¿Cómo garantizas aprendizaje y justicia?", [
        opt("A", "Evito reportar por miedo.", 1, "deontologia"),
        opt("B", "Reporto solo si es inevitable.", 2, "utilitarismo"),
        opt("C", "Reporto porque una buena persona debe hacerlo.", 3, "regla_de_oro"),
        opt("D", "Reporto completo según la norma institucional.", 4, "deontologia"),
        opt("E", "Promuevo transparencia, análisis de causa raíz y enfoque no punitivo.", 5, "contrato_social"),
        opt("F", "Centro la respuesta en justicia, reparación y paciente.", 6, "virtudes"),
    ]),
    ("H7", "Priorización quirúrgica", "Hay 3 cirugías pendientes y solo 1 cupo hoy. ¿Cómo priorizas?", [
        opt("A", "Elijo la que me evita más problemas.", 1, "utilitarismo"),
        opt("B", "Elijo la que maximiza el beneficio global.", 2, "utilitarismo"),
        opt("C", "Elijo a la persona más vulnerable.", 3, "cuidado"),
        opt("D", "Sigo el protocolo institucional.", 4, "deontologia"),
        opt("E", "Uso criterios públicos, documentados y revisables.", 5, "contrato_social"),
        opt("F", "Equilibro justicia, compasión, urgencia e integridad.", 6, "virtudes"),
    ]),
    ("H8", "Conflicto de interés con proveedor", "Un proveedor ofrece incentivo por usar su insumo. ¿Qué haces?", [
        opt("A", "Acepto para no perder la oportunidad.", 1, "utilitarismo"),
        opt("B", "Acepto si beneficia al servicio.", 2, "utilitarismo"),
        opt("C", "Rechazo por reputación y expectativas del equipo.", 3, "regla_de_oro"),
        opt("D", "Rechazo por deber ético y normativo.", 4, "deontologia"),
        opt("E", "Declaro el conflicto y sigo compras transparentes.", 5, "contrato_social"),
        opt("F", "Mantengo integridad y justicia distributiva; no acepto incentivos.", 6, "virtudes"),
    ]),
]

law = [
    ("L1", "Confidencialidad vs riesgo a tercero", "Recibes información que sugiere riesgo serio e inminente para un tercero. ¿Cómo actúas?", [
        opt("A", "No digo nada para evitar problemas.", 1, "deontologia"),
        opt("B", "Actúo según la consecuencia práctica esperada.", 2, "utilitarismo"),
        opt("C", "Busco consejo informal para mantener relaciones.", 3, "regla_de_oro"),
        opt("D", "Sigo la ley y el protocolo de excepciones.", 4, "deontologia"),
        opt("E", "Aplico proporcionalidad: mínima revelación, registro y protección de derechos.", 5, "contrato_social"),
        opt("F", "Protejo dignidad y vida con prudencia.", 6, "virtudes"),
    ]),
    ("L2", "Prueba obtenida irregularmente", "Te llega evidencia clave obtenida de forma irregular. ¿La usas?", [
        opt("A", "La uso si me conviene.", 1, "utilitarismo"),
        opt("B", "La uso si mejora el resultado del caso.", 2, "utilitarismo"),
        opt("C", "La uso si todos lo hacen.", 3, "regla_de_oro"),
        opt("D", "No la uso: debo proceso y legalidad importan.", 4, "deontologia"),
        opt("E", "Evalúo admisibilidad, garantías y justificación pública.", 5, "contrato_social"),
        opt("F", "No corroería el sistema por un resultado inmediato.", 6, "deontologia"),
    ]),
    ("L3", "Asignación de ayudas", "Tienes recursos limitados para ayudas comunitarias. ¿Cómo asignas?", [
        opt("A", "A quien me evite conflicto.", 1, "utilitarismo"),
        opt("B", "A quien permita maximizar el impacto total.", 2, "utilitarismo"),
        opt("C", "Priorizaría a los más vulnerables.", 3, "cuidado"),
        opt("D", "Aplicaría criterios normativos fijos.", 4, "deontologia"),
        opt("E", "Usaría criterios públicos, participativos y revisables.", 5, "contrato_social"),
        opt("F", "Mantendría el eje en justicia, dignidad y proporcionalidad.", 6, "virtudes"),
    ]),
    ("L4", "Plagio detectado", "Detectas plagio y el estudiante suplica una excepción por graduación. ¿Qué haces?", [
        opt("A", "Lo dejo pasar para evitar problemas.", 1, "utilitarismo"),
        opt("B", "Hago una excepción si creo que no afecta a nadie.", 2, "utilitarismo"),
        opt("C", "Dialogo de forma formativa por cuidado y aprendizaje.", 3, "cuidado"),
        opt("D", "Aplico el reglamento.", 4, "deontologia"),
        opt("E", "Garantizo debido proceso académico y posibilidades de reparación.", 5, "contrato_social"),
        opt("F", "Sostengo integridad y justicia, sin abandonar el acompañamiento.", 6, "virtudes"),
    ]),
    ("L5", "Investigación social con datos sensibles", "Vas a publicar resultados con datos sensibles. ¿Cómo gestionas anonimización y riesgo?", [
        opt("A", "Publico tal como está si me conviene.", 1, "utilitarismo"),
        opt("B", "Anonimizo lo mínimo para publicar rápido.", 2, "utilitarismo"),
        opt("C", "Consulto a participantes por cuidado y respeto.", 3, "cuidado"),
        opt("D", "Cumplo estrictamente la normativa de datos.", 4, "deontologia"),
        opt("E", "Hago evaluación de impacto, consentimiento y transparencia.", 5, "contrato_social"),
        opt("F", "Priorizo no daño, dignidad y justicia epistémica.", 6, "virtudes"),
    ]),
    ("L6", "Presión en mediación", "Te presionan para favorecer a un actor poderoso en una mediación. ¿Qué haces?", [
        opt("A", "Cedo para evitar represalias.", 1, "utilitarismo"),
        opt("B", "Cedo si eso trae paz rápida.", 2, "utilitarismo"),
        opt("C", "Busco equilibrio por regla de oro.", 3, "regla_de_oro"),
        opt("D", "Mantengo neutralidad, procedimiento y reglas.", 4, "deontologia"),
        opt("E", "Aseguro transparencia, participación y control.", 5, "contrato_social"),
        opt("F", "Defiendo la dignidad de la parte vulnerable.", 6, "virtudes"),
    ]),
    ("L7", "Conflicto de interés profesional", "Tu cliente ofrece un extra si aceleras un trámite mediante un atajo. ¿Qué haces?", [
        opt("A", "Acepto por beneficio personal.", 1, "utilitarismo"),
        opt("B", "Acepto si nadie se entera.", 2, "utilitarismo"),
        opt("C", "Rechazo por reputación.", 3, "regla_de_oro"),
        opt("D", "Rechazo por deber legal y ético.", 4, "deontologia"),
        opt("E", "Declaro el conflicto y sigo canales formales.", 5, "contrato_social"),
        opt("F", "Sostengo integridad y justicia; no uso atajos.", 6, "virtudes"),
    ]),
    ("L8", "Denuncia con riesgo de represalia", "Observas una práctica institucional injusta y denunciar implica riesgo. ¿Qué haces?", [
        opt("A", "Callo para protegerme.", 1, "utilitarismo"),
        opt("B", "Denuncio solo si me conviene.", 2, "utilitarismo"),
        opt("C", "Busco apoyo colectivo y relacional.", 3, "cuidado"),
        opt("D", "Sigo el procedimiento institucional.", 4, "deontologia"),
        opt("E", "Uso canales con protección, evidencia y transparencia.", 5, "contrato_social"),
        opt("F", "Actúo con valentía prudente en defensa de justicia y dignidad.", 6, "virtudes"),
    ]),
]

it_cases = [
    ("T1", "Seguridad vs costo", "Detectas una falla que puede causar accidentes. Corregirla retrasa y cuesta. Te piden callar. ¿Qué haces?", [
        opt("A", "Callo para no perder el trabajo.", 1, "utilitarismo"),
        opt("B", "Hago un arreglo mínimo sin frenar el proyecto.", 2, "utilitarismo"),
        opt("C", "Lo reporto por responsabilidad con otros.", 3, "regla_de_oro"),
        opt("D", "Realizo un reporte formal porque la seguridad es un deber.", 4, "deontologia"),
        opt("E", "Uso canal ético con evidencia y transparencia.", 5, "contrato_social"),
        opt("F", "No acepto poner vidas en riesgo.", 6, "deontologia"),
    ]),
    ("T2", "Sesgo algorítmico detectado", "Encuentras sesgo contra un grupo vulnerable. Corregirlo reduce desempeño y retrasa entrega. ¿Qué haces?", [
        opt("A", "Lo ignoro para cumplir.", 1, "utilitarismo"),
        opt("B", "Hago un ajuste mínimo para no afectar métricas.", 2, "utilitarismo"),
        opt("C", "Consulto al equipo desde una lógica de equidad relacional.", 3, "cuidado"),
        opt("D", "Cumplo políticas y estándares de equidad.", 4, "deontologia"),
        opt("E", "Hago auditoría, documento y evalúo impacto.", 5, "contrato_social"),
        opt("F", "La justicia y la no discriminación prevalecen sobre optimización ciega.", 6, "virtudes"),
    ]),
    ("T3", "Dataset con datos personales", "Te ofrecen un dataset con datos personales sin consentimiento claro para entrenar un modelo. ¿Qué haces?", [
        opt("A", "Lo uso si nadie se entera.", 1, "utilitarismo"),
        opt("B", "Lo anonimizo rápido y lo uso.", 2, "utilitarismo"),
        opt("C", "Pido consentimiento por cuidado y respeto.", 3, "cuidado"),
        opt("D", "Lo rechazo por normativa de datos.", 4, "deontologia"),
        opt("E", "Aplico evaluación de impacto, minimización y gobernanza.", 5, "contrato_social"),
        opt("F", "Protejo dignidad y autonomía informacional.", 6, "deontologia"),
    ]),
    ("T4", "Vulnerabilidad crítica", "Descubres una vulnerabilidad crítica en un sistema público. ¿Cómo reportas?", [
        opt("A", "La publico para ganar reputación.", 2, "utilitarismo"),
        opt("B", "La oculto para evitar problemas.", 1, "utilitarismo"),
        opt("C", "Aviso al responsable y acompaño la corrección.", 3, "regla_de_oro"),
        opt("D", "Sigo el protocolo de divulgación responsable.", 4, "deontologia"),
        opt("E", "Combino transparencia con tiempos prudentes y protección pública.", 5, "contrato_social"),
        opt("F", "Prevengo daño masivo con prudencia.", 6, "virtudes"),
    ]),
    ("T5", "IA clínica: explicación vs caja negra", "Un modelo caja negra rinde mejor, pero uno explicable rinde un poco peor. ¿Cuál despliegas?", [
        opt("A", "Caja negra si tiene más accuracy.", 2, "utilitarismo"),
        opt("B", "Caja negra para cumplir plazos.", 1, "utilitarismo"),
        opt("C", "Busco equilibrio por la confianza de los usuarios.", 3, "cuidado"),
        opt("D", "Sigo estándares de explicabilidad y seguridad.", 4, "deontologia"),
        opt("E", "Hago piloto con monitoreo, evaluación pública y gobernanza.", 5, "contrato_social"),
        opt("F", "No expondría vidas a un sistema opaco sin control suficiente.", 6, "virtudes"),
    ]),
    ("T6", "Filtración de datos", "Ocurre una filtración y el equipo sugiere ocultarla. ¿Qué haces?", [
        opt("A", "La oculto para evitar sanción.", 1, "utilitarismo"),
        opt("B", "La oculto si el daño parece pequeño.", 2, "utilitarismo"),
        opt("C", "Aviso internamente por cuidado con los afectados.", 3, "cuidado"),
        opt("D", "Notifico según política y ley.", 4, "deontologia"),
        opt("E", "Promuevo transparencia, notificación a afectados y mitigación.", 5, "contrato_social"),
        opt("F", "Asumo responsabilidad, reparación y respeto por la dignidad.", 6, "virtudes"),
    ]),
    ("T7", "Scraping prohibido", "Para un proyecto se propone scraping que viola términos o límites éticos. ¿Qué haces?", [
        opt("A", "Lo hago por resultados.", 2, "utilitarismo"),
        opt("B", "Lo hago si nadie se entera.", 1, "utilitarismo"),
        opt("C", "Propongo una alternativa respetuosa.", 3, "regla_de_oro"),
        opt("D", "No lo hago: debo cumplir normas y licencias.", 4, "deontologia"),
        opt("E", "Gestiono permisos, consentimiento y gobernanza.", 5, "contrato_social"),
        opt("F", "Respeto derechos y justicia informacional.", 6, "virtudes"),
    ]),
    ("T8", "Manipulación de métricas", "Te piden presentar métricas mejoradas sin base real. ¿Qué haces?", [
        opt("A", "Las manipulo para quedar bien.", 1, "utilitarismo"),
        opt("B", "Ajusto la narrativa y omito limitaciones.", 2, "utilitarismo"),
        opt("C", "Me niego por honestidad profesional.", 3, "virtudes"),
        opt("D", "No lo hago: la integridad profesional lo impide.", 4, "deontologia"),
        opt("E", "Exijo transparencia metodológica y revisión.", 5, "contrato_social"),
        opt("F", "Defiendo verdad y justicia en decisiones críticas.", 6, "virtudes"),
    ]),
]

lab_cases = [
    (
        "BM1",
        "Bioseguridad en cabina y exposición ocupacional",
        "Durante el procesamiento de una muestra respiratoria con sospecha de tuberculosis, un compañero propone abrir el contenedor fuera de la cabina para ganar tiempo, aunque eso rompe el protocolo y aumenta el riesgo de aerosolización. ¿Qué haces?",
        [
            opt("A", "Acepto para no retrasar el turno ni tener problemas con el coordinador.", 1, "utilitarismo"),
            opt("B", "Acepto si así el laboratorio libera resultados más rápido para varios pacientes.", 2, "utilitarismo"),
            opt("C", "Le pido que no lo haga porque nadie querría exponerse de esa manera.", 3, "regla_de_oro"),
            opt("D", "Detengo el procedimiento y exijo cumplir la bioseguridad y el uso de cabina certificada.", 4, "deontologia"),
            opt("E", "Escalo el incidente, documento el riesgo y reviso condiciones del proceso con el equipo.", 5, "contrato_social"),
            opt("F", "Protejo la vida del personal y de terceros aunque eso implique confrontar la presión operativa.", 6, "virtudes"),
        ],
        "Fortalece la deliberación sobre deberes de bioseguridad, protección mutua y liderazgo prudente frente a prácticas inseguras normalizadas.",
    ),
    (
        "BM2",
        "Transporte de muestras y error preanalítico",
        "Una muestra para hemocultivo llega sin cadena de frío documentada y con demora significativa en el transporte. El servicio clínico presiona para procesarla sin observaciones porque el paciente está inestable. ¿Cómo procedes?",
        [
            opt("A", "La proceso sin decir nada para evitar conflicto con el servicio solicitante.", 1, "utilitarismo"),
            opt("B", "La proceso y omito la novedad si eso agiliza una decisión terapéutica.", 2, "utilitarismo"),
            opt("C", "Llamo al servicio y explico el problema pensando en cómo me gustaría que manejaran una muestra mía.", 3, "cuidado"),
            opt("D", "Registro la no conformidad y sigo el protocolo preanalítico antes de liberar cualquier interpretación.", 4, "deontologia"),
            opt("E", "Coordino una nueva toma o una liberación condicionada con trazabilidad y justificación compartida.", 5, "contrato_social"),
            opt("F", "Priorizo veracidad analítica y no maleficencia: un dato dudoso no debe disfrazarse de certeza.", 6, "virtudes"),
        ],
        "Permite discutir calidad preanalítica, trazabilidad y el equilibrio entre urgencia clínica y validez técnica del resultado.",
    ),
    (
        "BM3",
        "Contaminación cruzada entre cultivos",
        "Detectas que una placa pudo contaminarse al manipularse muy cerca de otra muestra positiva de alto impacto epidemiológico. Si repites el procedimiento, el reporte se retrasará y el laboratorio perderá un indicador de oportunidad. ¿Qué haces?",
        [
            opt("A", "Mantengo el cultivo como está para no afectar el indicador del laboratorio.", 1, "utilitarismo"),
            opt("B", "Mantengo el resultado si probablemente beneficie la decisión clínica inmediata.", 2, "utilitarismo"),
            opt("C", "Pido una segunda revisión porque una colega responsable no debería cargar sola con esa duda.", 3, "cuidado"),
            opt("D", "Anulo o repito la prueba según el protocolo de calidad y contaminación cruzada.", 4, "deontologia"),
            opt("E", "Documento el evento, notifico control de calidad y explico el retraso con transparencia.", 5, "contrato_social"),
            opt("F", "Prefiero una demora honesta a sostener un resultado potencialmente falso que pueda dañar a un paciente.", 6, "virtudes"),
        ],
        "Desarrolla juicio sobre integridad analítica, cultura de seguridad y decisiones bajo tensión entre calidad y productividad.",
    ),
    (
        "BM4",
        "Confidencialidad y hallazgo crítico sensible",
        "Obtienes un resultado preliminar compatible con una infección transmisible de alto impacto social. Un directivo administrativo pide conocer el nombre del paciente antes de que el resultado crítico sea comunicado por el canal clínico establecido. ¿Qué haces?",
        [
            opt("A", "Entrego la información para evitar problemas con la dirección.", 1, "utilitarismo"),
            opt("B", "La entrego si creo que eso podría acelerar medidas institucionales.", 2, "utilitarismo"),
            opt("C", "Busco una salida cuidadosa que informe lo necesario sin exponer de más al paciente.", 3, "cuidado"),
            opt("D", "Mantengo la confidencialidad y comunico el hallazgo crítico solo por la ruta autorizada.", 4, "deontologia"),
            opt("E", "Escalo la decisión con registro, mínima revelación necesaria y protección de derechos.", 5, "contrato_social"),
            opt("F", "Protejo simultáneamente dignidad, privacidad y seguridad pública con proporcionalidad estricta.", 6, "virtudes"),
        ],
        "Integra confidencialidad, reporte de hallazgos críticos y manejo proporcional de información sensible en microbiología clínica.",
    ),
    (
        "BM5",
        "Resistencia antimicrobiana y reporte selectivo",
        "El antibiograma muestra un perfil de multirresistencia inesperado. Un médico cercano al laboratorio sugiere reportar solo antibióticos 'menos alarmantes' para evitar restricciones institucionales y discusiones con la familia. ¿Cómo respondes?",
        [
            opt("A", "Acepto para evitar conflicto con el médico solicitante.", 1, "utilitarismo"),
            opt("B", "Acepto si eso reduce ansiedad y permite iniciar tratamiento rápido.", 2, "utilitarismo"),
            opt("C", "Dialogo con el médico y explico que todos merecen un reporte honesto y responsable.", 3, "regla_de_oro"),
            opt("D", "Reporto el perfil real según lineamientos de susceptibilidad y vigilancia.", 4, "deontologia"),
            opt("E", "Activo el circuito de stewardship y documento por qué el reporte debe ser completo y trazable.", 5, "contrato_social"),
            opt("F", "Defiendo verdad, salud pública y justicia intergeneracional frente a la resistencia antimicrobiana.", 6, "virtudes"),
        ],
        "Permite analizar la responsabilidad del laboratorio en la contención de resistencia antimicrobiana y en la veracidad de la información clínica.",
    ),
    (
        "BM6",
        "Errores analíticos y postanalíticos en cadena",
        "Tras liberar un resultado negativo, descubres que el control interno del equipo falló y además el informe ya fue enviado al sistema sin comentario correctivo. El turno está saturado y te sugieren esperar al día siguiente para revisar. ¿Qué haces?",
        [
            opt("A", "Espero y veo si alguien más detecta el problema mañana.", 1, "deontologia"),
            opt("B", "Espero si con eso evito caos operativo y quizá el impacto clínico sea bajo.", 2, "utilitarismo"),
            opt("C", "Aviso al equipo inmediato porque una buena práctica cuida también a colegas y pacientes.", 3, "cuidado"),
            opt("D", "Activo corrección, bloqueo del resultado y reproceso según el procedimiento analítico y postanalítico.", 4, "deontologia"),
            opt("E", "Informo al servicio tratante, documento el error y abro análisis de causa con trazabilidad completa.", 5, "contrato_social"),
            opt("F", "Asumo responsabilidad temprana para evitar daño clínico por una falsa seguridad diagnóstica.", 6, "virtudes"),
        ],
        "Conecta errores analíticos y postanalíticos con responsabilidad profesional, cultura justa y prevención de daño derivado del reporte.",
    ),
    (
        "BM7",
        "Uso ético de cultivos y microorganismos",
        "Un investigador externo pide acceso rápido a una cepa aislada en el laboratorio para un ensayo no aprobado todavía por el comité correspondiente. Asegura que el uso será 'solo preliminar' y que después regularizará los permisos. ¿Qué haces?",
        [
            opt("A", "Se la entrego para no cerrar una oportunidad académica.", 1, "utilitarismo"),
            opt("B", "La entrego si eso podría generar un hallazgo útil pronto.", 2, "utilitarismo"),
            opt("C", "Le propongo esperar y tramitar lo necesario porque nadie querría que usaran su material biológico sin garantías.", 3, "regla_de_oro"),
            opt("D", "No entrego la cepa sin aprobación, custodia y condiciones de bioseguridad formalmente verificadas.", 4, "deontologia"),
            opt("E", "Gestiono acceso institucional con permisos, trazabilidad, evaluación de riesgo y acuerdo de uso.", 5, "contrato_social"),
            opt("F", "Protejo la responsabilidad científica y social del laboratorio frente al manejo de microorganismos.", 6, "virtudes"),
        ],
        "Sitúa al estudiante frente al uso responsable de cultivos, la gobernanza del material biológico y la relación entre ciencia, riesgo y control institucional.",
    ),
    (
        "BM8",
        "Investigación con muestras biológicas y presión institucional",
        "En un estudio con muestras remanentes aparece un hallazgo que puede afectar la reputación de la institución. La dirección propone retrasar, suavizar o reagrupar los resultados antes de informar al comité y a los investigadores principales. ¿Cómo actúas?",
        [
            opt("A", "Acepto para evitar problemas laborales con la institución.", 1, "utilitarismo"),
            opt("B", "Acepto si eso protege temporalmente la imagen institucional y la continuidad del proyecto.", 2, "utilitarismo"),
            opt("C", "Busco una conversación cuidadosa con el equipo para evitar injusticias y daño a participantes.", 3, "cuidado"),
            opt("D", "Mantengo los datos tal como surgieron y sigo el protocolo de investigación y reporte.", 4, "deontologia"),
            opt("E", "Activo comité, trazabilidad metodológica y comunicación transparente con resguardo de participantes.", 5, "contrato_social"),
            opt("F", "Pongo por delante verdad científica, respeto por las muestras y dignidad de quienes confiaron en la investigación.", 6, "virtudes"),
        ],
        "Integra ética de investigación, custodia de muestras biológicas y resistencia a presiones institucionales que distorsionan evidencia.",
    ),
    (
        "BM9",
        "Bacteriología: probable contaminante en hemocultivo",
        "Dos botellas de hemocultivo muestran crecimiento de un microorganismo que podría ser contaminante cutáneo o bacteriemia real. El servicio clínico exige que lo reportes como infección confirmada para justificar antibióticos de amplio espectro. ¿Qué haces?",
        [
            opt("A", "Lo reporto como infección confirmada para evitar discusiones con el clínico.", 1, "utilitarismo"),
            opt("B", "Lo reporto como infección si eso facilita una intervención rápida aunque la evidencia sea incompleta.", 2, "utilitarismo"),
            opt("C", "Dialogo con el equipo tratante y con el laboratorio para valorar el contexto antes de afirmar algo.", 3, "cuidado"),
            opt("D", "Reporto el hallazgo con la interpretación técnica que corresponde y sugiero correlación clínica.", 4, "deontologia"),
            opt("E", "Documento la incertidumbre, recomiendo confirmación y sostengo trazabilidad interpretativa.", 5, "contrato_social"),
            opt("F", "Prefiero una verdad prudente antes que una certeza ficticia que exponga al paciente a daño innecesario.", 6, "virtudes"),
        ],
        "Entrena la distinción entre contaminación y bacteriemia, así como la responsabilidad interpretativa del bacteriólogo frente al uso de antibióticos.",
    ),
    (
        "BM10",
        "Bacteriología: rechazo de muestra mal rotulada",
        "Recibes una muestra para coprocultivo con rotulación dudosa y datos clínicos incompletos. El personal asistencial pide no rechazarla porque repetir la toma será incómodo para el paciente y afectará los tiempos del servicio. ¿Qué haces?",
        [
            opt("A", "La proceso sin objeción para evitar reclamos del servicio.", 1, "utilitarismo"),
            opt("B", "La proceso si eso ahorra tiempo, aunque la identificación no sea completamente segura.", 2, "utilitarismo"),
            opt("C", "Intento aclarar la identificación con el servicio porque nadie querría ser diagnosticado con una muestra incierta.", 3, "regla_de_oro"),
            opt("D", "Aplico el criterio de rechazo o regularización formal según el protocolo de identificación de muestras.", 4, "deontologia"),
            opt("E", "Gestiono solución trazable con nueva toma o validación documental extraordinaria debidamente registrada.", 5, "contrato_social"),
            opt("F", "Defiendo la validez de la muestra como condición ética mínima para un reporte clínico responsable.", 6, "virtudes"),
        ],
        "Permite trabajar identificación inequívoca, calidad preanalítica y justicia diagnóstica en bacteriología clínica.",
    ),
    (
        "BM11",
        "Microbiología: cepa multirresistente en docencia",
        "Un docente quiere usar una cepa multirresistente de colección en una práctica con estudiantes antes de completar la verificación de condiciones de contención y entrenamiento. Argumenta que la oportunidad académica no puede perderse. ¿Qué haces?",
        [
            opt("A", "Acepto para no contrariar al docente responsable.", 1, "utilitarismo"),
            opt("B", "Acepto si la práctica beneficiará a muchos estudiantes y el riesgo parece bajo.", 2, "utilitarismo"),
            opt("C", "Propongo otra actividad temporal más segura porque nadie debería aprender exponiéndose indebidamente.", 3, "cuidado"),
            opt("D", "No autorizo el uso hasta cumplir bioseguridad, entrenamiento y contención requeridos.", 4, "deontologia"),
            opt("E", "Solicito evaluación de riesgo, aval formal y plan de contingencia antes de cualquier uso docente.", 5, "contrato_social"),
            opt("F", "La formación ética también exige proteger a estudiantes, personal y comunidad frente a microorganismos de alto riesgo.", 6, "virtudes"),
        ],
        "Aborda uso ético de microorganismos, formación segura y responsabilidad institucional en microbiología aplicada y docente.",
    ),
    (
        "BM12",
        "Microbiología: secuenciación y comunicación de brote",
        "La secuenciación preliminar sugiere un brote intrahospitalario por un mismo clon, pero la evidencia aún no es definitiva. La gerencia pide esperar silencio hasta tener confirmación total para evitar alarma reputacional. ¿Cómo actúas?",
        [
            opt("A", "Espero y no digo nada para no crear problemas institucionales.", 1, "utilitarismo"),
            opt("B", "Espero si eso evita alarma mientras se completa el análisis.", 2, "utilitarismo"),
            opt("C", "Comparto la preocupación con quienes pueden proteger a los pacientes sin exponer información innecesaria.", 3, "cuidado"),
            opt("D", "Activo la notificación técnica interna según el protocolo de vigilancia microbiológica.", 4, "deontologia"),
            opt("E", "Comunico el hallazgo como señal preliminar, con incertidumbre explícita y medidas preventivas proporcionales.", 5, "contrato_social"),
            opt("F", "La prudencia ética exige advertir a tiempo cuando el silencio puede ampliar un daño prevenible.", 6, "virtudes"),
        ],
        "Conecta microbiología molecular, vigilancia epidemiológica y comunicación responsable bajo incertidumbre.",
    ),
]

LAB_DILEMMA_CATALOG = [
    {"id": "BM1", "ruta_sugerida": "Microbiología", "foco": "Bioseguridad y exposición ocupacional"},
    {"id": "BM2", "ruta_sugerida": "Bacteriología", "foco": "Transporte de muestras y error preanalítico"},
    {"id": "BM3", "ruta_sugerida": "Bacteriología / Microbiología", "foco": "Contaminación cruzada y control de calidad"},
    {"id": "BM4", "ruta_sugerida": "Bacteriología / Microbiología", "foco": "Confidencialidad y reporte crítico"},
    {"id": "BM5", "ruta_sugerida": "Bacteriología / Microbiología", "foco": "Resistencia antimicrobiana y reporte selectivo"},
    {"id": "BM6", "ruta_sugerida": "Bacteriología", "foco": "Errores analíticos y postanalíticos"},
    {"id": "BM7", "ruta_sugerida": "Microbiología", "foco": "Uso ético de cultivos y microorganismos"},
    {"id": "BM8", "ruta_sugerida": "Bacteriología / Microbiología", "foco": "Investigación con muestras biológicas y presión institucional"},
    {"id": "BM9", "ruta_sugerida": "Bacteriología", "foco": "Interpretación de hemocultivos y antibioticoterapia"},
    {"id": "BM10", "ruta_sugerida": "Bacteriología", "foco": "Rotulación y validez de muestra"},
    {"id": "BM11", "ruta_sugerida": "Microbiología", "foco": "Uso docente de cepas multirresistentes"},
    {"id": "BM12", "ruta_sugerida": "Microbiología", "foco": "Secuenciación y comunicación de brotes"},
]

for item_id, title, prompt, options in health + law + it_cases:
    BANK.append(make_dilemma(item_id, title, prompt, options, STAGE_TEMPLATE))

for item_id, title, prompt, options, pedagogical_justification in lab_cases:
    BANK.append(make_dilemma(item_id, title, prompt, options, STAGE_TEMPLATE, pedagogical_justification))

LOOKUP = {d["id"]: d for d in BANK}
PROFESSION_DEFINITIONS = [
    {"label": "Medicina", "route_group": "medicina"},
    {"label": "Enfermería", "route_group": "enfermeria"},
    {"label": "Fisioterapia", "route_group": "fisioterapia"},
    {"label": "Instrumentación Quirúrgica", "route_group": "instrumentacion_quirurgica"},
    {"label": "Bacteriología", "route_group": "bacteriologia_laboratorio"},
    {"label": "Microbiología", "route_group": "microbiologia_laboratorio"},
    {"label": "Derecho", "route_group": "derecho"},
    {"label": "Ciencias Sociales", "route_group": "ciencias_sociales"},
    {"label": "Educación", "route_group": "educacion"},
    {"label": "Ingeniería", "route_group": "ingenieria"},
    {"label": "TI", "route_group": "ti"},
    {"label": "Datos", "route_group": "datos"},
    {"label": "Otra / Mixta", "route_group": "mixta"},
]

PROFESSION_OPTIONS = [definition["label"] for definition in PROFESSION_DEFINITIONS]

ROUTE_BANKS = {
    "salud": ["K1", "K2", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"],
    "medicina": ["K1", "K2", "H1", "H3", "H2", "H5", "H6", "H7", "H8", "H4"],
    "enfermeria": ["K1", "K2", "H2", "H5", "H6", "H3", "H1", "H7", "H4", "H8"],
    "fisioterapia": ["K1", "K2", "H3", "H5", "H7", "H2", "H6", "H4", "H1", "H8"],
    "instrumentacion_quirurgica": ["K1", "K2", "H4", "H7", "H2", "H6", "H1", "H5", "H8", "H3"],
    "bacteriologia_laboratorio": ["K1", "K2", "BM10", "BM2", "BM3", "BM9", "BM5", "BM4", "BM6", "BM8"],
    "microbiologia_laboratorio": ["K1", "K2", "BM1", "BM7", "BM3", "BM4", "BM5", "BM12", "BM11", "BM8"],
    "social_juridica": ["K1", "K2", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
    "derecho": ["K1", "K2", "L1", "L2", "L6", "L7", "L8", "L3", "L5", "L4"],
    "ciencias_sociales": ["K1", "K2", "L3", "L5", "L6", "L8", "L1", "L4", "L7", "L2"],
    "educacion": ["K1", "K2", "L4", "L3", "L5", "L8", "L6", "L1", "L2", "L7"],
    "ingenieria_ti_datos": ["K1", "K2", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
    "ingenieria": ["K1", "K2", "T1", "T4", "T5", "T8", "T2", "T6", "T3", "T7"],
    "ti": ["K1", "K2", "T4", "T6", "T3", "T7", "T8", "T1", "T2", "T5"],
    "datos": ["K1", "K2", "T2", "T3", "T5", "T8", "T6", "T7", "T4", "T1"],
    "mixta": ["K1", "K2", "H1", "L1", "T1", "H4", "L4", "T2"],
}

PROFESSION_ROUTE_GROUPS = {
    definition["label"]: definition["route_group"]
    for definition in PROFESSION_DEFINITIONS
}

LEGACY_PROFESSION_LABELS = {
    "Salud (legado)": "salud",
    "Derecho / Ciencias Sociales / Educación (legado)": "social_juridica",
    "Ingeniería / TI / Datos (legado)": "ingenieria_ti_datos",
}

PROFESSION_ALIASES = {
    "Salud (medicina/enfermería/fisioterapia/instrumentación)": "Salud (legado)",
    "Derecho / Ciencias sociales / Educación": "Derecho / Ciencias Sociales / Educación (legado)",
    "Ingeniería / TI / Datos": "Ingeniería / TI / Datos (legado)",
    "Bacteriologia": "Bacteriología",
    "Microbiologia": "Microbiología",
    "Instrumentacion Quirurgica": "Instrumentación Quirúrgica",
}

PROFESSION_ROUTES = {
    profession: ROUTE_BANKS[route_group]
    for profession, route_group in PROFESSION_ROUTE_GROUPS.items()
}


def validate_profession_routes() -> None:
    required_professions = ["Bacteriología", "Microbiología"]
    for profession in required_professions:
        route_group = PROFESSION_ROUTE_GROUPS.get(profession)
        if not route_group:
            raise ValueError(f"La profesión {profession} no tiene un grupo de ruta configurado.")

        route_ids = ROUTE_BANKS.get(route_group, [])
        if not route_ids:
            raise ValueError(f"La ruta {route_group} no tiene dilemas asignados para {profession}.")

        missing_ids = [item_id for item_id in route_ids if item_id not in LOOKUP]
        if missing_ids:
            raise ValueError(
                f"La ruta {route_group} de {profession} referencia dilemas inexistentes: {', '.join(missing_ids)}."
            )

        if len(route_ids) < DEFAULT_ROUTE_SIZE:
            raise ValueError(
                f"La ruta {route_group} de {profession} tiene {len(route_ids)} dilemas y requiere al menos {DEFAULT_ROUTE_SIZE}."
            )

    for profession, route_group in PROFESSION_ROUTE_GROUPS.items():
        route_ids = ROUTE_BANKS.get(route_group, [])
        missing_ids = [item_id for item_id in route_ids if item_id not in LOOKUP]
        if missing_ids:
            raise ValueError(
                f"La profesión {profession} tiene referencias inválidas en su ruta {route_group}: {', '.join(missing_ids)}."
            )


validate_profession_routes()

PROFESSION_DISPLAY_ORDER = PROFESSION_OPTIONS + list(LEGACY_PROFESSION_LABELS.keys())

PARTICIPANT_CONTEXT_COLUMNS = [
    "gender",
    "age",
    "semester",
    "works_for_studies",
    "children_count",
    "academic_program",
    "academic_shift",
    "prior_experience_area",
    "ethics_training",
    "work_hours_per_week",
    "caregiving_load",
    "study_funding_type",
]

CONTEXT_FIELD_LABELS = {
    "gender": "Género",
    "age": "Edad",
    "semester": "Semestre",
    "works_for_studies": "Trabaja para pagar estudios",
    "children_count": "Número de hijos",
    "academic_program": "Programa académico",
    "academic_shift": "Jornada académica",
    "prior_experience_area": "Experiencia laboral o clínica previa",
    "ethics_training": "Formación previa en ética o bioética",
    "work_hours_per_week": "Horas de trabajo por semana",
    "caregiving_load": "Carga de cuidado familiar",
    "study_funding_type": "Tipo de financiación de estudios",
    "years_experience": "Años de experiencia",
}

CONTEXT_CATEGORICAL_COLUMNS = [
    "gender",
    "works_for_studies",
    "academic_shift",
    "prior_experience_area",
    "ethics_training",
    "caregiving_load",
    "study_funding_type",
]

CONTEXT_NUMERIC_COLUMNS = [
    "age",
    "semester",
    "years_experience",
    "children_count",
    "work_hours_per_week",
]

GENDER_OPTIONS = ["Mujer", "Hombre", "No binario", "Otro", "Prefiere no responder"]
WORK_STUDY_OPTIONS = ["Sí", "No", "Parcialmente"]
ACADEMIC_SHIFT_OPTIONS = ["", "Diurna", "Nocturna", "Mixta", "Fin de semana", "Otra"]
PRIOR_EXPERIENCE_OPTIONS = ["", "Ninguna", "Laboral", "Clínica", "Laboral y clínica", "Otra"]
ETHICS_TRAINING_OPTIONS = ["", "Ninguna", "Curso corto", "Asignatura formal", "Diplomado o posgrado", "Otra"]
CAREGIVING_LOAD_OPTIONS = ["", "Ninguna", "Baja", "Moderada", "Alta", "Muy alta"]
STUDY_FUNDING_OPTIONS = ["", "Recursos propios", "Apoyo familiar", "Beca", "Crédito", "Mixta", "Institucional", "Otra"]

SEMESTER_LIMITS_BY_PROFESSION = {
    "Medicina": 14,
    "Enfermería": 10,
    "Fisioterapia": 10,
    "Instrumentación Quirúrgica": 10,
    "Bacteriología": 10,
    "Microbiología": 10,
    "Derecho": 10,
    "Ciencias Sociales": 10,
    "Educación": 10,
    "Ingeniería": 10,
    "TI": 10,
    "Datos": 10,
    "Otra / Mixta": 12,
    "Salud (legado)": 10,
    "Derecho / Ciencias Sociales / Educación (legado)": 10,
    "Ingeniería / TI / Datos (legado)": 10,
}


# =========================
# Utilidades de datos
# =========================
def ensure_storage() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    PERSISTENCE_STORE.ensure_storage()


@st.cache_data(show_spinner=False)
def load_df_cached(cache_key: str) -> pd.DataFrame:
    del cache_key
    ensure_storage()
    df = PERSISTENCE_STORE.load_all_rows()
    df["profession"] = df["profession"].apply(normalize_profession_label)
    return df[COLUMNS].copy()


def load_df() -> pd.DataFrame:
    return load_df_cached(PERSISTENCE_STORE.backend_detail)


def save_attempt_rows(df: pd.DataFrame) -> None:
    PERSISTENCE_STORE.save_rows(df)
    load_df_cached.clear()


def sha_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_text(text: str | None) -> str:
    raw = (text or "").lower()
    raw = re.sub(r"[^a-záéíóúñü\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def normalize_profession_key(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


PROFESSION_NORMALIZATION_MAP = {
    normalize_profession_key(label): label
    for label in PROFESSION_OPTIONS + list(LEGACY_PROFESSION_LABELS.keys())
}
PROFESSION_NORMALIZATION_MAP.update({
    normalize_profession_key(alias): canonical
    for alias, canonical in PROFESSION_ALIASES.items()
})


def normalize_profession_label(value: str | None) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    normalized = PROFESSION_NORMALIZATION_MAP.get(normalize_profession_key(str(value)))
    if normalized:
        return normalized
    stripped = str(value).strip()
    return stripped or None


def route_group_for_profession(label: str | None) -> str:
    normalized = normalize_profession_label(label)
    if normalized in PROFESSION_ROUTE_GROUPS:
        return PROFESSION_ROUTE_GROUPS[normalized]
    if normalized in LEGACY_PROFESSION_LABELS:
        return LEGACY_PROFESSION_LABELS[normalized]
    return "mixta"


def profession_category_order(values: pd.Series | None = None) -> List[str]:
    base = list(PROFESSION_DISPLAY_ORDER)
    if values is None:
        return base
    extras = []
    for value in values.dropna().astype(str).unique().tolist():
        if value not in base and value not in extras:
            extras.append(value)
    return base + sorted(extras)


def normalize_group_label(value: str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Sin grupo"
    stripped = str(value).strip()
    return stripped if stripped else "Sin grupo"


def semester_limit_for_profession(profession: str | None) -> int:
    normalized = normalize_profession_label(profession)
    return SEMESTER_LIMITS_BY_PROFESSION.get(normalized or "", 12)


def optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def optional_int_from_text(value: str | None, *, minimum: int = 0, maximum: int | None = None, field_label: str = "valor") -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"{field_label} debe ser un número entero.")
    parsed = int(text)
    if parsed < minimum:
        raise ValueError(f"{field_label} debe ser mayor o igual a {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field_label} debe ser menor o igual a {maximum}.")
    return parsed


def participant_context_from_row(row: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
    source = row if isinstance(row, dict) else row.to_dict()
    return {column: source.get(column) for column in PARTICIPANT_CONTEXT_COLUMNS}


def build_context_display_rows(context: Dict[str, Any]) -> List[tuple[str, Any]]:
    rows = []
    for column in PARTICIPANT_CONTEXT_COLUMNS:
        value = context.get(column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        rows.append((CONTEXT_FIELD_LABELS.get(column, column), value))
    return rows


def get_admin_password() -> str | None:
    for env_key in ["MORAL_TEST_ADMIN_PASSWORD", "ADMIN_PASSWORD"]:
        env_password = os.getenv(env_key)
        if env_password:
            return env_password
    try:
        for secret_key in ["MORAL_TEST_ADMIN_PASSWORD", "ADMIN_PASSWORD"]:
            secret_password = st.secrets.get(secret_key)
            if secret_password:
                return str(secret_password)
    except Exception:
        pass
    return None


def is_admin_authenticated() -> bool:
    return bool(st.session_state.get(ADMIN_SESSION_KEY, False))


def admin_login_panel() -> bool:
    configured_password = get_admin_password()
    if not configured_password:
        st.error(
            "El acceso administrativo está bloqueado hasta configurar `MORAL_TEST_ADMIN_PASSWORD` o `ADMIN_PASSWORD` en variables de entorno o secrets de Streamlit."
        )
        return False

    if is_admin_authenticated():
        return True

    st.warning("Contenido restringido para administración. Ingresa la contraseña para continuar.")
    with st.form("admin_login_form"):
        password = st.text_input("Contraseña de administrador", type="password")
        submitted = st.form_submit_button("Ingresar", use_container_width=True)

    if submitted:
        if hmac.compare_digest(password, configured_password):
            st.session_state[ADMIN_SESSION_KEY] = True
            st.rerun()
        st.error("Contraseña incorrecta.")
    return False


def render_admin_session_controls() -> None:
    if is_admin_authenticated() and st.sidebar.button("Cerrar sesión admin", use_container_width=True):
        st.session_state[ADMIN_SESSION_KEY] = False
        st.rerun()


def apply_dashboard_filters(
    students: pd.DataFrame,
    df_last: pd.DataFrame,
    selected_professions: List[str],
    selected_groups: List[str],
    year_range: Tuple[int, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filtered_students = students.copy()
    filtered_students["group_filter"] = filtered_students["group"].apply(normalize_group_label)

    if selected_professions:
        filtered_students = filtered_students[filtered_students["profession"].isin(selected_professions)]
    else:
        filtered_students = filtered_students.iloc[0:0].copy()

    if selected_groups:
        filtered_students = filtered_students[filtered_students["group_filter"].isin(selected_groups)]
    else:
        filtered_students = filtered_students.iloc[0:0].copy()

    min_year, max_year = year_range
    filtered_students = filtered_students[
        filtered_students["years_experience"].fillna(0).astype(float).between(min_year, max_year)
    ].copy()

    selected_ids = filtered_students["anon_id"].dropna().unique().tolist()
    filtered_df_last = df_last[df_last["anon_id"].isin(selected_ids)].copy()
    if not filtered_df_last.empty:
        filtered_df_last["group_filter"] = filtered_df_last["group"].apply(normalize_group_label)

    return filtered_students, filtered_df_last


def bootstrap_ci(values: np.ndarray | list, func=np.mean, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
    x = np.array([v for v in values if not pd.isna(v)], dtype=float)
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    stats = []
    for _ in range(n_boot):
        sample = np.random.choice(x, size=len(x), replace=True)
        stats.append(func(sample))
    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))
    return float(func(x)), low, high


def hedges_g(a: np.ndarray | list, b: np.ndarray | list) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    if sp == 0:
        return np.nan
    d = (a.mean() - b.mean()) / sp
    correction = 1 - (3 / (4 * (na + nb) - 9))
    return float(correction * d)


def route_for_profession(label: str, route_size: int) -> List[dict]:
    route_group = route_group_for_profession(label)
    ids = ROUTE_BANKS.get(route_group, ["K1", "K2"])[:route_size]
    return [LOOKUP[item_id] for item_id in ids]


def last_attempt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    idx = df.groupby("anon_id")["timestamp"].transform("max") == df["timestamp"]
    return df[idx].copy()


def kohlberg_from_stage_likert(df_stage: pd.DataFrame) -> Tuple[float, int | None, str | None, Dict[int, float]]:
    temp = df_stage.copy()
    if temp.empty:
        return np.nan, None, None, {k: np.nan for k in range(1, 7)}
    temp["sub_id"] = temp["sub_id"].astype(int)
    stage_means = temp.groupby("sub_id")["likert_value"].mean().reindex([1, 2, 3, 4, 5, 6])
    idx = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    weights = stage_means.to_numpy(dtype=float)
    if np.all(np.isnan(weights)):
        return np.nan, None, None, stage_means.to_dict()
    fill_value = np.nanmean(weights) if not np.isnan(np.nanmean(weights)) else 4.0
    weights = np.nan_to_num(weights, nan=fill_value)
    weights = np.clip(weights, 1e-6, None)
    estimate = float((idx * weights).sum() / weights.sum())
    stage = int(np.clip(round(estimate), 1, 6))
    level = "preconvencional" if stage in [1, 2] else ("convencional" if stage in [3, 4] else "postconvencional")
    return estimate, stage, level, stage_means.to_dict()


def framework_scores(df_fw: pd.DataFrame) -> Tuple[Dict[str, float], str | None]:
    scores = df_fw.groupby("sub_id")["likert_value"].mean().reindex(FRAMEWORKS)
    dominant = scores.idxmax() if scores.notna().any() else None
    return scores.to_dict(), dominant


def student_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), df
    df_last = last_attempt(df)
    choices = df_last[df_last["row_type"] == "choice"].copy()
    stage_lk = df_last[df_last["row_type"] == "stage_likert"].copy()
    fw_lk = df_last[df_last["row_type"] == "framework_likert"].copy()

    meta = df_last.groupby("anon_id").agg({
        "student_id": "first",
        "name": "first",
        "profession": "first",
        "years_experience": "first",
        "group": "first",
        "gender": "first",
        "age": "first",
        "semester": "first",
        "works_for_studies": "first",
        "children_count": "first",
        "academic_program": "first",
        "academic_shift": "first",
        "prior_experience_area": "first",
        "ethics_training": "first",
        "work_hours_per_week": "first",
        "caregiving_load": "first",
        "study_funding_type": "first",
        "timestamp": "first",
    }).reset_index()

    k_rows = []
    for aid, g in stage_lk.groupby("anon_id"):
        estimate, stage, level, stage_means = kohlberg_from_stage_likert(g)
        k_rows.append({
            "anon_id": aid,
            "k_est": estimate,
            "k_stage": stage,
            "k_level": level,
            "k_coherence_std": float(np.nanstd(list(stage_means.values()))),
        })
    k_df = pd.DataFrame(k_rows)

    fw_rows = []
    for aid, g in fw_lk.groupby("anon_id"):
        scores, dom = framework_scores(g)
        row = {"anon_id": aid, "fw_dom": dom}
        for key, value in scores.items():
            row[f"fw_{key}"] = value
        fw_rows.append(row)
    fw_df = pd.DataFrame(fw_rows)

    if choices.empty:
        choice_summary = pd.DataFrame(columns=["anon_id", "choice_level_mode", "choice_fw_mode", "n_dilemmas"])
    else:
        choice_summary = choices.groupby("anon_id").apply(
            lambda g: pd.Series({
                "choice_level_mode": g["choice_level"].mode().iloc[0] if not g["choice_level"].mode().empty else None,
                "choice_fw_mode": g["choice_framework"].mode().iloc[0] if not g["choice_framework"].mode().empty else None,
                "n_dilemmas": len(g),
            }),
            include_groups=False,
        ).reset_index()

    students = meta.merge(k_df, on="anon_id", how="left")
    students = students.merge(fw_df, on="anon_id", how="left")
    students = students.merge(choice_summary, on="anon_id", how="left")
    students["profession"] = students["profession"].apply(normalize_profession_label)
    return students, df_last


def build_rows(
    student_id: str,
    name: str,
    profession: str,
    years_experience: int,
    group: str,
    anonymize: bool,
    participant_context: Dict[str, Any],
    choices_payload: List[dict],
    stage_payload: List[dict],
    framework_payload: List[dict],
) -> pd.DataFrame:
    ts = now_iso()
    anon_id = sha_id(student_id) if anonymize else student_id
    rows = []
    common_fields = {
        "timestamp": ts,
        "anon_id": anon_id,
        "student_id": student_id,
        "name": name,
        "profession": profession,
        "years_experience": int(years_experience),
        "group": group,
    }
    for column in PARTICIPANT_CONTEXT_COLUMNS:
        common_fields[column] = participant_context.get(column)

    for row in choices_payload:
        rows.append({
            **common_fields,
            "row_type": "choice",
            "item_id": row["item_id"],
            "sub_id": "",
            "choice_key": row["choice_key"],
            "choice_stage": row["choice_stage"],
            "choice_level": row["choice_level"],
            "choice_framework": row["choice_framework"],
            "likert_value": np.nan,
            "text": row["text"],
        })

    for row in stage_payload:
        rows.append({
            **common_fields,
            "row_type": "stage_likert",
            "item_id": row["item_id"],
            "sub_id": str(row["sub_id"]),
            "choice_key": "",
            "choice_stage": np.nan,
            "choice_level": "",
            "choice_framework": "",
            "likert_value": int(row["likert_value"]),
            "text": "",
        })

    for row in framework_payload:
        rows.append({
            **common_fields,
            "row_type": "framework_likert",
            "item_id": row["item_id"],
            "sub_id": row["sub_id"],
            "choice_key": "",
            "choice_stage": np.nan,
            "choice_level": "",
            "choice_framework": "",
            "likert_value": int(row["likert_value"]),
            "text": row["text"],
        })

    return pd.DataFrame(rows)[COLUMNS]


# =========================
# Componentes visuales
# =========================
def render_global_styles() -> None:
    st.markdown(
        """
        <style>
        .hero-card {
            background:
                radial-gradient(circle at top right, rgba(192, 138, 43, 0.20), transparent 28%),
                linear-gradient(135deg, #081729 0%, #0b1f3a 38%, #12446c 100%);
            padding: 1.55rem 1.7rem;
            border-radius: 22px;
            color: #f7fbff;
            border: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: 0 14px 36px rgba(10, 25, 47, 0.24);
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }
        .hero-card::after {
            content: "";
            position: absolute;
            right: -30px;
            top: -30px;
            width: 180px;
            height: 180px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
        }
        .hero-card h1 {
            font-size: 2.3rem;
            margin: 0 0 .2rem 0;
            line-height: 1.15;
            letter-spacing: 0.08em;
            font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
        }
        .hero-card p {
            margin: 0.2rem 0;
            font-size: 1rem;
        }
        .hero-kicker {
            display: inline-block;
            margin-bottom: 0.7rem;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.16);
            color: #f8df9d;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-lead {
            max-width: 56rem;
            color: #dce8f5;
            font-size: 1.03rem;
        }
        .brand-signature {
            color: #f8df9d;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .author-card {
            background: linear-gradient(180deg, #f7fafc 0%, #eef4f9 100%);
            border: 1px solid #d9e5f2;
            border-left: 6px solid #c08a2b;
            padding: 1rem;
            border-radius: 14px;
            margin-bottom: 1rem;
            color: #0b1f3a;
        }
        .objective-card {
            background: #ffffff;
            border: 1px solid #e6eef7;
            border-radius: 14px;
            padding: .95rem 1rem;
            min-height: 120px;
            box-shadow: 0 4px 12px rgba(17, 35, 58, 0.06);
            color: #0b1f3a;
        }
        .objective-card strong {
            color: #0b1f3a;
        }
        .minor-note {
            color: #5a6b7c;
            font-size: 0.93rem;
        }
        .section-card {
            background: linear-gradient(180deg, #f9fbfd 0%, #f2f6fa 100%);
            border: 1px solid #d7e2ee;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
        }
        .section-card h3 {
            margin: 0 0 0.35rem 0;
            color: #0b1f3a;
        }
        .section-card p {
            margin: 0;
            color: #405569;
        }
        .interpretation-panel {
            background: #fbfcfe;
            border: 1px solid #d8e2ec;
            border-left: 6px solid #c08a2b;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            color: #0b1f3a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    render_global_styles()
    creds = " • ".join(AUTHOR_CREDENTIALS)
    st.markdown(
        f"""
        <div class="hero-card">
            <span class="hero-kicker">Software académico de bioética aplicada</span>
            <h1>{APP_TITLE}</h1>
            <p class="brand-signature">{APP_BRAND_LINE}</p>
            <p class="hero-lead">{APP_SUBTITLE}</p>
            <p><strong>Autor:</strong> {AUTHOR_NAME}</p>
            <p>{creds}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Instrumento formativo y reflexivo. No emite diagnósticos psicológicos ni reemplaza juicio ético profesional, "
        "comités, supervisión docente o valoración clínica/jurídica especializada."
    )


def render_sidebar_branding() -> None:
    st.sidebar.markdown(
        f"""
        <div class="author-card">
            <strong style="font-size:1.05rem; letter-spacing:.08em;">{APP_BRAND_NAME}</strong><br>
            <span class="minor-note">{APP_BRAND_LINE}</span>
            <hr style="margin:.6rem 0;">
            <strong>{AUTHOR_NAME}</strong><br>
            <span class="minor-note">{'<br>'.join(AUTHOR_CREDENTIALS)}</span>
            <hr style="margin:.6rem 0;">
            <strong>Función principal</strong><br>
            <span class="minor-note">{MAIN_FUNCTION}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**Objetivos del programa**")
    for objective in PROGRAM_OBJECTIVES:
        st.sidebar.markdown(f"- {objective}")


def make_program_flow_figure() -> go.Figure:
    labels = [
        "Identificación y\nruta profesional",
        "Dilemas éticos\ncontextualizados",
        "Justificación\nargumentativa",
        "Escalas Likert\npor estadio",
        "Inventario de\nmarcos éticos",
        "Reporte\nindividual",
        "Dashboard\ncolectivo",
    ]
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            pad=22,
            thickness=24,
            line=dict(color="#d7e2ee", width=1.2),
            color=[
                "#16324F",
                "#1F4E79",
                "#2A6B8F",
                "#3C7A89",
                "#7A6A4F",
                "#B78A3E",
                "#51697D",
            ],
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=[0, 1, 2, 3, 4],
            target=[1, 2, 3, 5, 6],
            value=[1, 1, 1, 1, 1],
            color=[
                "rgba(31, 78, 121, 0.22)",
                "rgba(42, 107, 143, 0.20)",
                "rgba(22, 50, 79, 0.18)",
                "rgba(183, 138, 62, 0.22)",
                "rgba(81, 105, 125, 0.24)",
            ],
            hovertemplate="%{source.label} → %{target.label}<extra></extra>",
        ),
    )])
    fig.update_layout(
        title={
            "text": "Arquitectura funcional del programa",
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 22, "family": "Aptos, Segoe UI, Arial, sans-serif", "color": "#16324F"},
        },
        height=480,
        font=dict(size=12, family="Aptos, Segoe UI, Arial, sans-serif", color="#F7FBFF"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=80, b=30),
        annotations=[
            dict(
                x=0,
                y=1.11,
                xref="paper",
                yref="paper",
                text="Ruta sintética del instrumento desde la identificación del participante hasta el análisis individual y colectivo.",
                showarrow=False,
                font=dict(size=12, color="#526577", family="Aptos, Segoe UI, Arial, sans-serif"),
                align="left",
            )
        ],
    )
    return fig


def make_route_summary_figure() -> go.Figure:
    rows = []
    for profession in PROFESSION_OPTIONS:
        rows.append({"ruta": profession, "dilemas": len(PROFESSION_ROUTES[profession])})
    data = pd.DataFrame(rows)
    fig = px.bar(
        data,
        x="ruta",
        y="dilemas",
        title="Cobertura de dilemas por ruta profesional",
        text="dilemas",
        height=420,
    )
    fig.update_layout(
        xaxis_title="Profesión",
        yaxis_title="Número de dilemas",
        xaxis={"categoryorder": "array", "categoryarray": PROFESSION_OPTIONS},
    )
    return fig


def page_program() -> None:
    st.subheader("Presentación del programa")
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        st.markdown(
            f"""
            <div class="author-card">
                <strong>Autor académico</strong><br>
                <span style="font-size:1.05rem;">{AUTHOR_NAME}</span><br><br>
                <strong>Credenciales</strong><br>
                <span class="minor-note">{'<br>'.join(AUTHOR_CREDENTIALS)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Función principal")
        st.write(MAIN_FUNCTION)
        st.markdown("### Alcance")
        st.write(
            "El programa está diseñado para formación, reflexión, docencia, análisis de cohortes y apoyo a la discusión ética aplicada. "
            "No debe emplearse como instrumento clínico-diagnóstico ni como criterio único para decisiones académicas o laborales."
        )
    with c2:
        render_plotly_figure(
            make_program_flow_figure(),
            "arquitectura_funcional_programa",
            caption="Diagrama de alto nivel del recorrido pedagógico y analítico de ETHOSCOPE, con mejor contraste y legibilidad para docencia, presentación y revisión institucional.",
        )

    st.markdown("### Objetivos del programa")
    cols = st.columns(2)
    for i, objective in enumerate(PROGRAM_OBJECTIVES):
        with cols[i % 2]:
            st.markdown(f'<div class="objective-card"><strong>Objetivo {i+1}</strong><br><br>{objective}</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(make_route_summary_figure(), use_container_width=True)
    with g2:
        route_mix = pd.DataFrame({
            "componente": [
                "Profesiones ruta salud",
                "Profesiones ruta bacteriología",
                "Profesiones ruta microbiología",
                "Profesiones ruta derecho/social",
                "Profesiones ruta ingeniería/TI/datos",
                "Profesiones ruta mixta",
            ],
            "cantidad": [4, 1, 1, 3, 3, 1],
        })
        fig = px.pie(route_mix, names="componente", values="cantidad", hole=0.48, title="Distribución de profesiones por familia de ruta", height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Qué produce la herramienta")
    st.markdown(
        "- **Reporte individual inmediato** con nivel global, estadio estimado, coherencia y marco dominante.\n"
        "- **Visualización colectiva** con distribuciones, brechas, clusters argumentativos y perfiles por profesión.\n"
        "- **Exportación de datos** para análisis docente, auditoría académica e investigación educativa exploratoria."
    )




def make_radar(scores: Dict[str, float], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[scores.get(fw, np.nan) for fw in FRAMEWORKS],
        theta=[FRAMEWORK_LABELS.get(fw, fw) for fw in FRAMEWORKS],
        fill="toself",
        name="Perfil",
        line=dict(color="#1c5f8d", width=3),
        fillcolor="rgba(28, 95, 141, 0.22)",
    ))
    return style_academic_figure(
        fig,
        title,
        height=450,
        showlegend=False,
        polar=dict(radialaxis=dict(visible=True, range=[1, 7], gridcolor="#d7e2ee")),
    )


def make_sankey_from_choices(choices_df: pd.DataFrame, title: str) -> go.Figure:
    if choices_df.empty:
        return go.Figure()
    links = choices_df.groupby(["item_id", "choice_framework"]).size().reset_index(name="value")
    links["choice_framework"] = links["choice_framework"].map(lambda value: FRAMEWORK_LABELS.get(value, value))
    nodes = list(links["item_id"].unique()) + list(links["choice_framework"].unique())
    idx = {label: i for i, label in enumerate(nodes)}
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            label=nodes,
            pad=18,
            thickness=18,
            line=dict(color="#d7e2ee", width=0.8),
            color=["#0b1f3a" if label in links["item_id"].unique() else "#c08a2b" for label in nodes],
        ),
        link=dict(
            source=[idx[val] for val in links["item_id"]],
            target=[idx[val] for val in links["choice_framework"]],
            value=links["value"].tolist(),
            color="rgba(28, 95, 141, 0.28)",
        ),
    )])
    return style_academic_figure(fig, title, height=500)


def framework_label(value: str) -> str:
    return FRAMEWORK_LABELS.get(value, value.replace("_", " ").strip().title())


def filename_slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return text.strip("_") or "grafica"


def serialize_for_ai(value):
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def selected_option_text(item_id: str, choice_key: str) -> str | None:
    dilemma = LOOKUP.get(item_id)
    if not dilemma:
        return None
    for option in dilemma["options"]:
        if option["key"] == choice_key:
            return option["text"]
    return None


def build_individual_ai_payload(student_row: pd.Series, rows: pd.DataFrame) -> Dict[str, object]:
    choice_df = rows[rows["row_type"] == "choice"].copy()
    stage_df = rows[rows["row_type"] == "stage_likert"].copy()
    fw_df = rows[rows["row_type"] == "framework_likert"].copy()
    k_est, k_stage, k_level, stage_means = kohlberg_from_stage_likert(stage_df)
    fw_scores, fw_dom = framework_scores(fw_df)
    coherence = float(np.nanstd(list(stage_means.values()))) if stage_means else np.nan

    dilemmas = []
    for _, row in choice_df.iterrows():
        dilemmas.append({
            "item_id": row["item_id"],
            "titulo": LOOKUP.get(row["item_id"], {}).get("title"),
            "planteamiento": LOOKUP.get(row["item_id"], {}).get("prompt"),
            "opcion_seleccionada": row["choice_key"],
            "texto_opcion": selected_option_text(str(row["item_id"]), str(row["choice_key"])),
            "estadio_opcion": serialize_for_ai(row["choice_stage"]),
            "nivel_opcion": row["choice_level"],
            "marco_opcion": row["choice_framework"],
            "justificacion": row["text"],
        })

    return {
        "tipo_analisis": "individual",
        "participante": {
            "anon_id": student_row["anon_id"],
            "student_id": student_row.get("student_id"),
            "nombre": student_row.get("name"),
            "profesion": student_row.get("profession"),
            "grupo": normalize_group_label(student_row.get("group")),
            "anos_experiencia": serialize_for_ai(student_row.get("years_experience")),
            "genero": student_row.get("gender"),
            "edad": serialize_for_ai(student_row.get("age")),
            "semestre": serialize_for_ai(student_row.get("semester")),
            "trabaja_para_pagar_estudios": student_row.get("works_for_studies"),
            "numero_hijos": serialize_for_ai(student_row.get("children_count")),
            "programa_academico": student_row.get("academic_program"),
            "jornada_academica": student_row.get("academic_shift"),
            "tipo_experiencia_previa": student_row.get("prior_experience_area"),
            "formacion_previa_etica": student_row.get("ethics_training"),
            "horas_trabajo_semana": serialize_for_ai(student_row.get("work_hours_per_week")),
            "carga_cuidado_familiar": student_row.get("caregiving_load"),
            "financiacion_estudios": student_row.get("study_funding_type"),
            "timestamp": student_row.get("timestamp"),
        },
        "dilemas_respondidos": dilemmas,
        "indicadores": {
            "indice_k_est": serialize_for_ai(k_est),
            "estadio_redondeado": serialize_for_ai(k_stage),
            "nivel_dominante": k_level,
            "marco_dominante": fw_dom,
            "coherencia": serialize_for_ai(coherence),
            "n_dilemas": int(len(choice_df)),
            "stage_means": {str(key): serialize_for_ai(value) for key, value in stage_means.items()},
            "framework_scores": {key: serialize_for_ai(value) for key, value in fw_scores.items()},
        },
    }


def build_group_ai_payload(
    students_filtered: pd.DataFrame,
    df_last_filtered: pd.DataFrame,
    quantitative_report: Dict[str, pd.DataFrame],
    keyword_df: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    trends_df: pd.DataFrame,
) -> Dict[str, object]:
    choice_df = df_last_filtered[df_last_filtered["row_type"] == "choice"].copy()
    sample_choices = []
    for _, row in choice_df.head(12).iterrows():
        sample_choices.append({
            "anon_id": row["anon_id"],
            "profesion": row["profession"],
            "item_id": row["item_id"],
            "titulo": LOOKUP.get(row["item_id"], {}).get("title"),
            "opcion": row["choice_key"],
            "texto_opcion": selected_option_text(str(row["item_id"]), str(row["choice_key"])),
            "nivel_opcion": row["choice_level"],
            "marco_opcion": row["choice_framework"],
            "justificacion": row["text"],
        })

    profession_distribution = quantitative_report.get("profession_distribution", pd.DataFrame())
    framework_summary = quantitative_report.get("framework_summary", pd.DataFrame())
    stage_summary = quantitative_report.get("stage_summary", pd.DataFrame())
    descriptive_summary = quantitative_report.get("descriptive_summary", pd.DataFrame())

    return {
        "tipo_analisis": "grupal",
        "resumen_muestra": {
            "n_participantes": int(len(students_filtered)),
            "profesiones": sorted(students_filtered["profession"].dropna().astype(str).unique().tolist()),
            "grupos": sorted(students_filtered["group"].fillna("Sin grupo").astype(str).unique().tolist()),
        },
        "variables_contextuales": {
            "resumen_categorico": students_filtered[CONTEXT_CATEGORICAL_COLUMNS].astype(object).fillna("No disponible").apply(
                lambda col: col.value_counts(dropna=False).head(8).to_dict()
            ).to_dict(),
            "resumen_numerico": students_filtered[[col for col in CONTEXT_NUMERIC_COLUMNS if col in students_filtered.columns]].describe(include="all").fillna("").to_dict(),
        },
        "distribucion_profesion": profession_distribution.to_dict(orient="records"),
        "marcos_por_profesion": framework_summary.to_dict(orient="records"),
        "estadios_por_profesion": stage_summary.to_dict(orient="records"),
        "resumen_descriptivo": descriptive_summary.to_dict(orient="records"),
        "tendencias_profesion": trends_df.to_dict(orient="records"),
        "palabras_clave": keyword_df.to_dict(orient="records"),
        "patrones_argumentativos": pattern_summary.to_dict(orient="records"),
        "muestra_justificaciones": sample_choices,
    }


def build_individual_choice_detail_df(choice_df: pd.DataFrame) -> pd.DataFrame:
    detail_df = choice_df.copy()
    detail_df["dilema"] = detail_df["item_id"].map(lambda item_id: LOOKUP.get(item_id, {}).get("title", item_id))
    detail_df["opcion"] = detail_df.apply(
        lambda row: selected_option_text(str(row["item_id"]), str(row["choice_key"])) or str(row["choice_key"]),
        axis=1,
    )
    detail_df["nivel_moral"] = detail_df["choice_level"].fillna("No disponible").astype(str).str.title()
    detail_df["marco_etico"] = detail_df["choice_framework"].map(framework_label)
    return detail_df[["item_id", "dilema", "opcion", "nivel_moral", "marco_etico", "text"]].rename(columns={
        "item_id": "dilema_id",
        "text": "justificacion",
    })


def build_framework_score_df(fw_scores: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for framework, score in fw_scores.items():
        rows.append({
            "marco_etico": framework_label(framework),
            "puntuacion": round(float(score), 2) if pd.notna(score) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("puntuacion", ascending=False, na_position="last")


def build_stage_score_df(stage_means: Dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "estadio": f"Estadio {stage}",
            "promedio_likert": round(float(stage_means.get(stage, np.nan)), 2) if pd.notna(stage_means.get(stage, np.nan)) else np.nan,
        }
        for stage in range(1, 7)
    ])


def build_individual_recommendations(fw_scores: Dict[str, float]) -> List[str]:
    if not fw_scores:
        return [
            "Explicita el principio protegido, la consecuencia esperada y la salvaguarda frente a daño colateral.",
            "Contrasta tu decisión con al menos un marco ético alternativo para fortalecer la deliberación.",
        ]
    ranked_scores = sorted(fw_scores.items(), key=lambda item: (pd.isna(item[1]), -(item[1] if pd.notna(item[1]) else -999)))
    strongest = framework_label(ranked_scores[0][0])
    weakest = framework_label(ranked_scores[-1][0])
    return [
        f"Reescribe uno de los casos desde el marco {weakest} para contrastarlo con tu tendencia dominante {strongest}.",
        "Explicita siempre el principio protegido, la consecuencia esperada y la salvaguarda frente a daño colateral.",
        "En los casos con alta afectación de terceros, incorpora proporcionalidad, transparencia y trazabilidad.",
    ]


def build_individual_summary_text(
    *,
    k_level: str | None,
    k_est: float,
    fw_dom: str | None,
    coherence: float,
    profession: str,
    route_group: str,
) -> str:
    level_text = str(k_level or "sin predominio claro")
    framework_text = framework_label(fw_dom) if fw_dom else "sin predominio claro"
    k_est_text = f"{k_est:.2f}" if pd.notna(k_est) else "NA"
    coherence_text = f"{coherence:.2f}" if pd.notna(coherence) else "NA"
    return (
        f"En la ruta individual de {profession} ({route_group}) se observa predominio del nivel {level_text} "
        f"con un índice global k_est de {k_est_text}. El marco ético dominante fue {framework_text} "
        f"y la dispersión interna entre estadios fue {coherence_text}, lo que orienta la lectura formativa del reporte."
    )


def create_individual_report_context(
    *,
    rows_df: pd.DataFrame,
    student_id: str,
    name: str,
    profession: str,
    years_experience: int,
    group: str,
    route_group: str,
) -> Dict[str, object]:
    person = rows_df.copy()
    choice_df = person[person["row_type"] == "choice"].copy()
    stage_df = person[person["row_type"] == "stage_likert"].copy()
    fw_df = person[person["row_type"] == "framework_likert"].copy()

    k_est, k_stage, k_level, stage_means = kohlberg_from_stage_likert(stage_df)
    fw_scores, fw_dom = framework_scores(fw_df)
    coherence = float(np.nanstd(list(stage_means.values()))) if stage_means else np.nan
    timestamp = str(person["timestamp"].iloc[0]) if not person.empty else now_iso()
    anon_id = str(person["anon_id"].iloc[0]) if not person.empty else sha_id(student_id)

    choice_export_df = choice_df[["item_id", "choice_key", "choice_level", "choice_framework", "text"]].copy()
    choice_export_df["opcion_texto"] = choice_export_df.apply(
        lambda row: selected_option_text(str(row["item_id"]), str(row["choice_key"])),
        axis=1,
    )
    choice_export_df["marco_etico"] = choice_export_df["choice_framework"].map(framework_label)
    participant_context = participant_context_from_row(person.iloc[0]) if not person.empty else {column: None for column in PARTICIPANT_CONTEXT_COLUMNS}

    return {
        "rows_df": person,
        "timestamp": timestamp,
        "anon_id": anon_id,
        "student_id": student_id,
        "name": name,
        "profession": profession,
        "years_experience": years_experience,
        "group": group,
        "route_group": route_group,
        "participant_context": participant_context,
        "choice_df": choice_df,
        "choice_export_df": choice_export_df,
        "choice_detail_df": build_individual_choice_detail_df(choice_df),
        "framework_score_df": build_framework_score_df(fw_scores),
        "stage_score_df": build_stage_score_df(stage_means),
        "k_est": k_est,
        "k_stage": k_stage,
        "k_level": k_level,
        "stage_means": stage_means,
        "fw_scores": fw_scores,
        "fw_dom": fw_dom,
        "coherence": coherence,
        "summary_text": build_individual_summary_text(
            k_level=k_level,
            k_est=k_est,
            fw_dom=fw_dom,
            coherence=coherence,
            profession=profession,
            route_group=route_group,
        ),
        "recommendations": build_individual_recommendations(fw_scores),
        "ai_result": None,
        "ai_error": None,
        "pdf_bytes": None,
        "saved_paths": {},
    }


def run_individual_ai_analysis(report_context: Dict[str, object], spinner_text: str | None = None) -> Dict[str, object]:
    student_row = pd.Series({
        "anon_id": report_context["anon_id"],
        "student_id": report_context["student_id"],
        "name": report_context["name"],
        "profession": report_context["profession"],
        "group": report_context["group"],
        "years_experience": report_context["years_experience"],
        "timestamp": report_context["timestamp"],
        **(report_context.get("participant_context") or {}),
    })
    payload = build_individual_ai_payload(student_row, report_context["rows_df"])
    try:
        if spinner_text:
            with st.spinner(spinner_text):
                result = interpret_payload(payload, scope="individual")
        else:
            result = interpret_payload(payload, scope="individual")
        report_context["ai_result"] = result
        report_context["ai_error"] = None
    except OpenAIInterpreterError as exc:
        report_context["ai_result"] = None
        report_context["ai_error"] = str(exc)
    except Exception as exc:
        report_context["ai_result"] = None
        report_context["ai_error"] = f"No fue posible generar la interpretación IA integrada: {exc}"
    return report_context


def rebuild_individual_report_artifacts(report_context: Dict[str, object]) -> Dict[str, object]:
    dominant_framework = framework_label(report_context["fw_dom"]) if report_context.get("fw_dom") else "No definido"
    participant_context_rows = build_context_display_rows(report_context.get("participant_context", {}))
    pdf_bytes = build_individual_report_pdf(
        app_title=APP_TITLE,
        app_brand_line=APP_BRAND_LINE,
        author_name=AUTHOR_NAME,
        author_credentials=AUTHOR_CREDENTIALS,
        main_function=MAIN_FUNCTION,
        generated_at=str(report_context["timestamp"]),
        participant_rows=[
            ("ID anonimizado", report_context["anon_id"]),
            ("ID institucional", report_context["student_id"]),
            ("Nombre o seudónimo", report_context["name"]),
            ("Profesión", report_context["profession"]),
            ("Grupo", normalize_group_label(report_context["group"])),
            ("Años de experiencia", report_context["years_experience"]),
            ("Ruta profesional", report_context["route_group"]),
            *participant_context_rows,
        ],
        metric_rows=[
            ("Nivel global", report_context["k_level"]),
            ("Estadio redondeado", report_context["k_stage"]),
            ("Índice k_est", report_context["k_est"]),
            ("Coherencia", report_context["coherence"]),
            ("Marco dominante", dominant_framework),
        ],
        narrative_summary=str(report_context["summary_text"]),
        recommendations=report_context["recommendations"],
        framework_scores_df=report_context["framework_score_df"],
        stage_scores_df=report_context["stage_score_df"],
        choice_detail_df=report_context["choice_detail_df"],
        interpretation_result=report_context.get("ai_result"),
        interpretation_note=report_context.get("ai_error"),
        figures=[
            (
                "Perfil individual de marcos éticos",
                "Afinidad relativa del participante con los marcos éticos evaluados por el software.",
                make_radar(report_context["fw_scores"], "Perfil individual de marcos éticos"),
            ),
            (
                "Perfil individual por estadios morales",
                "Promedios individuales en los ítems tipo Likert vinculados a estadios de razonamiento moral.",
                make_stage_profile_chart(report_context["stage_means"], "Perfil individual por estadios morales"),
            ),
            (
                "Flujo individual entre dilemas y marcos éticos",
                "Relación entre los dilemas respondidos y los marcos éticos seleccionados por el participante.",
                make_sankey_from_choices(report_context["choice_df"], "Dilema → marco ético del participante"),
            ),
        ],
    )
    report_context["pdf_bytes"] = pdf_bytes
    report_context["saved_paths"] = ADMIN_REPORT_STORE.save_individual_report(
        timestamp=str(report_context["timestamp"]),
        anon_id=str(report_context["anon_id"]),
        student_id=str(report_context["student_id"]),
        name=str(report_context["name"]),
        profession=str(report_context["profession"]),
        group=str(report_context["group"]),
        years_experience=int(report_context["years_experience"]),
        k_est=report_context["k_est"],
        k_stage=report_context["k_stage"],
        k_level=report_context["k_level"],
        coherence=report_context["coherence"],
        fw_dom=report_context["fw_dom"],
        fw_scores=report_context["fw_scores"],
        stage_means=report_context["stage_means"],
        choice_df=report_context["choice_export_df"],
        participant_context=report_context.get("participant_context"),
        ai_interpretation=report_context.get("ai_result"),
        pdf_bytes=pdf_bytes,
    )
    return report_context


def render_individual_report(report_context: Dict[str, object]) -> None:
    export_name = filename_slug(
        f"ethoscope_reporte_individual_{report_context['anon_id']}_{str(report_context['timestamp']).replace(':', '-').replace('T', '_')}"
    )
    dominant_framework = framework_label(report_context["fw_dom"]) if report_context.get("fw_dom") else "No definido"

    if not report_context.get("pdf_bytes"):
        report_context = rebuild_individual_report_artifacts(report_context)
        st.session_state[INDIVIDUAL_REPORT_SESSION_KEY] = report_context

    st.subheader("Reporte individual")
    st.caption(
        f"Participante: {report_context['name']} | Profesión: {report_context['profession']} | "
        f"Grupo: {normalize_group_label(report_context['group'])} | Ruta: {report_context['route_group']}"
    )
    context_rows = build_context_display_rows(report_context.get("participant_context", {}))
    if context_rows:
        with st.expander("Ver variables contextuales del participante"):
            st.dataframe(
                pd.DataFrame(context_rows, columns=["Variable", "Valor"]),
                use_container_width=True,
                hide_index=True,
            )

    action_col1, action_col2 = st.columns(2)
    retry_label = "Regenerar análisis IA integrado" if report_context.get("ai_result") else "Generar o reintentar análisis IA integrado"
    if action_col1.button(
        retry_label,
        key=f"rerun_ai_{report_context['anon_id']}_{report_context['timestamp']}",
        use_container_width=True,
    ):
        updated_context = run_individual_ai_analysis(
            report_context,
            spinner_text="Generando análisis IA integrado al reporte individual...",
        )
        updated_context = rebuild_individual_report_artifacts(updated_context)
        st.session_state[INDIVIDUAL_REPORT_SESSION_KEY] = updated_context
        st.rerun()

    action_col2.download_button(
        "Descargar reporte completo en PDF",
        data=report_context["pdf_bytes"],
        file_name=f"{export_name}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nivel global", str(report_context["k_level"]))
    m2.metric("Estadio redondeado", str(report_context["k_stage"]))
    m3.metric("Índice k_est", f"{report_context['k_est']:.2f}" if pd.notna(report_context["k_est"]) else "NA")
    m4.metric("Coherencia (DE)", f"{report_context['coherence']:.2f}" if pd.notna(report_context["coherence"]) else "NA")
    st.caption(f"Marco dominante: {dominant_framework}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_plotly_figure(
            make_radar(report_context["fw_scores"], "Perfil individual de marcos éticos"),
            f"radar_reporte_individual_{report_context['anon_id']}",
            data_df=report_context["framework_score_df"],
            caption="Afinidad relativa del participante con los marcos éticos evaluados.",
        )
    with chart_col2:
        render_plotly_figure(
            make_stage_profile_chart(report_context["stage_means"], "Perfil individual por estadios morales"),
            f"estadios_reporte_individual_{report_context['anon_id']}",
            data_df=report_context["stage_score_df"],
            caption="Promedios individuales en los ítems tipo Likert vinculados a estadios morales.",
        )

    render_plotly_figure(
        make_sankey_from_choices(report_context["choice_df"], "Dilema → marco ético del participante"),
        f"sankey_reporte_individual_{report_context['anon_id']}",
        data_df=report_context["choice_detail_df"],
        caption="Flujo individual entre dilemas respondidos y marcos éticos escogidos.",
    )

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        st.markdown("### Tabla de marcos éticos")
        st.dataframe(report_context["framework_score_df"], use_container_width=True, hide_index=True)
    with table_col2:
        st.markdown("### Tabla de estadios morales")
        st.dataframe(report_context["stage_score_df"], use_container_width=True, hide_index=True)

    st.markdown("### Síntesis interpretativa")
    st.write(report_context["summary_text"])

    st.markdown("### Recomendaciones de mejora argumentativa")
    st.markdown("\n".join(f"- {item}" for item in report_context["recommendations"]))

    st.markdown("### Análisis IA integrado")
    if report_context.get("ai_result"):
        render_interpretation_report(report_context["ai_result"], "Interpretación IA integrada al reporte")
    elif report_context.get("ai_error"):
        st.info(report_context["ai_error"])

    with st.expander("Ver detalle de respuestas"):
        st.dataframe(report_context["choice_detail_df"], use_container_width=True, hide_index=True)

    saved_paths = report_context.get("saved_paths") or {}
    if saved_paths:
        st.caption(
            f"El reporte individual quedó guardado en la carpeta administrativa del servidor: {ADMIN_REPORT_STORE.base_dir}."
        )


def page_ai_interpretation(df: pd.DataFrame) -> None:
    st.subheader("Interpretación IA")
    st.write("Genera interpretación individual y grupal asistida por OpenAI a partir de los resultados consolidados del instrumento.")

    if df.empty:
        st.warning("Aún no hay respuestas registradas para interpretar.")
        return

    students, df_last = student_table(df)
    if students.empty:
        st.warning("No fue posible consolidar participantes para la interpretación IA.")
        return

    tabs = st.tabs(["1. Interpretación individual", "2. Interpretación grupal"])

    with tabs[0]:
        participant_df = students[["anon_id", "name", "student_id", "profession", "group", "timestamp"]].copy()
        participant_df["display_label"] = participant_df.apply(
            lambda row: f"{row['name'] or row['student_id'] or row['anon_id'][:8]} | {row['profession']} | {normalize_group_label(row['group'])}",
            axis=1,
        )
        selected_label = st.selectbox("Participante para interpretación IA", options=participant_df["display_label"].tolist(), key="ai_individual_select")
        selected_anon = participant_df.loc[participant_df["display_label"] == selected_label, "anon_id"].iloc[0]
        student_row = students[students["anon_id"] == selected_anon].iloc[0]
        selected_rows = df_last[df_last["anon_id"] == selected_anon].copy()

        if st.button("Generar interpretación IA individual", use_container_width=True):
            payload = build_individual_ai_payload(student_row, selected_rows)
            try:
                with st.spinner("Consultando OpenAI para interpretación individual..."):
                    result = interpret_payload(payload, scope="individual")
                render_interpretation_report(result, "Interpretación IA individual")
            except OpenAIInterpreterError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"No fue posible generar la interpretación IA individual: {exc}")

    with tabs[1]:
        students = students.copy()
        df_last = df_last.copy()
        students["group_filter"] = students["group"].apply(normalize_group_label)
        profession_options = [
            value for value in profession_category_order(students["profession"])
            if value in students["profession"].dropna().astype(str).unique().tolist()
        ]
        group_options = sorted(students["group_filter"].dropna().astype(str).unique().tolist())
        year_values = students["years_experience"].dropna().astype(float)
        year_min = int(year_values.min()) if not year_values.empty else 0
        year_max = int(year_values.max()) if not year_values.empty else 0

        filter_col1, filter_col2, filter_col3 = st.columns([1.4, 1.2, 1.2])
        selected_professions = filter_col1.multiselect("Profesiones", options=profession_options, default=profession_options, key="ai_group_professions")
        selected_groups = filter_col2.multiselect("Grupos / cohortes", options=group_options, default=group_options, key="ai_group_groups")
        if year_min < year_max:
            selected_year_range = filter_col3.slider("Años de experiencia", min_value=year_min, max_value=year_max, value=(year_min, year_max), key="ai_group_years")
        else:
            filter_col3.caption(f"Años de experiencia: {year_min}")
            selected_year_range = (year_min, year_max)

        students_filtered, df_last_filtered = apply_dashboard_filters(
            students,
            df_last,
            selected_professions,
            selected_groups,
            selected_year_range,
        )
        if students_filtered.empty:
            st.info("Los filtros actuales no dejan muestra disponible para interpretación grupal.")
        elif st.button("Generar interpretación IA grupal", use_container_width=True):
            quantitative_report = build_quantitative_report(
                students_filtered.drop(columns=["group_filter"], errors="ignore"),
                df_last_filtered.drop(columns=["group_filter"], errors="ignore"),
                FRAMEWORKS,
            )
            keyword_df = extract_keywords_by_group(df_last_filtered, BASE_STOPWORDS, group_col="profession")
            pattern_summary = argumentative_pattern_table(df_last_filtered, BASE_STOPWORDS)
            trends_df = profession_interpretive_trends(
                students_filtered.drop(columns=["group_filter"], errors="ignore"),
                keyword_df,
                pattern_summary,
            )
            payload = build_group_ai_payload(
                students_filtered.drop(columns=["group_filter"], errors="ignore"),
                df_last_filtered.drop(columns=["group_filter"], errors="ignore"),
                quantitative_report,
                keyword_df,
                pattern_summary,
                trends_df,
            )
            try:
                with st.spinner("Consultando OpenAI para interpretación grupal..."):
                    result = interpret_payload(payload, scope="grupal")
                render_interpretation_report(result, "Interpretación IA grupal")
            except OpenAIInterpreterError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"No fue posible generar la interpretación IA grupal: {exc}")


def dataframe_to_excel_bytes(sheet_map: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, data in sheet_map.items():
            safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(sheet_name))[:31] or "Hoja"
            export_df = data.copy()
            if export_df.empty:
                export_df = pd.DataFrame({"mensaje": ["Sin datos disponibles"]})
            export_df.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output.getvalue()


def style_academic_figure(fig: go.Figure, title: str, height: int = 480, showlegend: bool = True, **layout_updates) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Georgia, Times New Roman, serif", size=14, color="#0b1f3a"),
        colorway=ACADEMIC_COLOR_SEQUENCE,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=40, r=20, t=70, b=50),
        height=height,
        showlegend=showlegend,
        **layout_updates,
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#c9d6e2", gridcolor="#ecf1f6")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="#c9d6e2", gridcolor="#ecf1f6")
    return fig


def render_plotly_figure(fig: go.Figure, export_name: str, data_df: pd.DataFrame | None = None, caption: str | None = None) -> None:
    export_slug = filename_slug(export_name)
    config = {
        **PLOTLY_EXPORT_CONFIG,
        "toImageButtonOptions": {
            **PLOTLY_EXPORT_CONFIG["toImageButtonOptions"],
            "filename": export_slug,
        },
    }
    st.plotly_chart(fig, use_container_width=True, config=config)
    controls = st.columns([1, 1, 5]) if data_df is not None else st.columns([1, 5])
    controls[0].download_button(
        "Descargar HTML",
        data=pio.to_html(fig, include_plotlyjs="cdn", full_html=True).encode("utf-8"),
        file_name=f"{export_slug}.html",
        mime="text/html",
        key=f"html_{export_slug}",
        use_container_width=True,
    )
    if data_df is not None:
        controls[1].download_button(
            "Descargar CSV",
            data=data_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{export_slug}.csv",
            mime="text/csv",
            key=f"csv_{export_slug}",
            use_container_width=True,
        )
    if caption:
        st.caption(caption)


def make_profession_distribution_bar(profession_distribution: pd.DataFrame, profession_order: List[str]) -> go.Figure:
    plot_df = profession_distribution.copy()
    fig = px.bar(
        plot_df,
        x="profession",
        y="n",
        color="profession",
        text="n",
        title="Distribución de participantes por profesión",
        category_orders={"profession": profession_order},
        color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
    )
    fig.update_layout(showlegend=False, xaxis_title="Profesión", yaxis_title="Participantes")
    fig.update_traces(textposition="outside")
    return style_academic_figure(fig, "Distribución de participantes por profesión", height=430, showlegend=False)


def make_framework_heatmap(framework_summary: pd.DataFrame, profession_order: List[str]) -> go.Figure:
    plot_df = framework_summary.copy()
    plot_df["framework_label"] = plot_df["framework"].map(framework_label)
    fig = px.density_heatmap(
        plot_df,
        x="framework_label",
        y="profession",
        z="framework_mean",
        histfunc="avg",
        color_continuous_scale=["#f3f7fb", "#9db9d3", "#1c5f8d", "#0b1f3a"],
        title="Intensidad promedio de marcos éticos por profesión",
        category_orders={"profession": profession_order},
    )
    fig.update_layout(xaxis_title="Marco ético", yaxis_title="Profesión", coloraxis_colorbar_title="Promedio")
    return style_academic_figure(fig, "Intensidad promedio de marcos éticos por profesión", height=500)


def make_collective_radar(framework_summary: pd.DataFrame, professions: List[str]) -> go.Figure:
    fig = go.Figure()
    if framework_summary.empty:
        return fig
    subset = framework_summary[framework_summary["profession"].isin(professions)].copy()
    for profession in professions:
        prof_data = subset[subset["profession"] == profession]
        if prof_data.empty:
            continue
        values = []
        for fw in FRAMEWORKS:
            match = prof_data.loc[prof_data["framework"] == fw, "framework_mean"]
            values.append(float(match.iloc[0]) if not match.empty else np.nan)
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[framework_label(fw) for fw in FRAMEWORKS],
            fill="toself",
            name=profession,
            opacity=0.35,
        ))
    return style_academic_figure(
        fig,
        "Radar comparativo de marcos éticos por profesión",
        height=520,
        polar=dict(radialaxis=dict(visible=True, range=[1, 7], gridcolor="#d7e2ee")),
    )


def make_collective_cluster_bar(cluster_summary: pd.DataFrame) -> go.Figure:
    plot_df = cluster_summary.copy()
    plot_df["cluster_label"] = plot_df["cluster"].map(lambda value: f"Cluster {value}")
    fig = px.bar(
        plot_df,
        x="cluster_label",
        y="n",
        color="dominant_profession",
        text="n",
        title="Peso relativo de clusters cualitativos",
        color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
    )
    fig.update_layout(xaxis_title="Cluster cualitativo", yaxis_title="Justificaciones")
    fig.update_traces(textposition="outside")
    return style_academic_figure(fig, "Peso relativo de clusters cualitativos", height=420)


def make_stage_profile_chart(stage_means: Dict[int, float], title: str) -> go.Figure:
    plot_df = pd.DataFrame({
        "stage": [f"Estadio {stage}" for stage in range(1, 7)],
        "mean": [stage_means.get(stage, np.nan) for stage in range(1, 7)],
    })
    fig = px.bar(
        plot_df,
        x="stage",
        y="mean",
        text="mean",
        title=title,
        color="mean",
        color_continuous_scale=["#d8e2ec", "#8ab0c9", "#1c5f8d"],
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_title="Estadio", yaxis_title="Promedio Likert", coloraxis_showscale=False)
    fig.update_yaxes(range=[0, 7.4])
    return style_academic_figure(fig, title, height=420, showlegend=False)


def make_frequency_summary_chart(frequency_summary: pd.DataFrame) -> go.Figure:
    if frequency_summary.empty:
        return go.Figure()
    plot_df = frequency_summary[frequency_summary["variable"].isin(["k_level", "fw_dom"])].copy()
    if plot_df.empty:
        return go.Figure()
    variable_map = {"k_level": "Nivel moral dominante", "fw_dom": "Marco ético dominante"}
    plot_df["variable_label"] = plot_df["variable"].map(variable_map)
    fig = px.bar(
        plot_df,
        x="category",
        y="pct",
        color="variable_label",
        barmode="group",
        text="n",
        title="Distribución de frecuencias: nivel moral y marco dominante",
        color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
        labels={"pct": "Porcentaje (%)", "category": "Categoría observada", "variable_label": "Variable"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Categoría", yaxis_title="% dentro de la variable")
    return style_academic_figure(fig, "Distribución de frecuencias: nivel moral y marco dominante", height=450)


def make_stage_ci_chart(stage_summary: pd.DataFrame, profession_order: List[str]) -> go.Figure:
    if stage_summary.empty:
        return go.Figure()
    plot_df = stage_summary.copy()
    plot_df["stage"] = pd.to_numeric(plot_df["stage"], errors="coerce")
    plot_df = plot_df.dropna(subset=["stage", "stage_mean"])
    if plot_df.empty:
        return go.Figure()
    plot_df["stage_label"] = plot_df["stage"].apply(lambda s: f"E{int(s)}")
    plot_df["error_up"] = (plot_df["ci_high"] - plot_df["stage_mean"]).clip(lower=0).fillna(0)
    plot_df["error_down"] = (plot_df["stage_mean"] - plot_df["ci_low"]).clip(lower=0).fillna(0)
    fig = px.scatter(
        plot_df,
        x="stage_label",
        y="stage_mean",
        color="profession",
        error_y="error_up",
        error_y_minus="error_down",
        title="Promedios por estadio moral con IC 95% (bootstrap)",
        category_orders={
            "profession": profession_order,
            "stage_label": [f"E{i}" for i in range(1, 7)],
        },
        color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
        labels={"stage_mean": "Promedio Likert", "stage_label": "Estadio"},
    )
    fig.update_layout(yaxis=dict(range=[0.5, 7.5]))
    return style_academic_figure(fig, "Promedios por estadio moral con IC 95% (bootstrap)", height=500)


def make_framework_ci_chart(framework_ci_summary: pd.DataFrame, profession_order: List[str]) -> go.Figure:
    if framework_ci_summary.empty:
        return go.Figure()
    plot_df = framework_ci_summary.copy()
    plot_df["framework_label"] = plot_df["framework"].map(framework_label)
    plot_df["error_up"] = (plot_df["ci_high"] - plot_df["fw_mean"]).clip(lower=0).fillna(0)
    plot_df["error_down"] = (plot_df["fw_mean"] - plot_df["ci_low"]).clip(lower=0).fillna(0)
    fig = px.scatter(
        plot_df,
        x="framework_label",
        y="fw_mean",
        color="profession",
        error_y="error_up",
        error_y_minus="error_down",
        title="Medias de marcos éticos por profesión con IC 95% (bootstrap)",
        category_orders={"profession": profession_order},
        color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
        labels={"fw_mean": "Media Likert", "framework_label": "Marco ético"},
    )
    fig.update_layout(yaxis=dict(range=[0.5, 7.5]))
    return style_academic_figure(fig, "Medias de marcos éticos por profesión con IC 95% (bootstrap)", height=500)


def make_keyword_bubble_chart(keyword_df: pd.DataFrame, group_col: str = "profession") -> go.Figure:
    if keyword_df.empty:
        return go.Figure()
    rows = []
    for _, row in keyword_df.head(10).iterrows():
        terms = [t.strip() for t in str(row["top_keywords"]).split(",") if t.strip()]
        for rank, term in enumerate(terms[:6], start=1):
            rows.append({
                group_col: str(row[group_col]),
                "termino": term,
                "rank": rank,
                "peso": max(1, 7 - rank),
            })
    if not rows:
        return go.Figure()
    plot_df = pd.DataFrame(rows)
    fig = px.scatter(
        plot_df,
        x="rank",
        y=group_col,
        text="termino",
        size="peso",
        color=group_col,
        title="Vocabulario clave por grupo (relevancia TF-IDF posicional)",
        color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
        labels={"rank": "Rango de relevancia (1 = más distintivo)", group_col: ""},
    )
    fig.update_traces(textposition="middle center", mode="markers+text", marker_opacity=0.4)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(tickmode="linear", dtick=1, range=[0.3, 6.7]),
        yaxis_title="",
    )
    n_groups = len(plot_df[group_col].unique())
    return style_academic_figure(
        fig,
        "Vocabulario clave por grupo (relevancia TF-IDF posicional)",
        height=max(360, n_groups * 52 + 180),
        showlegend=False,
    )


def build_contextual_summary_tables(students: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    categorical_rows = []
    for column in CONTEXT_CATEGORICAL_COLUMNS:
        if column not in students.columns:
            continue
        counts = students[column].fillna("No disponible").astype(str).value_counts(dropna=False)
        for category, count in counts.items():
            categorical_rows.append({
                "variable": CONTEXT_FIELD_LABELS.get(column, column),
                "categoria": category,
                "n": int(count),
                "pct": round(float(count) / max(len(students), 1) * 100, 2),
            })

    numeric_rows = []
    for column in CONTEXT_NUMERIC_COLUMNS:
        if column not in students.columns:
            continue
        values = pd.to_numeric(students[column], errors="coerce").dropna()
        if values.empty:
            continue
        numeric_rows.append({
            "variable": CONTEXT_FIELD_LABELS.get(column, column),
            "n": int(values.count()),
            "media": round(float(values.mean()), 2),
            "mediana": round(float(values.median()), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
        })

    return pd.DataFrame(categorical_rows), pd.DataFrame(numeric_rows)


def build_context_outcome_summary(students: pd.DataFrame, context_column: str) -> pd.DataFrame:
    if context_column not in students.columns:
        return pd.DataFrame()
    summary_df = students.copy()
    summary_df[context_column] = summary_df[context_column].fillna("No disponible")
    grouped = summary_df.groupby(context_column).agg(
        participantes=("anon_id", "count"),
        k_est_promedio=("k_est", "mean"),
        edad_promedio=("age", "mean"),
        semestre_promedio=("semester", "mean"),
        marco_dominante=("fw_dom", lambda s: s.mode().iloc[0] if s.notna().any() else None),
        nivel_dominante=("k_level", lambda s: s.mode().iloc[0] if s.notna().any() else None),
    ).reset_index().rename(columns={context_column: "categoria"})
    return grouped.sort_values("participantes", ascending=False)


def make_context_boxplot(students: pd.DataFrame, context_column: str) -> go.Figure:
    plot_df = students.copy()
    plot_df[context_column] = plot_df[context_column].fillna("No disponible")
    fig = px.box(
        plot_df,
        x=context_column,
        y="k_est",
        color=context_column,
        points="all",
        title=f"{CONTEXT_FIELD_LABELS.get(context_column, context_column)} vs k_est",
        color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
        labels={context_column: CONTEXT_FIELD_LABELS.get(context_column, context_column), "k_est": "Índice k_est"},
    )
    fig.update_layout(showlegend=False)
    return style_academic_figure(fig, f"{CONTEXT_FIELD_LABELS.get(context_column, context_column)} vs k_est", height=500, showlegend=False)


def make_context_level_chart(students: pd.DataFrame, context_column: str) -> go.Figure:
    plot_df = students.copy()
    plot_df[context_column] = plot_df[context_column].fillna("No disponible")
    level_distribution = pd.crosstab(plot_df[context_column], plot_df["k_level"], normalize="index") * 100
    level_distribution = level_distribution.reset_index().melt(id_vars=context_column, var_name="k_level", value_name="pct")
    fig = px.bar(
        level_distribution,
        x=context_column,
        y="pct",
        color="k_level",
        barmode="stack",
        title=f"Distribución de niveles morales por {CONTEXT_FIELD_LABELS.get(context_column, context_column)}",
        color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
        labels={context_column: CONTEXT_FIELD_LABELS.get(context_column, context_column), "pct": "Porcentaje (%)", "k_level": "Nivel moral"},
    )
    return style_academic_figure(fig, f"Distribución de niveles morales por {CONTEXT_FIELD_LABELS.get(context_column, context_column)}", height=470)


def make_numeric_context_scatter(students: pd.DataFrame, context_column: str) -> go.Figure:
    plot_df = students.copy()
    plot_df[context_column] = pd.to_numeric(plot_df[context_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[context_column, "k_est"])
    if plot_df.empty:
        return go.Figure()
    fig = px.scatter(
        plot_df,
        x=context_column,
        y="k_est",
        color="profession",
        title=f"{CONTEXT_FIELD_LABELS.get(context_column, context_column)} vs k_est",
        color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
        labels={context_column: CONTEXT_FIELD_LABELS.get(context_column, context_column), "k_est": "Índice k_est", "profession": "Profesión"},
        hover_data=["k_level", "fw_dom", "group"],
    )
    return style_academic_figure(fig, f"{CONTEXT_FIELD_LABELS.get(context_column, context_column)} vs k_est", height=500)


def build_executive_kpi_table(students: pd.DataFrame, quantitative_report: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    k_mean, k_lo, k_hi = bootstrap_ci(students["k_est"].dropna().values, np.mean)
    c_mean, _, _ = bootstrap_ci(students["k_coherence_std"].dropna().values, np.mean)
    top_framework = students["fw_dom"].mode().iloc[0] if students["fw_dom"].notna().any() else "Sin predominio"
    top_level = students["k_level"].mode().iloc[0] if students["k_level"].notna().any() else "Sin predominio"
    profession_distribution = quantitative_report.get("profession_distribution", pd.DataFrame())
    top_profession = profession_distribution.iloc[0]["profession"] if not profession_distribution.empty else "Sin datos"
    return pd.DataFrame([
        {"KPI": "Participantes consolidados", "Valor": int(len(students)), "Lectura": "Base efectiva del análisis colectivo"},
        {"KPI": "Profesiones representadas", "Valor": int(students["profession"].nunique()), "Lectura": "Diversidad disciplinar presente"},
        {"KPI": "Cohortes activas", "Valor": int(students["group"].fillna("Sin grupo").nunique()), "Lectura": "Número de grupos comparables"},
        {"KPI": "k_est promedio", "Valor": f"{k_mean:.2f}" if pd.notna(k_mean) else "NA", "Lectura": f"IC95% [{k_lo:.2f}, {k_hi:.2f}]" if pd.notna(k_mean) else "Sin estimación"},
        {"KPI": "Coherencia promedio", "Valor": f"{c_mean:.2f}" if pd.notna(c_mean) else "NA", "Lectura": "Dispersión interna entre estadios"},
        {"KPI": "Marco dominante más frecuente", "Valor": framework_label(str(top_framework)), "Lectura": "Predominio agregado observado"},
        {"KPI": "Nivel moral más frecuente", "Valor": str(top_level).title(), "Lectura": "Tendencia descriptiva agregada"},
        {"KPI": "Profesión con mayor peso muestral", "Valor": str(top_profession), "Lectura": "Mayor aporte relativo a la muestra"},
    ])


def metrics_section(students: pd.DataFrame) -> None:
    n = len(students)
    k_mean, k_lo, k_hi = bootstrap_ci(students["k_est"].dropna().values, np.mean)
    c_mean, c_lo, c_hi = bootstrap_ci(students["k_coherence_std"].dropna().values, np.mean)
    top_fw = students["fw_dom"].value_counts().idxmax() if students["fw_dom"].notna().any() else "sin datos"
    top_prof = students["profession"].value_counts().idxmax() if students["profession"].notna().any() else "sin datos"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Participantes", f"{n}")
    m2.metric("k_est promedio", f"{k_mean:.2f}" if pd.notna(k_mean) else "NA")
    m3.metric("Coherencia promedio", f"{c_mean:.2f}" if pd.notna(c_mean) else "NA")
    m4.metric("Marco dominante más frecuente", str(top_fw))
    st.caption(
        f"IC95% k_est: [{k_lo:.2f}, {k_hi:.2f}] | IC95% coherencia: [{c_lo:.2f}, {c_hi:.2f}] | "
        f"Profesión más frecuente: {top_prof}."
    )


def profession_gap_table(students: pd.DataFrame) -> pd.DataFrame:
    if students["profession"].nunique() < 2:
        return pd.DataFrame()
    p1, p2 = students["profession"].value_counts().index[:2]
    rows = []
    for fw in FRAMEWORKS:
        a = students.loc[students["profession"] == p1, f"fw_{fw}"].values
        b = students.loc[students["profession"] == p2, f"fw_{fw}"].values
        ma, la, ha = bootstrap_ci(a, np.mean)
        mb, lb, hb = bootstrap_ci(b, np.mean)
        rows.append({
            "framework": fw,
            f"{p1}_mean": ma,
            f"{p1}_IC95": f"[{la:.2f}, {ha:.2f}]" if pd.notna(ma) else "NA",
            f"{p2}_mean": mb,
            f"{p2}_IC95": f"[{lb:.2f}, {hb:.2f}]" if pd.notna(mb) else "NA",
            "hedges_g": hedges_g(a, b),
        })
    return pd.DataFrame(rows).sort_values("hedges_g", key=lambda s: s.abs(), ascending=False)


def qualitative_clusters(df_last: pd.DataFrame, profession_order: List[str] | None = None) -> Tuple[pd.DataFrame, go.Figure | None]:
    choices = df_last[df_last["row_type"] == "choice"].copy()
    choices["clean"] = choices["text"].fillna("").astype(str).map(clean_text)
    choices = choices[choices["clean"].str.len() > 0].copy()
    if len(choices) < 6:
        return pd.DataFrame(), None

    vectorizer = TfidfVectorizer(stop_words=list(BASE_STOPWORDS), max_features=1200, ngram_range=(1, 2))
    X = vectorizer.fit_transform(choices["clean"])
    k = min(6, max(2, int(np.sqrt(X.shape[0]))))
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X)
    choices["cluster"] = labels

    terms = np.array(vectorizer.get_feature_names_out())
    rows = []
    for cluster_id in range(k):
        top_idx = np.argsort(km.cluster_centers_[cluster_id])[::-1][:10]
        rows.append({
            "cluster": cluster_id,
            "top_terms": ", ".join(terms[top_idx]),
            "n": int((choices["cluster"] == cluster_id).sum()),
        })
    summary = pd.DataFrame(rows)

    heat = pd.crosstab(choices["profession"], choices["cluster"], normalize="index") * 100
    heat = heat.reset_index().melt(id_vars="profession", var_name="cluster", value_name="pct")
    fig = px.density_heatmap(
        heat,
        x="cluster",
        y="profession",
        z="pct",
        histfunc="avg",
        title="Distribución de clusters argumentativos por profesión (%)",
        height=450,
        category_orders={"profession": profession_order or profession_category_order(heat["profession"])},
    )
    return summary, fig


# =========================
# Páginas
# =========================
def page_apply(df: pd.DataFrame) -> None:
    st.subheader("Aplicación individual")
    st.write("Completa el instrumento, genera el reporte individual inmediato y obtén análisis IA integrado con exportación PDF.")
    a1, a2, a3 = st.columns(3)
    a1.metric("Profesiones disponibles", str(len(PROFESSION_ROUTES)))
    a2.metric("Marcos éticos evaluados", str(len(FRAMEWORKS)))
    a3.metric("Justificación mínima", f"{MIN_JUSTIFICATION_CHARS} caracteres")
    st.caption("La app valora razonamiento argumentativo básico: opción elegida, calidad de la justificación, patrones por estadio y afinidad con marcos éticos.")

    with st.sidebar:
        st.markdown("### Parámetros de aplicación")
        route_size = st.slider("Número de dilemas por ruta", min_value=4, max_value=10, value=DEFAULT_ROUTE_SIZE, step=1)
        anonymize = st.checkbox("Anonimizar identificador", value=True)

    st.markdown("### Configuración de la ruta profesional")
    route_col1, route_col2 = st.columns([2, 1])
    profession = route_col1.selectbox(
        "Profesión",
        options=PROFESSION_OPTIONS,
        key="apply_profession_selector",
    )
    years_experience = route_col2.number_input(
        "Años de experiencia laboral o clínica",
        min_value=0,
        max_value=60,
        value=0,
        step=1,
        key="apply_years_experience",
    )

    route_group = route_group_for_profession(profession)
    full_route_ids = ROUTE_BANKS.get(route_group, [])
    dilemmas = route_for_profession(profession, route_size)

    st.caption(
        f"Profesión seleccionada: {profession} | Ruta activa: {route_group} | Dilemas visibles: {len(dilemmas)}"
    )
    semester_max = semester_limit_for_profession(profession)

    with st.form("moral_test_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1, 1, 1])
        student_id = c1.text_input("ID institucional o código", max_chars=60)
        name = c2.text_input("Nombre o seudónimo", max_chars=120)
        group = c3.text_input("Grupo / cohorte / curso", max_chars=120)

        st.markdown("### Perfil sociodemográfico y académico")
        d1, d2, d3 = st.columns([1.1, 0.9, 0.9])
        gender = d1.selectbox("Género", options=[""] + GENDER_OPTIONS, key="apply_gender")
        age = d2.number_input(
            "Edad",
            min_value=MIN_AGE,
            max_value=MAX_AGE,
            value=max(MIN_AGE, 18),
            step=1,
            key="apply_age",
        )
        semester = d3.number_input(
            f"Semestre (1 a {semester_max})",
            min_value=1,
            max_value=semester_max,
            value=min(1, semester_max),
            step=1,
            key="apply_semester",
        )

        d4, d5 = st.columns([1.2, 0.8])
        works_for_studies = d4.selectbox(
            "Trabaja para pagar sus estudios",
            options=[""] + WORK_STUDY_OPTIONS,
            key="apply_work_study",
        )
        children_count = d5.number_input(
            "Número de hijos",
            min_value=0,
            max_value=MAX_CHILDREN,
            value=0,
            step=1,
            key="apply_children_count",
        )

        with st.expander("Variables contextuales opcionales"):
            o1, o2, o3 = st.columns(3)
            academic_program = o1.text_input("Programa académico o denominación específica", max_chars=120)
            academic_shift = o2.selectbox("Jornada académica", options=ACADEMIC_SHIFT_OPTIONS, key="apply_academic_shift")
            prior_experience_area = o3.selectbox(
                "Experiencia laboral o clínica previa",
                options=PRIOR_EXPERIENCE_OPTIONS,
                key="apply_prior_experience_area",
            )

            o4, o5, o6 = st.columns(3)
            ethics_training = o4.selectbox(
                "Formación previa en ética o bioética",
                options=ETHICS_TRAINING_OPTIONS,
                key="apply_ethics_training",
            )
            work_hours_per_week_input = o5.text_input(
                "Horas de trabajo por semana",
                max_chars=3,
                placeholder="0-80",
            )
            caregiving_load = o6.selectbox(
                "Carga de cuidado familiar",
                options=CAREGIVING_LOAD_OPTIONS,
                key="apply_caregiving_load",
            )

            study_funding_type = st.selectbox(
                "Tipo de financiación de estudios",
                options=STUDY_FUNDING_OPTIONS,
                key="apply_study_funding_type",
            )

        st.markdown("### A. Dilemas y justificación argumentativa")
        st.caption("Se exige una justificación breve con razonamiento suficiente; evita respuestas telegráficas.")
        if route_group in {"bacteriologia_laboratorio", "microbiologia_laboratorio"}:
            st.info(
                f"Ruta específica de {profession}: se cargaron {len(dilemmas)} de {len(full_route_ids)} dilemas disponibles. "
                "Los dilemas específicos de la profesión aparecen primero; si quieres ver toda la ruta, aumenta el valor del control 'Número de dilemas por ruta'."
            )
            with st.expander(f"Ver catálogo completo de la ruta de {profession}"):
                for item_id in full_route_ids:
                    st.markdown(f"- **{item_id}**: {LOOKUP[item_id]['title']}")

        collected_choices = []
        for i, dilemma in enumerate(dilemmas, start=1):
            st.markdown(f"**{i}. {dilemma['title']}**")
            st.write(dilemma["prompt"])
            if dilemma.get("pedagogical_justification"):
                st.caption(f"Justificación pedagógica: {dilemma['pedagogical_justification']}")
            options_map = {f"{o['key']}. {o['text']}": o for o in dilemma["options"]}
            selected_label = st.radio(
                f"Selecciona la opción para {dilemma['id']}",
                options=list(options_map.keys()),
                key=f"choice_{dilemma['id']}",
            )
            justification = st.text_area(
                f"Justificación de {dilemma['id']}",
                key=f"just_{dilemma['id']}",
                height=100,
                placeholder="Explica la decisión con argumentos, principios, consecuencias y salvaguardas.",
            )
            collected_choices.append((dilemma, selected_label, justification))
            st.divider()

        st.markdown("### B. Likert por estadio (1 a 7)")
        st.caption("1 = nada de acuerdo | 7 = totalmente de acuerdo")
        stage_values = []
        for dilemma in dilemmas:
            with st.expander(f"Valora los estadios para {dilemma['id']} — {dilemma['title']}"):
                for stage in range(1, 7):
                    value = st.slider(
                        f"{dilemma['id']} · Estadio {stage}",
                        min_value=1,
                        max_value=7,
                        value=4,
                        key=f"stage_{dilemma['id']}_{stage}",
                    )
                    st.caption(STAGE_TEMPLATE[stage])
                    stage_values.append({"item_id": dilemma["id"], "sub_id": stage, "likert_value": value})

        st.markdown("### C. Inventario de marcos éticos (1 a 7)")
        framework_values = []
        for item_id, framework, text in FRAMEWORK_INVENTORY:
            value = st.slider(
                f"{item_id} · {framework}",
                min_value=1,
                max_value=7,
                value=4,
                key=f"fw_{item_id}",
            )
            st.caption(text)
            framework_values.append({"item_id": item_id, "sub_id": framework, "likert_value": value, "text": text})

        submitted = st.form_submit_button("Guardar respuesta y generar reporte", use_container_width=True)

    report_context = st.session_state.get(INDIVIDUAL_REPORT_SESSION_KEY)

    if submitted:
        errors = []
        if not student_id.strip():
            errors.append("Debes diligenciar el ID institucional o código.")
        if not name.strip():
            errors.append("Debes diligenciar el nombre o seudónimo.")
        if not gender:
            errors.append("Debes seleccionar el género.")
        if not works_for_studies:
            errors.append("Debes indicar si trabajas para pagar tus estudios.")
        if int(age) < MIN_AGE or int(age) > MAX_AGE:
            errors.append(f"La edad debe estar entre {MIN_AGE} y {MAX_AGE} años.")
        if int(semester) < 1 or int(semester) > semester_max:
            errors.append(f"El semestre para {profession} debe estar entre 1 y {semester_max}.")
        if int(children_count) < 0 or int(children_count) > MAX_CHILDREN:
            errors.append(f"El número de hijos debe estar entre 0 y {MAX_CHILDREN}.")

        try:
            work_hours_per_week = optional_int_from_text(
                work_hours_per_week_input,
                minimum=0,
                maximum=MAX_WORK_HOURS,
                field_label="Las horas de trabajo por semana",
            )
        except ValueError as exc:
            errors.append(str(exc))
            work_hours_per_week = None

        participant_context = {
            "gender": gender,
            "age": int(age),
            "semester": int(semester),
            "works_for_studies": works_for_studies,
            "children_count": int(children_count),
            "academic_program": optional_text(academic_program),
            "academic_shift": optional_text(academic_shift),
            "prior_experience_area": optional_text(prior_experience_area),
            "ethics_training": optional_text(ethics_training),
            "work_hours_per_week": work_hours_per_week,
            "caregiving_load": optional_text(caregiving_load),
            "study_funding_type": optional_text(study_funding_type),
        }

        choices_payload = []
        for dilemma, selected_label, justification in collected_choices:
            if not selected_label:
                errors.append(f"Falta elegir una opción en {dilemma['id']}.")
                continue
            clean_just = justification.strip()
            if len(clean_just) < MIN_JUSTIFICATION_CHARS:
                errors.append(
                    f"La justificación de {dilemma['id']} es demasiado corta. Usa al menos {MIN_JUSTIFICATION_CHARS} caracteres."
                )
                continue
            chosen = selected_label.split(".")[0].strip()
            option = {o["key"]: o for o in dilemma["options"]}[chosen]
            choices_payload.append({
                "item_id": dilemma["id"],
                "choice_key": chosen,
                "choice_stage": int(option["stage"]),
                "choice_level": option["level"],
                "choice_framework": option["framework"],
                "text": clean_just,
            })

        if errors:
            for error in errors:
                st.error(error)
        else:
            new_rows = build_rows(
                student_id=student_id.strip(),
                name=name.strip(),
                profession=profession,
                years_experience=int(years_experience),
                group=group.strip(),
                anonymize=anonymize,
                participant_context=participant_context,
                choices_payload=choices_payload,
                stage_payload=stage_values,
                framework_payload=framework_values,
            )
            save_attempt_rows(new_rows)
            report_context = create_individual_report_context(
                rows_df=new_rows,
                student_id=student_id.strip(),
                name=name.strip(),
                profession=profession,
                years_experience=int(years_experience),
                group=group.strip(),
                route_group=route_group,
            )
            report_context = run_individual_ai_analysis(
                report_context,
                spinner_text="Generando análisis IA integrado al reporte individual...",
            )
            report_context = rebuild_individual_report_artifacts(report_context)
            st.session_state[INDIVIDUAL_REPORT_SESSION_KEY] = report_context
            st.success("Respuesta guardada correctamente. El reporte individual ya incluye análisis IA integrado y exportación PDF.")

    if report_context:
        render_individual_report(report_context)


def page_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dashboard colectivo")
    if df.empty:
        st.warning("Aún no hay respuestas registradas.")
        return

    students, df_last = student_table(df)
    if students.empty:
        st.warning("No fue posible consolidar participantes.")
        return

    students = students.copy()
    df_last = df_last.copy()
    students["group_filter"] = students["group"].apply(normalize_group_label)

    profession_options = [
        value for value in profession_category_order(students["profession"])
        if value in students["profession"].dropna().astype(str).unique().tolist()
    ]
    group_options = sorted(students["group_filter"].dropna().astype(str).unique().tolist())

    year_values = students["years_experience"].dropna().astype(float)
    year_min = int(year_values.min()) if not year_values.empty else 0
    year_max = int(year_values.max()) if not year_values.empty else 0

    st.markdown("### Filtros")
    f1, f2, f3 = st.columns([1.4, 1.2, 1.2])
    selected_professions = f1.multiselect(
        "Profesiones",
        options=profession_options,
        default=profession_options,
    )
    selected_groups = f2.multiselect(
        "Grupos / cohortes",
        options=group_options,
        default=group_options,
    )
    if year_min < year_max:
        selected_year_range = f3.slider(
            "Años de experiencia",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
        )
    else:
        f3.caption(f"Años de experiencia: {year_min}")
        selected_year_range = (year_min, year_max)

    students_filtered, df_last_filtered = apply_dashboard_filters(
        students,
        df_last,
        selected_professions,
        selected_groups,
        selected_year_range,
    )

    if students_filtered.empty:
        st.warning("Los filtros actuales no devuelven participantes. Ajusta la selección para continuar.")
        return

    st.caption(f"Mostrando {len(students_filtered)} de {len(students)} participantes consolidados.")

    ADMIN_REPORT_STORE.save_collective_snapshot(
        students_df=students.drop(columns=["group_filter"], errors="ignore"),
        last_attempt_df=df_last,
        snapshot_name="latest_global",
    )
    ADMIN_REPORT_STORE.save_collective_snapshot(
        students_df=students_filtered.drop(columns=["group_filter"], errors="ignore"),
        last_attempt_df=df_last_filtered.drop(columns=["group_filter"], errors="ignore"),
        snapshot_name="latest_filtered",
    )

    quantitative_report = build_quantitative_report(
        students_filtered.drop(columns=["group_filter"], errors="ignore"),
        df_last_filtered.drop(columns=["group_filter"], errors="ignore"),
        FRAMEWORKS,
    )
    keyword_df = extract_keywords_by_group(df_last_filtered, BASE_STOPWORDS, group_col="profession")
    cluster_summary, cluster_distribution = cluster_thematic_justifications(df_last_filtered, BASE_STOPWORDS, RANDOM_SEED)
    pattern_summary = argumentative_pattern_table(df_last_filtered, BASE_STOPWORDS)
    trends_df = profession_interpretive_trends(
        students_filtered.drop(columns=["group_filter"], errors="ignore"),
        keyword_df,
        pattern_summary,
    )
    collective_synthesis = automatic_interpretive_synthesis(
        students_filtered.drop(columns=["group_filter"], errors="ignore"),
        quantitative_report,
        trends_df,
        cluster_summary,
    )
    processed_texts = df_last_filtered[df_last_filtered["row_type"] == "choice"].copy()
    if not processed_texts.empty:
        processed_texts["texto_limpio"] = clean_text_series(processed_texts["text"], BASE_STOPWORDS)
    profession_order = profession_category_order(students_filtered["profession"])
    profession_distribution = quantitative_report["profession_distribution"].copy()
    framework_summary = quantitative_report["framework_summary"].copy()
    profession_comparisons = quantitative_report["profession_comparisons"].copy()
    cohort_summary = quantitative_report["cohort_summary"].copy()
    executive_kpis = build_executive_kpi_table(students_filtered, quantitative_report)
    framework_ci_summary = quantitative_report.get("framework_ci_summary", pd.DataFrame()).copy()
    stage_summary_df = quantitative_report.get("stage_summary", pd.DataFrame()).copy()
    consistency_df = internal_consistency_estimate(df_last_filtered)
    contextual_categorical_df, contextual_numeric_df = build_contextual_summary_tables(students_filtered)
    collective_choice_df = df_last_filtered[df_last_filtered["row_type"] == "choice"].copy()
    if not collective_choice_df.empty:
        collective_choice_df["framework_label"] = collective_choice_df["choice_framework"].map(framework_label)

    metrics_section(students_filtered)

    tabs = st.tabs([
        "1. Resumen general",
        "2. Análisis individual",
        "3. Análisis por profesión",
        "4. Análisis cualitativo",
        "5. Interpretación integrada",
        "6. Contexto ampliado",
    ])

    with tabs[0]:
        st.markdown(
            """
            <div class="section-card">
                <h3>Resumen general</h3>
                <p>Vista ejecutiva del comportamiento agregado de la cohorte, con indicadores clave y visualizaciones de distribución de alto nivel para docencia, seguimiento y análisis académico.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Tabla ejecutiva con KPIs")
        st.dataframe(executive_kpis, use_container_width=True)

        summary_col1, summary_col2 = st.columns([1, 1])
        with summary_col1:
            profession_bar = make_profession_distribution_bar(profession_distribution, profession_order)
            render_plotly_figure(
                profession_bar,
                "barras_profesion",
                data_df=profession_distribution,
                caption="Distribución absoluta y relativa de la muestra consolidada por profesión.",
            )
        with summary_col2:
            sun = students_filtered.dropna(subset=["profession", "k_level", "fw_dom"]).copy()
            if sun.empty:
                st.info("No hay datos suficientes para construir el sunburst agregado.")
            else:
                sun["marco_dominante_label"] = sun["fw_dom"].map(framework_label)
                sunburst_fig = px.sunburst(
                    sun,
                    path=["profession", "k_level", "marco_dominante_label"],
                    title="Profesión → nivel → marco dominante",
                    color="k_level",
                    color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
                )
                sunburst_fig = style_academic_figure(sunburst_fig, "Profesión → nivel → marco dominante", height=560)
                render_plotly_figure(
                    sunburst_fig,
                    "sunburst_profesion_nivel_marco",
                    data_df=sun[["profession", "k_level", "marco_dominante_label"]],
                    caption="Jerarquía agregada entre profesión, nivel moral predominante y marco ético dominante.",
                )

        st.markdown("#### Flujo agregado de dilemas hacia marcos éticos")
        if collective_choice_df.empty:
            st.info("No hay respuestas de dilemas suficientes para construir el Sankey colectivo.")
        else:
            collective_sankey = make_sankey_from_choices(collective_choice_df, "Dilema → marco ético")
            render_plotly_figure(
                collective_sankey,
                "sankey_dilema_marco",
                data_df=collective_choice_df[["profession", "item_id", "choice_framework", "framework_label", "choice_level"]],
                caption="Mapa de decisión agregado entre los dilemas respondidos y los marcos éticos finalmente seleccionados.",
            )

        st.markdown("#### Distribución de frecuencias por categoría analítica")
        freq_data = quantitative_report.get("frequency_summary", pd.DataFrame())
        if freq_data.empty:
            st.info("No hay datos suficientes para construir el gráfico de frecuencias.")
        else:
            freq_fig = make_frequency_summary_chart(freq_data)
            render_plotly_figure(
                freq_fig,
                "frecuencias_nivel_marco",
                data_df=freq_data,
                caption="Frecuencias relativas de nivel moral y marco ético dominante en el conjunto filtrado. Las barras agrupadas permiten comparar la distribución observada entre categorias.",
            )

    with tabs[1]:
        st.markdown(
            """
            <div class="section-card">
                <h3>Análisis individual</h3>
                <p>Explora el perfil de un participante dentro del conjunto filtrado para apoyar retroalimentación tutorial, discusión en aula y seguimiento formativo individual.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        participant_df = students_filtered[["anon_id", "name", "student_id", "profession", "group", "timestamp"]].copy()
        participant_df["display_label"] = participant_df.apply(
            lambda row: f"{row['name'] or row['student_id'] or row['anon_id'][:8]} | {row['profession']} | {normalize_group_label(row['group'])}",
            axis=1,
        )
        selected_label = st.selectbox("Selecciona participante", options=participant_df["display_label"].tolist())
        selected_anon = participant_df.loc[participant_df["display_label"] == selected_label, "anon_id"].iloc[0]
        selected_student = students_filtered[students_filtered["anon_id"] == selected_anon].iloc[0]
        selected_rows = df_last_filtered[df_last_filtered["anon_id"] == selected_anon].copy()
        selected_choice_df = selected_rows[selected_rows["row_type"] == "choice"].copy()
        selected_stage_df = selected_rows[selected_rows["row_type"] == "stage_likert"].copy()
        selected_fw_df = selected_rows[selected_rows["row_type"] == "framework_likert"].copy()

        ind_k_est, ind_k_stage, ind_k_level, ind_stage_means = kohlberg_from_stage_likert(selected_stage_df)
        ind_fw_scores, ind_fw_dom = framework_scores(selected_fw_df)
        ind_coherence = float(np.nanstd(list(ind_stage_means.values()))) if ind_stage_means else np.nan

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Profesión", str(selected_student["profession"]))
        i2.metric("Nivel global", str(ind_k_level))
        i3.metric("Índice k_est", f"{ind_k_est:.2f}" if pd.notna(ind_k_est) else "NA")
        i4.metric("Marco dominante", framework_label(str(ind_fw_dom)) if ind_fw_dom else "NA")
        st.caption(
            f"Grupo: {normalize_group_label(selected_student['group'])} | Estadio redondeado: {ind_k_stage} | Coherencia interna: {ind_coherence:.2f}"
        )
        selected_context_rows = build_context_display_rows(participant_context_from_row(selected_student))
        if selected_context_rows:
            with st.expander("Ver contexto sociodemográfico, académico y laboral"):
                st.dataframe(
                    pd.DataFrame(selected_context_rows, columns=["Variable", "Valor"]),
                    use_container_width=True,
                    hide_index=True,
                )

        ind_col1, ind_col2 = st.columns([1, 1])
        with ind_col1:
            individual_radar = make_radar(ind_fw_scores, "Perfil individual de marcos éticos")
            render_plotly_figure(
                individual_radar,
                f"radar_individual_{selected_anon}",
                data_df=pd.DataFrame([{"framework": framework_label(key), "score": value} for key, value in ind_fw_scores.items()]),
                caption="Perfil individual de afinidad relativa con los marcos éticos evaluados.",
            )
        with ind_col2:
            stage_chart = make_stage_profile_chart(ind_stage_means, "Perfil individual por estadios morales")
            render_plotly_figure(
                stage_chart,
                f"estadios_individual_{selected_anon}",
                data_df=pd.DataFrame([{"stage": stage, "mean": value} for stage, value in ind_stage_means.items()]),
                caption="Promedios individuales en los ítems tipo Likert vinculados a estadios de razonamiento moral.",
            )

        if selected_choice_df.empty:
            st.info("Este participante no tiene elecciones de dilemas disponibles para Sankey o detalle argumentativo.")
        else:
            individual_sankey = make_sankey_from_choices(selected_choice_df, "Dilema → marco ético del participante")
            render_plotly_figure(
                individual_sankey,
                f"sankey_individual_{selected_anon}",
                data_df=selected_choice_df[["item_id", "choice_framework", "choice_level", "text"]],
                caption="Flujo individual entre dilemas respondidos y marcos éticos escogidos.",
            )
            with st.expander("Ver justificaciones y elecciones del participante"):
                st.dataframe(
                    selected_choice_df[["item_id", "choice_level", "choice_framework", "text"]],
                    use_container_width=True,
                )

    with tabs[2]:
        st.markdown(
            """
            <div class="section-card">
                <h3>Análisis por profesión</h3>
                <p>Compara dispersiones, promedios éticos y perfiles agregados por profesión para apoyar decisiones docentes, lectura comparativa entre grupos y análisis de tendencias disciplinares.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        violin_fig = px.violin(
            students_filtered,
            x="profession",
            y="k_est",
            color="profession",
            box=True,
            points="all",
            category_orders={"profession": profession_order},
            color_discrete_sequence=ACADEMIC_COLOR_SEQUENCE,
            title="Distribución de k_est por profesión",
        )
        violin_fig.update_layout(showlegend=False, xaxis_title="Profesión", yaxis_title="Índice k_est")
        violin_fig = style_academic_figure(violin_fig, "Distribución de k_est por profesión", height=520, showlegend=False)
        render_plotly_figure(
            violin_fig,
            "violin_kest_profesion",
            data_df=students_filtered[["profession", "k_est", "k_level", "fw_dom"]],
            caption="La forma y amplitud de cada distribución permiten revisar dispersión, asimetrías y concentración de puntajes por profesión.",
        )

        if framework_summary.empty:
            st.info("No hay datos suficientes para consolidar el heatmap y el radar comparativo por profesión.")
        else:
            prof_col1, prof_col2 = st.columns([1.15, 1])
            with prof_col1:
                framework_heatmap = make_framework_heatmap(framework_summary, profession_order)
                render_plotly_figure(
                    framework_heatmap,
                    "heatmap_marcos_profesion",
                    data_df=framework_summary,
                    caption="Comparación de la intensidad promedio de cada marco ético entre profesiones.",
                )
            with prof_col2:
                default_radar_professions = profession_distribution.head(min(4, len(profession_distribution)))["profession"].tolist()
                selected_radar_professions = st.multiselect(
                    "Profesiones para radar comparativo",
                    options=profession_order,
                    default=default_radar_professions,
                )
                if len(selected_radar_professions) < 2:
                    st.info("Selecciona al menos dos profesiones para el radar comparativo.")
                else:
                    radar_fig = make_collective_radar(framework_summary, selected_radar_professions)
                    radar_export = framework_summary[framework_summary["profession"].isin(selected_radar_professions)].copy()
                    render_plotly_figure(
                        radar_fig,
                        "radar_comparativo_profesiones",
                        data_df=radar_export,
                        caption="Comparación de perfiles medios de marcos éticos entre profesiones seleccionadas.",
                    )

        st.markdown("#### Comparaciones descriptivas entre profesiones")
        if profession_comparisons.empty:
            st.info("Se requieren al menos dos profesiones con suficiente tamaño muestral para estimar comparaciones y tamaños de efecto.")
        else:
            st.dataframe(profession_comparisons, use_container_width=True)
            st.caption("Los tamaños de efecto y los intervalos de confianza se interpretan como apoyo descriptivo-formativo y no como prueba concluyente por sí sola.")

        st.markdown("#### Perfil por estadio moral con intervalos de confianza (IC 95%)")
        if stage_summary_df.empty:
            st.info("No hay datos de estadio suficientes para el perfil con IC por profesión.")
        else:
            stage_ci_fig = make_stage_ci_chart(stage_summary_df, profession_order)
            render_plotly_figure(
                stage_ci_fig,
                "estadios_ic_profesion",
                data_df=stage_summary_df,
                caption="Medias Likert de cada estadio Kohlberg por profesión con IC bootstrap al 95%. Mayor promedio en un estadio no implica mayor mérito moral; es una señal argumentativa descriptiva.",
            )

        st.markdown("#### Marcos éticos por profesión con intervalos de confianza (IC 95%)")
        if framework_ci_summary.empty:
            st.info("No hay datos suficientes para estimar IC de marcos por profesión.")
        else:
            fw_ci_fig = make_framework_ci_chart(framework_ci_summary, profession_order)
            render_plotly_figure(
                fw_ci_fig,
                "marcos_ic_profesion",
                data_df=framework_ci_summary,
                caption="Medias de afinidad con cada marco ético por profesión con IC 95% bootstrap. Las barras de error expresan incertidumbre en la estimación, no variabilidad individual.",
            )

    with tabs[3]:
        st.markdown(
            """
            <div class="section-card">
                <h3>Análisis cualitativo</h3>
                <p>Integra agrupamientos temáticos, patrones argumentativos y repertorios léxicos para reconocer tendencias discursivas de interés docente e investigador.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        qual_col1, qual_col2 = st.columns([1, 1])
        with qual_col1:
            if cluster_summary.empty:
                st.info("Se requieren más justificaciones textuales para construir gráficos cualitativos robustos.")
            else:
                cluster_bar = make_collective_cluster_bar(cluster_summary)
                render_plotly_figure(
                    cluster_bar,
                    "clusters_cualitativos",
                    data_df=cluster_summary,
                    caption="Peso relativo y perfil profesional dominante de los clusters cualitativos identificados.",
                )
        with qual_col2:
            if cluster_distribution.empty:
                st.info("No hay distribución suficiente por profesión para el heatmap cualitativo.")
            else:
                cluster_heatmap = px.density_heatmap(
                    cluster_distribution,
                    x="cluster",
                    y="profession",
                    z="pct",
                    histfunc="avg",
                    color_continuous_scale=["#f3f7fb", "#9db9d3", "#1c5f8d", "#0b1f3a"],
                    title="Distribución de clusters temáticos por profesión (%)",
                    category_orders={"profession": profession_order},
                )
                cluster_heatmap.update_layout(xaxis_title="Cluster", yaxis_title="Profesión", coloraxis_colorbar_title="%")
                cluster_heatmap = style_academic_figure(cluster_heatmap, "Distribución de clusters temáticos por profesión (%)", height=460)
                render_plotly_figure(
                    cluster_heatmap,
                    "heatmap_clusters_profesion",
                    data_df=cluster_distribution,
                    caption="Presencia relativa de cada cluster cualitativo dentro de cada profesión.",
                )

        st.markdown("#### Vocabulario clave por grupo (TF-IDF)")
        if keyword_df.empty:
            st.info("No fue posible extraer palabras clave por profesión con la información disponible.")
        else:
            kw_fig = make_keyword_bubble_chart(keyword_df)
            render_plotly_figure(
                kw_fig,
                "vocabulario_clave_grupos",
                data_df=keyword_df,
                caption="Términos más distintivos por grupo según relevancia TF-IDF relativa. El rango 1 indica el término más diferenciador respecto a los demás grupos.",
            )
            with st.expander("Ver tabla de palabras clave por grupo"):
                st.dataframe(keyword_df, use_container_width=True)

        st.markdown("#### Patrones argumentativos y tendencias interpretativas")
        q_table1, q_table2 = st.columns([1, 1])
        with q_table1:
            if pattern_summary.empty:
                st.info("No fue posible identificar patrones argumentativos con la información disponible.")
            else:
                st.dataframe(pattern_summary, use_container_width=True)
        with q_table2:
            if trends_df.empty:
                st.info("No hay datos suficientes para consolidar tendencias interpretativas por profesión.")
            else:
                st.dataframe(trends_df, use_container_width=True)

        with st.expander("Ver muestra de textos procesados"):
            if processed_texts.empty:
                st.info("No hay justificaciones textuales suficientes para mostrar ejemplos procesados.")
            else:
                st.dataframe(
                    processed_texts[["profession", "item_id", "text", "texto_limpio"]].head(30),
                    use_container_width=True,
                )

    with tabs[4]:
        st.markdown(
            """
            <div class="section-card">
                <h3>Interpretación integrada</h3>
                <p>Articula los hallazgos cuantitativos y cualitativos en una lectura final prudente, útil para acompañamiento pedagógico, docencia, investigación educativa y revisión curricular.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        interp_notes = []
        if not profession_distribution.empty:
            top_row = profession_distribution.iloc[0]
            interp_notes.append(
                f"La mayor representación muestral corresponde a {top_row['profession']} (n={int(top_row['n'])}, {float(top_row['pct']):.1f}%)."
            )
        if not profession_comparisons.empty:
            strongest_gap = profession_comparisons.iloc[0]
            if pd.notna(strongest_gap.get("hedges_g")):
                interp_notes.append(
                    f"La comparación descriptiva más visible en k_est aparece entre {strongest_gap['profession_a']} y {strongest_gap['profession_b']} con Hedges g={strongest_gap['hedges_g']:.2f}."
                )
        if not cluster_summary.empty:
            lead_cluster = cluster_summary.iloc[0]
            interp_notes.append(
                f"El cluster cualitativo de mayor peso es el {int(lead_cluster['cluster'])}, asociado a términos como {lead_cluster['top_terms']}."
            )
        if not trends_df.empty:
            lead_trend = trends_df.iloc[0]
            interp_notes.append(
                f"En {lead_trend['profession']} destaca una combinación entre nivel {lead_trend['nivel_dominante']}, marco {lead_trend['marco_dominante']} y patrón {lead_trend['patron_argumentativo']}."
            )

        interp_col1, interp_col2 = st.columns([1.2, 1])
        with interp_col1:
            st.markdown(f"<div class='interpretation-panel'>{collective_synthesis}</div>", unsafe_allow_html=True)
        with interp_col2:
            st.markdown("#### Hallazgos clave")
            if interp_notes:
                for note in interp_notes:
                    st.write(f"- {note}")
            else:
                st.info("Aún no hay suficientes regularidades para resumir hallazgos integrados.")

            st.markdown("#### Implicaciones de uso")
            st.write("- Priorizar lectura formativa y comparativa prudente, especialmente cuando algunas profesiones concentran mayor peso muestral.")
            st.write("- Usar los perfiles éticos y argumentativos como insumo para retroalimentación docente, no como veredicto definitivo sobre el estudiante o profesional.")
            st.write("- Integrar los hallazgos en revisión curricular, discusión de casos y diseño de actividades de deliberación ética contextualizada.")

        st.markdown("#### Soportes para interpretación")
        interp_table1, interp_table2 = st.columns([1, 1])
        with interp_table1:
            st.dataframe(cohort_summary, use_container_width=True)
        with interp_table2:
            st.dataframe(quantitative_report["descriptive_summary"], use_container_width=True)

        st.markdown("#### Niveles dominantes por participante")
        dominant_levels_df = quantitative_report.get("dominant_levels", pd.DataFrame())
        if dominant_levels_df.empty:
            st.info("No hay datos de niveles dominantes disponibles.")
        else:
            st.dataframe(dominant_levels_df, use_container_width=True)
            st.caption("Nivel moral global dominante, estadio redondeado, marco ético dominante y modo de elección observados para cada participante en su último intento registrado.")

        st.markdown("#### Señal de consistencia interna del bloque Likert por estadio")
        if consistency_df.empty:
            st.info("Se requieren al menos 3 participantes por profesión para estimar la consistencia interna del bloque de estadios.")
        else:
            st.dataframe(consistency_df, use_container_width=True)
            st.caption(
                "Correlación media interitem y proxy de Spearman-Brown (alpha_proxy) dentro del bloque Likert de estadios morales. "
                "Se interpreta como señal descriptiva de cohesión interna del instrumento en esta muestra, no como diagnóstico psicométrico definitivo."
            )

        st.markdown("#### Exportaciones y trazabilidad")
        export_col1, export_col2 = st.columns([1, 1])
        students_export = students_filtered.drop(columns=["group_filter"], errors="ignore")
        df_last_export = df_last_filtered.drop(columns=["group_filter"], errors="ignore")
        export_col1.download_button(
            "Descargar resumen consolidado (CSV)",
            data=students_export.to_csv(index=False).encode("utf-8"),
            file_name="students_summary_filtered.csv",
            mime="text/csv",
            key="students_summary_filtered_csv",
            use_container_width=True,
        )
        export_col2.download_button(
            "Descargar filas del último intento (CSV)",
            data=df_last_export.to_csv(index=False).encode("utf-8"),
            file_name="last_attempt_rows_filtered.csv",
            mime="text/csv",
            key="last_attempt_rows_filtered_csv",
            use_container_width=True,
        )
        st.info(
            f"Los archivos del análisis individual y colectivo también se guardan automáticamente en la carpeta administrativa del servidor: {ADMIN_REPORT_STORE.base_dir}."
        )

    with tabs[5]:
        st.markdown(
            """
            <div class="section-card">
                <h3>Contexto sociodemográfico, académico y laboral</h3>
                <p>Relaciona variables contextuales del participante con k_est, nivel moral dominante y marco ético predominante para apoyar lecturas comparativas prudentes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        context_col1, context_col2 = st.columns([1, 1])
        with context_col1:
            st.markdown("#### Resumen categórico")
            if contextual_categorical_df.empty:
                st.info("No hay variables categóricas contextuales suficientes para resumir.")
            else:
                st.dataframe(contextual_categorical_df, use_container_width=True, hide_index=True)
        with context_col2:
            st.markdown("#### Resumen numérico")
            if contextual_numeric_df.empty:
                st.info("No hay variables numéricas contextuales suficientes para resumir.")
            else:
                st.dataframe(contextual_numeric_df, use_container_width=True, hide_index=True)

        st.markdown("#### Cruces contextuales con resultados")
        analysis_mode = st.radio(
            "Tipo de variable contextual",
            options=["Categórica", "Numérica"],
            horizontal=True,
        )

        if analysis_mode == "Categórica":
            available_columns = [column for column in CONTEXT_CATEGORICAL_COLUMNS if column in students_filtered.columns and students_filtered[column].notna().any()]
            if not available_columns:
                st.info("No hay variables categóricas con datos disponibles para análisis relacional.")
            else:
                selected_context = st.selectbox(
                    "Variable categórica",
                    options=available_columns,
                    format_func=lambda column: CONTEXT_FIELD_LABELS.get(column, column),
                )
                outcome_summary = build_context_outcome_summary(students_filtered, selected_context)
                context_chart_col1, context_chart_col2 = st.columns([1, 1])
                with context_chart_col1:
                    box_fig = make_context_boxplot(students_filtered, selected_context)
                    render_plotly_figure(
                        box_fig,
                        f"contexto_kest_{selected_context}",
                        data_df=outcome_summary,
                        caption="Distribución de k_est en cada categoría contextual observada.",
                    )
                with context_chart_col2:
                    level_fig = make_context_level_chart(students_filtered, selected_context)
                    render_plotly_figure(
                        level_fig,
                        f"contexto_nivel_{selected_context}",
                        data_df=outcome_summary,
                        caption="Composición porcentual de niveles morales dominantes por categoría contextual.",
                    )
                st.dataframe(outcome_summary, use_container_width=True, hide_index=True)
        else:
            available_numeric_columns = [column for column in CONTEXT_NUMERIC_COLUMNS if column in students_filtered.columns and pd.to_numeric(students_filtered[column], errors="coerce").notna().any()]
            if not available_numeric_columns:
                st.info("No hay variables numéricas con datos disponibles para análisis relacional.")
            else:
                selected_numeric_context = st.selectbox(
                    "Variable numérica",
                    options=available_numeric_columns,
                    format_func=lambda column: CONTEXT_FIELD_LABELS.get(column, column),
                )
                numeric_fig = make_numeric_context_scatter(students_filtered, selected_numeric_context)
                numeric_df = students_filtered[[selected_numeric_context, "k_est", "k_level", "fw_dom", "profession", "group"]].copy()
                numeric_df = numeric_df.rename(columns={selected_numeric_context: CONTEXT_FIELD_LABELS.get(selected_numeric_context, selected_numeric_context)})
                render_plotly_figure(
                    numeric_fig,
                    f"contexto_numerico_{selected_numeric_context}",
                    data_df=numeric_df,
                    caption="Relación descriptiva entre la variable contextual seleccionada y el índice k_est.",
                )
                corr_df = students_filtered[[selected_numeric_context, "k_est"]].copy()
                corr_df[selected_numeric_context] = pd.to_numeric(corr_df[selected_numeric_context], errors="coerce")
                corr_df = corr_df.dropna()
                correlation = corr_df[selected_numeric_context].corr(corr_df["k_est"]) if len(corr_df) >= 3 else np.nan
                st.caption(
                    f"Correlación descriptiva con k_est: {correlation:.2f}" if pd.notna(correlation) else "No hay suficientes casos para estimar una correlación descriptiva estable."
                )


def page_deployment() -> None:
    st.subheader("Guía operativa y de despliegue")
    st.markdown(f"**Autor del programa:** {AUTHOR_NAME}")
    st.markdown(
        """
### Qué mejoré respecto al prototipo original
- Migración de `ipywidgets`/Colab a **Streamlit**.
- Eliminación de dependencias frágiles como `drive.mount()` y `!pip install` dentro del script.
- Validación mínima de calidad: se exige justificación argumentativa suficiente.
- Persistencia desacoplada con **Supabase/Postgres** como backend principal cuando existe `SUPABASE_DB_URL`, y **SQLite** local como fallback de desarrollo.
- Dashboard descriptivo más robusto y menos sobreprometedor: mantuve análisis exploratorio y removí predicción poco estable para muestras pequeñas.
- Exportación directa de resultados para docencia, auditoría o análisis institucional.

### Advertencia de rigor
- Esta app funciona bien para enseñanza, formación y análisis exploratorio.
- En **Streamlit Community Cloud**, SQLite local sigue siendo útil para desarrollo o demos, pero la opción recomendada para persistencia real es **Supabase/Postgres**.
- Para uso institucional serio, configura `SUPABASE_DB_URL` y evita depender del disco efímero del contenedor.
        """
    )

    st.code(
        "streamlit run app.py",
        language="bash",
    )

    st.markdown(
        """
### Estructura esperada del repositorio
```text
moral_test_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    └── .gitkeep
```
        """
    )


def page_admin(df: pd.DataFrame) -> None:
    st.subheader("Administración rápida")
    st.write("Úsalo para revisar el backend activo y el estado actual de la persistencia.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas almacenadas", str(len(df)))
    c2.metric("Backend activo", PERSISTENCE_STORE.backend_name)
    c3.metric("Detalle", PERSISTENCE_STORE.backend_detail)
    st.caption("Si existe `SUPABASE_DB_URL`, la app usa Supabase/Postgres. Si no existe, usa SQLite local y puede migrar automáticamente un CSV legado en el primer arranque.")
    st.caption(f"Carpeta administrativa de reportes: {ADMIN_REPORT_STORE.base_dir}")
    st.caption("Las secciones Dashboard colectivo y Administración están protegidas por `MORAL_TEST_ADMIN_PASSWORD` o `ADMIN_PASSWORD`.")
    if ADMIN_REPORT_STORE.drive_url:
        st.caption(f"Referencia de carpeta administrativa en Drive: {ADMIN_REPORT_STORE.drive_url}")
    else:
        st.caption("Para guardar en una carpeta de Google Drive, sincroniza o monta esa carpeta en el servidor y define `MORAL_TEST_ADMIN_REPORTS_DIR` con esa ruta local. Un enlace web de Drive por sí solo no es escribible desde la app.")
    if st.button("Recargar datos", use_container_width=True):
        load_df_cached.clear()
        st.rerun()

    st.markdown("### Consultas persistentes y exportación")
    st.write("Consulta el backend consolidado sin depender del CSV legado y descarga tablas administrativas listas para revisión académica.")
    attempt_summaries = PERSISTENCE_STORE.run_query("attempt_summaries", limit=5000)
    if attempt_summaries.empty:
        st.info("Aún no hay intentos almacenados en el backend activo para consultas administrativas.")
    else:
        attempt_summaries = attempt_summaries.copy()
        attempt_summaries["timestamp_dt"] = pd.to_datetime(attempt_summaries["timestamp"], errors="coerce")
        attempt_lookup = attempt_summaries.drop_duplicates(subset=["anon_id"]).copy()
        attempt_lookup["display_label"] = attempt_lookup.apply(
            lambda row: f"{row['name'] or row['student_id'] or str(row['anon_id'])[:8]} | {row['profession'] or 'Sin profesión'} | {row['group'] or 'Sin grupo'}",
            axis=1,
        )
        profession_options = sorted(attempt_summaries["profession"].dropna().astype(str).unique().tolist())
        group_options = sorted(attempt_summaries["group"].dropna().astype(str).unique().tolist())

        st.markdown("#### Vista de auditoría")
        date_col1, date_col2, date_col3 = st.columns([1, 1, 1])
        valid_dates = attempt_summaries["timestamp_dt"].dropna()
        min_date = valid_dates.min().date() if not valid_dates.empty else datetime.now().date()
        max_date = valid_dates.max().date() if not valid_dates.empty else datetime.now().date()
        selected_date_range = date_col1.date_input(
            "Rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        audit_profession = date_col2.selectbox("Profesión auditoría", options=["Todas"] + profession_options)
        audit_group = date_col3.selectbox("Cohorte auditoría", options=["Todas"] + group_options)

        audit_df = attempt_summaries.copy()
        if isinstance(selected_date_range, (tuple, list)) and len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
            audit_df = audit_df[
                audit_df["timestamp_dt"].dt.date.between(start_date, end_date, inclusive="both")
            ].copy()
        if audit_profession != "Todas":
            audit_df = audit_df[audit_df["profession"] == audit_profession].copy()
        if audit_group != "Todas":
            audit_df = audit_df[audit_df["group"] == audit_group].copy()

        audit_metrics = st.columns(4)
        audit_metrics[0].metric("Intentos filtrados", str(len(audit_df)))
        audit_metrics[1].metric("Participantes únicos", str(audit_df["anon_id"].nunique()))
        audit_metrics[2].metric("Profesiones visibles", str(audit_df["profession"].dropna().nunique()))
        audit_metrics[3].metric("Cohortes visibles", str(audit_df["group"].dropna().nunique()))

        if audit_df.empty:
            st.info("Los filtros de auditoría no devuelven intentos en el rango seleccionado.")
        else:
            st.dataframe(
                audit_df.drop(columns=["timestamp_dt"], errors="ignore"),
                use_container_width=True,
            )
            audit_professions = audit_df["profession"].dropna().astype(str).unique().tolist()
            audit_comparison_df = PERSISTENCE_STORE.run_query(
                "profession_comparison",
                professions=audit_professions or None,
            )
            raw_rows_filtered = df.copy()
            raw_rows_filtered["timestamp_dt"] = pd.to_datetime(raw_rows_filtered["timestamp"], errors="coerce")
            if isinstance(selected_date_range, (tuple, list)) and len(selected_date_range) == 2:
                start_date, end_date = selected_date_range
                raw_rows_filtered = raw_rows_filtered[
                    raw_rows_filtered["timestamp_dt"].dt.date.between(start_date, end_date, inclusive="both")
                ].copy()
            if audit_profession != "Todas":
                raw_rows_filtered = raw_rows_filtered[raw_rows_filtered["profession"] == audit_profession].copy()
            if audit_group != "Todas":
                raw_rows_filtered = raw_rows_filtered[raw_rows_filtered["group"] == audit_group].copy()
            raw_rows_filtered = raw_rows_filtered.drop(columns=["timestamp_dt"], errors="ignore")

            export_col1, export_col2 = st.columns([1, 1])
            export_col1.download_button(
                "Descargar auditoría filtrada (CSV)",
                data=audit_df.drop(columns=["timestamp_dt"], errors="ignore").to_csv(index=False).encode("utf-8"),
                file_name="auditoria_filtrada.csv",
                mime="text/csv",
                use_container_width=True,
            )
            export_col2.download_button(
                "Descargar paquete administrativo (Excel)",
                data=dataframe_to_excel_bytes({
                    "auditoria_intentos": audit_df.drop(columns=["timestamp_dt"], errors="ignore"),
                    "comparacion_profesion": audit_comparison_df,
                    "filas_respuestas": raw_rows_filtered,
                    "catalogo_lab": pd.DataFrame([
                        {
                            "id": item["id"],
                            "ruta_sugerida": item["ruta_sugerida"],
                            "foco": item["foco"],
                            "titulo": LOOKUP[item["id"]]["title"],
                            "planteamiento": LOOKUP[item["id"]]["prompt"],
                            "justificacion_pedagogica": LOOKUP[item["id"]].get("pedagogical_justification", ""),
                        }
                        for item in LAB_DILEMMA_CATALOG
                    ]),
                }),
                file_name="paquete_administrativo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        query_labels = {
            "attempt_summaries": "Resumen de intentos",
            "last_attempt": "Último intento por usuario",
            "user_history": "Histórico completo por usuario",
            "cohort_history": "Histórico por cohorte",
            "profession_comparison": "Comparación agregada por profesión",
        }
        selected_query = st.selectbox("Tipo de consulta", options=list(query_labels.keys()), format_func=lambda key: query_labels[key])

        query_df = pd.DataFrame()
        export_name = selected_query
        if selected_query == "attempt_summaries":
            selected_profession = st.selectbox("Filtrar profesión", options=["Todas"] + profession_options)
            selected_group = st.selectbox("Filtrar cohorte", options=["Todas"] + group_options)
            limit = st.slider("Máximo de intentos a mostrar", min_value=20, max_value=500, value=200, step=20)
            query_df = PERSISTENCE_STORE.run_query(
                "attempt_summaries",
                profession=None if selected_profession == "Todas" else selected_profession,
                group_name=None if selected_group == "Todas" else selected_group,
                limit=limit,
            )
            export_name = "resumen_intentos"
        elif selected_query == "last_attempt":
            selected_label = st.selectbox("Usuario", options=attempt_lookup["display_label"].tolist())
            selected_anon = attempt_lookup.loc[attempt_lookup["display_label"] == selected_label, "anon_id"].iloc[0]
            query_df = PERSISTENCE_STORE.run_query("last_attempt", anon_id=selected_anon)
            export_name = f"ultimo_intento_{selected_anon}"
        elif selected_query == "user_history":
            selected_label = st.selectbox("Usuario", options=attempt_lookup["display_label"].tolist(), key="admin_user_history")
            selected_anon = attempt_lookup.loc[attempt_lookup["display_label"] == selected_label, "anon_id"].iloc[0]
            query_df = PERSISTENCE_STORE.run_query("user_history", anon_id=selected_anon)
            export_name = f"historico_usuario_{selected_anon}"
        elif selected_query == "cohort_history":
            selected_group = st.selectbox("Cohorte / grupo", options=group_options)
            query_df = PERSISTENCE_STORE.run_query("cohort_history", group_name=selected_group)
            export_name = f"historico_cohorte_{filename_slug(selected_group)}"
        elif selected_query == "profession_comparison":
            selected_professions = st.multiselect("Profesiones", options=profession_options, default=profession_options)
            query_df = PERSISTENCE_STORE.run_query("profession_comparison", professions=selected_professions or None)
            export_name = "comparacion_profesiones"

        if query_df.empty:
            st.info("La consulta seleccionada no devolvió resultados con los filtros actuales.")
        else:
            st.dataframe(query_df, use_container_width=True)
            st.download_button(
                "Descargar resultado de la consulta (CSV)",
                data=query_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{filename_slug(export_name)}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("### Catálogo de dilemas de laboratorio")
    st.write("Revisión docente de los casos específicos para Bacteriología y Microbiología.")
    route_filter = st.multiselect(
        "Filtrar por ruta sugerida",
        options=["Bacteriología", "Microbiología", "Bacteriología / Microbiología"],
        default=["Bacteriología", "Microbiología", "Bacteriología / Microbiología"],
    )
    lab_catalog_rows = []
    for item in LAB_DILEMMA_CATALOG:
        if route_filter and item["ruta_sugerida"] not in route_filter:
            continue
        dilemma = LOOKUP[item["id"]]
        lab_catalog_rows.append({
            "id": item["id"],
            "ruta_sugerida": item["ruta_sugerida"],
            "foco": item["foco"],
            "titulo": dilemma["title"],
            "planteamiento": dilemma["prompt"],
            "justificacion_pedagogica": dilemma.get("pedagogical_justification", ""),
        })

    lab_catalog_df = pd.DataFrame(lab_catalog_rows)
    st.dataframe(lab_catalog_df, use_container_width=True)
    st.download_button(
        "Descargar catálogo de dilemas de laboratorio (CSV)",
        data=lab_catalog_df.to_csv(index=False).encode("utf-8"),
        file_name="lab_dilemmas_catalog.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    ensure_storage()
    render_header()
    render_sidebar_branding()
    render_admin_session_controls()
    df = load_df()

    page = st.sidebar.radio(
        "Navegación",
        ["Presentación del programa", "Aplicación individual", "Dashboard colectivo", "Interpretación IA", "Guía de despliegue", "Administración"],
    )

    if page == "Presentación del programa":
        page_program()
    elif page == "Aplicación individual":
        page_apply(df)
    elif page == "Dashboard colectivo":
        if admin_login_panel():
            page_dashboard(df)
    elif page == "Interpretación IA":
        if admin_login_panel():
            page_ai_interpretation(df)
    elif page == "Guía de despliegue":
        page_deployment()
    else:
        if admin_login_panel():
            page_admin(df)


if __name__ == "__main__":
    main()
