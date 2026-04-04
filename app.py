from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import portalocker
except Exception:  # pragma: no cover
    portalocker = None

# =========================
# Configuración general
# =========================
st.set_page_config(
    page_title="Moral Test — Kohlberg & Ethical Frameworks",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
APP_TITLE = "Instrumento formativo de razonamiento moral y marcos éticos"
APP_SUBTITLE = (
    "Versión en Streamlit, orientada a estudiantes y profesionales. "
    "No es diagnóstica; apoya reflexión, docencia y análisis colectivo."
)
DEFAULT_ROUTE_SIZE = 6
MIN_JUSTIFICATION_CHARS = 25
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

DATA_PATH = Path(os.getenv("MORAL_TEST_DATA_PATH", "data/responses.csv"))
LOCK_PATH = DATA_PATH.with_suffix(".lock")
EXPORT_DIR = Path("data/exports")

BASE_STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "para", "por", "con",
    "del", "se", "que", "al", "lo", "como", "su", "sus", "si", "no", "es", "son", "o",
    "u", "mi", "tu", "yo", "me", "te", "le", "les", "ha", "han", "ser", "hacer", "porque",
    "más", "mas", "muy", "ya", "cuando", "donde", "qué", "este", "esta", "estos", "estas",
    "ese", "esa", "eso", "desde", "entre", "sin", "sobre", "ante", "bajo", "durante", "hasta",
    "contra", "todo", "toda", "todos", "todas", "uno", "dos", "tres", "pero", "también",
}

COLUMNS = [
    "timestamp", "anon_id", "student_id", "name", "profession", "years_experience", "group",
    "row_type", "item_id", "sub_id", "choice_key", "choice_stage", "choice_level", "choice_framework",
    "likert_value", "text",
]

FRAMEWORKS = ["utilitarismo", "deontologia", "regla_de_oro", "virtudes", "contrato_social", "cuidado"]

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


def make_dilemma(item_id: str, title: str, prompt: str, options: List[dict], stage_statements: Dict[int, str]) -> dict:
    return {"id": item_id, "title": title, "prompt": prompt, "options": options, "stage_statements": stage_statements}


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

for item_id, title, prompt, options in health + law + it_cases:
    BANK.append(make_dilemma(item_id, title, prompt, options, STAGE_TEMPLATE))

LOOKUP = {d["id"]: d for d in BANK}
PROFESSION_ROUTES = {
    "Salud (medicina/enfermería/fisioterapia/instrumentación)": ["K1", "K2", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"],
    "Derecho / Ciencias sociales / Educación": ["K1", "K2", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
    "Ingeniería / TI / Datos": ["K1", "K2", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
    "Otra / Mixta": ["K1", "K2", "H1", "L1", "T1", "H4", "L4", "T2"],
}


# =========================
# Utilidades de datos
# =========================
def ensure_storage() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(DATA_PATH, index=False)


def write_locked(df: pd.DataFrame) -> None:
    ensure_storage()
    if portalocker is None:
        df.to_csv(DATA_PATH, index=False)
        return
    with portalocker.Lock(str(LOCK_PATH), timeout=15):
        df.to_csv(DATA_PATH, index=False)


@st.cache_data(show_spinner=False)
def load_df_cached(file_mtime: float | None) -> pd.DataFrame:
    del file_mtime
    ensure_storage()
    df = pd.read_csv(DATA_PATH)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLUMNS].copy()


def load_df() -> pd.DataFrame:
    mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else None
    return load_df_cached(mtime)


def save_df(df: pd.DataFrame) -> None:
    write_locked(df)
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
    ids = PROFESSION_ROUTES.get(label, ["K1", "K2"])[:route_size]
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
    return students, df_last


def build_rows(
    student_id: str,
    name: str,
    profession: str,
    years_experience: int,
    group: str,
    anonymize: bool,
    choices_payload: List[dict],
    stage_payload: List[dict],
    framework_payload: List[dict],
) -> pd.DataFrame:
    ts = now_iso()
    anon_id = sha_id(student_id) if anonymize else student_id
    rows = []

    for row in choices_payload:
        rows.append({
            "timestamp": ts,
            "anon_id": anon_id,
            "student_id": student_id,
            "name": name,
            "profession": profession,
            "years_experience": int(years_experience),
            "group": group,
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
            "timestamp": ts,
            "anon_id": anon_id,
            "student_id": student_id,
            "name": name,
            "profession": profession,
            "years_experience": int(years_experience),
            "group": group,
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
            "timestamp": ts,
            "anon_id": anon_id,
            "student_id": student_id,
            "name": name,
            "profession": profession,
            "years_experience": int(years_experience),
            "group": group,
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
            background: linear-gradient(135deg, #0b1f3a 0%, #133a63 55%, #1c5f8d 100%);
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            color: #f7fbff;
            box-shadow: 0 10px 28px rgba(10, 25, 47, 0.22);
            margin-bottom: 1rem;
        }
        .hero-card h1 {
            font-size: 2rem;
            margin: 0 0 .35rem 0;
            line-height: 1.15;
        }
        .hero-card p {
            margin: 0.2rem 0;
            font-size: 1rem;
        }
        .author-card {
            background: #f6f9fc;
            border: 1px solid #d9e5f2;
            border-left: 6px solid #1c5f8d;
            padding: 1rem;
            border-radius: 14px;
            margin-bottom: 1rem;
        }
        .objective-card {
            background: #ffffff;
            border: 1px solid #e6eef7;
            border-radius: 14px;
            padding: .95rem 1rem;
            min-height: 120px;
            box-shadow: 0 4px 12px rgba(17, 35, 58, 0.06);
        }
        .minor-note {
            color: #5a6b7c;
            font-size: 0.93rem;
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
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
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
        "Identificación y ruta profesional",
        "Dilemas éticos contextualizados",
        "Justificación argumentativa",
        "Escalas Likert por estadio",
        "Inventario de marcos éticos",
        "Reporte individual",
        "Dashboard colectivo",
    ]
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=20, thickness=20),
        link=dict(
            source=[0, 1, 2, 3, 4],
            target=[1, 2, 3, 5, 6],
            value=[1, 1, 1, 1, 1],
        ),
    )])
    fig.update_layout(title="Arquitectura funcional del programa", height=420)
    return fig


def make_route_summary_figure() -> go.Figure:
    rows = []
    for route, items in PROFESSION_ROUTES.items():
        rows.append({"ruta": route, "dilemas": len(items)})
    data = pd.DataFrame(rows)
    fig = px.bar(
        data,
        x="ruta",
        y="dilemas",
        title="Cobertura de dilemas por ruta profesional",
        text="dilemas",
        height=420,
    )
    fig.update_layout(xaxis_title="Ruta profesional", yaxis_title="Número de dilemas")
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
        st.plotly_chart(make_program_flow_figure(), use_container_width=True)

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
            "componente": ["Dilemas núcleo", "Ruta salud", "Ruta derecho/social", "Ruta ingeniería/TI"],
            "cantidad": [2, 8, 8, 8],
        })
        fig = px.pie(route_mix, names="componente", values="cantidad", hole=0.48, title="Composición del banco base de dilemas", height=420)
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
        theta=FRAMEWORKS,
        fill="toself",
        name="Perfil",
    ))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[1, 7])),
        showlegend=False,
        height=450,
    )
    return fig


def make_sankey_from_choices(choices_df: pd.DataFrame, title: str) -> go.Figure:
    nodes = list(choices_df["item_id"].unique()) + list(choices_df["choice_framework"].unique())
    idx = {label: i for i, label in enumerate(nodes)}
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=nodes),
        link=dict(
            source=[idx[val] for val in choices_df["item_id"]],
            target=[idx[val] for val in choices_df["choice_framework"]],
            value=[1] * len(choices_df),
        ),
    )])
    fig.update_layout(title=title, height=500)
    return fig


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


def qualitative_clusters(df_last: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure | None]:
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
    )
    return summary, fig


# =========================
# Páginas
# =========================
def page_apply(df: pd.DataFrame) -> None:
    st.subheader("Aplicación individual")
    st.write("Completa el instrumento y genera un reporte individual inmediato.")
    a1, a2, a3 = st.columns(3)
    a1.metric("Rutas disponibles", str(len(PROFESSION_ROUTES)))
    a2.metric("Marcos éticos evaluados", str(len(FRAMEWORKS)))
    a3.metric("Justificación mínima", f"{MIN_JUSTIFICATION_CHARS} caracteres")
    st.caption("La app valora razonamiento argumentativo básico: opción elegida, calidad de la justificación, patrones por estadio y afinidad con marcos éticos.")

    with st.sidebar:
        st.markdown("### Parámetros de aplicación")
        route_size = st.slider("Número de dilemas por ruta", min_value=4, max_value=10, value=DEFAULT_ROUTE_SIZE, step=1)
        anonymize = st.checkbox("Anonimizar identificador", value=True)

    with st.form("moral_test_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1, 1, 1])
        student_id = c1.text_input("ID institucional o código", max_chars=60)
        name = c2.text_input("Nombre o seudónimo", max_chars=120)
        group = c3.text_input("Grupo / cohorte / curso", max_chars=120)

        c4, c5 = st.columns([2, 1])
        profession = c4.selectbox("Área profesional", options=list(PROFESSION_ROUTES.keys()))
        years_experience = c5.number_input("Años de experiencia", min_value=0, max_value=60, value=0, step=1)

        dilemmas = route_for_profession(profession, route_size)
        st.markdown("### A. Dilemas y justificación argumentativa")
        st.caption("Se exige una justificación breve con razonamiento suficiente; evita respuestas telegráficas.")

        collected_choices = []
        for i, dilemma in enumerate(dilemmas, start=1):
            st.markdown(f"**{i}. {dilemma['title']}**")
            st.write(dilemma["prompt"])
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

    if not submitted:
        return

    errors = []
    if not student_id.strip():
        errors.append("Debes diligenciar el ID institucional o código.")
    if not name.strip():
        errors.append("Debes diligenciar el nombre o seudónimo.")

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
        return

    new_rows = build_rows(
        student_id=student_id.strip(),
        name=name.strip(),
        profession=profession,
        years_experience=int(years_experience),
        group=group.strip(),
        anonymize=anonymize,
        choices_payload=choices_payload,
        stage_payload=stage_values,
        framework_payload=framework_values,
    )

    full_df = pd.concat([df, new_rows], ignore_index=True)
    save_df(full_df)
    person = new_rows.copy()
    choice_df = person[person["row_type"] == "choice"].copy()
    stage_df = person[person["row_type"] == "stage_likert"].copy()
    fw_df = person[person["row_type"] == "framework_likert"].copy()

    k_est, k_stage, k_level, stage_means = kohlberg_from_stage_likert(stage_df)
    fw_scores, fw_dom = framework_scores(fw_df)
    coherence = float(np.nanstd(list(stage_means.values())))

    st.success("Respuesta guardada correctamente.")
    st.subheader("Reporte individual")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nivel global", str(k_level))
    m2.metric("Estadio redondeado", str(k_stage))
    m3.metric("Índice k_est", f"{k_est:.2f}" if pd.notna(k_est) else "NA")
    m4.metric("Coherencia (DE)", f"{coherence:.2f}")
    st.caption(f"Marco dominante: {fw_dom}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_radar(fw_scores, "Perfil de marcos éticos"), use_container_width=True)
    with c2:
        st.plotly_chart(make_sankey_from_choices(choice_df, "Flujo de dilemas hacia marcos seleccionados"), use_container_width=True)

    st.markdown("### Síntesis interpretativa")
    st.write(
        f"Se observa predominio del nivel **{k_level}** con un índice global de **{k_est:.2f}**. "
        f"El marco dominante fue **{fw_dom}** y la dispersión interna de estadios fue **{coherence:.2f}**."
    )

    recommendation_top = sorted(fw_scores.items(), key=lambda x: (pd.isna(x[1]), -(x[1] or -999)))
    if recommendation_top:
        strongest = recommendation_top[0][0]
        weakest = recommendation_top[-1][0]
        st.markdown("### Recomendaciones de mejora argumentativa")
        st.markdown(
            f"- Reescribe un caso desde el marco **{weakest}** para contrastarlo con tu tendencia dominante **{strongest}**.\n"
            "- Explicita siempre el principio protegido, la consecuencia esperada y la salvaguarda frente a daño colateral.\n"
            "- En los casos con alta afectación de terceros, incorpora proporcionalidad, transparencia y trazabilidad."
        )

    with st.expander("Ver detalle de respuestas"):
        st.dataframe(choice_df[["item_id", "choice_level", "choice_framework", "text"]], use_container_width=True)


def page_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dashboard colectivo")
    if df.empty:
        st.warning("Aún no hay respuestas registradas.")
        return

    students, df_last = student_table(df)
    if students.empty:
        st.warning("No fue posible consolidar participantes.")
        return

    metrics_section(students)

    tab1, tab2, tab3, tab4 = st.tabs(["Distribuciones", "Brechas", "Cualitativo", "Datos"])

    with tab1:
        fig1 = px.violin(
            students,
            x="profession",
            y="k_est",
            box=True,
            points="all",
            title="k_est por profesión",
            height=500,
        )
        st.plotly_chart(fig1, use_container_width=True)

        fw_cols = [f"fw_{fw}" for fw in FRAMEWORKS]
        heat = students.groupby("profession")[fw_cols].mean().reset_index()
        heat_m = heat.melt(id_vars="profession", var_name="framework", value_name="mean")
        heat_m["framework"] = heat_m["framework"].str.replace("fw_", "", regex=False)
        fig2 = px.density_heatmap(
            heat_m,
            x="framework",
            y="profession",
            z="mean",
            histfunc="avg",
            title="Promedio de marcos éticos por profesión",
            height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)

        sun = students.dropna(subset=["k_level", "fw_dom"]).copy()
        if not sun.empty:
            fig3 = px.sunburst(
                sun,
                path=["profession", "k_level", "fw_dom"],
                title="Profesión → nivel → marco dominante",
                height=550,
            )
            st.plotly_chart(fig3, use_container_width=True)

        sank = students.dropna(subset=["profession", "fw_dom"]).copy()
        if not sank.empty:
            left = sank["profession"].unique().tolist()
            right = sank["fw_dom"].unique().tolist()
            nodes = left + right
            idx = {label: i for i, label in enumerate(nodes)}
            links = sank.groupby(["profession", "fw_dom"]).size().reset_index(name="value")
            fig4 = go.Figure(data=[go.Sankey(
                node=dict(label=nodes),
                link=dict(
                    source=[idx[s] for s in links["profession"]],
                    target=[idx[t] for t in links["fw_dom"]],
                    value=links["value"].tolist(),
                ),
            )])
            fig4.update_layout(title="Profesión → marco dominante", height=500)
            st.plotly_chart(fig4, use_container_width=True)

        pca_data = students.dropna(subset=[f"fw_{fw}" for fw in FRAMEWORKS]).copy()
        if len(pca_data) >= 6:
            X = pca_data[[f"fw_{fw}" for fw in FRAMEWORKS]].to_numpy(float)
            Xc = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
            coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(Xc)
            k = min(4, max(2, int(np.sqrt(len(pca_data)))))
            clusters = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(coords)
            plot_df = pca_data[["profession", "k_level"]].copy()
            plot_df["PC1"] = coords[:, 0]
            plot_df["PC2"] = coords[:, 1]
            plot_df["cluster"] = clusters.astype(str)
            fig5 = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="profession",
                symbol="cluster",
                title="Mapa exploratorio de perfiles morales (PCA + clusters)",
                height=520,
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Se requieren al menos 6 participantes completos para PCA y clustering.")

    with tab2:
        prof_table = students.groupby("profession").agg(
            n=("anon_id", "count"),
            k_est_promedio=("k_est", "mean"),
            k_est_de=("k_est", "std"),
            coherencia_promedio=("k_coherence_std", "mean"),
            marco_dominante=("fw_dom", lambda x: x.mode().iloc[0] if x.notna().any() else None),
        ).reset_index().sort_values("n", ascending=False)
        st.dataframe(prof_table, use_container_width=True)

        gap = profession_gap_table(students)
        if gap.empty:
            st.info("Se requieren al menos dos profesiones con datos para comparar brechas.")
        else:
            st.dataframe(gap, use_container_width=True)
            p1, p2 = students["profession"].value_counts().index[:2]
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=[gap.loc[gap["framework"] == fw, f"{p1}_mean"].values[0] for fw in FRAMEWORKS],
                theta=FRAMEWORKS,
                fill="toself",
                name=p1,
            ))
            radar.add_trace(go.Scatterpolar(
                r=[gap.loc[gap["framework"] == fw, f"{p2}_mean"].values[0] for fw in FRAMEWORKS],
                theta=FRAMEWORKS,
                fill="toself",
                name=p2,
            ))
            radar.update_layout(
                title=f"Comparación de marcos: {p1} vs {p2}",
                polar=dict(radialaxis=dict(visible=True, range=[1, 7])),
                height=500,
            )
            st.plotly_chart(radar, use_container_width=True)
            st.caption("Hedges g cuantifica magnitud de diferencia; úsalo como insumo exploratorio, no como veredicto definitivo.")

    with tab3:
        summary, fig = qualitative_clusters(df_last)
        if summary.empty:
            st.info("Se requieren más justificaciones textuales para análisis cualitativo exploratorio.")
        else:
            st.dataframe(summary, use_container_width=True)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.dataframe(students, use_container_width=True)
        st.download_button(
            "Descargar resumen de participantes (CSV)",
            data=students.to_csv(index=False).encode("utf-8"),
            file_name="students_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Descargar última respuesta consolidada (CSV)",
            data=df_last.to_csv(index=False).encode("utf-8"),
            file_name="last_attempt_rows.csv",
            mime="text/csv",
            use_container_width=True,
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
- Persistencia en CSV con **bloqueo de archivo** cuando `portalocker` está disponible.
- Dashboard descriptivo más robusto y menos sobreprometedor: mantuve análisis exploratorio y removí predicción poco estable para muestras pequeñas.
- Exportación directa de resultados para docencia, auditoría o análisis institucional.

### Advertencia de rigor
- Esta app funciona bien para enseñanza, formación y análisis exploratorio.
- En **Streamlit Community Cloud**, los archivos locales pueden perderse si la app reinicia o se redespliega.
- Para uso institucional serio, conviene reemplazar el CSV por una base persistente: **Supabase, PostgreSQL, Google Sheets o S3**.
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
    st.write("Úsalo para revisar el estado actual del almacenamiento local.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas almacenadas", str(len(df)))
    c2.metric("Archivo de datos", str(DATA_PATH))
    c3.metric("Tamaño del archivo", f"{DATA_PATH.stat().st_size / 1024:.1f} KB" if DATA_PATH.exists() else "0 KB")
    st.caption("En Streamlit Cloud este almacenamiento es efímero. Para persistencia real, cambia el backend.")
    if st.button("Recargar datos", use_container_width=True):
        load_df_cached.clear()
        st.rerun()


def main() -> None:
    ensure_storage()
    render_header()
    render_sidebar_branding()
    df = load_df()

    page = st.sidebar.radio(
        "Navegación",
        ["Presentación del programa", "Aplicación individual", "Dashboard colectivo", "Guía de despliegue", "Administración"],
    )

    if page == "Presentación del programa":
        page_program()
    elif page == "Aplicación individual":
        page_apply(df)
    elif page == "Dashboard colectivo":
        page_dashboard(df)
    elif page == "Guía de despliegue":
        page_deployment()
    else:
        page_admin(df)


if __name__ == "__main__":
    main()
