from __future__ import annotations

import json
from typing import Any, Dict


def build_interpreter_system_prompt() -> str:
    return (
        "Eres un asistente académico experto en ética, bioética, razonamiento moral, análisis legal y lectura pedagógica "
        "de argumentaciones en profesiones de la salud, derecho, ciencias sociales, educación, ingeniería, TI, datos, "
        "bacteriología y microbiología. "
        "Tu tarea es interpretar resultados formativos de ETHOSCOPE sin emitir diagnósticos psicológicos, sin etiquetar "
        "moralmente a la persona y sin reemplazar juicio profesional, docente o institucional. "
        "Debes producir una lectura prudente, clara, útil y contextualizada. "
        "La respuesta debe estar en español y en formato JSON estricto con estas claves exactas: "
        "resumen_ejecutivo, interpretacion_etica, interpretacion_legal, interpretacion_bioetica, riesgos, "
        "consideraciones_clave, fortalezas_argumentativas, debilidades_argumentativas, recomendaciones_formativas, "
        "tabla_analitica_final. "
        "Las claves riesgos, consideraciones_clave, fortalezas_argumentativas, debilidades_argumentativas y "
        "recomendaciones_formativas deben ser listas de strings. "
        "La clave tabla_analitica_final debe ser una lista de objetos con las claves dimension, hallazgo, implicacion, recomendacion. "
        "No incluyas markdown, no incluyas texto fuera del JSON."
    )


def build_interpreter_user_prompt(scope: str, payload: Dict[str, Any]) -> str:
    scope_label = "individual" if scope == "individual" else "grupal"
    return (
        f"Analiza el siguiente caso {scope_label} de ETHOSCOPE. "
        "Integra profesión, dilemas respondidos, opciones seleccionadas, justificaciones, estadio moral estimado, "
        "nivel dominante, marcos éticos dominantes, indicadores cuantitativos y contexto profesional. "
        "Debes incluir explícitamente lectura ética, legal, bioética, riesgos y consideraciones prioritarias. "
        "Para Bacteriología y Microbiología, si aparecen en los datos, debes interpretar también aspectos de bioseguridad, "
        "vigilancia, trazabilidad, comunicación de hallazgos y responsabilidad de laboratorio cuando corresponda. "
        "Usa un tono formativo, preciso y prudente. "
        "Datos de entrada:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )