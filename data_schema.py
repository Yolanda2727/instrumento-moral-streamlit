from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


ATTEMPT_IDENTITY_COLUMNS = [
    "timestamp",
    "anon_id",
    "student_id",
    "name",
    "profession",
    "years_experience",
    "group",
]

RESPONSE_COLUMNS = [
    "row_type",
    "item_id",
    "sub_id",
    "choice_key",
    "choice_stage",
    "choice_level",
    "choice_framework",
    "likert_value",
    "text",
]

ATTEMPT_METADATA_EXPORT_COLUMNS = [
    "n_dilemmas_answered",
    "n_justifications",
    "responded_item_ids",
    "created_at",
]

MIN_AGE = 16
MAX_AGE = 85
MAX_CHILDREN = 20
MAX_WORK_HOURS = 80

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


@dataclass(frozen=True)
class ParticipantContextField:
    key: str
    label: str
    storage_type: str
    category: str
    required: bool = False
    options: tuple[str, ...] = ()


PARTICIPANT_CONTEXT_FIELDS: tuple[ParticipantContextField, ...] = (
    ParticipantContextField("gender", "Género", "TEXT", "categorical", True, tuple(GENDER_OPTIONS)),
    ParticipantContextField("age", "Edad", "INTEGER", "numeric", True),
    ParticipantContextField("semester", "Semestre", "INTEGER", "numeric", True),
    ParticipantContextField("works_for_studies", "Trabaja para pagar estudios", "TEXT", "categorical", True, tuple(WORK_STUDY_OPTIONS)),
    ParticipantContextField("children_count", "Número de hijos", "INTEGER", "numeric", True),
    ParticipantContextField("academic_program", "Programa académico", "TEXT", "categorical"),
    ParticipantContextField("academic_shift", "Jornada académica", "TEXT", "categorical", False, tuple(option for option in ACADEMIC_SHIFT_OPTIONS if option)),
    ParticipantContextField("prior_experience_area", "Experiencia laboral o clínica previa", "TEXT", "categorical", False, tuple(option for option in PRIOR_EXPERIENCE_OPTIONS if option)),
    ParticipantContextField("ethics_training", "Formación previa en ética o bioética", "TEXT", "categorical", False, tuple(option for option in ETHICS_TRAINING_OPTIONS if option)),
    ParticipantContextField("work_hours_per_week", "Horas de trabajo por semana", "INTEGER", "numeric"),
    ParticipantContextField("caregiving_load", "Carga de cuidado familiar", "TEXT", "categorical", False, tuple(option for option in CAREGIVING_LOAD_OPTIONS if option)),
    ParticipantContextField("study_funding_type", "Tipo de financiación de estudios", "TEXT", "categorical", False, tuple(option for option in STUDY_FUNDING_OPTIONS if option)),
)

PARTICIPANT_CONTEXT_COLUMNS = [field.key for field in PARTICIPANT_CONTEXT_FIELDS]
PARTICIPANT_CONTEXT_COLUMN_TYPES = {field.key: field.storage_type for field in PARTICIPANT_CONTEXT_FIELDS}
CONTEXT_FIELD_LABELS = {field.key: field.label for field in PARTICIPANT_CONTEXT_FIELDS}
CONTEXT_FIELD_LABELS["years_experience"] = "Años de experiencia"
CONTEXT_CATEGORICAL_COLUMNS = [field.key for field in PARTICIPANT_CONTEXT_FIELDS if field.category == "categorical"]
CONTEXT_NUMERIC_COLUMNS = [field.key for field in PARTICIPANT_CONTEXT_FIELDS if field.category == "numeric"] + ["years_experience"]

COLUMNS = ATTEMPT_IDENTITY_COLUMNS + PARTICIPANT_CONTEXT_COLUMNS + RESPONSE_COLUMNS


def semester_limit_for_profession(profession: str | None) -> int:
    return SEMESTER_LIMITS_BY_PROFESSION.get((profession or "").strip(), 12)


def attempt_select_columns(alias: str = "a") -> list[str]:
    group_expr = f'{alias}.group_name AS "group"'
    select_columns = [
        f"{alias}.timestamp",
        f"{alias}.anon_id",
        f"{alias}.student_id",
        f"{alias}.name",
        f"{alias}.profession",
        f"{alias}.years_experience",
        group_expr,
    ]
    select_columns.extend(f"{alias}.{column}" for column in PARTICIPANT_CONTEXT_COLUMNS)
    return select_columns


def response_select_columns(alias: str = "r") -> list[str]:
    return [
        f"{alias}.row_type",
        f"{alias}.item_id",
        f"{alias}.sub_id",
        f"{alias}.choice_key",
        f"{alias}.choice_stage",
        f"{alias}.choice_level",
        f"{alias}.choice_framework",
        f"{alias}.likert_value",
        f"{alias}.text_value AS text",
    ]


def attempt_summary_select_columns(alias: str = "a", include_raw_payload: bool = False) -> list[str]:
    columns = attempt_select_columns(alias)
    columns.extend(f"{alias}.{column}" for column in ATTEMPT_METADATA_EXPORT_COLUMNS)
    if include_raw_payload:
        columns.append(f"{alias}.raw_payload")
    return columns


def aggregate_metric_columns() -> dict[str, str]:
    return {
        "avg_age": "AVG(age)",
        "avg_semester": "AVG(semester)",
        "avg_years_experience": "AVG(years_experience)",
        "avg_work_hours_per_week": "AVG(work_hours_per_week)",
        "avg_children_count": "AVG(children_count)",
        "avg_dilemmas_answered": "AVG(n_dilemmas_answered)",
        "avg_justifications": "AVG(n_justifications)",
    }
