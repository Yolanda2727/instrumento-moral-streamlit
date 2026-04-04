from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


ARGUMENT_PATTERN_LEXICONS = {
    "consecuencias": {
        "beneficio", "beneficios", "riesgo", "riesgos", "impacto", "resultado", "resultados",
        "consecuencia", "consecuencias", "efecto", "efectos", "eficacia", "daño", "danos", "bienestar",
    },
    "normas": {
        "norma", "normas", "regla", "reglas", "protocolo", "protocolos", "deber", "deberes",
        "obligacion", "obligaciones", "cumplir", "legal", "ley", "leyes", "procedimiento",
    },
    "cuidado": {
        "cuidado", "vulnerable", "vulnerables", "empatía", "empatia", "acompanamiento", "acompañamiento",
        "relacion", "relaciones", "sufrimiento", "proteger", "apoyo", "familia", "equipo",
    },
    "justicia": {
        "justicia", "equidad", "derechos", "derecho", "dignidad", "proporcionalidad", "imparcialidad",
        "transparencia", "responsabilidad", "garantia", "garantias", "publico", "consenso",
    },
    "integridad": {
        "integridad", "honestidad", "verdad", "prudencia", "caracter", "carácter", "virtud", "virtudes",
        "coherencia", "responsable", "responsabilidad", "etico", "ético",
    },
}


def bootstrap_ci(values: Sequence[float] | np.ndarray, func=np.mean, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
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


def hedges_g(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return np.nan
    na, nb = len(arr_a), len(arr_b)
    sa2, sb2 = arr_a.var(ddof=1), arr_b.var(ddof=1)
    sp = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    if sp == 0:
        return np.nan
    d = (arr_a.mean() - arr_b.mean()) / sp
    correction = 1 - (3 / (4 * (na + nb) - 9))
    return float(correction * d)


def build_quantitative_report(students: pd.DataFrame, df_last: pd.DataFrame, frameworks: Sequence[str]) -> Dict[str, pd.DataFrame]:
    report: Dict[str, pd.DataFrame] = {}
    if students.empty:
        empty = pd.DataFrame()
        return {
            "profession_distribution": empty,
            "stage_summary": empty,
            "framework_summary": empty,
            "dominant_levels": empty,
            "profession_comparisons": empty,
            "cohort_summary": empty,
            "frequency_summary": empty,
            "descriptive_summary": empty,
        }

    profession_distribution = students["profession"].value_counts(dropna=False).rename_axis("profession").reset_index(name="n")
    profession_distribution["pct"] = (profession_distribution["n"] / profession_distribution["n"].sum() * 100).round(2)
    report["profession_distribution"] = profession_distribution

    stage_rows = df_last[df_last["row_type"] == "stage_likert"].copy()
    if not stage_rows.empty:
        stage_rows["sub_id"] = pd.to_numeric(stage_rows["sub_id"], errors="coerce")
        stage_summary = stage_rows.groupby(["profession", "sub_id"])["likert_value"].agg(["count", "mean", "std"]).reset_index()
        ci_rows = []
        for (profession, stage), g in stage_rows.groupby(["profession", "sub_id"]):
            mean, low, high = bootstrap_ci(g["likert_value"].values)
            ci_rows.append({"profession": profession, "sub_id": stage, "ci_low": low, "ci_high": high, "ci_mean": mean})
        stage_ci = pd.DataFrame(ci_rows)
        stage_summary = stage_summary.merge(stage_ci, on=["profession", "sub_id"], how="left")
        stage_summary = stage_summary.rename(columns={"sub_id": "stage", "mean": "stage_mean", "std": "stage_sd"})
    else:
        stage_summary = pd.DataFrame()
    report["stage_summary"] = stage_summary

    fw_cols = [f"fw_{fw}" for fw in frameworks if f"fw_{fw}" in students.columns]
    if fw_cols:
        framework_summary = students.groupby("profession")[fw_cols].mean().reset_index()
        framework_summary = framework_summary.melt(id_vars="profession", var_name="framework", value_name="framework_mean")
        framework_summary["framework"] = framework_summary["framework"].str.replace("fw_", "", regex=False)
    else:
        framework_summary = pd.DataFrame()
    report["framework_summary"] = framework_summary

    dominant_levels = students[["anon_id", "profession", "group", "k_level", "k_stage", "fw_dom", "choice_level_mode", "choice_fw_mode"]].copy()
    dominant_levels = dominant_levels.rename(columns={
        "k_level": "nivel_global_dominante",
        "k_stage": "estadio_redondeado",
        "fw_dom": "marco_dominante",
        "choice_level_mode": "nivel_mas_elegido",
        "choice_fw_mode": "marco_mas_elegido",
    })
    report["dominant_levels"] = dominant_levels

    comparison_rows = []
    valid_professions = students["profession"].dropna().value_counts()
    relevant_professions = valid_professions[valid_professions >= 2].index.tolist()
    for p1, p2 in combinations(relevant_professions, 2):
        a = students.loc[students["profession"] == p1, "k_est"].values
        b = students.loc[students["profession"] == p2, "k_est"].values
        mean_a, low_a, high_a = bootstrap_ci(a)
        mean_b, low_b, high_b = bootstrap_ci(b)
        comparison_rows.append({
            "profession_a": p1,
            "profession_b": p2,
            "metric": "k_est",
            "mean_a": mean_a,
            "ci_a": f"[{low_a:.2f}, {high_a:.2f}]" if pd.notna(mean_a) else "NA",
            "mean_b": mean_b,
            "ci_b": f"[{low_b:.2f}, {high_b:.2f}]" if pd.notna(mean_b) else "NA",
            "mean_diff": mean_a - mean_b if pd.notna(mean_a) and pd.notna(mean_b) else np.nan,
            "hedges_g": hedges_g(a, b),
        })
    profession_comparisons = pd.DataFrame(comparison_rows)
    report["profession_comparisons"] = profession_comparisons.sort_values("hedges_g", key=lambda s: s.abs(), ascending=False) if not profession_comparisons.empty else profession_comparisons

    cohort_summary = students.groupby("group").agg(
        participantes=("anon_id", "count"),
        profesiones_distintas=("profession", "nunique"),
        k_est_promedio=("k_est", "mean"),
        k_est_sd=("k_est", "std"),
        coherencia_promedio=("k_coherence_std", "mean"),
        marco_dominante=("fw_dom", lambda x: x.mode().iloc[0] if x.notna().any() else None),
    ).reset_index().sort_values("participantes", ascending=False)
    report["cohort_summary"] = cohort_summary

    frequency_summary = pd.concat([
        students["profession"].value_counts(dropna=False).rename("n").rename_axis("category").reset_index().assign(variable="profession"),
        students["k_level"].value_counts(dropna=False).rename("n").rename_axis("category").reset_index().assign(variable="k_level"),
        students["fw_dom"].value_counts(dropna=False).rename("n").rename_axis("category").reset_index().assign(variable="fw_dom"),
    ], ignore_index=True)
    frequency_summary["pct"] = frequency_summary.groupby("variable")["n"].transform(lambda s: (s / s.sum() * 100).round(2))
    report["frequency_summary"] = frequency_summary

    descriptive_targets = {"k_est": "Indice k_est", "k_coherence_std": "Coherencia", **{col: col.replace("fw_", "Marco ") for col in fw_cols}}
    descriptive_rows = []
    for column, label in descriptive_targets.items():
        values = students[column].dropna().astype(float).values if column in students.columns else np.array([])
        if len(values) == 0:
            continue
        mean, low, high = bootstrap_ci(values)
        descriptive_rows.append({
            "measure": label,
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "ic95_mean": f"[{low:.2f}, {high:.2f}]" if pd.notna(mean) else "NA",
        })
    report["descriptive_summary"] = pd.DataFrame(descriptive_rows)
    return report


def clean_text_series(text_series: pd.Series, stopwords: Sequence[str]) -> pd.Series:
    cleaned = text_series.fillna("").astype(str).str.lower()
    cleaned = cleaned.str.replace(r"[^a-záéíóúñü\s]", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    if stopwords:
        stopword_pattern = r"\b(?:" + "|".join(sorted({w for w in stopwords if w})) + r")\b"
        cleaned = cleaned.str.replace(stopword_pattern, " ", regex=True)
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def extract_keywords_by_group(df_last: pd.DataFrame, stopwords: Sequence[str], group_col: str = "profession", top_n: int = 12) -> pd.DataFrame:
    choices = df_last[df_last["row_type"] == "choice"].copy()
    if choices.empty:
        return pd.DataFrame()
    choices["clean"] = clean_text_series(choices["text"], stopwords)
    rows = []
    for group_value, subset in choices.groupby(group_col):
        terms = Counter()
        for text in subset["clean"]:
            tokens = [token for token in text.split() if len(token) > 2]
            terms.update(tokens)
        top_terms = ", ".join([token for token, _ in terms.most_common(top_n)])
        rows.append({group_col: group_value, "top_keywords": top_terms, "n_texts": int(len(subset))})
    return pd.DataFrame(rows).sort_values("n_texts", ascending=False)


def cluster_thematic_justifications(
    df_last: pd.DataFrame,
    stopwords: Sequence[str],
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    choices = df_last[df_last["row_type"] == "choice"].copy()
    if choices.empty:
        return pd.DataFrame(), pd.DataFrame()
    choices["clean"] = clean_text_series(choices["text"], stopwords)
    choices = choices[choices["clean"].str.len() > 0].copy()
    if len(choices) < 6:
        return pd.DataFrame(), pd.DataFrame()
    vectorizer = TfidfVectorizer(stop_words=list(stopwords), max_features=1200, ngram_range=(1, 2))
    X = vectorizer.fit_transform(choices["clean"])
    k = min(6, max(2, int(np.sqrt(X.shape[0]))))
    km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = km.fit_predict(X)
    choices["cluster"] = labels
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_rows = []
    for cluster_id in range(k):
        top_idx = np.argsort(km.cluster_centers_[cluster_id])[::-1][:10]
        subset = choices[choices["cluster"] == cluster_id]
        dominant_profession = subset["profession"].mode().iloc[0] if subset["profession"].notna().any() else None
        cluster_rows.append({
            "cluster": int(cluster_id),
            "n": int(len(subset)),
            "dominant_profession": dominant_profession,
            "top_terms": ", ".join(terms[top_idx]),
        })
    cluster_summary = pd.DataFrame(cluster_rows).sort_values("n", ascending=False)
    cluster_distribution = pd.crosstab(choices["profession"], choices["cluster"], normalize="index") * 100
    cluster_distribution = cluster_distribution.reset_index().melt(id_vars="profession", var_name="cluster", value_name="pct")
    return cluster_summary, cluster_distribution


def argumentative_pattern_table(df_last: pd.DataFrame, stopwords: Sequence[str]) -> pd.DataFrame:
    choices = df_last[df_last["row_type"] == "choice"].copy()
    if choices.empty:
        return pd.DataFrame()
    choices["clean"] = clean_text_series(choices["text"], stopwords)
    pattern_rows = []
    for _, row in choices.iterrows():
        tokens = set(token for token in row["clean"].split() if len(token) > 2)
        scores = {pattern: len(tokens & lexicon) for pattern, lexicon in ARGUMENT_PATTERN_LEXICONS.items()}
        dominant_pattern = max(scores, key=scores.get) if any(scores.values()) else "indeterminado"
        pattern_rows.append({
            "anon_id": row["anon_id"],
            "profession": row["profession"],
            "item_id": row["item_id"],
            "argument_pattern": dominant_pattern,
        })
    pattern_df = pd.DataFrame(pattern_rows)
    summary = pattern_df.groupby(["profession", "argument_pattern"]).size().reset_index(name="n")
    summary["pct_profession"] = summary.groupby("profession")["n"].transform(lambda s: (s / s.sum() * 100).round(2))
    return summary.sort_values(["profession", "n"], ascending=[True, False])


def profession_interpretive_trends(
    students: pd.DataFrame,
    keyword_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for profession, subset in students.groupby("profession"):
        dominant_level = subset["k_level"].mode().iloc[0] if subset["k_level"].notna().any() else None
        dominant_framework = subset["fw_dom"].mode().iloc[0] if subset["fw_dom"].notna().any() else None
        keywords = keyword_df.loc[keyword_df["profession"] == profession, "top_keywords"]
        keywords_text = keywords.iloc[0] if not keywords.empty else ""
        patterns = pattern_df.loc[pattern_df["profession"] == profession].sort_values("n", ascending=False)
        top_pattern = patterns.iloc[0]["argument_pattern"] if not patterns.empty else None
        rows.append({
            "profession": profession,
            "n": int(len(subset)),
            "nivel_dominante": dominant_level,
            "marco_dominante": dominant_framework,
            "patron_argumentativo": top_pattern,
            "palabras_clave": keywords_text,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def automatic_interpretive_synthesis(
    students: pd.DataFrame,
    quantitative_report: Dict[str, pd.DataFrame],
    trends_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> str:
    if students.empty:
        return "No hay datos suficientes para elaborar una síntesis interpretativa colectiva de carácter descriptivo."

    profession_dist = quantitative_report.get("profession_distribution", pd.DataFrame())
    top_profession = profession_dist.iloc[0]["profession"] if not profession_dist.empty else "sin datos"
    top_n = int(profession_dist.iloc[0]["n"]) if not profession_dist.empty else 0
    level_mode = students["k_level"].mode().iloc[0] if students["k_level"].notna().any() else "sin predominio claro"
    fw_mode = students["fw_dom"].mode().iloc[0] if students["fw_dom"].notna().any() else "sin predominio claro"
    k_mean = float(students["k_est"].dropna().mean()) if students["k_est"].notna().any() else np.nan
    stage_summary = quantitative_report.get("stage_summary", pd.DataFrame())
    profession_comparisons = quantitative_report.get("profession_comparisons", pd.DataFrame())
    cohort_summary = quantitative_report.get("cohort_summary", pd.DataFrame())

    sentences = [
        (
            f"Desde una perspectiva de bioética aplicada e investigación educativa con alcance descriptivo-exploratorio, la cohorte consolidada comprende {len(students)} participantes; "
            f"la mayor representación corresponde a {top_profession} (n={top_n}), de modo que cualquier lectura curricular o comparativa debe situarse en la composición efectiva de la muestra."
        ),
        (
            f"En el plano agregado, el nivel moral más frecuente se ubica en {level_mode} y el marco ético con mayor presencia relativa es {fw_mode}; "
            "en clave de deliberación bioética, estos predominantes se interpretan como regularidades argumentativas observables en las respuestas y no como atributos fijos o concluyentes de los participantes."
        ),
    ]
    if pd.notna(k_mean):
        sentences.append(
            (
                f"El índice promedio k_est se sitúa en {k_mean:.2f}; dentro del instrumento, este valor opera como un indicador pedagógico de organización del razonamiento moral "
                "útil para seguimiento formativo, reflexión docente y evaluación curricular de baja inferencia, antes que como una medición psicométrica cerrada."
            )
        )
    if not stage_summary.empty:
        dominant_stage = stage_summary.groupby("stage")["count"].sum().sort_values(ascending=False).index[0]
        sentences.append(
            (
                f"Al considerar los ítems tipo Likert asociados a estadios, el mayor peso descriptivo se concentra en el estadio {int(dominant_stage)}; "
                "esta señal sugiere una orientación argumentativa más visible en la muestra y puede leerse como un insumo para comprender trayectorias de deliberación moral en contextos de formación, siempre en relación con el carácter situado de los dilemas."
            )
        )
    if not trends_df.empty:
        trend = trends_df.iloc[0]
        sentences.append(
            (
                f"En la profesión {trend['profession']} se aprecia, de manera más visible, una articulación entre nivel {trend['nivel_dominante']}, marco {trend['marco_dominante']} "
                f"y patrón argumentativo {trend['patron_argumentativo']}; en el plano discursivo, ello se acompaña de una recurrencia léxica asociada a {trend['palabras_clave'] or 'una señal léxica sin predominio claro'}, "
                "lo que aporta pistas para interpretar énfasis formativos y repertorios de justificación profesional."
            )
        )
    if not cluster_summary.empty:
        cluster = cluster_summary.iloc[0]
        sentences.append(
            (
                f"El análisis de agrupamiento temático exploratorio identifica un núcleo discursivo recurrente caracterizado por términos como {cluster['top_terms']}; "
                f"su mayor presencia relativa en {cluster['dominant_profession'] or 'diversas profesiones'} sugiere focos de preocupación moral y énfasis justificativos que pueden dialogar con contenidos de bioética aplicada y con decisiones de ajuste curricular."
            )
        )
    if not profession_comparisons.empty:
        strongest_gap = profession_comparisons.iloc[0]
        if pd.notna(strongest_gap.get("hedges_g")):
            sentences.append(
                (
                    f"En las comparaciones entre profesiones, la diferencia descriptiva de mayor magnitud observada en k_est aparece entre {strongest_gap['profession_a']} y {strongest_gap['profession_b']} "
                    f"(Hedges g = {strongest_gap['hedges_g']:.2f}); esta magnitud orienta la lectura comparativa y puede servir como señal para priorizar acompañamiento pedagógico o revisión curricular, "
                    "pero no sustituye una interpretación contextual ni autoriza jerarquizaciones normativas entre grupos."
                )
            )
    if not cohort_summary.empty:
        cohort_count = int(cohort_summary["group"].nunique()) if "group" in cohort_summary.columns else len(cohort_summary)
        sentences.append(
            (
                f"La consolidación por cohorte muestra variación entre {cohort_count} grupos analíticos, lo que refuerza la conveniencia de leer los resultados desde una lógica comparativa prudente, sensible al tamaño muestral, a las condiciones locales de formación y a los posibles usos en evaluación curricular."
            )
        )
    sentences.append(
        (
            "En conjunto, esta síntesis debe entenderse como una lectura analítica de apoyo para docencia, reflexión en bioética aplicada, investigación educativa y evaluación curricular de carácter formativo; "
            "describe regularidades argumentativas observables en respuestas escritas, pero no autoriza inferencias clínicas, diagnósticas ni juicios definitivos sobre desarrollo o competencia moral individual."
        )
    )
    return " ".join(sentences)
