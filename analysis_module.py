from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ARGUMENT_PATTERN_LEXICONS = {
    "consecuencias": {
        "beneficio", "beneficios", "riesgo", "riesgos", "impacto", "resultado", "resultados",
        "consecuencia", "consecuencias", "efecto", "efectos", "eficacia", "daño", "danos", "bienestar",
        "perjuicio", "utilidad", "seguridad", "calidad", "costo", "costos", "beneficioso",
        "perjudicial", "perdida", "pérdida", "ganancia", "maximizar", "minimizar",
    },
    "normas": {
        "norma", "normas", "regla", "reglas", "protocolo", "protocolos", "deber", "deberes",
        "obligacion", "obligaciones", "cumplir", "legal", "ley", "leyes", "procedimiento",
        "reglamento", "normativa", "política", "politica", "estandar", "estándar", "codigo",
        "código", "cumplimiento", "sancion", "sanción", "disposicion", "disposición",
        "incumplimiento", "autoridad", "mandato",
    },
    "cuidado": {
        "cuidado", "vulnerable", "vulnerables", "empatía", "empatia", "acompanamiento", "acompañamiento",
        "relacion", "relaciones", "sufrimiento", "proteger", "apoyo", "familia", "equipo",
        "compasion", "compasión", "sensibilidad", "atencion", "atención", "solidaridad",
        "humanizar", "humano", "confianza", "proteccion", "protección",
    },
    "justicia": {
        "justicia", "equidad", "derechos", "derecho", "dignidad", "proporcionalidad", "imparcialidad",
        "transparencia", "responsabilidad", "garantia", "garantias", "publico", "consenso",
        "igualdad", "justo", "injusto", "acceso", "inclusion", "inclusión", "discriminacion",
        "discriminación", "autonomia", "autonomía", "consentimiento", "trato",
    },
    "integridad": {
        "integridad", "honestidad", "verdad", "prudencia", "caracter", "carácter", "virtud", "virtudes",
        "coherencia", "responsable", "responsabilidad", "etico", "ético",
        "veracidad", "transparente", "confiable", "principio", "principios",
        "compromiso", "lealtad", "autenticidad", "credibilidad",
    },
    "bioseguridad_calidad": {
        "bioseguridad", "esterilidad", "contaminacion", "contaminación", "muestra", "muestras",
        "cabina", "aislamiento", "resistencia", "antibiotico", "antibiótico", "cultivo", "cultivos",
        "cepa", "cepas", "laboratorio", "preanalitico", "preanalítico", "analitico", "analítico",
        "postanalitico", "postanalítico", "trazabilidad", "validez", "confiabilidad",
        "precision", "precisión", "informe", "reporte",
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


def _safe_mode(series: pd.Series) -> Any:
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else None


def _is_approximately_normal(values: Sequence[float] | np.ndarray, alpha: float = 0.05) -> bool:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 8:
        return False
    if np.ptp(arr) == 0:
        return False
    if len(arr) > 5000:
        arr = np.random.choice(arr, size=5000, replace=False)
    try:
        _, p_value = stats.shapiro(arr)
    except Exception:
        return False
    return bool(p_value > alpha)


def _rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    if n_a <= 0 or n_b <= 0:
        return np.nan
    return float(1 - (2 * u_stat) / (n_a * n_b))


def _eta_squared_from_f(f_stat: float, groups_n: int, total_n: int) -> float:
    if total_n <= groups_n or pd.isna(f_stat):
        return np.nan
    numerator = f_stat * (groups_n - 1)
    denominator = numerator + (total_n - groups_n)
    return float(numerator / denominator) if denominator else np.nan


def _epsilon_squared_from_h(h_stat: float, groups_n: int, total_n: int) -> float:
    if total_n <= groups_n or pd.isna(h_stat):
        return np.nan
    return float((h_stat - groups_n + 1) / (total_n - groups_n))


def _cramers_v(contingency: pd.DataFrame) -> float:
    if contingency.empty:
        return np.nan
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    if n == 0:
        return np.nan
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))


def _sanitize_feature_names(feature_names: Sequence[str]) -> list[str]:
    return [str(name).replace("num__", "").replace("cat__", "") for name in feature_names]


def build_argument_pattern_profiles(df_last: pd.DataFrame, stopwords: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    choices = df_last[df_last["row_type"] == "choice"].copy()
    if choices.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    choices["clean"] = clean_text_series(choices["text"], stopwords)
    pattern_rows = []
    for _, row in choices.iterrows():
        tokens = set(token for token in str(row["clean"]).split() if len(token) > 2)
        scores = {pattern: len(tokens & lexicon) for pattern, lexicon in ARGUMENT_PATTERN_LEXICONS.items()}
        dominant_pattern = max(scores, key=scores.get) if any(scores.values()) else "indeterminado"
        pattern_rows.append({
            "anon_id": row["anon_id"],
            "profession": row.get("profession"),
            "group": row.get("group"),
            "item_id": row.get("item_id"),
            "argument_pattern": dominant_pattern,
        })
    pattern_df = pd.DataFrame(pattern_rows)
    if pattern_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    participant_summary = pattern_df.groupby("anon_id").agg(
        profession=("profession", "first"),
        group=("group", "first"),
        argument_pattern_dom=("argument_pattern", _safe_mode),
        argument_pattern_diversity=("argument_pattern", "nunique"),
        n_argument_rows=("argument_pattern", "size"),
    ).reset_index()
    profession_summary = pattern_df.groupby(["profession", "argument_pattern"]).size().reset_index(name="n")
    profession_summary["pct_profession"] = profession_summary.groupby("profession")["n"].transform(
        lambda s: (s / s.sum() * 100).round(2)
    )
    return pattern_df, participant_summary, profession_summary.sort_values(["profession", "n"], ascending=[True, False])


def build_frequency_tables(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        subset = df[column].copy()
        if subset.dropna().empty:
            continue
        freq = subset.fillna("No disponible").astype(str).value_counts(dropna=False).rename_axis("category").reset_index(name="n")
        freq["pct"] = (freq["n"] / freq["n"].sum() * 100).round(2)
        freq["variable"] = column
        rows.append(freq[["variable", "category", "n", "pct"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["variable", "category", "n", "pct"])


def build_numeric_descriptives(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna().values
        if len(values) == 0:
            continue
        rows.append({
            "variable": column,
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    return pd.DataFrame(rows)


def build_correlation_analysis(
    df: pd.DataFrame,
    predictors: Sequence[str],
    outcomes: Sequence[str],
    ordinal_predictors: Sequence[str] | None = None,
    ordinal_outcomes: Sequence[str] | None = None,
) -> pd.DataFrame:
    ordinal_predictors = set(ordinal_predictors or [])
    ordinal_outcomes = set(ordinal_outcomes or [])
    rows = []
    for predictor in predictors:
        if predictor not in df.columns:
            continue
        for outcome in outcomes:
            if outcome not in df.columns:
                continue
            subset = df[[predictor, outcome]].copy()
            subset[predictor] = pd.to_numeric(subset[predictor], errors="coerce")
            subset[outcome] = pd.to_numeric(subset[outcome], errors="coerce")
            subset = subset.dropna()
            if len(subset) < 5 or subset[predictor].nunique() < 2 or subset[outcome].nunique() < 2:
                continue
            use_spearman = (
                predictor in ordinal_predictors
                or outcome in ordinal_outcomes
                or not _is_approximately_normal(subset[predictor].values)
                or not _is_approximately_normal(subset[outcome].values)
            )
            if use_spearman:
                statistic, p_value = stats.spearmanr(subset[predictor].values, subset[outcome].values)
                method = "Spearman"
            else:
                statistic, p_value = stats.pearsonr(subset[predictor].values, subset[outcome].values)
                method = "Pearson"
            rows.append({
                "predictor": predictor,
                "outcome": outcome,
                "method": method,
                "n": int(len(subset)),
                "statistic": float(statistic) if pd.notna(statistic) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["p_value", "predictor", "outcome"], na_position="last")
    return result


def build_group_comparison_analysis(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    outcome_columns: Sequence[str],
    ordinal_outcomes: Sequence[str] | None = None,
) -> pd.DataFrame:
    ordinal_outcomes = set(ordinal_outcomes or [])
    rows = []
    for group_column in group_columns:
        if group_column not in df.columns:
            continue
        for outcome_column in outcome_columns:
            if outcome_column not in df.columns:
                continue
            subset = df[[group_column, outcome_column]].copy()
            subset[outcome_column] = pd.to_numeric(subset[outcome_column], errors="coerce")
            subset = subset.dropna()
            if subset.empty or subset[outcome_column].nunique() < 2:
                continue
            grouped = [(name, values[outcome_column].values.astype(float)) for name, values in subset.groupby(group_column) if len(values) >= 2]
            if len(grouped) < 2:
                continue
            group_names = [name for name, _ in grouped]
            grouped_values = [values for _, values in grouped]
            normality_ok = all(_is_approximately_normal(values) for values in grouped_values)
            if len(grouped_values) == 2:
                a, b = grouped_values
                if outcome_column not in ordinal_outcomes and normality_ok:
                    _, levene_p = stats.levene(a, b)
                    statistic, p_value = stats.ttest_ind(a, b, equal_var=bool(levene_p > 0.05))
                    effect_size = hedges_g(a, b)
                    test_name = "t-test"
                    effect_label = "Hedges g"
                else:
                    statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
                    effect_size = _rank_biserial_from_u(statistic, len(a), len(b))
                    test_name = "Mann-Whitney U"
                    effect_label = "rango-biserial"
                rows.append({
                    "group_variable": group_column,
                    "outcome": outcome_column,
                    "groups": f"{group_names[0]} vs {group_names[1]}",
                    "test": test_name,
                    "n": int(len(a) + len(b)),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "effect_size": effect_size,
                    "effect_label": effect_label,
                })
                continue
            if outcome_column not in ordinal_outcomes and normality_ok:
                statistic, p_value = stats.f_oneway(*grouped_values)
                effect_size = _eta_squared_from_f(statistic, len(grouped_values), sum(len(values) for values in grouped_values))
                test_name = "ANOVA"
                effect_label = "eta^2"
            else:
                statistic, p_value = stats.kruskal(*grouped_values)
                effect_size = _epsilon_squared_from_h(statistic, len(grouped_values), sum(len(values) for values in grouped_values))
                test_name = "Kruskal-Wallis"
                effect_label = "epsilon^2"
            rows.append({
                "group_variable": group_column,
                "outcome": outcome_column,
                "groups": ", ".join(map(str, group_names)),
                "test": test_name,
                "n": int(sum(len(values) for values in grouped_values)),
                "statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size": effect_size,
                "effect_label": effect_label,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["p_value", "group_variable", "outcome"], na_position="last")
    return result


def build_categorical_association_analysis(
    df: pd.DataFrame,
    predictor_columns: Sequence[str],
    outcome_columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for predictor in predictor_columns:
        if predictor not in df.columns:
            continue
        for outcome in outcome_columns:
            if outcome not in df.columns:
                continue
            subset = df[[predictor, outcome]].dropna().copy()
            if subset.empty:
                continue
            contingency = pd.crosstab(subset[predictor], subset[outcome])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue
            chi2, p_value, degrees, expected = stats.chi2_contingency(contingency)
            rows.append({
                "predictor": predictor,
                "outcome": outcome,
                "n": int(contingency.values.sum()),
                "degrees_freedom": int(degrees),
                "chi2": float(chi2),
                "p_value": float(p_value),
                "cramers_v": _cramers_v(contingency),
                "min_expected": float(np.min(expected)),
                "pct_expected_lt5": float((expected < 5).mean() * 100),
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["p_value", "predictor", "outcome"], na_position="last")
    return result


def build_predictive_models(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    classification_targets: Sequence[str],
    regression_targets: Sequence[str],
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usable_features = [column for column in feature_columns if column in df.columns and df[column].notna().any()]
    if not usable_features:
        empty = pd.DataFrame()
        return empty, empty

    numeric_features = [column for column in usable_features if pd.api.types.is_numeric_dtype(df[column])]
    categorical_features = [column for column in usable_features if column not in numeric_features]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    score_rows = []
    importance_rows = []

    for target in classification_targets:
        if target not in df.columns:
            continue
        subset = df[usable_features + [target]].dropna(subset=[target]).copy()
        if len(subset) < 20 or subset[target].nunique() < 2:
            continue
        class_counts = subset[target].astype(str).value_counts()
        min_class_n = int(class_counts.min()) if not class_counts.empty else 0
        if min_class_n < 2:
            continue
        cv = StratifiedKFold(n_splits=min(5, min_class_n), shuffle=True, random_state=random_seed)
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_seed, class_weight="balanced"),
        }
        X = subset[usable_features]
        y = subset[target].astype(str)
        for model_name, estimator in models.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring={"balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
            )
            score_rows.append({
                "target": target,
                "task": "classification",
                "model": model_name,
                "n": int(len(subset)),
                "classes": int(y.nunique()),
                "metric": "balanced_accuracy",
                "mean_score": float(np.mean(scores["test_balanced_accuracy"])),
                "sd_score": float(np.std(scores["test_balanced_accuracy"])),
            })
            score_rows.append({
                "target": target,
                "task": "classification",
                "model": model_name,
                "n": int(len(subset)),
                "classes": int(y.nunique()),
                "metric": "f1_macro",
                "mean_score": float(np.mean(scores["test_f1_macro"])),
                "sd_score": float(np.std(scores["test_f1_macro"])),
            })
            if model_name != "random_forest":
                continue
            fitted = pipeline.fit(X, y)
            feature_names = _sanitize_feature_names(fitted.named_steps["preprocessor"].get_feature_names_out())
            importances = fitted.named_steps["model"].feature_importances_
            top_indices = np.argsort(importances)[::-1][:20]
            for index in top_indices:
                importance_rows.append({
                    "target": target,
                    "task": "classification",
                    "model": model_name,
                    "feature": feature_names[index],
                    "importance": float(importances[index]),
                })

    for target in regression_targets:
        if target not in df.columns:
            continue
        subset = df[usable_features + [target]].copy()
        subset[target] = pd.to_numeric(subset[target], errors="coerce")
        subset = subset.dropna(subset=[target])
        if len(subset) < 25:
            continue
        cv = KFold(n_splits=min(5, len(subset)), shuffle=True, random_state=random_seed)
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(n_estimators=300, random_state=random_seed),
        }
        X = subset[usable_features]
        y = subset[target].astype(float)
        for model_name, estimator in models.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
            )
            score_rows.append({
                "target": target,
                "task": "regression",
                "model": model_name,
                "n": int(len(subset)),
                "classes": np.nan,
                "metric": "r2",
                "mean_score": float(np.mean(scores["test_r2"])),
                "sd_score": float(np.std(scores["test_r2"])),
            })
            score_rows.append({
                "target": target,
                "task": "regression",
                "model": model_name,
                "n": int(len(subset)),
                "classes": np.nan,
                "metric": "mae",
                "mean_score": float(-np.mean(scores["test_neg_mae"])),
                "sd_score": float(np.std(-scores["test_neg_mae"])),
            })
            if model_name != "random_forest_regressor":
                continue
            fitted = pipeline.fit(X, y)
            feature_names = _sanitize_feature_names(fitted.named_steps["preprocessor"].get_feature_names_out())
            importances = fitted.named_steps["model"].feature_importances_
            top_indices = np.argsort(importances)[::-1][:20]
            for index in top_indices:
                importance_rows.append({
                    "target": target,
                    "task": "regression",
                    "model": model_name,
                    "feature": feature_names[index],
                    "importance": float(importances[index]),
                })

    return pd.DataFrame(score_rows), pd.DataFrame(importance_rows)


def build_statistical_association_report(
    students: pd.DataFrame,
    df_last: pd.DataFrame,
    stopwords: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    empty = pd.DataFrame()
    if students.empty:
        return {
            "analysis_frame": empty,
            "sample_overview": empty,
            "independent_frequencies": empty,
            "outcome_frequencies": empty,
            "numeric_descriptives": empty,
            "correlations": empty,
            "group_comparisons": empty,
            "categorical_associations": empty,
            "predictive_models": empty,
            "feature_importance": empty,
            "methodological_notes": empty,
        }

    analysis_df = students.copy()
    _, participant_patterns, _ = build_argument_pattern_profiles(df_last, stopwords)
    if not participant_patterns.empty:
        analysis_df = analysis_df.merge(
            participant_patterns[["anon_id", "argument_pattern_dom", "argument_pattern_diversity"]],
            on="anon_id",
            how="left",
        )

    independent_numeric = [
        column for column in ["age", "semester", "children_count", "work_hours_per_week", "years_experience"]
        if column in analysis_df.columns
    ]
    independent_categorical = [
        column for column in [
            "gender",
            "works_for_studies",
            "profession",
            "academic_program",
            "academic_shift",
            "prior_experience_area",
            "ethics_training",
            "caregiving_load",
            "study_funding_type",
            "group",
        ]
        if column in analysis_df.columns
    ]
    numeric_outcomes = [
        column for column in ["k_est", "k_stage", "k_coherence_std", "argument_pattern_diversity"]
        if column in analysis_df.columns
    ]
    numeric_outcomes.extend(column for column in analysis_df.columns if column.startswith("fw_") and column != "fw_dom")
    categorical_outcomes = [
        column for column in ["k_level", "fw_dom", "argument_pattern_dom"]
        if column in analysis_df.columns
    ]

    sample_overview = pd.DataFrame([
        {"metric": "participantes", "value": int(len(analysis_df))},
        {"metric": "profesiones", "value": int(analysis_df["profession"].dropna().nunique()) if "profession" in analysis_df.columns else 0},
        {"metric": "cohortes", "value": int(analysis_df["group"].dropna().nunique()) if "group" in analysis_df.columns else 0},
        {"metric": "variables_independientes_categoricas", "value": int(len(independent_categorical))},
        {"metric": "variables_independientes_numericas", "value": int(len(independent_numeric))},
        {"metric": "resultados_numericos", "value": int(len(numeric_outcomes))},
        {"metric": "resultados_categoricos", "value": int(len(categorical_outcomes))},
    ])

    correlations = build_correlation_analysis(
        analysis_df,
        predictors=independent_numeric,
        outcomes=numeric_outcomes,
        ordinal_predictors=["semester", "children_count"],
        ordinal_outcomes=["k_stage", "argument_pattern_diversity"],
    )
    group_comparisons = build_group_comparison_analysis(
        analysis_df,
        group_columns=independent_categorical,
        outcome_columns=numeric_outcomes,
        ordinal_outcomes=["k_stage", "argument_pattern_diversity"],
    )
    categorical_associations = build_categorical_association_analysis(
        analysis_df,
        predictor_columns=independent_categorical,
        outcome_columns=categorical_outcomes,
    )
    predictive_models, feature_importance = build_predictive_models(
        analysis_df,
        feature_columns=independent_numeric + independent_categorical,
        classification_targets=[column for column in ["k_level", "fw_dom", "argument_pattern_dom"] if column in analysis_df.columns],
        regression_targets=[column for column in ["k_est"] if column in analysis_df.columns],
    )
    methodological_notes = pd.DataFrame([
        {"tema": "Interpretación", "nota": "Los resultados deben leerse como asociaciones, correlaciones o capacidad predictiva exploratoria; no implican causalidad."},
        {"tema": "Selección de prueba", "nota": "Spearman se prioriza para variables ordinales o distribuciones no normales; Pearson solo se usa cuando la forma de los datos lo permite."},
        {"tema": "Comparaciones entre grupos", "nota": "Se usan pruebas no paramétricas cuando la normalidad no se sostiene o cuando el resultado es ordinal; ANOVA y t-test quedan restringidos a supuestos compatibles."},
        {"tema": "Modelos predictivos", "nota": "Los modelos multivariables y de bosque aleatorio funcionan como apoyo exploratorio para priorizar hipótesis analíticas, no como diagnóstico individual."},
        {"tema": "Muestra", "nota": "P-valores, tamaños de efecto y métricas predictivas dependen del tamaño muestral, del desbalance entre clases y de la calidad de los registros disponibles."},
    ])

    return {
        "analysis_frame": analysis_df,
        "sample_overview": sample_overview,
        "independent_frequencies": build_frequency_tables(analysis_df, independent_categorical),
        "outcome_frequencies": build_frequency_tables(analysis_df, categorical_outcomes),
        "numeric_descriptives": build_numeric_descriptives(analysis_df, independent_numeric + numeric_outcomes),
        "correlations": correlations,
        "group_comparisons": group_comparisons,
        "categorical_associations": categorical_associations,
        "predictive_models": predictive_models,
        "feature_importance": feature_importance,
        "methodological_notes": methodological_notes,
    }


def build_quantitative_report(students: pd.DataFrame, df_last: pd.DataFrame, frameworks: Sequence[str]) -> Dict[str, pd.DataFrame]:
    report: Dict[str, pd.DataFrame] = {}
    if students.empty:
        empty = pd.DataFrame()
        return {
            "profession_distribution": empty,
            "stage_summary": empty,
            "framework_summary": empty,
            "framework_ci_summary": empty,
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

    fw_ci_rows = []
    for fw in frameworks:
        fw_col = f"fw_{fw}"
        if fw_col not in students.columns:
            continue
        for profession, g in students.groupby("profession"):
            values = g[fw_col].dropna().values
            mean, low, high = bootstrap_ci(values)
            fw_ci_rows.append({
                "profession": profession,
                "framework": fw,
                "fw_mean": round(float(mean), 3) if pd.notna(mean) else np.nan,
                "ci_low": round(float(low), 3) if pd.notna(low) else np.nan,
                "ci_high": round(float(high), 3) if pd.notna(high) else np.nan,
            })
    report["framework_ci_summary"] = pd.DataFrame(fw_ci_rows)

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
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
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
    choices = choices[choices["clean"].str.len() > 0].copy()
    if choices.empty:
        return pd.DataFrame()
    n_texts_map = choices.groupby(group_col).size().to_dict()
    if len(n_texts_map) >= 2:
        group_corpus = choices.groupby(group_col)["clean"].apply(lambda s: " ".join(s))
        try:
            vectorizer = TfidfVectorizer(
                stop_words=list(stopwords),
                min_df=1,
                max_features=2000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            tfidf_matrix = vectorizer.fit_transform(group_corpus)
            terms_arr = np.array(vectorizer.get_feature_names_out())
            rows = []
            for i, group_value in enumerate(group_corpus.index):
                scores = tfidf_matrix[i].toarray().flatten()
                top_idx = np.argsort(scores)[::-1][:top_n]
                top_terms = [terms_arr[j] for j in top_idx if scores[j] > 0]
                rows.append({group_col: group_value, "top_keywords": ", ".join(top_terms), "n_texts": n_texts_map.get(group_value, 0)})
            return pd.DataFrame(rows).sort_values("n_texts", ascending=False)
        except Exception:
            pass
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
    _, _, summary = build_argument_pattern_profiles(df_last, stopwords)
    return summary


def internal_consistency_estimate(df_last: pd.DataFrame) -> pd.DataFrame:
    """Proxy de consistencia interna del bloque Likert por estadio usando correlación media
    interitem y fórmula de Spearman-Brown. Señal descriptiva-formativa, no diagnóstico psicométrico."""
    stage_rows = df_last[df_last["row_type"] == "stage_likert"].copy()
    if stage_rows.empty:
        return pd.DataFrame()
    stage_rows["sub_id_num"] = pd.to_numeric(stage_rows["sub_id"], errors="coerce")
    stage_rows = stage_rows.dropna(subset=["sub_id_num"])
    pivoted = stage_rows.pivot_table(
        index=["anon_id", "profession"],
        columns="sub_id_num",
        values="likert_value",
        aggfunc="mean",
    ).dropna()
    if pivoted.shape[0] < 3 or pivoted.shape[1] < 2:
        return pd.DataFrame()
    n_items = int(pivoted.shape[1])
    rows = []
    for profession, g in pivoted.groupby(level="profession"):
        mat = g.droplevel("profession")
        if len(mat) < 2:
            continue
        corr_matrix = mat.corr()
        off_diag = corr_matrix.values[~np.eye(corr_matrix.shape[0], dtype=bool)]
        valid = off_diag[~np.isnan(off_diag)]
        if len(valid) == 0:
            continue
        mean_r = float(valid.mean())
        denom = 1 + (n_items - 1) * mean_r
        alpha_proxy = (n_items * mean_r / denom) if denom != 0 and not np.isnan(mean_r) else np.nan
        rows.append({
            "profesion": profession,
            "n_participantes": int(len(mat)),
            "r_media_interitem": round(mean_r, 3),
            "alpha_proxy": round(float(alpha_proxy), 3) if pd.notna(alpha_proxy) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_participantes", ascending=False)


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
    frequency_summary = quantitative_report.get("frequency_summary", pd.DataFrame())
    if not frequency_summary.empty:
        level_freq = frequency_summary[frequency_summary["variable"] == "k_level"].copy()
        if not level_freq.empty:
            top_level_row = level_freq.sort_values("pct", ascending=False).iloc[0]
            sentences.append(
                f"El análisis de frecuencias sobre los niveles morales registrados indica que la categoría '{top_level_row['category']}' concentra el {top_level_row['pct']:.1f}% de las observaciones; "
                "este dato de frecuencia relativa puede orientar la discusión curricular sobre qué estadios de argumentación conviene reforzar con mayor énfasis en el programa de formación, "
                "sin convertir el porcentaje en un veredicto sobre la calidad moral de los participantes."
            )
    sentences.append(
        (
            "En conjunto, esta síntesis debe entenderse como una lectura analítica de apoyo para docencia, reflexión en bioética aplicada, investigación educativa y evaluación curricular de carácter formativo; "
            "describe regularidades argumentativas observables en respuestas escritas, pero no autoriza inferencias clínicas, diagnósticas ni juicios definitivos sobre desarrollo o competencia moral individual."
        )
    )
    return " ".join(sentences)
