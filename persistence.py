from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from data_schema import (
    COLUMNS,
    PARTICIPANT_CONTEXT_COLUMNS,
    PARTICIPANT_CONTEXT_COLUMN_TYPES,
    aggregate_metric_columns,
    attempt_select_columns,
    attempt_summary_select_columns,
    response_select_columns,
)

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

ATTEMPTS_TABLE = "participant_attempts"
RESPONSES_TABLE = "attempt_responses"
ATTEMPT_METADATA_COLUMNS = {
    "n_dilemmas_answered": "INTEGER",
    "n_justifications": "INTEGER",
    "responded_item_ids": "TEXT",
}


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)


def _to_optional_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value)
    return text


def _to_optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    return int(value)


@dataclass(frozen=True)
class PersistenceConfig:
    backend: str
    sqlite_path: Path
    legacy_csv_path: Path | None
    supabase_db_url: str | None


class PersistenceStore:
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self._sqlite_conn: sqlite3.Connection | None = None

    @property
    def backend_name(self) -> str:
        return self.config.backend

    @property
    def backend_detail(self) -> str:
        if self.config.backend == "supabase":
            return "Supabase Postgres"
        return f"SQLite local ({self.config.sqlite_path})"

    def ensure_storage(self) -> None:
        self.config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.legacy_csv_path is not None:
            self.config.legacy_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._ensure_schema(conn)
        self._migrate_legacy_csv_if_needed()

    def save_rows(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        self.ensure_storage()
        payload = rows[COLUMNS].copy()
        attempt_meta = self._extract_attempt_meta(payload)
        response_records = self._extract_response_records(payload)
        with self._connect() as conn:
            self._ensure_schema(conn)
            self._upsert_attempt(conn, attempt_meta)
            self._replace_attempt_rows(conn, attempt_meta["attempt_id"], response_records)

    def load_all_rows(self) -> pd.DataFrame:
        self.ensure_storage()
        with self._connect() as conn:
            query = f"""
                SELECT
                    {self._attempt_response_select_sql()}
                FROM {ATTEMPTS_TABLE} a
                JOIN {RESPONSES_TABLE} r ON r.attempt_id = a.attempt_id
                ORDER BY a.timestamp, r.response_order
            """
            rows = self._read_sql(conn, query)
        return self._normalize_dataframe(rows)

    def get_analysis_frame(
        self,
        profession: str | None = None,
        group_name: str | None = None,
        anon_id: str | None = None,
        limit: int | None = None,
        include_raw_payload: bool = False,
    ) -> pd.DataFrame:
        self.ensure_storage()
        with self._connect() as conn:
            query = f"""
                SELECT
                    {self._attempt_summary_select_sql(include_raw_payload=include_raw_payload)}
                FROM {ATTEMPTS_TABLE} a
            """
            query, params = self._append_attempt_filters(
                query,
                profession=profession,
                group_name=group_name,
                anon_id=anon_id,
                limit=limit,
            )
            return self._read_sql(conn, query, params)

    def get_attempt_summaries(
        self,
        profession: str | None = None,
        group_name: str | None = None,
        anon_id: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        return self.get_analysis_frame(
            profession=profession,
            group_name=group_name,
            anon_id=anon_id,
            limit=limit,
            include_raw_payload=False,
        )

    def get_last_attempt_rows(self, anon_id: str) -> pd.DataFrame:
        self.ensure_storage()
        with self._connect() as conn:
            placeholder = self._placeholder
            query = f"""
                SELECT
                    {self._attempt_response_select_sql()}
                FROM {ATTEMPTS_TABLE} a
                JOIN {RESPONSES_TABLE} r ON r.attempt_id = a.attempt_id
                WHERE a.anon_id = {placeholder}
                  AND a.timestamp = (
                    SELECT MAX(timestamp) FROM {ATTEMPTS_TABLE} WHERE anon_id = {placeholder}
                  )
                ORDER BY r.response_order
            """
            params = [anon_id, anon_id]
            rows = self._read_sql(conn, query, params)
        return self._normalize_dataframe(rows)

    def get_user_history(self, anon_id: str) -> pd.DataFrame:
        self.ensure_storage()
        with self._connect() as conn:
            placeholder = self._placeholder
            query = f"""
                SELECT
                    {self._attempt_response_select_sql()}
                FROM {ATTEMPTS_TABLE} a
                JOIN {RESPONSES_TABLE} r ON r.attempt_id = a.attempt_id
                WHERE a.anon_id = {placeholder}
                ORDER BY a.timestamp, r.response_order
            """
            rows = self._read_sql(conn, query, [anon_id])
        return self._normalize_dataframe(rows)

    def get_cohort_history(self, group_name: str) -> pd.DataFrame:
        self.ensure_storage()
        with self._connect() as conn:
            placeholder = self._placeholder
            query = f"""
                SELECT
                    {self._attempt_response_select_sql()}
                FROM {ATTEMPTS_TABLE} a
                JOIN {RESPONSES_TABLE} r ON r.attempt_id = a.attempt_id
                WHERE a.group_name = {placeholder}
                ORDER BY a.timestamp, r.response_order
            """
            rows = self._read_sql(conn, query, [group_name])
        return self._normalize_dataframe(rows)

    def get_profession_comparison(self, professions: Sequence[str] | None = None) -> pd.DataFrame:
        return self._comparison_frame(group_column="profession", values=professions, order_label="profession")

    def get_cohort_comparison(self, groups: Sequence[str] | None = None) -> pd.DataFrame:
        return self._comparison_frame(group_column="group_name", values=groups, order_label="group_name")

    def run_query(self, query_name: str, **filters: Any) -> pd.DataFrame:
        if query_name == "last_attempt":
            anon_id = filters.get("anon_id")
            if not anon_id:
                raise ValueError("anon_id es obligatorio para consultar el último intento.")
            return self.get_last_attempt_rows(str(anon_id))
        if query_name == "user_history":
            anon_id = filters.get("anon_id")
            if not anon_id:
                raise ValueError("anon_id es obligatorio para consultar el histórico por usuario.")
            return self.get_user_history(str(anon_id))
        if query_name == "cohort_history":
            group_name = filters.get("group_name")
            if not group_name:
                raise ValueError("group_name es obligatorio para consultar el histórico por cohorte.")
            return self.get_cohort_history(str(group_name))
        if query_name == "profession_comparison":
            professions = filters.get("professions")
            return self.get_profession_comparison(professions)
        if query_name == "cohort_comparison":
            groups = filters.get("groups")
            return self.get_cohort_comparison(groups)
        if query_name == "analysis_frame":
            return self.get_analysis_frame(
                profession=filters.get("profession"),
                group_name=filters.get("group_name"),
                anon_id=filters.get("anon_id"),
                limit=filters.get("limit"),
                include_raw_payload=bool(filters.get("include_raw_payload", False)),
            )
        if query_name == "attempt_summaries":
            return self.get_attempt_summaries(
                profession=filters.get("profession"),
                group_name=filters.get("group_name"),
                anon_id=filters.get("anon_id"),
                limit=filters.get("limit"),
            )
        raise ValueError(f"Consulta no soportada: {query_name}")

    def _migrate_legacy_csv_if_needed(self) -> None:
        legacy_path = self.config.legacy_csv_path
        if legacy_path is None or not legacy_path.exists():
            return
        if self._attempt_count() > 0:
            return
        legacy_df = pd.read_csv(legacy_path)
        if legacy_df.empty:
            return
        for column in COLUMNS:
            if column not in legacy_df.columns:
                legacy_df[column] = np.nan
        for _, attempt_rows in legacy_df[COLUMNS].groupby(["anon_id", "timestamp"], dropna=False):
            self.save_rows(attempt_rows.reset_index(drop=True))

    def _attempt_count(self) -> int:
        with self._connect() as conn:
            self._ensure_schema(conn)
            cursor = conn.execute(f"SELECT COUNT(*) FROM {ATTEMPTS_TABLE}")
            result = cursor.fetchone()
        return int(result[0]) if result else 0

    def _extract_attempt_meta(self, rows: pd.DataFrame) -> Dict[str, Any]:
        first = rows.iloc[0]
        choice_rows = rows[rows["row_type"] == "choice"].copy()
        responded_item_ids = sorted(choice_rows["item_id"].dropna().astype(str).unique().tolist())
        n_justifications = int(choice_rows["text"].fillna("").astype(str).str.strip().ne("").sum())
        meta = {
            "attempt_id": f"{first['anon_id']}_{first['timestamp']}_{uuid4().hex[:8]}",
            "timestamp": str(first["timestamp"]),
            "anon_id": _to_optional_str(first["anon_id"]),
            "student_id": _to_optional_str(first["student_id"]),
            "name": _to_optional_str(first["name"]),
            "profession": _to_optional_str(first["profession"]),
            "years_experience": _to_optional_int(first["years_experience"]),
            "group_name": _to_optional_str(first["group"]),
            "n_dilemmas_answered": int(len(responded_item_ids)),
            "n_justifications": n_justifications,
            "responded_item_ids": ", ".join(responded_item_ids),
            "raw_payload": rows.to_json(orient="records", force_ascii=False),
        }
        for column in PARTICIPANT_CONTEXT_COLUMNS:
            value = first.get(column)
            if PARTICIPANT_CONTEXT_COLUMN_TYPES.get(column) == "INTEGER":
                meta[column] = _to_optional_int(value)
            else:
                meta[column] = _to_optional_str(value)
        return meta

    def _extract_response_records(self, rows: pd.DataFrame) -> list[Dict[str, Any]]:
        records = []
        for index, row in rows.reset_index(drop=True).iterrows():
            records.append({
                "response_order": index,
                "row_type": _to_optional_str(row["row_type"]),
                "item_id": _to_optional_str(row["item_id"]),
                "sub_id": _to_optional_str(row["sub_id"]),
                "choice_key": _to_optional_str(row["choice_key"]),
                "choice_stage": _to_optional_int(row["choice_stage"]),
                "choice_level": _to_optional_str(row["choice_level"]),
                "choice_framework": _to_optional_str(row["choice_framework"]),
                "likert_value": _to_optional_int(row["likert_value"]),
                "text_value": _to_optional_str(row["text"]),
            })
        return records

    @property
    def _placeholder(self) -> str:
        return "%s" if self.config.backend == "supabase" else "?"

    def _connect(self):
        if self.config.backend == "supabase":
            if not self.config.supabase_db_url:
                raise RuntimeError("SUPABASE_DB_URL no está configurada.")
            if psycopg is None:
                raise RuntimeError("psycopg no está instalado para usar Supabase Postgres.")
            return psycopg.connect(self.config.supabase_db_url)
        if self._sqlite_conn is None:
            self._sqlite_conn = sqlite3.connect(self.config.sqlite_path, check_same_thread=False)
            self._sqlite_conn.row_factory = sqlite3.Row
        return self._sqlite_conn

    def _ensure_schema(self, conn) -> None:
        response_id_column = "BIGSERIAL PRIMARY KEY" if self.config.backend == "supabase" else "INTEGER PRIMARY KEY AUTOINCREMENT"
        context_columns_sql = ",\n                ".join(
            f"{column_name} {column_type}" for column_name, column_type in PARTICIPANT_CONTEXT_COLUMN_TYPES.items()
        )
        attempts_sql = f"""
            CREATE TABLE IF NOT EXISTS {ATTEMPTS_TABLE} (
                attempt_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                anon_id TEXT NOT NULL,
                student_id TEXT,
                name TEXT,
                profession TEXT,
                years_experience INTEGER,
                group_name TEXT,
                {context_columns_sql},
                n_dilemmas_answered INTEGER,
                n_justifications INTEGER,
                responded_item_ids TEXT,
                raw_payload TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        responses_sql = f"""
            CREATE TABLE IF NOT EXISTS {RESPONSES_TABLE} (
                response_id {response_id_column},
                attempt_id TEXT NOT NULL,
                response_order INTEGER NOT NULL,
                row_type TEXT NOT NULL,
                item_id TEXT,
                sub_id TEXT,
                choice_key TEXT,
                choice_stage INTEGER,
                choice_level TEXT,
                choice_framework TEXT,
                likert_value INTEGER,
                text_value TEXT,
                FOREIGN KEY(attempt_id) REFERENCES {ATTEMPTS_TABLE}(attempt_id) ON DELETE CASCADE
            )
        """
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_attempts_anon_id ON {ATTEMPTS_TABLE}(anon_id)",
            f"CREATE INDEX IF NOT EXISTS idx_attempts_group_name ON {ATTEMPTS_TABLE}(group_name)",
            f"CREATE INDEX IF NOT EXISTS idx_attempts_profession ON {ATTEMPTS_TABLE}(profession)",
            f"CREATE INDEX IF NOT EXISTS idx_responses_attempt_id ON {RESPONSES_TABLE}(attempt_id)",
            f"CREATE INDEX IF NOT EXISTS idx_responses_item_id ON {RESPONSES_TABLE}(item_id)",
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_responses_attempt_order ON {RESPONSES_TABLE}(attempt_id, response_order)",
        ]
        statements = [attempts_sql, responses_sql, *indexes]
        cursor = conn.cursor()
        if self.config.backend == "sqlite":
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
        for statement in statements:
            cursor.execute(statement)
        self._ensure_attempt_profile_columns(conn)
        self._ensure_attempt_metadata_columns(conn)
        conn.commit()

    def _ensure_attempt_profile_columns(self, conn) -> None:
        existing_columns = self._existing_columns(conn, ATTEMPTS_TABLE)
        cursor = conn.cursor()
        for column_name, column_type in PARTICIPANT_CONTEXT_COLUMN_TYPES.items():
            if column_name in existing_columns:
                continue
            cursor.execute(f"ALTER TABLE {ATTEMPTS_TABLE} ADD COLUMN {column_name} {column_type}")

    def _ensure_attempt_metadata_columns(self, conn) -> None:
        existing_columns = self._existing_columns(conn, ATTEMPTS_TABLE)
        cursor = conn.cursor()
        for column_name, column_type in ATTEMPT_METADATA_COLUMNS.items():
            if column_name in existing_columns:
                continue
            cursor.execute(f"ALTER TABLE {ATTEMPTS_TABLE} ADD COLUMN {column_name} {column_type}")

    def _existing_columns(self, conn, table_name: str) -> set[str]:
        if self.config.backend == "supabase":
            query = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
            """
            rows = self._read_sql(conn, query, [table_name])
            return set(rows["column_name"].astype(str).tolist()) if not rows.empty else set()
        rows = self._read_sql(conn, f"PRAGMA table_info({table_name})")
        return set(rows["name"].astype(str).tolist()) if not rows.empty else set()

    def _upsert_attempt(self, conn, meta: Dict[str, Any]) -> None:
        placeholder = self._placeholder
        columns = [
            "attempt_id", "timestamp", "anon_id", "student_id", "name",
            "profession", "years_experience", "group_name",
            *PARTICIPANT_CONTEXT_COLUMNS,
            "n_dilemmas_answered",
            "n_justifications", "responded_item_ids", "raw_payload",
        ]
        values = [meta[column] for column in columns]
        if self.config.backend == "supabase":
            conflict_sql = ", ".join([f"{col}=EXCLUDED.{col}" for col in columns[1:]])
            sql = f"""
                INSERT INTO {ATTEMPTS_TABLE} ({', '.join(columns)})
                VALUES ({', '.join([placeholder] * len(columns))})
                ON CONFLICT (attempt_id) DO UPDATE SET {conflict_sql}
            """
        else:
            sql = f"""
                INSERT OR REPLACE INTO {ATTEMPTS_TABLE} ({', '.join(columns)})
                VALUES ({', '.join([placeholder] * len(columns))})
            """
        conn.execute(sql, values)
        conn.commit()

    def _attempt_response_select_sql(self) -> str:
        return ",\n                    ".join([*attempt_select_columns("a"), *response_select_columns("r")])

    def _attempt_summary_select_sql(self, include_raw_payload: bool = False) -> str:
        summary_columns = ["a.attempt_id", *attempt_summary_select_columns("a", include_raw_payload=include_raw_payload)]
        return ",\n                    ".join(summary_columns)

    def _append_attempt_filters(
        self,
        query: str,
        *,
        profession: str | None = None,
        group_name: str | None = None,
        anon_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[str, list[Any]]:
        placeholder = self._placeholder
        filters: list[str] = []
        params: list[Any] = []
        if profession:
            filters.append(f"a.profession = {placeholder}")
            params.append(profession)
        if group_name:
            filters.append(f"a.group_name = {placeholder}")
            params.append(group_name)
        if anon_id:
            filters.append(f"a.anon_id = {placeholder}")
            params.append(anon_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY a.timestamp DESC"
        if limit is not None and limit > 0:
            query += f" LIMIT {int(limit)}"
        return query, params

    def _comparison_frame(self, group_column: str, values: Sequence[str] | None = None, order_label: str | None = None) -> pd.DataFrame:
        self.ensure_storage()
        label = order_label or group_column
        aggregate_sql = aggregate_metric_columns()
        metrics = ",\n                    ".join([f"{expression} AS {alias}" for alias, expression in aggregate_sql.items()])
        with self._connect() as conn:
            query = f"""
                SELECT
                    {group_column} AS comparison_group,
                    COUNT(*) AS attempts,
                    COUNT(DISTINCT anon_id) AS participants,
                    {metrics},
                    MIN(timestamp) AS first_attempt_at,
                    MAX(timestamp) AS last_attempt_at
                FROM {ATTEMPTS_TABLE}
            """
            params: list[Any] = []
            if values:
                placeholders = ", ".join(self._placeholder for _ in values)
                query += f" WHERE {group_column} IN ({placeholders})"
                params.extend(list(values))
            query += f" GROUP BY {group_column} ORDER BY attempts DESC, {label} ASC"
            return self._read_sql(conn, query, params)

    def _replace_attempt_rows(self, conn, attempt_id: str, records: Iterable[Dict[str, Any]]) -> None:
        placeholder = self._placeholder
        conn.execute(f"DELETE FROM {RESPONSES_TABLE} WHERE attempt_id = {placeholder}", [attempt_id])
        insert_sql = f"""
            INSERT INTO {RESPONSES_TABLE} (
                attempt_id, response_order, row_type, item_id, sub_id, choice_key,
                choice_stage, choice_level, choice_framework, likert_value, text_value
            ) VALUES ({', '.join([placeholder] * 11)})
        """
        for record in records:
            conn.execute(insert_sql, [
                attempt_id,
                record["response_order"],
                record["row_type"],
                record["item_id"],
                record["sub_id"],
                record["choice_key"],
                record["choice_stage"],
                record["choice_level"],
                record["choice_framework"],
                record["likert_value"],
                record["text_value"],
            ])
        conn.commit()

    def _read_sql(self, conn, query: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        params = params or []
        if self.config.backend == "supabase":
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [desc.name for desc in cursor.description] if cursor.description else []
            return pd.DataFrame(rows, columns=columns)
        return pd.read_sql_query(query, conn, params=params)

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)
        for column in COLUMNS:
            if column not in df.columns:
                df[column] = np.nan
        return df[COLUMNS].copy()


def load_persistence_store(
    sqlite_path: str | Path = "data/responses.db",
    legacy_csv_path: str | Path | None = "data/responses.csv",
) -> PersistenceStore:
    sqlite_path = Path(sqlite_path)
    csv_path = Path(legacy_csv_path) if legacy_csv_path else None
    supabase_db_url = (
        os.getenv("SUPABASE_DB_URL")
        or os.getenv("SUPABASE_DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("DATABASE_URL")
    )
    backend = "supabase" if supabase_db_url else "sqlite"
    store = PersistenceStore(
        PersistenceConfig(
            backend=backend,
            sqlite_path=sqlite_path,
            legacy_csv_path=csv_path,
            supabase_db_url=supabase_db_url,
        )
    )
    store.ensure_storage()
    return store
