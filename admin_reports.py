from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _slugify(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "na"


def _safe_timestamp(value: str) -> str:
    return value.replace(":", "-").replace("T", "_")


@dataclass(frozen=True)
class AdminReportConfig:
    base_dir: Path
    drive_url: str | None


class AdminReportStore:
    def __init__(self, config: AdminReportConfig):
        self.config = config

    @property
    def base_dir(self) -> Path:
        return self.config.base_dir

    @property
    def drive_url(self) -> str | None:
        return self.config.drive_url

    def ensure_storage(self) -> None:
        (self.base_dir / "individual").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "collective").mkdir(parents=True, exist_ok=True)

    def save_individual_report(
        self,
        *,
        timestamp: str,
        anon_id: str,
        student_id: str,
        name: str,
        profession: str,
        group: str,
        years_experience: int,
        k_est: float,
        k_stage: int | None,
        k_level: str | None,
        coherence: float,
        fw_dom: str | None,
        fw_scores: Dict[str, float],
        stage_means: Dict[int, float],
        choice_df: pd.DataFrame,
        participant_context: Dict[str, Any] | None = None,
        ai_interpretation: Dict[str, Any] | None = None,
        pdf_bytes: bytes | None = None,
        docx_bytes: bytes | None = None,
    ) -> Dict[str, str]:
        self.ensure_storage()
        report_id = f"{_safe_timestamp(timestamp)}_{_slugify(anon_id)}"
        report_dir = self.base_dir / "individual"
        json_path = report_dir / f"{report_id}.json"
        csv_path = report_dir / f"{report_id}_choices.csv"
        interpretation_path = report_dir / f"{report_id}_interpretation.json"
        pdf_path = report_dir / f"{report_id}_report.pdf"
        docx_path = report_dir / f"{report_id}_report.docx"

        payload = {
            "timestamp": timestamp,
            "anon_id": anon_id,
            "student_id": student_id,
            "name": name,
            "profession": profession,
            "group": group,
            "years_experience": years_experience,
            "k_est": k_est,
            "k_stage": k_stage,
            "k_level": k_level,
            "coherence": coherence,
            "framework_dominant": fw_dom,
            "framework_scores": fw_scores,
            "stage_means": stage_means,
            "participant_context": participant_context or {},
        }

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        choice_df.to_csv(csv_path, index=False)
        output = {"json": str(json_path), "csv": str(csv_path)}

        if ai_interpretation is not None:
            interpretation_path.write_text(
                json.dumps(ai_interpretation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output["interpretation_json"] = str(interpretation_path)

        if pdf_bytes is not None:
            pdf_path.write_bytes(pdf_bytes)
            output["pdf"] = str(pdf_path)

        if docx_bytes is not None:
            docx_path.write_bytes(docx_bytes)
            output["docx"] = str(docx_path)

        return output

    def save_collective_snapshot(
        self,
        *,
        students_df: pd.DataFrame,
        last_attempt_df: pd.DataFrame,
        snapshot_name: str = "latest",
    ) -> Dict[str, str]:
        self.ensure_storage()
        collective_dir = self.base_dir / "collective"
        students_path = collective_dir / f"{snapshot_name}_students_summary.csv"
        attempts_path = collective_dir / f"{snapshot_name}_last_attempt_rows.csv"
        metadata_path = collective_dir / f"{snapshot_name}_metadata.json"

        students_df.to_csv(students_path, index=False)
        last_attempt_df.to_csv(attempts_path, index=False)
        metadata = {
            "snapshot_name": snapshot_name,
            "students_rows": int(len(students_df)),
            "last_attempt_rows": int(len(last_attempt_df)),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "students": str(students_path),
            "attempts": str(attempts_path),
            "metadata": str(metadata_path),
        }


def load_admin_report_store(base_dir: str | Path = "data/admin_reports") -> AdminReportStore:
    base_path_value = os.getenv("MORAL_TEST_ADMIN_REPORTS_DIR")
    drive_url = os.getenv("MORAL_TEST_ADMIN_DRIVE_URL")

    if not base_path_value or not drive_url:
        try:
            import streamlit as st

            base_path_value = base_path_value or st.secrets.get("MORAL_TEST_ADMIN_REPORTS_DIR")
            drive_url = drive_url or st.secrets.get("MORAL_TEST_ADMIN_DRIVE_URL")
        except Exception:
            pass

    base_path = Path(str(base_path_value) if base_path_value else str(base_dir))
    drive_url = str(drive_url) if drive_url else None
    store = AdminReportStore(AdminReportConfig(base_dir=base_path, drive_url=drive_url))
    store.ensure_storage()
    return store
