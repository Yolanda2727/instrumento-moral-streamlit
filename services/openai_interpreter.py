from __future__ import annotations

import json
from typing import Any, Dict

import streamlit as st
from openai import OpenAI

from prompts.interpreter_prompt import build_interpreter_system_prompt, build_interpreter_user_prompt


class OpenAIInterpreterError(RuntimeError):
    pass


def get_openai_api_key() -> str:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception as exc:
        raise OpenAIInterpreterError(
            "No se encontró OPENAI_API_KEY en st.secrets. Configura la clave antes de usar Interpretación IA."
        ) from exc
    if not api_key:
        raise OpenAIInterpreterError("OPENAI_API_KEY está vacío en st.secrets.")
    return str(api_key)


def get_openai_model() -> str:
    try:
        model = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")
    except Exception:
        model = "gpt-4.1-mini"
    return str(model)


def _parse_json_response(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenAIInterpreterError("La respuesta del modelo no pudo parsearse como JSON válido.") from exc


def interpret_payload(payload: Dict[str, Any], scope: str) -> Dict[str, Any]:
    client = OpenAI(api_key=get_openai_api_key())
    completion = client.chat.completions.create(
        model=get_openai_model(),
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": build_interpreter_system_prompt()},
            {"role": "user", "content": build_interpreter_user_prompt(scope, payload)},
        ],
    )
    content = completion.choices[0].message.content if completion.choices else ""
    result = _parse_json_response(content)
    result["_model"] = get_openai_model()
    result["_scope"] = scope
    return result