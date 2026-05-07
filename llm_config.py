import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

PRIVATE_CONFIG_FILE = Path("llm.private.json")


def _read_private_config() -> dict[str, Any]:
    if not PRIVATE_CONFIG_FILE.exists():
        return {}
    try:
        with PRIVATE_CONFIG_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw
    except Exception:
        return {}
    return {}


def _normalize_profile(name: str, profile: dict[str, Any]) -> dict[str, str]:
    api_key = str(profile.get("api_key", "")).strip()
    base_url = str(profile.get("base_url", "")).strip()
    model = str(profile.get("model", "")).strip() or "gpt-4o-mini"
    return {
        "provider": name,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def _settings_from_multi_profiles(private_cfg: dict[str, Any]) -> dict[str, str] | None:
    profiles = private_cfg.get("profiles")
    active_profile = str(private_cfg.get("active_profile", "")).strip()
    if not isinstance(profiles, dict) or not active_profile:
        return None
    selected = profiles.get(active_profile)
    if not isinstance(selected, dict):
        return None
    return _normalize_profile(active_profile, selected)


def _settings_from_profile_key(private_cfg: dict[str, Any], profile_key: str) -> dict[str, str] | None:
    profiles = private_cfg.get("profiles")
    if not isinstance(profiles, dict):
        return None
    selected = profiles.get(profile_key)
    if not isinstance(selected, dict):
        return None
    return _normalize_profile(profile_key, selected)


def _settings_from_legacy(private_cfg: dict[str, Any]) -> dict[str, str]:
    provider = str(private_cfg.get("provider", os.getenv("LLM_PROVIDER", "openai"))).strip() or "openai"
    api_key = str(private_cfg.get("api_key", "")).strip()
    base_url = str(private_cfg.get("base_url", "")).strip()
    model = str(private_cfg.get("model", "")).strip() or "gpt-4o-mini"
    if not api_key:
        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    if not base_url:
        base_url = str(os.getenv("OPENAI_BASE_URL", "")).strip()
    if not model:
        model = str(os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini"
    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def get_llm_settings() -> dict[str, str]:
    private_cfg = _read_private_config()
    settings = _settings_from_multi_profiles(private_cfg)
    if settings:
        return settings
    return _settings_from_legacy(private_cfg)


def get_asr_model() -> str:
    private_cfg = _read_private_config()
    asr_model = str(private_cfg.get("asr_model", "")).strip()
    if asr_model:
        return asr_model
    return str(os.getenv("ASR_MODEL", "gpt-4o-mini-transcribe")).strip() or "gpt-4o-mini-transcribe"


def get_speech_settings() -> dict[str, str]:
    private_cfg = _read_private_config()
    provider = str(private_cfg.get("speech_provider", os.getenv("SPEECH_PROVIDER", ""))).strip()
    api_key = str(private_cfg.get("speech_api_key", os.getenv("SPEECH_API_KEY", ""))).strip()
    base_url = str(private_cfg.get("speech_base_url", os.getenv("SPEECH_BASE_URL", ""))).strip()
    model = str(private_cfg.get("speech_model", os.getenv("SPEECH_MODEL", ""))).strip()
    diarization = str(private_cfg.get("speech_diarization", "true")).strip().lower()
    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "speech_diarization": "true" if diarization in {"1", "true", "yes", "y"} else "false",
    }


def get_asr_settings() -> dict[str, str]:
    private_cfg = _read_private_config()
    asr_profile = str(private_cfg.get("asr_profile", "")).strip()
    if asr_profile:
        settings = _settings_from_profile_key(private_cfg, asr_profile)
        if settings:
            return settings
    return get_llm_settings()


def build_llm_client() -> tuple[OpenAI | None, str, str]:
    settings = get_llm_settings()
    api_key = settings["api_key"]
    model = settings["model"]
    provider = settings["provider"]
    if not api_key:
        return None, model, provider

    kwargs: dict[str, str] = {"api_key": api_key}
    if settings["base_url"]:
        kwargs["base_url"] = settings["base_url"]
    return OpenAI(**kwargs), model, provider


def build_asr_client() -> tuple[OpenAI | None, str]:
    settings = get_asr_settings()
    api_key = settings["api_key"]
    if not api_key:
        return None, settings["provider"]
    kwargs: dict[str, str] = {"api_key": api_key}
    if settings["base_url"]:
        kwargs["base_url"] = settings["base_url"]
    return OpenAI(**kwargs), settings["provider"]


def build_speech_client() -> tuple[OpenAI | None, dict[str, str]]:
    settings = get_speech_settings()
    if not settings["api_key"]:
        return None, settings
    kwargs: dict[str, str] = {"api_key": settings["api_key"]}
    if settings["base_url"]:
        kwargs["base_url"] = settings["base_url"]
    return OpenAI(**kwargs), settings
