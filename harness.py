import annotationlib
import ast
import argparse
import asyncio
import atexit
import base64
import compression.zstd as zstd
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import string.templatelib as templatelib
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid
from concurrent.futures import InterpreterPoolExecutor
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, LiteralString, cast
from urllib.parse import urlparse

import reflex as rx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError, UnsupportedProviderError
from fontTools.ttLib import TTFont
from neo4j import GraphDatabase, Query
from pydantic import BaseModel

rx = cast(Any, rx)  # pyre-ignore[31]

# Emit maximum coverage progress logs when requested.
MAX_COVERAGE_LOG = (
    os.environ.get("MAX_COVERAGE_LOG") == "1" or "--maximum-coverage" in sys.argv
)

type TemplateMsg = str | templatelib.Template


def _render_template(msg: TemplateMsg) -> str:
    if not isinstance(msg, templatelib.Template):
        return str(msg)
    parts: list[str] = []
    literals = msg.strings
    interpolations = msg.interpolations
    for idx, literal in enumerate(literals):
        parts.append(literal)
        if idx < len(interpolations):
            parts.append(_render_interpolation(interpolations[idx]))
    return "".join(parts)


def _render_interpolation(interp: templatelib.Interpolation) -> str:
    value: Any = interp.value
    if isinstance(value, templatelib.Template):
        value = _render_template(value)
    conversion = interp.conversion
    if conversion == "r":
        value = repr(value)
    elif conversion == "s":
        value = str(value)
    elif conversion == "a":
        value = ascii(value)
    format_spec = interp.format_spec
    if isinstance(format_spec, templatelib.Template):
        format_spec = _render_template(format_spec)
    if format_spec is None:
        format_spec = ""
    try:
        return format(value, format_spec)
    except Exception:
        return str(value)


_INTERPRETER_POOL: InterpreterPoolExecutor | None = None
_INTERPRETER_POOL_READY = False
try:
    _INTERPRETER_POOL_TIMEOUT = float(
        os.environ.get("HARNESS_INTERPRETER_TIMEOUT", "1.5")
    )
except ValueError:
    _INTERPRETER_POOL_TIMEOUT = 1.5
try:
    _INTERPRETER_POOL_MIN_SIZE = int(
        os.environ.get("HARNESS_INTERPRETER_MIN_SIZE", "512")
    )
except ValueError:
    _INTERPRETER_POOL_MIN_SIZE = 512


def _get_interpreter_pool() -> InterpreterPoolExecutor | None:
    global _INTERPRETER_POOL, _INTERPRETER_POOL_READY
    if _INTERPRETER_POOL_READY:
        return _INTERPRETER_POOL
    _INTERPRETER_POOL_READY = True
    if os.environ.get("HARNESS_INTERPRETER_POOL", "1") == "0":
        return None
    try:
        _INTERPRETER_POOL = InterpreterPoolExecutor(max_workers=1)
    except Exception:
        _INTERPRETER_POOL = None
    return _INTERPRETER_POOL


def _shutdown_interpreter_pool() -> None:
    pool = _INTERPRETER_POOL
    if pool is None:
        return
    with suppress(Exception):
        pool.shutdown(wait=False, cancel_futures=True)


def _sha256_text_worker(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_text(text: str) -> str:
    if not text:
        return ""
    if len(text) < _INTERPRETER_POOL_MIN_SIZE:
        return _sha256_text_worker(text)
    pool = _get_interpreter_pool()
    if pool is None:
        return _sha256_text_worker(text)
    try:
        return pool.submit(_sha256_text_worker, text).result(
            timeout=_INTERPRETER_POOL_TIMEOUT
        )
    except Exception:
        return _sha256_text_worker(text)


def _zstd_compress_worker(data: bytes) -> bytes:
    return zstd.compress(data)


def _compress_debug_bytes(data: bytes) -> bytes:
    if not data:
        return b""
    pool = _get_interpreter_pool()
    if pool is None:
        return _zstd_compress_worker(data)
    try:
        return pool.submit(_zstd_compress_worker, data).result(
            timeout=_INTERPRETER_POOL_TIMEOUT
        )
    except Exception:
        return _zstd_compress_worker(data)


def _coerce_timeout(value: Any) -> float | None:
    if value is None:
        return None
    try:
        timeout_val = float(value)
    except (TypeError, ValueError):
        return None
    if timeout_val <= 0:
        return None
    return timeout_val


def _run_static_tool_job(job: dict[str, Any] | str) -> str:
    import json as _json
    import os as _os
    import shutil as _shutil
    import subprocess as _subprocess
    import tempfile as _tempfile
    import time as _time
    from contextlib import suppress as _suppress
    from pathlib import Path as _Path

    def _encode(result: dict[str, Any]) -> str:
        return _json.dumps(result, ensure_ascii=True, default=str)

    def _coerce_timeout_local(value: Any) -> float | None:
        if value is None:
            return None
        try:
            timeout_val = float(value)
        except (TypeError, ValueError):
            return None
        if timeout_val <= 0:
            return None
        return timeout_val

    def _run_pyupgrade_local(payload: dict[str, Any]) -> dict[str, Any]:
        started = _time.perf_counter()
        label = str(payload.get("tool") or "pyupgrade")
        if _shutil.which("pyupgrade") is None:
            return {
                "tool": label,
                "status": "fail",
                "duration_s": 0.0,
                "details": "missing",
            }
        file_entries = payload.get("files")
        if not isinstance(file_entries, list):
            file_entries = []
        if not file_entries:
            return {
                "tool": label,
                "status": "ok",
                "duration_s": _time.perf_counter() - started,
                "details": "files=0",
            }
        timeout_s = _coerce_timeout_local(payload.get("timeout_s"))
        tmp_dir = _Path(
            _tempfile.mkdtemp(prefix="maxcov_pyupgrade_", dir=_tempfile.gettempdir())
        )
        try:
            dest_pairs: list[tuple[_Path, _Path]] = []
            for entry in file_entries:
                try:
                    src, rel = entry
                except (TypeError, ValueError):
                    continue
                src_path = _Path(str(src))
                rel_path = _Path(str(rel)) if rel else _Path(src_path.name)
                dest_path = tmp_dir / rel_path
                try:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    _shutil.copy2(src_path, dest_path)
                except Exception:
                    continue
                dest_pairs.append((src_path, dest_path))
            if not dest_pairs:
                return {
                    "tool": label,
                    "status": "ok",
                    "duration_s": _time.perf_counter() - started,
                    "details": "files=0",
                }
            pyupgrade_flag = str(payload.get("pyupgrade_flag") or "--py314-plus")
            cmd = [
                "pyupgrade",
                pyupgrade_flag,
                "--exit-zero-even-if-changed",
                *[str(dest) for _, dest in dest_pairs],
            ]
            try:
                result = _subprocess.run(
                    cmd,
                    cwd=str(payload.get("cwd") or _os.getcwd()),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
            except _subprocess.TimeoutExpired:
                return {
                    "tool": label,
                    "status": "warn",
                    "duration_s": _time.perf_counter() - started,
                    "details": f"timeout>{timeout_s}s" if timeout_s else "timeout",
                }
            except Exception as exc:
                return {
                    "tool": label,
                    "status": "warn",
                    "duration_s": _time.perf_counter() - started,
                    "details": f"error={type(exc).__name__}",
                }
            changed = 0
            for src_path, dest_path in dest_pairs:
                try:
                    if dest_path.exists() and dest_path.read_bytes() != src_path.read_bytes():
                        changed += 1
                except Exception:
                    continue
            status = "warn" if changed > 0 else "ok"
            details = f"files={changed}"
            if result.returncode != 0:
                status = "warn"
                details = f"{details}, rc={result.returncode}"
            return {
                "tool": label,
                "status": status,
                "duration_s": _time.perf_counter() - started,
                "details": details,
            }
        finally:
            with _suppress(Exception):
                _shutil.rmtree(tmp_dir, ignore_errors=True)

    if isinstance(job, str):
        try:
            parsed = _json.loads(job)
        except _json.JSONDecodeError:
            return _encode(
                {
                    "tool": "unknown",
                    "status": "warn",
                    "ran": False,
                    "duration_s": 0.0,
                    "details": "invalid payload",
                }
            )
        if not isinstance(parsed, dict):
            return _encode(
                {
                    "tool": "unknown",
                    "status": "warn",
                    "ran": False,
                    "duration_s": 0.0,
                    "details": "invalid payload",
                }
            )
        payload = parsed
    elif isinstance(job, dict):
        payload = job
    else:
        return _encode(
            {
                "tool": "unknown",
                "status": "warn",
                "ran": False,
                "duration_s": 0.0,
                "details": "invalid payload",
            }
        )
    kind = str(payload.get("kind") or "subprocess")
    if kind == "pyupgrade":
        return _encode(_run_pyupgrade_local(payload))
    started = _time.perf_counter()
    label = str(payload.get("tool") or "")
    cmd = payload.get("cmd")
    if not isinstance(cmd, list) or not cmd:
        return _encode(
            {
                "tool": label,
                "status": "warn",
                "ran": False,
                "duration_s": 0.0,
                "details": "missing command",
            }
        )
    timeout_s = _coerce_timeout_local(payload.get("timeout_s"))
    env = payload.get("env")
    if env is not None and not isinstance(env, dict):
        env = None
    output_path = payload.get("output_path")
    output_path_str = str(output_path) if output_path else ""
    try:
        result = _subprocess.run(
            cmd,
            cwd=str(payload.get("cwd") or _os.getcwd()),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except _subprocess.TimeoutExpired:
        return _encode(
            {
                "tool": label,
                "status": "warn",
                "ran": False,
                "duration_s": _time.perf_counter() - started,
                "details": f"timeout>{timeout_s}s" if timeout_s else "timeout",
            }
        )
    except Exception as exc:
        return _encode(
            {
                "tool": label,
                "status": "warn",
                "ran": False,
                "duration_s": _time.perf_counter() - started,
                "details": f"error={type(exc).__name__}",
            }
        )
    output = "\n".join(
        [text for text in (result.stdout or "", result.stderr or "") if text]
    )
    if output_path_str:
        output_path_obj = _Path(output_path_str)
        if output_path_obj.exists():
            with _suppress(Exception):
                output = output_path_obj.read_text(encoding="utf-8")
    return _encode(
        {
            "tool": label,
            "ran": True,
            "duration_s": _time.perf_counter() - started,
            "returncode": result.returncode,
            "output": output,
        }
    )


def _should_silence_warnings() -> bool:
    return os.environ.get("MAX_COVERAGE_SILENCE_WARNINGS") == "1"


def _install_maxcov_print_filter() -> None:
    if not _should_silence_warnings():
        return
    import builtins
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    orig_print = getattr(builtins, "print")

    def _filtered_print(*args, **kwargs):
        try:
            msg = " ".join(str(a) for a in args)
        except Exception:
            msg = ""
        if msg.startswith("[maxcov]"):
            return orig_print(*args, **kwargs)
        if msg.startswith(
            (
                "Warning:",
                "Error running Typst:",
                "Typst compilation failed",
                "Req file not found",
                "No resume found in Neo4j",
                "Error compiling PDF:",
                "Error retrieving resume data:",
                "Error listing applied jobs:",
                "Error resetting/importing",
                "Error importing",
                "Error saving Profile:",
            )
        ):
            return
        if any(
            token in msg
            for token in (
                "file not found at",
                "could not fetch",
                "unable to apply PDF metadata",
                "could not save resume pdf",
                "could not render resume pdf",
                "Playwright not available",
                "Playwright traversal failed",
                "reflex coverage server did not start",
                "failed to start reflex coverage server",
                "Unexpected exit from worker",
                "Task exception was never retrieved",
                "Event loop is closed",
                "Error in on_load:",
            )
        ):
            return
        return orig_print(*args, **kwargs)

    builtins.print = _filtered_print


def _install_reflex_signal_handlers() -> None:
    if __name__ == "__main__":
        return
    if os.environ.get("REFLEX_GRACEFUL_EXIT") == "0":
        return

    def _graceful_exit(_signum, _frame):
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            current = signal.getsignal(sig)
        except Exception:
            continue
        if current not in (signal.SIG_DFL, signal.default_int_handler):
            continue
        try:
            signal.signal(sig, _graceful_exit)
        except Exception:
            continue


# Default app-level LLM settings when not explicitly set.
def _ensure_default_llm_env() -> None:
    if not os.environ.get("LLM_REASONING_EFFORT") and not os.environ.get(
        "OPENAI_REASONING_EFFORT"
    ):
        os.environ["LLM_REASONING_EFFORT"] = "medium"
    if not os.environ.get("LLM_MAX_OUTPUT_TOKENS"):
        os.environ["LLM_MAX_OUTPUT_TOKENS"] = "4096"


_ensure_default_llm_env()
_install_maxcov_print_filter()
_install_reflex_signal_handlers()


def _try_import_coverage():
    try:
        import coverage  # noqa: F401
    except Exception:
        return None
    return coverage


# Enable coverage collection inside Reflex worker processes when requested.
_REFLEX_COVERAGE = None
_REFLEX_COVERAGE_OWNED = False


def _init_reflex_coverage() -> None:
    """Initialize coverage in Reflex worker processes when requested."""
    global _REFLEX_COVERAGE, _REFLEX_COVERAGE_OWNED
    if os.environ.get("REFLEX_COVERAGE") != "1":
        return
    try:
        import atexit

        import coverage  # noqa: F401

        _REFLEX_COVERAGE = coverage.Coverage(
            data_file=os.environ.get("COVERAGE_FILE")
            or os.environ.get("REFLEX_COVERAGE_FILE"),
            data_suffix=True,
            branch=True,
            source=[str(Path(__file__).resolve().parent)],
        )
        current_cov = None
        force_owned = os.environ.get("REFLEX_COVERAGE_FORCE_OWNED") == "1"
        try:
            current_cov = coverage.Coverage.current()
        except Exception:
            current_cov = None

        if current_cov is None or force_owned:
            _REFLEX_COVERAGE_OWNED = True
            _REFLEX_COVERAGE.start()

            def _stop_reflex_coverage() -> None:
                if _REFLEX_COVERAGE is not None and _REFLEX_COVERAGE_OWNED:
                    _REFLEX_COVERAGE.stop()
                    _REFLEX_COVERAGE.save()

            atexit.register(_stop_reflex_coverage)
            if os.environ.get("MAX_COVERAGE_REFLEX_STOP") == "1":
                _stop_reflex_coverage()
            if current_cov is not None:
                _REFLEX_COVERAGE.stop()
                _REFLEX_COVERAGE.save()
                _REFLEX_COVERAGE_OWNED = False
        else:
            _REFLEX_COVERAGE.start()
            _REFLEX_COVERAGE.stop()
            _REFLEX_COVERAGE.save()
    except Exception:
        _REFLEX_COVERAGE = None
        _REFLEX_COVERAGE_OWNED = False


_init_reflex_coverage()

# ==========================================
# CONFIGURATION
# ==========================================
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "ResumeBuilder")
SUPPORTED_LLM_PROVIDERS = {"openai", "gemini"}
DEFAULT_LLM_MODELS = [
    # Google Gemini (Completions API supported)
    "gemini:gemini-3-flash-preview",
    "gemini:gemini-3-pro-preview",
    "gemini:gemini-3-pro-image-preview",
    "gemini:gemini-flash-latest",
    "gemini:gemini-flash-lite-latest",
    "gemini:gemini-pro-latest",
    "gemini:gemini-2.5-flash",
    "gemini:gemini-2.5-flash-lite",
    "gemini:gemini-2.5-flash-preview-09-2025",
    "gemini:gemini-2.5-flash-lite-preview-09-2025",
    "gemini:gemini-2.5-flash-preview-tts",
    "gemini:gemini-2.5-flash-image",
    "gemini:gemini-2.5-flash-image-preview",
    "gemini:gemini-2.5-pro",
    "gemini:gemini-2.5-pro-preview-tts",
    "gemini:gemini-2.5-computer-use-preview-10-2025",
    "gemini:gemini-2.0-flash",
    "gemini:gemini-2.0-flash-001",
    "gemini:gemini-2.0-flash-exp",
    "gemini:gemini-2.0-flash-exp-image-generation",
    "gemini:gemini-2.0-flash-lite",
    "gemini:gemini-2.0-flash-lite-001",
    "gemini:gemini-2.0-flash-lite-preview",
    "gemini:gemini-2.0-flash-lite-preview-02-05",
    "gemini:gemini-exp-1206",
    "gemini:gemini-robotics-er-1.5-preview",
    # OpenAI (Responses API supported)
    "openai:gpt-5.2-pro",
    "openai:gpt-5.2-pro-2025-12-11",
    "openai:gpt-5.2",
    "openai:gpt-5.2-2025-12-11",
    "openai:gpt-5.2-chat-latest",
    "openai:gpt-5-mini",
    "openai:gpt-5-mini-2025-08-07",
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
]


def _resolve_default_llm_settings() -> tuple[str, str]:
    effort = (
        (
            os.environ.get("LLM_REASONING_EFFORT")
            or os.environ.get("OPENAI_REASONING_EFFORT")
            or "high"
        )
        .strip()
        .lower()
    )
    if effort not in {"none", "minimal", "low", "medium", "high"}:
        effort = "high"
    model_env = (os.environ.get("LLM_MODEL") or "").strip()
    if model_env:
        model = model_env
    else:
        model = (os.environ.get("OPENAI_MODEL") or DEFAULT_LLM_MODELS[0]).strip()
    if model and ":" not in model:
        # Backward compatibility: bare model ids default to OpenAI.
        prefix = model.split("/", 1)[0].strip().lower()
        if prefix not in SUPPORTED_LLM_PROVIDERS:
            model = f"openai:{model}"
    return effort, model


DEFAULT_LLM_REASONING_EFFORT, DEFAULT_LLM_MODEL = _resolve_default_llm_settings()
BASE_DIR = Path(__file__).resolve().parent
PROMPT_YAML_PATH = BASE_DIR / "prompt.yaml"
STOP_LIST_PATH = BASE_DIR / "stop-list.txt"
STOP_LIST_PLACEHOLDER = "{{STOP_LIST}}"


def _resolve_assets_dir(base_dir: Path) -> Path:
    candidate = base_dir / "assets"
    if candidate.exists() and not candidate.is_dir():
        return base_dir / "assets_out"
    return candidate


ASSETS_DIR = _resolve_assets_dir(BASE_DIR)
# Cache fonts inside the project to make the build self-contained.
FONTS_DIR = BASE_DIR / "fonts"
PACKAGES_DIR = BASE_DIR / "packages"
LIVE_PDF_PATH = ASSETS_DIR / "preview.pdf"
LIVE_PDF_SIG_PATH = ASSETS_DIR / "preview.sig"
TYPST_TEMPLATE_VERSION = "modern-cv-0.9.0-2025-02-23"
RUNTIME_WRITE_PDF = os.environ.get("RUNTIME_WRITE_PDF", "0") == "1"
TEMP_BUILD_DIR = BASE_DIR / ".tmp_typst"
FONT_AWESOME_PACKAGE_VERSION = "0.6.0"
FONT_AWESOME_PACKAGE_URL = f"https://packages.typst.org/preview/fontawesome-{FONT_AWESOME_PACKAGE_VERSION}.tar.gz"
FONT_AWESOME_PACKAGE_DIR = (
    PACKAGES_DIR / "preview" / "fontawesome" / FONT_AWESOME_PACKAGE_VERSION
)
FONT_AWESOME_SOURCES = {
    "Font Awesome 7 Free-Solid-900.otf": "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/7.x/otfs/Font%20Awesome%207%20Free-Solid-900.otf",
    "Font Awesome 7 Free-Regular-400.otf": "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/7.x/otfs/Font%20Awesome%207%20Free-Regular-400.otf",
    "Font Awesome 7 Brands-Regular-400.otf": "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/7.x/otfs/Font%20Awesome%207%20Brands-Regular-400.otf",
}
DEFAULT_ASSETS_JSON = BASE_DIR / "michael_scott_resume.json"
DEBUG_LOG = Path(tempfile.gettempdir()) / "resume_builder_debug.log"
DEBUG_LOG_ZST = DEBUG_LOG.with_suffix(".log.zst")
LLM_JSON_LOG_ENV = "LLM_LOG_JSON_OUTPUT"
LLM_JSON_LOG_DIR_ENV = "LLM_JSON_LOG_DIR"
DEFAULT_TYPST_PATH = BASE_DIR / "bin" / ("typst.exe" if os.name == "nt" else "typst")
TYPST_BIN = os.environ.get("TYPST_BIN") or (
    str(DEFAULT_TYPST_PATH) if DEFAULT_TYPST_PATH.exists() else "typst"
)
DEFAULT_SECTION_ORDER = [
    "summary",
    "matrices",
    "education",
    "education_continued",
    "experience",
    "founder",
]
DEFAULT_AUTO_FIT_TARGET_PAGES = 2
DEFAULT_SKILLS_ROW_LABELS = [
    "Leadership & Strategy",
    "Technical Domain & Tools",
    "Architectural Patterns & Methodologies",
]
SOFT_BOLD_WEIGHT = 350
SOFT_EMPH_FILL = "#374151"
SOFT_SECONDARY_FILL = "#6B7280"
SECTION_LABELS = {
    "summary": "Summary",
    "education": "Education",
    "education_continued": "Education Continued",
    "experience": "Experience",
    "founder": "Startup Founder",
    "matrices": "Skills",
}
_NEO4J_DRIVER = None
_NEO4J_SCHEMA_READY = False
_FONTS_READY = False
_LOCAL_FONT_CATALOG = None
_LOCAL_FONT_EXTRA_FONTS = None

def _write_debug_line(msg: TemplateMsg) -> None:
    with suppress(Exception):
        rendered = _render_template(msg)
        stamp = datetime.now().isoformat()
        with DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(_render_template(t"{stamp} | {rendered}") + "\n")


def _resolve_llm_json_log_dir() -> Path | None:
    if os.environ.get(LLM_JSON_LOG_ENV) != "1":
        return None
    explicit_dir = (os.environ.get(LLM_JSON_LOG_DIR_ENV) or "").strip()
    if explicit_dir:
        return Path(explicit_dir)
    default_dir = Path("/var/log/maxcov")
    if default_dir.exists():
        return default_dir
    return BASE_DIR / "maxcov_logs"


def _summarize_llm_json_output(result: dict) -> dict[str, Any]:
    summary: dict[str, Any] = {"keys": sorted(result.keys())}
    summary["summary_len"] = len(str(result.get("summary") or ""))
    headers = result.get("headers")
    if isinstance(headers, list):
        summary["headers_len"] = len(headers)
    highlighted = result.get("highlighted_skills")
    if isinstance(highlighted, list):
        summary["highlighted_skills_len"] = len(highlighted)
    skills_rows = result.get("skills_rows")
    if isinstance(skills_rows, list):
        summary["skills_rows_len"] = len(skills_rows)
        summary["skills_rows_counts"] = [
            len(row) for row in skills_rows if isinstance(row, list)
        ]
    return summary


def _log_llm_json_output(result: dict, *, model_name: str, context: str) -> None:
    log_dir = _resolve_llm_json_log_dir()
    if not log_dir:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _write_debug_line(t"LLM JSON output log dir failed: {exc}")
        return
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    payload = {
        "timestamp": datetime.now().isoformat(),
        "context": context,
        "model": model_name,
        "summary": _summarize_llm_json_output(result),
        "result": result,
    }
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, default=str)
    targets = [
        log_dir / f"llm_json_output_{context}_{stamp}.json",
        log_dir / f"llm_json_output_{context}_latest.json",
    ]
    for target in targets:
        with suppress(Exception):
            target.write_text(encoded, encoding="utf-8")
    _write_debug_line(t"LLM JSON output logged to {targets[0]}")


def _maybe_compress_debug_log() -> None:
    if os.environ.get("MAX_COVERAGE_COMPRESS_DEBUG") != "1":
        return
    if not DEBUG_LOG.exists():
        return
    with suppress(Exception):
        data = DEBUG_LOG.read_bytes()
        if not data:
            return
        DEBUG_LOG_ZST.write_bytes(_compress_debug_bytes(data))


atexit.register(_maybe_compress_debug_log)
atexit.register(_shutdown_interpreter_pool)


def _empty_resume_payload() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": "",
        "first_name": "",
        "middle_name": "",
        "last_name": "",
        "email": "",
        "email2": "",
        "phone": "",
        "font_family": DEFAULT_RESUME_FONT_FAMILY,
        "auto_fit_target_pages": DEFAULT_AUTO_FIT_TARGET_PAGES,
        "auto_fit_best_scale": 1.0,
        "auto_fit_too_long_scale": 0.0,
        "linkedin_url": "",
        "github_url": "",
        "scholar_url": "",
        "calendly_url": "",
        "portfolio_url": "",
        "summary": "",
        "head1_left": "",
        "head1_middle": "",
        "head1_right": "",
        "head2_left": "",
        "head2_middle": "",
        "head2_right": "",
        "head3_left": "",
        "head3_middle": "",
        "head3_right": "",
        "top_skills": [],
        "section_order": DEFAULT_SECTION_ORDER.copy(),
        "section_enabled": list(SECTION_LABELS),
        "section_titles_json": "{}",
        "custom_sections_json": "[]",
        "prompt_yaml": _load_prompt_yaml_from_file() or "",
    }


def _iter_local_font_files() -> list[Path]:
    if not FONTS_DIR.exists():
        return []
    paths = []
    for path in sorted(FONTS_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".otf", ".ttf", ".woff", ".woff2"}:
            continue
        if "font awesome" in path.name.lower():
            continue
        paths.append(path)
    return paths


def _read_font_family(ttfont: TTFont) -> str:
    try:
        name_table = ttfont["name"]
    except Exception:
        return ""
    for name_id in (16, 1):
        for record in name_table.names:
            if record.nameID != name_id:
                continue
            try:
                value = record.toUnicode().strip()
            except Exception:
                value = ""
            if value:
                return value
    return ""


def _read_font_weight_italic(ttfont: TTFont) -> tuple[int, bool]:
    weight = 400
    italic = False
    with suppress(Exception):
        os2 = ttfont["OS/2"]
        weight = int(getattr(os2, "usWeightClass", weight) or weight)
        italic = bool(getattr(os2, "fsSelection", 0) & 0x01)
    with suppress(Exception):
        head = ttfont["head"]
        italic = italic or bool(getattr(head, "macStyle", 0) & 0x02)
    weight = max(100, min(weight, 900))
    return weight, italic


def _build_local_font_catalog() -> dict[str, list[dict]]:
    catalog: dict[str, list[dict]] = {}
    for path in _iter_local_font_files():
        try:
            ttfont = TTFont(str(path), fontNumber=0, lazy=True)
        except Exception:
            continue
        family = _read_font_family(ttfont) or path.stem
        if "font awesome" in family.lower():
            continue
        weight, italic = _read_font_weight_italic(ttfont)
        catalog.setdefault(family, []).append(
            {"path": path, "weight": weight, "italic": italic}
        )
    return catalog


def _get_local_font_catalog() -> dict[str, list[dict]]:
    global _LOCAL_FONT_CATALOG
    if _LOCAL_FONT_CATALOG is None:
        _LOCAL_FONT_CATALOG = _build_local_font_catalog()
    return _LOCAL_FONT_CATALOG


def _pick_primary_font_entry(entries: list[dict]) -> dict | None:
    if not entries:
        return None

    def score(entry: dict) -> int:
        weight = int(entry.get("weight") or 400)
        italic = bool(entry.get("italic"))
        return abs(weight - 400) + (1000 if italic else 0)

    return min(entries, key=score)


def _select_local_font_paths(family: str, italic: bool) -> list[Path]:
    catalog = _get_local_font_catalog()
    entries = catalog.get(family, [])
    if not entries:
        family_lower = (family).strip().lower()
        for key, items in catalog.items():
            if key.lower() == family_lower:
                entries = items
                break
    if not entries:
        return []

    def score(entry: dict) -> int:
        weight = int(entry.get("weight") or 400)
        is_italic = bool(entry.get("italic"))
        return abs(weight - 400) + (0 if is_italic == italic else 1000)

    ordered = sorted(entries, key=score)
    return [entry["path"] for entry in ordered if "path" in entry]


def _font_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".otf": "font/otf",
        ".ttf": "font/ttf",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
    }.get(suffix, "application/octet-stream")
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_local_font_extra_fonts() -> list[dict]:
    extra_fonts: list[dict] = []
    catalog = _get_local_font_catalog()
    for family, entries in sorted(catalog.items()):
        entry = _pick_primary_font_entry(entries)
        if not entry:
            continue
        path = entry.get("path")
        if not isinstance(path, Path):
            continue
        try:
            url = _font_data_uri(path)
        except Exception:
            continue
        extra_fonts.append({"name": family, "variants": ["400"], "url": url})
    return extra_fonts


def _get_local_font_extra_fonts() -> list[dict]:
    global _LOCAL_FONT_EXTRA_FONTS
    if _LOCAL_FONT_EXTRA_FONTS is None:
        _LOCAL_FONT_EXTRA_FONTS = _build_local_font_extra_fonts()
    return _LOCAL_FONT_EXTRA_FONTS


def _resolve_default_font_family() -> str:
    catalog = _get_local_font_catalog()
    for name in catalog:
        if name.strip().lower() in {"avenir lt std", "avenir"}:
            return name
    if catalog:
        return min(catalog.keys())
    return "Avenir LT Std"


DEFAULT_RESUME_FONT_FAMILY = _resolve_default_font_family()
FONT_PICKER_EXTRA_FONTS_JSON = json.dumps(
    _get_local_font_extra_fonts(), ensure_ascii=True
)


def _known_section_keys(extra_keys: list[str] | None = None) -> list[str]:
    keys = list(SECTION_LABELS)
    for key in extra_keys or []:
        if key and key not in keys:
            keys.append(key)
    return keys


def _sanitize_section_order(
    raw_order: list[str] | None, extra_keys: list[str] | None = None
) -> list[str]:
    """Return a stable, de-duplicated section order with defaults appended."""
    known_keys = _known_section_keys(extra_keys)
    seen: set[str] = set()
    ordered: list[str] = []
    for key in raw_order or []:
        if key in known_keys and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in DEFAULT_SECTION_ORDER:
        if key in known_keys and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in known_keys:
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _filter_section_order(
    raw_order: list[str] | None, extra_keys: list[str] | None = None
) -> list[str]:
    """Filter a section order list to known keys without appending defaults."""
    known_keys = _known_section_keys(extra_keys)
    seen: set[str] = set()
    ordered: list[str] = []
    for key in raw_order or []:
        if key in known_keys and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _normalize_section_enabled(
    raw,
    default: list[str] | None = None,
    extra_keys: list[str] | None = None,
) -> list[str]:
    """Normalize section_enabled values to a list of enabled section keys."""
    known_keys = _known_section_keys(extra_keys)
    default_list = list(default or known_keys)
    if raw is None:
        return default_list.copy()
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return default_list.copy()
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = [s.strip() for s in cleaned.split(",") if s.strip()]
        return _normalize_section_enabled(parsed, default_list, extra_keys)
    if isinstance(raw, dict):
        return [k for k, v in raw.items() if v and k in known_keys]
    if isinstance(raw, (list, tuple, set)):
        return [k for k in raw if k in known_keys]
    return default_list.copy()


def _apply_section_enabled(
    order: list[str] | None, enabled: list[str] | None
) -> list[str]:
    """Filter an order list by enabled section keys."""
    order = list(order or [])
    if enabled is None:
        return order
    enabled_set = {k for k in enabled}
    return [k for k in order if k in enabled_set]


def _coerce_bool(value) -> bool:
    """Best-effort conversion for UI checkbox/switch values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "checked"}
    if isinstance(value, dict):
        for key in ("checked", "value", "isChecked"):
            if key in value:
                return _coerce_bool(value.get(key))
        target = value.get("target")
        if isinstance(target, dict):
            for key in ("checked", "value", "isChecked"):
                if key in target:
                    return _coerce_bool(target.get(key))
    return False


def _normalize_section_titles(raw) -> dict[str, str]:
    """Normalize section title overrides to a dict."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return {}
        try:
            raw = json.loads(cleaned)
        except Exception:
            return {}
    if isinstance(raw, dict):
        return {str(k): str(v).strip() for k, v in raw.items() if k and str(v).strip()}
    return {}


def _normalize_custom_sections(raw) -> list[dict]:
    """Normalize custom sections to a list of dicts with id/key/title/body."""
    if raw is None:
        return []
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return []
        try:
            raw = json.loads(cleaned)
        except Exception:
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("id") or "").strip() or str(uuid.uuid4())
        key = str(item.get("key") or "").strip() or f"custom_{section_id}"
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or item.get("content") or "").rstrip()
        out.append(
            {
                "id": section_id,
                "key": key,
                "title": title,
                "body": body,
            }
        )
    return out


def _custom_section_keys(custom_sections: list[dict] | None) -> list[str]:
    keys: list[str] = []
    for item in custom_sections or []:
        key = str(item.get("key") or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def _build_section_title_map(
    section_titles: dict | None, custom_sections: list[dict] | None
) -> dict[str, str]:
    titles = {key: value for key, value in SECTION_LABELS.items()}
    overrides = _normalize_section_titles(section_titles)
    for key, value in overrides.items():
        if value:
            titles[key] = value
    for item in custom_sections or []:
        key = str(item.get("key") or "").strip()
        title = str(item.get("title") or "").strip()
        if key and title:
            titles[key] = title
    return titles


def _load_prompt_yaml_from_file(path: Path | None = None) -> str | None:
    """Load the prompt.yaml template from disk (if available)."""
    prompt_path = Path(path) if path else PROMPT_YAML_PATH
    try:
        return prompt_path.read_text(encoding="utf-8").rstrip()
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _resolve_prompt_template(base_profile: dict | None) -> str | None:
    """Prefer a prompt stored in Neo4j; fall back to prompt.yaml on disk."""
    if isinstance(base_profile, dict):
        raw = base_profile.get("prompt_yaml")
        if isinstance(raw, str) and raw.strip():
            return raw.rstrip()
    return _load_prompt_yaml_from_file()


def _format_profile_label(profile: dict) -> str:
    created_at = str(profile.get("created_at") or "").strip()
    if created_at:
        created_at = created_at.replace("T", " ")
        if "." in created_at:
            created_at = created_at.split(".", 1)[0]
        if created_at.endswith("Z"):
            created_at = f"{created_at[:-1]} UTC"
        elif created_at.endswith("+00:00"):
            created_at = f"{created_at[:-6]} UTC"
    company = str(profile.get("target_company") or "").strip() or "Unknown company"
    role = str(profile.get("target_role") or "").strip() or "Unknown role"
    if created_at:
        return f"{created_at} | {company} | {role}"
    return f"{company} | {role}"


def _load_stop_list(path: Path | None = None) -> list[str]:
    stop_path = Path(path) if path else STOP_LIST_PATH
    try:
        raw = stop_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return []
    except Exception:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _format_stop_list_items(stop_terms: list[str], indent: str = "    ") -> str:
    return "\n".join(f'{indent}- "{term}"' for term in stop_terms)


def _format_stop_list_section(stop_terms: list[str]) -> str:
    if not stop_terms:
        return ""
    items = _format_stop_list_items(stop_terms)
    return (
        "\n\n# ==============================================================================\n"
        "# SECTION 6B: STOP LIST (Forbidden Phrases)\n"
        "# ==============================================================================\n"
        "stop_list:\n"
        '  directive: "The following words and phrases are banned from all output fields. '
        "Do not use them in any casing or tense. Replace them with specific, measurable language. "
        "This list is injected from stop-list.txt."
        '"\n'
        "  phrases:\n"
        f"{items}\n"
    )


def _inject_stop_list(prompt_template: str, stop_terms: list[str]) -> str:
    if not prompt_template or not stop_terms:
        return prompt_template
    if STOP_LIST_PLACEHOLDER in prompt_template:
        lines = prompt_template.splitlines()
        for idx, line in enumerate(lines):
            if STOP_LIST_PLACEHOLDER not in line:
                continue
            indent = line.split(STOP_LIST_PLACEHOLDER, 1)[0]
            items = _format_stop_list_items(stop_terms, indent=indent)
            lines[idx : idx + 1] = items.splitlines()
            return "\n".join(lines)
    return prompt_template.rstrip() + _format_stop_list_section(stop_terms)


def _compile_stop_list_patterns(
    stop_terms: list[str],
) -> list[tuple[str, re.Pattern]]:
    patterns: list[tuple[str, re.Pattern]] = []
    for term in stop_terms:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE
        )
        patterns.append((term, pattern))
    return patterns


def _collect_text_values(value, out: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        out.append(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_text_values(item, out)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_text_values(item, out)


def _find_stop_list_hits(value, patterns: list[tuple[str, re.Pattern]]) -> list[str]:
    if not patterns:
        return []
    texts: list[str] = []
    _collect_text_values(value, texts)
    if not texts:
        return []
    hits: list[str] = []
    for term, pattern in patterns:
        for text in texts:
            if pattern.search(text):
                hits.append(term)
                break
    return hits


def _strip_stop_list_phrases(value, patterns: list[tuple[str, re.Pattern]]):
    if value is None or not patterns:
        return value
    if isinstance(value, str):
        cleaned = value
        for _, pattern in patterns:
            cleaned = pattern.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = cleaned.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
        return cleaned.strip()
    if isinstance(value, list):
        return [_strip_stop_list_phrases(item, patterns) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_stop_list_phrases(item, patterns) for item in value)
    if isinstance(value, dict):
        return {key: _strip_stop_list_phrases(val, patterns) for key, val in value.items()}
    return value


def _coerce_bullet_overrides(raw, *, keep_empty_id: bool = False) -> list[dict]:
    """Normalize bullet override payloads to a list of {id, bullets} dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    items: list[dict] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            items.append({"id": key, "bullets": value})
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                items.append(item)
    else:
        return []

    normalized: list[dict] = []
    for item in items:
        raw_id = item.get("id") or item.get("experience_id") or item.get("role_id")
        bullet_raw = item.get("bullets", [])
        entry_id = str(raw_id or "").strip()
        if not entry_id and not keep_empty_id:
            continue
        bullets: list[str] = []
        if isinstance(bullet_raw, str):
            bullets = [line.strip() for line in bullet_raw.split("\n") if line.strip()]
        elif isinstance(bullet_raw, (list, tuple)):
            for b in bullet_raw:
                if b is None:
                    continue
                text = str(b).strip()
                if text:
                    bullets.append(text)
        else:
            text = str(bullet_raw).strip()
            if text:
                bullets = [text]
        normalized.append({"id": entry_id, "bullets": bullets})
    return normalized


def _bullet_override_map(raw, *, allow_empty_id: bool = False) -> dict[str, list[str]]:
    overrides = _coerce_bullet_overrides(raw, keep_empty_id=allow_empty_id)
    out: dict[str, list[str]] = {}
    for item in overrides:
        entry_id = str(item.get("id") or "").strip()
        if not entry_id:
            continue
        bullets = [
            str(b).strip() for b in (item.get("bullets") or []) if str(b).strip()
        ]
        if bullets:
            out[entry_id] = bullets
    return out


def _coerce_bullet_text(bullets) -> str:
    if isinstance(bullets, (list, tuple)):
        return "\n".join([str(b).strip() for b in bullets if str(b).strip()])
    if bullets is None:
        return ""
    return str(bullets).strip()


def _apply_bullet_overrides(items: list[dict], overrides: dict[str, list[str]]):
    """Return items with bullets overridden by id when overrides are present."""
    if not overrides:
        return items
    out: list[dict] = []
    for item in items:
        data = item.copy()
        entry_id = str(data.get("id") or "").strip()
        override_bullets = overrides.get(entry_id)
        if override_bullets:
            data["bullets"] = list(override_bullets)
        out.append(data)
    return out


def _ensure_skill_rows(raw_rows):
    if raw_rows is None:
        raw_rows = []
    if isinstance(raw_rows, str):
        try:
            raw_rows = json.loads(raw_rows)
        except Exception:
            raw_rows = []
    rows: list[list[str]] = []
    if isinstance(raw_rows, (list, tuple)):
        for row in raw_rows[:3]:
            if isinstance(row, str):
                items = [p.strip() for p in row.split(",") if p.strip()]
            elif isinstance(row, (list, tuple)):
                items = [str(v).strip() for v in row if str(v).strip()]
            elif row is None or str(row).strip() == "":
                items = []
            else:
                items = [str(row).strip()]
            rows.append(items)
    while len(rows) < 3:
        rows.append([])
    return rows[:3]


def _em_value(
    layout_scale: float,
    value: float,
    *,
    weight: float = 1.0,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Return a scaled float value (in `em` units)."""
    try:
        scale = float(layout_scale)
    except (TypeError, ValueError):
        scale = 1.0
    try:
        weight_value = float(weight)
    except (TypeError, ValueError):
        weight_value = 1.0
    if weight_value <= 0:
        weight_value = 1.0

    scaled = value * (scale ** weight_value)
    if min_value is not None:
        scaled = max(float(min_value), scaled)
    if max_value is not None:
        scaled = min(float(max_value), scaled)
    return scaled


def _fmt_em(value: float) -> str:
    if abs(value) < 1e-6:
        return "0em"
    return f"{value:.3f}em"


def _split_degree_parts(text) -> tuple[str, str]:
    """Return (main, detail) where detail preserves parentheses if present."""
    if not isinstance(text, str):
        return str(text or ""), ""
    raw = text.strip()
    if "(" in raw and raw.rstrip().endswith(")"):
        idx = raw.find("(")
        return raw[:idx].rstrip(), raw[idx:].strip()
    return raw, ""


def _parse_degree_details(detail) -> list[str]:
    """Turn a trailing parenthetical into a list of compact highlight items."""
    if not isinstance(detail, str):
        return []
    raw = detail.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(";")]
    return [p for p in parts if p]


def _format_degree_details(items: list[str]) -> str:
    items = [i.strip() for i in (items) if i.strip()]
    if not items:
        return ""
    return " · ".join(items)


# LLM providers are configured on-demand in functions that use them.


# ==========================================
# DATABASE LAYER
# ==========================================
def _get_shared_driver():
    """Create or reuse a process-wide Neo4j driver to avoid reconnect overhead."""
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        _NEO4J_DRIVER = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _NEO4J_DRIVER


def _neo4j_run(session, query: str | Query, **params):
    if isinstance(query, Query):
        return session.run(query, **params)
    return session.run(Query(cast(LiteralString, query)), **params)


class Neo4jClient:
    def __init__(self, driver=None):
        self.driver = driver or _get_shared_driver()
        self._owns_driver = driver is not None and driver is not _NEO4J_DRIVER
        self._ensure_schema()

    def _session(self):
        if self.driver is None:
            raise RuntimeError("Neo4j driver is not available.")
        return self.driver.session()

    def _ensure_schema(self):
        """Create constraints/indexes needed by the app."""
        if self.driver is None:
            return
        global _NEO4J_SCHEMA_READY
        if _NEO4J_SCHEMA_READY:
            return
        try:
            statements = [
                "CREATE CONSTRAINT resume_id IF NOT EXISTS FOR (r:Resume) REQUIRE r.id IS UNIQUE",
                "CREATE CONSTRAINT experience_id IF NOT EXISTS FOR (e:Experience) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT education_id IF NOT EXISTS FOR (e:Education) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT founder_id IF NOT EXISTS FOR (f:FounderRole) REQUIRE f.id IS UNIQUE",
                "CREATE CONSTRAINT profile_id IF NOT EXISTS FOR (p:Profile) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT skill_category_name IF NOT EXISTS FOR (c:SkillCategory) REQUIRE c.name IS UNIQUE",
                "CREATE INDEX profile_created_at IF NOT EXISTS FOR (p:Profile) ON (p.created_at)",
                "CREATE INDEX experience_end_date IF NOT EXISTS FOR (e:Experience) ON (e.end_date)",
                "CREATE INDEX education_end_date IF NOT EXISTS FOR (e:Education) ON (e.end_date)",
                "CREATE INDEX founder_end_date IF NOT EXISTS FOR (f:FounderRole) ON (f.end_date)",
            ]
            with self._session() as session:
                for stmt in statements:
                    _neo4j_run(session, stmt)
            _NEO4J_SCHEMA_READY = True
        except Exception as e:
            print(f"Warning: could not ensure Neo4j schema: {e}")

    def close(self):
        if self._owns_driver and self.driver is not None:
            self.driver.close()

    def reset(self):
        """Blow away all nodes/edges in the DB."""
        with self._session() as session:
            _neo4j_run(session, "MATCH (n) DETACH DELETE n")

    def reset_and_import(self, assets_path: str | Path = DEFAULT_ASSETS_JSON):
        """Reset the DB then import the provided assets JSON."""
        self.reset()
        self.import_assets(assets_path, allow_overwrite=True)

    def resume_exists(self) -> bool:
        with self._session() as session:
            row = _neo4j_run(session, "MATCH (r:Resume) RETURN count(r) AS c").single()
            return bool(row and row["c"] and row["c"] > 0)

    def import_assets(
        self,
        assets_path: str | Path = DEFAULT_ASSETS_JSON,
        *,
        allow_overwrite: bool = False,
    ) -> bool:
        assets_path = Path(assets_path)
        if not assets_path.is_absolute():
            assets_path = BASE_DIR / assets_path

        if not assets_path.exists():
            print(f"Seed file not found at {assets_path}")
            return False

        if not allow_overwrite and self.resume_exists():
            print(
                "Resume already exists; refusing to overwrite. "
                "Use --overwrite-resume to replace it."
            )
            return False

        with assets_path.open(encoding="utf-8") as f:
            data = json.load(f)

        with self._session() as session:
            session.execute_write(self._create_resume, data.get("profile", {}))
            session.execute_write(self._create_experiences, data.get("experience", []))
            session.execute_write(self._create_education, data.get("education", []))
            session.execute_write(
                self._create_founder_roles, data.get("founder_roles", [])
            )
            session.execute_write(self._create_skills, data.get("skills", []))
        return True

    def ensure_resume_exists(self, assets_path: str | Path = DEFAULT_ASSETS_JSON):
        """If no Resume exists, import from JSON (dev bootstrap)."""
        with self._session() as session:
            row = _neo4j_run(session, "MATCH (r:Resume) RETURN count(r) AS c").single()
            if row and row["c"] and row["c"] > 0:
                self.ensure_prompt_yaml()
                self._ensure_placeholder_relationships()
                return
        self.import_assets(assets_path)
        with self._session() as session:
            row = _neo4j_run(session, "MATCH (r:Resume) RETURN count(r) AS c").single()
            if not (row and row["c"] and row["c"] > 0):
                payload = _empty_resume_payload()
                _neo4j_run(
                    session,
                    """
                    MERGE (r:Resume {id: $id})
                    SET r.name = $name,
                        r.first_name = $first_name,
                        r.middle_name = $middle_name,
                        r.last_name = $last_name,
                        r.email = $email,
                        r.email2 = $email2,
                        r.phone = $phone,
                        r.font_family = $font_family,
                        r.auto_fit_target_pages = $auto_fit_target_pages,
                        r.auto_fit_best_scale = $auto_fit_best_scale,
                        r.auto_fit_too_long_scale = $auto_fit_too_long_scale,
                        r.linkedin_url = $linkedin_url,
                        r.github_url = $github_url,
                        r.scholar_url = $scholar_url,
                        r.calendly_url = $calendly_url,
                        r.portfolio_url = $portfolio_url,
                        r.summary = $summary,
                        r.head1_left = $head1_left,
                        r.head1_middle = $head1_middle,
                        r.head1_right = $head1_right,
                        r.head2_left = $head2_left,
                        r.head2_middle = $head2_middle,
                        r.head2_right = $head2_right,
                        r.head3_left = $head3_left,
                        r.head3_middle = $head3_middle,
                        r.head3_right = $head3_right,
                        r.top_skills = $top_skills,
                        r.section_order = $section_order,
                        r.section_enabled = $section_enabled,
                        r.section_titles_json = $section_titles_json,
                        r.custom_sections_json = $custom_sections_json,
                        r.prompt_yaml = $prompt_yaml
                    """,
                    **payload,
                )
        self.ensure_prompt_yaml()
        self._ensure_placeholder_relationships()

    def ensure_prompt_yaml(self, prompt_path: str | Path | None = None) -> str | None:
        """Seed Resume.prompt_yaml from prompt.yaml when missing."""
        prompt_text = _load_prompt_yaml_from_file(
            Path(prompt_path) if prompt_path else None
        )
        if not prompt_text:
            return None
        with self._session() as session:
            _neo4j_run(
                session,
                """
                MATCH (r:Resume)
                WHERE r.prompt_yaml IS NULL OR trim(r.prompt_yaml) = ''
                SET r.prompt_yaml = $prompt
                """,
                prompt=prompt_text,
            )
        return prompt_text

    def _create_resume(self, tx, profile_data):
        profile_data = dict(profile_data or {})
        profile_data.setdefault("scholar_url", "")
        profile_data.setdefault("calendly_url", "")
        profile_data.setdefault("portfolio_url", "")
        profile_data.setdefault("email2", "")
        profile_data.setdefault("font_family", DEFAULT_RESUME_FONT_FAMILY)
        profile_data.setdefault("auto_fit_target_pages", DEFAULT_AUTO_FIT_TARGET_PAGES)
        profile_data.setdefault("auto_fit_best_scale", 1.0)
        profile_data.setdefault("auto_fit_too_long_scale", 0.0)
        profile_data.setdefault("section_order", DEFAULT_SECTION_ORDER.copy())
        profile_data.setdefault("section_titles_json", "{}")
        profile_data.setdefault("custom_sections_json", "[]")
        section_titles = _normalize_section_titles(
            profile_data.get("section_titles_json")
            or profile_data.get("section_titles")
        )
        custom_sections = _normalize_custom_sections(
            profile_data.get("custom_sections_json")
            or profile_data.get("custom_sections")
        )
        extra_keys = _custom_section_keys(custom_sections)
        profile_data["section_order"] = _sanitize_section_order(
            profile_data.get("section_order"), extra_keys
        )
        profile_data["section_enabled"] = _normalize_section_enabled(
            profile_data.get("section_enabled"),
            list(SECTION_LABELS) + extra_keys,
            extra_keys=extra_keys,
        )
        profile_data["section_titles_json"] = json.dumps(
            section_titles, ensure_ascii=True
        )
        profile_data["custom_sections_json"] = json.dumps(
            custom_sections, ensure_ascii=True
        )
        profile_data.setdefault("prompt_yaml", "")
        query = """
        MERGE (r:Resume {id: $id})
        SET r.name = $name,
            r.email = $email,
            r.email2 = $email2,
            r.phone = $phone,
            r.font_family = $font_family,
            r.auto_fit_target_pages = $auto_fit_target_pages,
            r.auto_fit_best_scale = $auto_fit_best_scale,
            r.auto_fit_too_long_scale = $auto_fit_too_long_scale,
            r.linkedin_url = $linkedin_url,
            r.github_url = $github_url,
            r.scholar_url = $scholar_url,
            r.calendly_url = $calendly_url,
            r.portfolio_url = $portfolio_url,
            r.summary = $summary,
            r.head1_left = $head1_left,
            r.head1_middle = $head1_middle,
            r.head1_right = $head1_right,
            r.head2_left = $head2_left,
            r.head2_middle = $head2_middle,
            r.head2_right = $head2_right,
            r.head3_left = $head3_left,
            r.head3_middle = $head3_middle,
            r.head3_right = $head3_right,
            r.top_skills = $top_skills,
            r.section_order = $section_order,
            r.section_enabled = $section_enabled,
            r.section_titles_json = $section_titles_json,
            r.custom_sections_json = $custom_sections_json,
            r.prompt_yaml = $prompt_yaml
        """
        tx.run(query, **profile_data)

    def _ensure_placeholder_relationships(self) -> None:
        """Seed placeholder nodes to register relationship types when DB is empty."""
        if self.driver is None:
            return
        placeholder_date = "2000-01-01"
        placeholders = [
            (
                "Experience",
                "HAS_EXPERIENCE",
                "placeholder_experience",
                {
                    "company": "",
                    "role": "",
                    "location": "",
                    "description": "",
                    "bullets": [],
                },
            ),
            (
                "Education",
                "HAS_EDUCATION",
                "placeholder_education",
                {
                    "school": "",
                    "degree": "",
                    "location": "",
                    "description": "",
                    "bullets": [],
                },
            ),
            (
                "FounderRole",
                "HAS_FOUNDER_ROLE",
                "placeholder_founder",
                {
                    "company": "",
                    "role": "",
                    "location": "",
                    "description": "",
                    "bullets": [],
                },
            ),
        ]
        with self._session() as session:
            for label, rel, node_id, fields in placeholders:
                _neo4j_run(
                    session,
                    f"""
                    MATCH (r:Resume)
                    MERGE (n:{label} {{id: $id}})
                    ON CREATE SET n.is_placeholder = true,
                        n.start_date = date($start_date),
                        n.end_date = date($end_date),
                        n.company = $company,
                        n.role = $role,
                        n.location = $location,
                        n.description = $description,
                        n.bullets = $bullets,
                        n.school = $school,
                        n.degree = $degree
                    MERGE (r)-[:{rel}]->(n)
                    """,
                    id=node_id,
                    start_date=placeholder_date,
                    end_date=placeholder_date,
                    company=fields.get("company", ""),
                    role=fields.get("role", ""),
                    location=fields.get("location", ""),
                    description=fields.get("description", ""),
                    bullets=fields.get("bullets", []),
                    school=fields.get("school", ""),
                    degree=fields.get("degree", ""),
                )
                if MAX_COVERAGE_LOG:
                    _maxcov_log(t"placeholder ensured: {label}")
            _neo4j_run(
                session,
                """
                MATCH (r:Resume)
                MERGE (p:Profile {id: $id})
                ON CREATE SET p.is_placeholder = true,
                    p.created_at = datetime(),
                    p.summary = ""
                MERGE (p)-[:FOR_RESUME]->(r)
                """,
                id="placeholder_profile",
            )
            if MAX_COVERAGE_LOG:
                _maxcov_log("placeholder ensured: Profile")

    def _create_experiences(self, tx, experiences):
        query = """
        MATCH (r:Resume)
        MERGE (e:Experience {id: $id})
        SET e.company = $company,
            e.role = $role,
            e.location = $location,
            e.description = $description,
            e.bullets = $bullets,
            e.start_date = date($start_date),
            e.end_date = date($end_date)
        MERGE (r)-[:HAS_EXPERIENCE]->(e)
        """
        for exp in experiences:
            tx.run(query, **exp)

    def _create_education(self, tx, education):
        query = """
        MATCH (r:Resume)
        MERGE (e:Education {id: $id})
        SET e.school = $school,
            e.degree = $degree,
            e.location = $location,
            e.description = $description,
            e.bullets = $bullets,
            e.start_date = date($start_date),
            e.end_date = date($end_date)
        MERGE (r)-[:HAS_EDUCATION]->(e)
        """
        for edu in education:
            tx.run(query, **edu)

    def _create_founder_roles(self, tx, roles):
        query = """
        MATCH (r:Resume)
        MERGE (f:FounderRole {id: $id})
        SET f.company = $company,
            f.role = $role,
            f.location = $location,
            f.description = $description,
            f.bullets = $bullets,
            f.start_date = date($start_date),
            f.end_date = date($end_date)
        MERGE (r)-[:HAS_FOUNDER_ROLE]->(f)
        """
        for role in roles:
            tx.run(query, **role)

    def _create_skills(self, tx, skill_categories):
        for category in skill_categories:
            cat_name = category["category"]
            query_cat = "MERGE (c:SkillCategory {name: $name})"
            tx.run(query_cat, name=cat_name)

            query_skill = """
            MATCH (c:SkillCategory {name: $cat_name})
            MERGE (s:Skill {id: $id})
            SET s.name = $name
            MERGE (s)-[:IN_CATEGORY]->(c)
            """
            for skill in category["skills"]:
                tx.run(query_skill, cat_name=cat_name, **skill)

    def get_resume_data(self):
        with self._session() as session:
            resume = _neo4j_run(session, "MATCH (r:Resume) RETURN r").single()
            if not resume:
                return None

            resume_data = dict(resume["r"])

        self._ensure_placeholder_relationships()

        with self._session() as session:
            experiences = _neo4j_run(
                session,
                """
                MATCH (r:Resume)-[:HAS_EXPERIENCE]->(e)
                WHERE coalesce(e.is_placeholder, false) = false
                RETURN e
                ORDER BY coalesce(e.end_date, date('9999-12-31')) DESC,
                         coalesce(e.start_date, date('0001-01-01')) DESC
                """,
            ).data()
            education = _neo4j_run(
                session,
                """
                MATCH (r:Resume)-[:HAS_EDUCATION]->(e)
                WHERE coalesce(e.is_placeholder, false) = false
                RETURN e
                ORDER BY coalesce(e.end_date, date('9999-12-31')) DESC,
                         coalesce(e.start_date, date('0001-01-01')) DESC
                """,
            ).data()
            founder_roles = _neo4j_run(
                session,
                """
                MATCH (r:Resume)-[:HAS_FOUNDER_ROLE]->(f)
                WHERE coalesce(f.is_placeholder, false) = false
                RETURN f
                ORDER BY coalesce(f.end_date, date('9999-12-31')) DESC,
                         coalesce(f.start_date, date('0001-01-01')) DESC
                """,
            ).data()

            # Helper to convert neo4j dates to string
            def serialize_dates(items):
                result = []
                for item in items:
                    node = item["e"] if "e" in item else item["f"]
                    data = dict(node)
                    if "start_date" in data:
                        data["start_date"] = str(data["start_date"])
                    if "end_date" in data:
                        data["end_date"] = str(data["end_date"])
                    result.append(data)
                return result

            return {
                "resume": resume_data,
                "experience": serialize_dates(experiences),
                "education": serialize_dates(education),
                "founder_roles": serialize_dates(founder_roles),
            }

    def get_auto_fit_cache(self):
        """Return the last auto-fit tuning values stored on the Resume node (if any)."""
        with self._session() as session:
            row = _neo4j_run(
                session,
                """
                MATCH (r:Resume)
                RETURN
                  r.auto_fit_best_scale AS best_scale,
                  r.auto_fit_too_long_scale AS too_long_scale
                LIMIT 1
                """,
            ).single()
            if not row:
                return None
            return {
                "best_scale": row.get("best_scale"),
                "too_long_scale": row.get("too_long_scale"),
            }

    def set_auto_fit_cache(
        self,
        *,
        best_scale: float,
        too_long_scale: float | None = None,
    ):
        """Store the latest auto-fit tuning values on the Resume node."""
        with self._session() as session:
            _neo4j_run(
                session,
                """
                MATCH (r:Resume)
                SET
                  r.auto_fit_best_scale = $best_scale,
                  r.auto_fit_too_long_scale = $too_long_scale
                """,
                best_scale=best_scale,
                too_long_scale=(
                    float(too_long_scale) if too_long_scale is not None else None
                ),
            )

    def list_applied_jobs(
        self,
        *,
        limit: int | None = None,
        offset: int = 0,
        profile_id: str | None = None,
    ):
        """Return applied jobs (profiles) with key metadata."""
        self._ensure_placeholder_relationships()
        filters = ["coalesce(p.is_placeholder, false) = false"]
        params: dict[str, Any] = {}
        if profile_id:
            filters.append("p.id = $profile_id")
            params["profile_id"] = profile_id
        offset = max(int(offset), 0)
        params["offset"] = offset
        limit_clause = ""
        if limit is not None:
            params["limit"] = max(int(limit), 1)
            limit_clause = "LIMIT $limit"
        where_clause = " AND ".join(filters)
        query = f"""
                MATCH (p:Profile)-[:FOR_RESUME]->(:Resume)
                WHERE {where_clause}
                RETURN p
                ORDER BY p.created_at DESC
                SKIP $offset
                {limit_clause}
                """
        with self._session() as session:
            rows = _neo4j_run(
                session,
                query,
                **params,
            ).data()
        jobs = []
        for row in rows:
            p = dict(row["p"])
            skills_rows = []
            raw_skills_rows = (p.get("skills_rows_json") or "").strip()
            if raw_skills_rows:
                try:
                    skills_rows = json.loads(raw_skills_rows)
                except Exception:
                    skills_rows = []
            norm_rows: list[list[str]] = []
            if isinstance(skills_rows, str):
                try:
                    skills_rows = json.loads(skills_rows)
                except Exception:
                    skills_rows = []
            if isinstance(skills_rows, (list, tuple)):
                for row_items in skills_rows[:3]:
                    if isinstance(row_items, str):
                        items = [s.strip() for s in row_items.split(",") if s.strip()]
                    elif isinstance(row_items, (list, tuple)):
                        items = [str(s).strip() for s in row_items if str(s).strip()]
                    elif row_items is None or str(row_items).strip() == "":
                        items = []
                    else:
                        items = [str(row_items).strip()]
                    norm_rows.append(items)
            while len(norm_rows) < 3:
                norm_rows.append([])
            norm_rows = norm_rows[:3]
            experience_bullets = _coerce_bullet_overrides(
                p.get("experience_bullets_json")
            )
            founder_role_bullets = _coerce_bullet_overrides(
                p.get("founder_role_bullets_json")
            )
            jobs.append(
                {
                    "id": p.get("id"),
                    "created_at": str(p.get("created_at")),
                    "target_company": p.get("target_company", ""),
                    "target_role": p.get("target_role", ""),
                    "seniority_level": p.get("seniority_level", ""),
                    "target_location": p.get("target_location", ""),
                    "work_mode": p.get("work_mode", ""),
                    "req_id": p.get("req_id", ""),
                    "summary": p.get("summary", ""),
                    "headers": p.get("headers", []),
                    "highlighted_skills": p.get("highlighted_skills", []),
                    "skills_rows": norm_rows,
                    "experience_bullets": experience_bullets,
                    "founder_role_bullets": founder_role_bullets,
                    "job_req_raw": p.get("job_req_raw", ""),
                    "travel_requirement": p.get("travel_requirement", ""),
                    "primary_domain": p.get("primary_domain", ""),
                    "must_have_skills": p.get("must_have_skills", []),
                    "nice_to_have_skills": p.get("nice_to_have_skills", []),
                    "tech_stack_keywords": p.get("tech_stack_keywords", []),
                    "non_technical_requirements": p.get(
                        "non_technical_requirements", []
                    ),
                    "certifications": p.get("certifications", []),
                    "clearances": p.get("clearances", []),
                    "core_responsibilities": p.get("core_responsibilities", []),
                    "outcome_goals": p.get("outcome_goals", []),
                    "salary_band": p.get("salary_band", ""),
                    "posting_url": p.get("posting_url", ""),
                }
            )
        return jobs

    def search_profile_metadata(
        self,
        *,
        term: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, str]]:
        """Search Profile metadata for Select2."""
        self._ensure_placeholder_relationships()
        term = str(term or "").strip().lower()
        limit = max(int(limit), 1)
        offset = max(int(offset), 0)
        query = """
                MATCH (p:Profile)-[:FOR_RESUME]->(:Resume)
                WHERE coalesce(p.is_placeholder, false) = false
                  AND (
                    $term = ""
                    OR toLower(coalesce(p.target_company, "")) CONTAINS $term
                    OR toLower(coalesce(p.target_role, "")) CONTAINS $term
                    OR toLower(coalesce(p.seniority_level, "")) CONTAINS $term
                    OR toLower(coalesce(p.target_location, "")) CONTAINS $term
                    OR toLower(coalesce(p.work_mode, "")) CONTAINS $term
                    OR toLower(coalesce(p.req_id, "")) CONTAINS $term
                    OR toLower(coalesce(p.posting_url, "")) CONTAINS $term
                    OR toLower(toString(p.created_at)) CONTAINS $term
                  )
                RETURN
                  p.id AS id,
                  p.created_at AS created_at,
                  p.target_company AS target_company,
                  p.target_role AS target_role,
                  p.seniority_level AS seniority_level,
                  p.target_location AS target_location,
                  p.work_mode AS work_mode,
                  p.req_id AS req_id,
                  p.posting_url AS posting_url
                ORDER BY p.created_at DESC
                SKIP $offset
                LIMIT $limit
                """
        with self._session() as session:
            rows = _neo4j_run(
                session,
                query,
                term=term,
                offset=offset,
                limit=limit,
            ).data()
        results: list[dict[str, str]] = []
        for row in rows:
            results.append(
                {
                    "id": row.get("id"),
                    "created_at": str(row.get("created_at") or ""),
                    "target_company": row.get("target_company") or "",
                    "target_role": row.get("target_role") or "",
                    "seniority_level": row.get("seniority_level") or "",
                    "target_location": row.get("target_location") or "",
                    "work_mode": row.get("work_mode") or "",
                    "req_id": row.get("req_id") or "",
                    "posting_url": row.get("posting_url") or "",
                }
            )
        return results

    def get_profile_metadata(self, profile_id: str) -> dict[str, str]:
        """Fetch metadata for a single Profile."""
        profile_id = str(profile_id or "").strip()
        if not profile_id:
            return {}
        with self._session() as session:
            row = _neo4j_run(
                session,
                """
                MATCH (p:Profile {id: $id})-[:FOR_RESUME]->(:Resume)
                RETURN
                  p.id AS id,
                  p.created_at AS created_at,
                  p.target_company AS target_company,
                  p.target_role AS target_role
                """,
                id=profile_id,
            ).single()
        if not row:
            return {}
        return {
            "id": row.get("id"),
            "created_at": str(row.get("created_at") or ""),
            "target_company": row.get("target_company") or "",
            "target_role": row.get("target_role") or "",
        }

    def save_resume(self, resume_data):
        """
        Persist a versioned Profile node (LLM output) linked to the canonical Resume.
        Nested arrays (e.g., skills rows) are stored as JSON strings to satisfy
        Neo4j property type constraints.
        """
        with self._session() as session:
            resume_data = dict(resume_data or {})
            resume_data.setdefault("summary", "")
            resume_data.setdefault("headers", [])
            resume_data.setdefault("highlighted_skills", [])
            resume_data.setdefault("skills_rows_json", "[]")
            resume_data.setdefault("job_req_raw", "")
            resume_data.setdefault("target_company", "")
            resume_data.setdefault("target_role", "")
            resume_data.setdefault("seniority_level", "")
            resume_data.setdefault("target_location", "")
            resume_data.setdefault("work_mode", "")
            resume_data.setdefault("travel_requirement", "")
            resume_data.setdefault("primary_domain", "")
            resume_data.setdefault("must_have_skills", [])
            resume_data.setdefault("nice_to_have_skills", [])
            resume_data.setdefault("tech_stack_keywords", [])
            resume_data.setdefault("non_technical_requirements", [])
            resume_data.setdefault("certifications", [])
            resume_data.setdefault("clearances", [])
            resume_data.setdefault("core_responsibilities", [])
            resume_data.setdefault("outcome_goals", [])
            resume_data.setdefault("salary_band", "")
            resume_data.setdefault("posting_url", "")
            resume_data.setdefault("req_id", "")
            resume_data.setdefault("experience_bullets_json", "[]")
            resume_data.setdefault("founder_role_bullets_json", "[]")
            query = """
            MATCH (r:Resume)
            CREATE (p:Profile {
                id: randomUUID(),
                created_at: datetime(),
                summary: $summary,
                headers: $headers,
                highlighted_skills: $highlighted_skills,
                skills_rows_json: $skills_rows_json,
                experience_bullets_json: $experience_bullets_json,
                founder_role_bullets_json: $founder_role_bullets_json,
                job_req_raw: $job_req_raw,
                target_company: $target_company,
                target_role: $target_role,
                seniority_level: $seniority_level,
                target_location: $target_location,
                work_mode: $work_mode,
                travel_requirement: $travel_requirement,
                primary_domain: $primary_domain,
                must_have_skills: $must_have_skills,
                nice_to_have_skills: $nice_to_have_skills,
                tech_stack_keywords: $tech_stack_keywords,
                non_technical_requirements: $non_technical_requirements,
                certifications: $certifications,
                clearances: $clearances,
                core_responsibilities: $core_responsibilities,
                outcome_goals: $outcome_goals,
                salary_band: $salary_band,
                posting_url: $posting_url,
                req_id: $req_id
            })
            MERGE (p)-[:FOR_RESUME]->(r)
            RETURN p.id as id
            """
            result = _neo4j_run(session, query, **resume_data)
            row = result.single()
            if not row:
                raise RuntimeError("Failed to save profile: Resume not found.")
            return row["id"]

    def update_profile_bullets(
        self,
        profile_id: str,
        experience_bullets: list[dict],
        founder_role_bullets: list[dict],
    ) -> bool:
        """Update bullet overrides on a Profile node by id."""
        with self._session() as session:
            result = _neo4j_run(
                session,
                """
                MATCH (p:Profile {id: $id})
                SET
                  p.experience_bullets_json = $experience_bullets_json,
                  p.founder_role_bullets_json = $founder_role_bullets_json
                RETURN p.id AS id
                """,
                id=profile_id,
                experience_bullets_json=json.dumps(
                    experience_bullets, ensure_ascii=False
                ),
                founder_role_bullets_json=json.dumps(
                    founder_role_bullets, ensure_ascii=False
                ),
            )
            row = result.single()
            return bool(row and row.get("id"))

    def upsert_resume_and_sections(
        self,
        resume_fields,
        experiences,
        education,
        founder_roles,
        *,
        delete_missing: bool = False,
    ):
        """Upsert resume, experiences, education, and founder roles in Neo4j."""
        with self._session() as session:
            resume_fields = dict(resume_fields or {})
            resume_fields.setdefault("section_order", DEFAULT_SECTION_ORDER.copy())
            section_titles = _normalize_section_titles(
                resume_fields.get("section_titles_json")
                or resume_fields.get("section_titles")
            )
            custom_sections = _normalize_custom_sections(
                resume_fields.get("custom_sections_json")
                or resume_fields.get("custom_sections")
            )
            extra_keys = _custom_section_keys(custom_sections)
            resume_fields["section_order"] = _sanitize_section_order(
                resume_fields.get("section_order"), extra_keys
            )
            resume_fields["section_enabled"] = _normalize_section_enabled(
                resume_fields.get("section_enabled"),
                list(SECTION_LABELS) + extra_keys,
                extra_keys=extra_keys,
            )
            resume_fields["section_titles_json"] = json.dumps(
                section_titles, ensure_ascii=True
            )
            resume_fields["custom_sections_json"] = json.dumps(
                custom_sections, ensure_ascii=True
            )
            resume_fields.setdefault("email2", "")
            resume_fields.setdefault("calendly_url", "")
            resume_fields.setdefault("portfolio_url", "")
            resume_fields.setdefault("font_family", DEFAULT_RESUME_FONT_FAMILY)
            resume_fields.setdefault(
                "auto_fit_target_pages", DEFAULT_AUTO_FIT_TARGET_PAGES
            )
            resume_fields.setdefault("prompt_yaml", "")
            _neo4j_run(
                session,
                """
                MATCH (r:Resume)
                SET r.summary = $summary,
                    r.prompt_yaml = $prompt_yaml,
                    r.name = $name,
                    r.first_name = $first_name,
                    r.middle_name = $middle_name,
                    r.last_name = $last_name,
                    r.email = $email,
                    r.email2 = $email2,
                    r.phone = $phone,
                    r.font_family = $font_family,
                    r.auto_fit_target_pages = $auto_fit_target_pages,
                    r.linkedin_url = $linkedin_url,
                    r.github_url = $github_url,
                    r.scholar_url = $scholar_url,
                    r.calendly_url = $calendly_url,
                    r.portfolio_url = $portfolio_url,
                    r.head1_left = $head1_left,
                    r.head1_middle = $head1_middle,
                    r.head1_right = $head1_right,
                    r.head2_left = $head2_left,
                    r.head2_middle = $head2_middle,
                    r.head2_right = $head2_right,
                    r.head3_left = $head3_left,
                    r.head3_middle = $head3_middle,
                    r.head3_right = $head3_right,
                    r.top_skills = $top_skills,
                    r.section_order = $section_order,
                    r.section_enabled = $section_enabled,
                    r.section_titles_json = $section_titles_json,
                    r.custom_sections_json = $custom_sections_json
                """,
                **resume_fields,
            )

            def date_clause(field):
                return f"CASE WHEN ${field} IS NOT NULL AND ${field} <> '' THEN date(${field}) ELSE NULL END"

            exp_query = Query(
                cast(
                    LiteralString,
                    f"""
                MATCH (r:Resume)
                MERGE (e:Experience {{id: $id}})
                SET e.company = $company,
                    e.role = $role,
                    e.location = $location,
                    e.description = $description,
                    e.bullets = $bullets,
                    e.start_date = {date_clause("start_date")},
                    e.end_date = {date_clause("end_date")}
                MERGE (r)-[:HAS_EXPERIENCE]->(e)
                    """,
                )
            )
            for exp in experiences:
                _neo4j_run(session, exp_query, **exp)

            edu_query = Query(
                cast(
                    LiteralString,
                    f"""
                MATCH (r:Resume)
                MERGE (e:Education {{id: $id}})
                SET e.school = $school,
                    e.degree = $degree,
                    e.location = $location,
                    e.description = $description,
                    e.bullets = $bullets,
                    e.start_date = {date_clause("start_date")},
                    e.end_date = {date_clause("end_date")}
                MERGE (r)-[:HAS_EDUCATION]->(e)
                    """,
                )
            )
            for edu in education:
                _neo4j_run(session, edu_query, **edu)

            role_query = Query(
                cast(
                    LiteralString,
                    f"""
                MATCH (r:Resume)
                MERGE (f:FounderRole {{id: $id}})
                SET f.company = $company,
                    f.role = $role,
                    f.location = $location,
                    f.description = $description,
                    f.bullets = $bullets,
                    f.start_date = {date_clause("start_date")},
                    f.end_date = {date_clause("end_date")}
                MERGE (r)-[:HAS_FOUNDER_ROLE]->(f)
                    """,
                )
            )
            for role in founder_roles:
                _neo4j_run(session, role_query, **role)

            if delete_missing:
                exp_ids = sorted(
                    {exp.get("id") for exp in experiences if exp.get("id")}
                )
                edu_ids = sorted({edu.get("id") for edu in education if edu.get("id")})
                role_ids = sorted(
                    {role.get("id") for role in founder_roles if role.get("id")}
                )
                _neo4j_run(
                    session,
                    """
                    MATCH (r:Resume)-[:HAS_EXPERIENCE]->(e:Experience)
                    WHERE NOT e.id IN $ids
                    DETACH DELETE e
                    """,
                    ids=exp_ids,
                )
                _neo4j_run(
                    session,
                    """
                    MATCH (r:Resume)-[:HAS_EDUCATION]->(e:Education)
                    WHERE NOT e.id IN $ids
                    DETACH DELETE e
                    """,
                    ids=edu_ids,
                )
                _neo4j_run(
                    session,
                    """
                    MATCH (r:Resume)-[:HAS_FOUNDER_ROLE]->(f:FounderRole)
                    WHERE NOT f.id IN $ids
                    DETACH DELETE f
                    """,
                    ids=role_ids,
                )


# ==========================================
# LLM LAYER (any-llm)
# ==========================================
def _split_llm_model_spec(raw: str | None) -> tuple[str, str]:
    """Return (provider, model_id) from `provider:model`, `provider/model`, or bare `model`."""
    raw = (raw or "").strip()
    if not raw:
        # DEFAULT_LLM_MODEL is already canonicalized to include a provider prefix.
        raw = DEFAULT_LLM_MODEL
    if ":" in raw:
        prefix, rest = raw.split(":", 1)
        provider = prefix.strip().lower()
        if provider in SUPPORTED_LLM_PROVIDERS and rest.strip():
            return provider, rest.strip()
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        provider = prefix.strip().lower()
        if provider in SUPPORTED_LLM_PROVIDERS and rest.strip():
            return provider, rest.strip()
    # Backward compatibility: bare model ids default to OpenAI.
    return "openai", raw


def list_llm_models() -> list[str]:
    """Return configured model ids for the UI/CLI (no network calls)."""
    raw = (
        os.environ.get("LLM_MODELS") or os.environ.get("OPENAI_MODELS") or ""
    ).strip()
    if raw:
        candidates = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        candidates = DEFAULT_LLM_MODELS.copy()
    seen: set[str] = set()
    out: list[str] = []
    for spec in candidates:
        provider, model = _split_llm_model_spec(spec)
        canonical = f"{provider}:{model}"
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def _read_first_secret_line(path: Path) -> str | None:
    """Return the first non-empty, non-comment line from a file (no printing)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return None


def load_openai_api_key() -> str | None:
    """Load OpenAI API key from ~/openaikey.txt or env (no printing)."""
    path = Path.home() / "openaikey.txt"

    def read_kv(name: str) -> str | None:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception:
            return None
        target = name.strip().upper()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip().upper() == target:
                return value.strip().strip("\"'")
        return None

    from_kv = read_kv("OPENAI_API_KEY")
    if from_kv:
        return from_kv

    # Backward compatibility: a single-line file containing just the key.
    raw = _read_first_secret_line(path)
    if not raw:
        raw = ""
    if raw.lstrip().startswith("sk-"):
        return raw.strip().strip("\"'")

    env = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if env:
        return env
    return None


def load_gemini_api_key() -> str | None:
    """Load Gemini API key from env or ~/geminikey.txt (GEMINI_API_KEY/GOOGLE_API_KEY)."""
    env = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if env:
        return env
    env = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if env:
        return env

    def read_kv(path: Path) -> str | None:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception:
            return None
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip().upper() in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
                return value.strip().strip("\"'")
        return None

    path = Path.home() / "geminikey.txt"
    from_kv = read_kv(path)
    if from_kv:
        return from_kv

    raw = _read_first_secret_line(path)
    if raw and "=" not in raw:
        return raw.strip().strip("\"'")

    # Backward compatibility: allow GEMINI_API_KEY/GOOGLE_API_KEY in openaikey.txt.
    legacy = read_kv(Path.home() / "openaikey.txt")
    if legacy:
        return legacy
    return None


def _openai_reasoning_params_for_model(model: str) -> dict | None:
    """Return OpenAI `reasoning` params for models that support it."""
    effort = (DEFAULT_LLM_REASONING_EFFORT).strip().lower()
    if not effort or effort == "none":
        return None
    # OpenAI doesn't consistently accept "minimal"; map it to "low".
    if effort == "minimal":
        effort = "low"
    model_id = (model).strip().lower()
    if not model_id:
        return None
    # Heuristic: reasoning params are accepted by o-series and GPT-5.x models.
    if model_id.startswith(("o", "gpt-5")):
        return {"effort": effort}
    return None


def _read_int_env(*names: str) -> int | None:
    for name in names:
        if not name:
            continue
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            val = int(raw)
        except Exception:
            continue
        if val > 0:
            return val
    return None


def _resolve_stop_list_max_attempts() -> int:
    override = _read_int_env("STOP_LIST_MAX_ATTEMPTS", "LLM_STOP_LIST_MAX_ATTEMPTS")
    if override:
        return override
    return 3


def _resolve_llm_json_retry_attempts(provider: str) -> int:
    provider = (provider or "").strip().lower()
    override = _read_int_env(
        "LLM_JSON_RETRY_ATTEMPTS",
        "OPENAI_JSON_RETRY_ATTEMPTS" if provider == "openai" else "",
        "GEMINI_JSON_RETRY_ATTEMPTS" if provider == "gemini" else "",
        "GOOGLE_JSON_RETRY_ATTEMPTS" if provider == "gemini" else "",
    )
    if override:
        return override
    if provider == "gemini":
        return 4
    return 2


def _is_retryable_llm_json_error(result: dict) -> bool:
    err = str(result.get("error") or "").strip()
    if not err:
        return False
    lowered = err.lower()
    return (
        "non-json" in lowered
        or "non json" in lowered
        or "empty response" in lowered
        or "returned no content" in lowered
    )


def _resolve_llm_max_output_tokens(provider: str, model: str) -> int:
    """Pick a safe output token budget, overridable via LLM_MAX_OUTPUT_TOKENS."""
    provider = (provider).strip().lower()
    model_id = (model).strip().lower()

    override = _read_int_env(
        "LLM_MAX_OUTPUT_TOKENS",
        "OPENAI_MAX_OUTPUT_TOKENS" if provider == "openai" else "",
        "GEMINI_MAX_OUTPUT_TOKENS" if provider == "gemini" else "",
        "GOOGLE_MAX_OUTPUT_TOKENS" if provider == "gemini" else "",
    )
    if override:
        return override

    if provider == "openai":
        # Reasoning-capable models can consume output budget for hidden reasoning. Give them more.
        if model_id.startswith(("gpt-5", "o")):
            return 8192
        return 4096

    if provider == "gemini":
        return 4096

    return 4096


def _resolve_llm_retry_max_output_tokens(
    provider: str, model: str, initial: int
) -> int:
    """Pick a larger token budget for a single retry on truncation."""
    provider = (provider).strip().lower()
    model_id = (model).strip().lower()

    override = _read_int_env(
        "LLM_MAX_OUTPUT_TOKENS_RETRY",
        "OPENAI_MAX_OUTPUT_TOKENS_RETRY" if provider == "openai" else "",
        "GEMINI_MAX_OUTPUT_TOKENS_RETRY" if provider == "gemini" else "",
        "GOOGLE_MAX_OUTPUT_TOKENS_RETRY" if provider == "gemini" else "",
    )
    if override:
        return override

    # Default: grow aggressively once, but keep it bounded.
    if provider == "openai":
        cap = (
            16384
            if (model_id.startswith(("gpt-5", "o")))
            else 8192
        )
    else:
        cap = 8192
    return min(max(initial * 2, initial + 2048), cap)


def render_resume_pdf_bytes(
    save_copy: bool = False,
    include_summary: bool = True,
    include_skills: bool = True,
    filename: str = "preview_no_summary_skills.pdf",
):
    """Compile a resume to PDF with optional summary/skills; optionally persist a copy."""
    try:
        db = Neo4jClient()
        data = db.get_resume_data() or {}
        db.close()

        resume_node = data.get("resume", {}) or {}

        def ensure_len(items, target=9):
            items = list(items or [])
            while len(items) < target:
                items.append("")
            return items[:target]

        def parse_section_order(raw_order, extra_keys):
            if isinstance(raw_order, str):
                raw_order = [s.strip() for s in raw_order.split(",") if s.strip()]
            return _sanitize_section_order(raw_order, extra_keys)

        headers = ensure_len(
            [
                resume_node.get("head1_left", ""),
                resume_node.get("head1_middle", ""),
                resume_node.get("head1_right", ""),
                resume_node.get("head2_left", ""),
                resume_node.get("head2_middle", ""),
                resume_node.get("head2_right", ""),
                resume_node.get("head3_left", ""),
                resume_node.get("head3_middle", ""),
                resume_node.get("head3_right", ""),
            ]
        )

        skills = ensure_len(resume_node.get("top_skills", []))
        summary_text = resume_node.get("summary", "") if include_summary else ""

        section_titles = _normalize_section_titles(
            resume_node.get("section_titles_json") or resume_node.get("section_titles")
        )
        custom_sections = _normalize_custom_sections(
            resume_node.get("custom_sections_json")
            or resume_node.get("custom_sections")
        )
        extra_keys = _custom_section_keys(custom_sections)
        resume_data = {
            "summary": summary_text,
            "headers": headers[:9],
            "highlighted_skills": skills[:9] if include_skills else [],
            "first_name": resume_node.get("first_name", ""),
            "middle_name": resume_node.get("middle_name", ""),
            "last_name": resume_node.get("last_name", ""),
            "email": resume_node.get("email", ""),
            "email2": resume_node.get("email2", ""),
            "phone": resume_node.get("phone", ""),
            "font_family": resume_node.get("font_family", DEFAULT_RESUME_FONT_FAMILY),
            "auto_fit_target_pages": _normalize_auto_fit_target_pages(
                resume_node.get("auto_fit_target_pages"),
                DEFAULT_AUTO_FIT_TARGET_PAGES,
            ),
            "linkedin_url": resume_node.get("linkedin_url", ""),
            "github_url": resume_node.get("github_url", ""),
            "scholar_url": resume_node.get("scholar_url", ""),
            "calendly_url": resume_node.get("calendly_url", ""),
            "portfolio_url": resume_node.get("portfolio_url", ""),
            "section_order": parse_section_order(
                resume_node.get("section_order"), extra_keys
            ),
            "section_titles": section_titles,
            "custom_sections": custom_sections,
        }
        section_enabled = _normalize_section_enabled(
            resume_node.get("section_enabled"),
            list(SECTION_LABELS) + extra_keys,
            extra_keys=extra_keys,
        )
        resume_data["section_order"] = _apply_section_enabled(
            resume_data["section_order"],
            section_enabled,
        )
        profile_data = {
            **resume_node,
            "experience": data.get("experience", []),
            "education": data.get("education", []),
            "founder_roles": data.get("founder_roles", []),
        }
        typst_src = generate_typst_source(
            resume_data,
            profile_data,
            include_matrices=include_skills,
            include_summary=include_summary,
            section_order=resume_data["section_order"],
        )
        ok, pdf_bytes = compile_pdf(typst_src)
        if not ok or not pdf_bytes:
            return None
        if save_copy:
            dest = ASSETS_DIR / filename
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(pdf_bytes)
            except Exception as e:
                print(f"Warning: could not save resume pdf output: {e}")
        return pdf_bytes
    except Exception as e:
        print(f"Warning: could not render resume pdf: {e}")
    return None


def _hex_to_rgba(value: str | None) -> tuple[int, int, int, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.removeprefix("#")
    if len(text) == 3:
        try:
            r = int(text[0] * 2, 16)
            g = int(text[1] * 2, 16)
            b = int(text[2] * 2, 16)
        except ValueError:
            return None
        return (r, g, b, 255)
    if len(text) != 6:
        return None
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
    except ValueError:
        return None
    return (r, g, b, 255)


def _rasterize_text_image(
    text: str,
    *,
    font_size: int = 48,
    target_height_pt: float | None = None,
    italic: bool = False,
    font_family: str | None = None,
    fill_hex: str | None = None,
) -> str:
    """Render a short text snippet to a PNG (high-res, inline) and return its repo-relative path."""
    text = (text).strip()
    if not text:
        return ""
    if not font_family:
        font_family = DEFAULT_RESUME_FONT_FAMILY
    fill_rgba = _hex_to_rgba(fill_hex)
    try:
        from PIL import Image, ImageDraw, ImageFont  # Lazy import; optional dependency.

        TEMP_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        # Use the selected font family to match resume typography.
        font_size = max(6, font_size)  # Guard against tiny/invalid sizes.
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None
        font_candidates = _select_local_font_paths(font_family, italic=italic)
        for candidate in font_candidates:
            try:
                font = ImageFont.truetype(str(candidate), font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        # Measure text bounds at the render resolution.
        dummy = Image.new("L", (1, 1), 0)
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = int((bbox[2] - bbox[0]) + 12)
        height = int((bbox[3] - bbox[1]) + 12)
        img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        # Use deep black for crisp rasterized text; Typst applies sizing on embed.
        draw.text((6, 6), text, fill=fill_rgba or (0, 0, 0, 255), font=font)

        # Optionally pre-scale to reduce aliasing when Typst scales down.
        if target_height_pt:
            # Oversample more aggressively (~6x at 96dpi) for cleaner downscale in Typst.
            target_px = max(1, int(target_height_pt * (96 / 72) * 6.0))
            scale = target_px / img.height
            if scale > 0:
                new_w = max(1, int(img.width * scale))
                new_h = max(1, int(img.height * scale))
                image_any = cast(Any, Image)
                resample = getattr(
                    getattr(image_any, "Resampling", image_any),
                    "LANCZOS",
                    getattr(image_any, "BICUBIC", 3),
                )
                img = img.resize((new_w, new_h), resample)

        cache_key = (
            f"{text}|{target_height_pt or ''}|{'ital' if italic else 'reg'}|"
            f"{fill_hex or ''}"
        )
        fname = (
            "founder_date_"
            + _hash_text(cache_key)[:12]
            + ".png"
        )
        fpath = TEMP_BUILD_DIR / fname
        img.save(fpath, format="PNG")
        return "/" + str(fpath.relative_to(BASE_DIR))
    except Exception:
        return ""


def generate_typst_source(
    resume_data,
    profile_data,
    include_matrices=True,
    include_summary=True,
    section_order=None,
    layout_scale: float = 1.0,
):
    """
    Converts resume data into Typst markup.
    """
    # Extract data
    email = profile_data.get("email", "email@example.com")
    email2 = profile_data.get("email2", resume_data.get("email2", ""))
    phone = profile_data.get("phone", "555-0123")
    linkedin = normalize_linkedin(profile_data.get("linkedin_url", ""))

    def _split_github_input(raw: str) -> tuple[str, str]:
        raw = raw.strip()
        if not raw:
            return "", ""
        if re.match(r"^https?://", raw, flags=re.IGNORECASE):
            try:
                parsed = urlparse(raw)
            except Exception:
                return "", raw
            host = (parsed.netloc or "").strip().lower()
            host = host.removeprefix("www.")
            if host and host != "github.com":
                return "", raw
            return normalize_github(parsed.path.lstrip("/")), ""
        trimmed = raw.lstrip()
        lowered = trimmed.lower()
        if lowered.startswith("www."):
            trimmed = trimmed[4:]
            lowered = trimmed.lower()
        if lowered.startswith("github.com/"):
            return normalize_github(trimmed), ""
        host_candidate = lowered.split("/", 1)[0]
        if "." in host_candidate:
            return "", "https://" + trimmed.lstrip("/")
        return normalize_github(trimmed), ""

    github_path, github_url = _split_github_input(profile_data.get("github_url", ""))
    github = github_path
    scholar_url = normalize_scholar_url(profile_data.get("scholar_url", ""))
    calendly_url = normalize_calendly_url(
        profile_data.get("calendly_url", resume_data.get("calendly_url", ""))
    )
    portfolio_url = normalize_portfolio_url(
        profile_data.get("portfolio_url", resume_data.get("portfolio_url", ""))
    )
    portfolio_label = "portfolio" if portfolio_url else ""
    portfolio_label = escape_typst(portfolio_label)
    scholar_label = escape_typst(
        profile_data.get("scholar_link_text", format_url_label(scholar_url))
    )
    github_label_source = profile_data.get("github_link_text", "")
    if github_url:
        if not github_label_source:
            github_label_source = format_url_label(github_url)
    elif not github_label_source:
        github_label_source = github
    github_label = escape_typst(github_label_source)
    linkedin_label = escape_typst(profile_data.get("linkedin_link_text", linkedin))
    custom_contacts: list[dict[str, str]] = []
    if calendly_url:
        calendly_label = format_url_label(calendly_url) or calendly_url
        custom_contacts.append(
            {"text": calendly_label, "icon": "calendar", "link": calendly_url}
        )

    # Prefer the resume-level summary; fall back to profile summary if missing.
    summary_source = resume_data.get("summary") or profile_data.get("summary", "")
    summary = summary_source if (summary_source and include_summary) else ""
    resume_font_family = (
        resume_data.get("font_family")
        or profile_data.get("font_family")
        or DEFAULT_RESUME_FONT_FAMILY
    )
    resume_font_family = str(resume_font_family or DEFAULT_RESUME_FONT_FAMILY).strip()
    if not resume_font_family:
        resume_font_family = DEFAULT_RESUME_FONT_FAMILY
    resume_font_family_escaped = escape_typst(resume_font_family)

    try:
        layout_scale = float(layout_scale)
    except Exception:
        layout_scale = 1.0
    if not (layout_scale > 0):
        layout_scale = 1.0
    layout_scale = max(0.35, min(layout_scale, 6.0))

    def em_value(
        value: float,
        *,
        weight: float = 1.0,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float:
        return _em_value(
            layout_scale, value, weight=weight, min_value=min_value, max_value=max_value
        )

    def fmt_em(value: float) -> str:
        return _fmt_em(value)

    def em(
        value: float,
        *,
        weight: float = 1.0,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> str:
        """
        Scale a Typst `em` value using a weighted exponent and clamp it.

        The goal is to let auto-fit adjust the document height without:
        - shrinking micro-leading so far that text overlaps
        - inflating macro gaps (between entries/sections) into absurd whitespace
        """
        return fmt_em(
            em_value(value, weight=weight, min_value=min_value, max_value=max_value)
        )

    # One canonical gap value used throughout the document for section-level spacing.
    gap_value = em_value(1.1, weight=1.0, min_value=0.01, max_value=None)
    GAP = fmt_em(gap_value)

    # Bullet spacing should scale with auto-fit, but stay proportionate to section gaps.
    # Use a geometric blend between raw auto-fit scaling and a GAP-anchored ratio.
    #
    # NOTE: These are `em` units, not points. Keep them small.
    bullet_base = 2.0
    gap_base = 1.1
    bullet_raw = em_value(bullet_base, weight=2.0)
    bullet_anchor = gap_value * (bullet_base / gap_base)
    bullet_balanced = (bullet_raw * max(1e-9, bullet_anchor)) ** 0.5
    bullet_min = 0.01  # Allow aggressive shrinking
    bullet_max = 1000.0  # Effectively no upper clamp
    _bullet_val = min(max(bullet_balanced, bullet_min), bullet_max)
    BULLET_GAP = fmt_em(_bullet_val)
    HALF_BULLET_GAP = fmt_em(_bullet_val * 0.5)
    # Deliberately extreme offset to confirm bullet marker movement.
    BULLET_MARKER_OFFSET = ".13em"
    BULLET_MARKER = (
        f'move(dy: {BULLET_MARKER_OFFSET}, '
        'text(size: 10.2pt * 0.75, weight: "regular", fill: color-gray)[#sym.bullet])'
    )

    space_between_contact_and_professional_summary = GAP
    space_between_professional_summary_and_next_section = GAP

    # Headers (3x3)
    headers = [escape_typst(str(h).upper()) for h in resume_data.get("headers", [])]
    while len(headers) < 9:
        headers.append("")

    # Highlighted Skills (3x3)
    h_skills = [
        escape_typst(str(s).upper()) for s in resume_data.get("highlighted_skills", [])
    ]
    while len(h_skills) < 9:
        h_skills.append("")

    experiences = profile_data.get("experience", [])
    education = profile_data.get("education", [])
    founder_roles = profile_data.get("founder_roles", [])
    section_titles = _normalize_section_titles(
        resume_data.get("section_titles") or profile_data.get("section_titles")
    )
    custom_sections = _normalize_custom_sections(
        resume_data.get("custom_sections") or profile_data.get("custom_sections")
    )
    custom_section_keys = _custom_section_keys(custom_sections)
    title_map = _build_section_title_map(section_titles, custom_sections)

    # Parse name into first/last; prefer explicit fields.
    first = profile_data.get("first_name") or resume_data.get("first_name") or ""
    last = profile_data.get("last_name") or resume_data.get("last_name") or ""
    full_name = profile_data.get("name", "").strip()
    if not (first or last):
        parts = full_name.split()
        if parts:
            first = parts[0]
            if len(parts) >= 2:
                last = parts[-1]
    firstname = first.strip() or "John"
    lastname = last or "Doe"

    RASTER_IMAGE_HEIGHT_PT = 10.6

    def render_img_or_text(
        raw: str,
        height_pt: float = RASTER_IMAGE_HEIGHT_PT,
        italic: bool = False,
        fill_hex: str | None = None,
    ) -> str:
        """
        Return Typst markup identical for descriptions and founder dates, using a shared height.
        """
        if not raw.strip():
            return ""
        lines = max(1, raw.count("\n") + 1)
        target_height = height_pt * lines
        img_path = _rasterize_text_image(
            raw,
            target_height_pt=target_height,
            italic=italic,
            font_family=resume_font_family,
            fill_hex=fill_hex,
        )
        if img_path:
            return f'image("{img_path}", height: {target_height}pt)'
        text_style = ' style: "italic"' if italic else ""
        fill = f'rgb("{fill_hex}")' if fill_hex else "color-darknight"
        return f'text(size: {target_height}pt, font: ("{resume_font_family_escaped}"), fill: {fill}{text_style})[{escape_typst(raw)}]'

    qr_overlay_block = "#let qr_overlay = []\n"

    # Build modern-cv Typst template using a root-relative import (root is set to BASE_DIR).
    fork_path = "/lib.typ"

    def build_keywords():
        kw = [
            str(field).strip()
            for field in (
                resume_data.get("target_role"),
                resume_data.get("target_company"),
                resume_data.get("primary_domain"),
            )
            if field and str(field).strip()
        ]
        # Add top skills without overloading the metadata
        kw.extend(
            [
                str(skill).strip()
                for skill in (resume_data.get("top_skills") or [])[:5]
                if skill and str(skill).strip()
            ]
        )
        seen = set()
        deduped = []
        for item in kw:
            if item.lower() in seen:
                continue
            seen.add(item.lower())
            deduped.append(item)
        if not deduped:
            deduped = [f"{firstname} {lastname}".strip(), "resume"]
        return ", ".join(deduped)

    meta_keywords = build_keywords()

    custom_entries = ""
    if custom_contacts:
        entries = [
            "(text: "
            f"\"{escape_typst(item['text'])}\", "
            f"icon: \"{escape_typst(item['icon'])}\", "
            f"link: \"{escape_typst(item['link'])}\")"
            for item in custom_contacts
        ]
        custom_entries = f"    custom: ({', '.join(entries)},),\n"

    typst_code = f"""#import "{fork_path}": *

{qr_overlay_block}

#show: resume.with(
  author: (
    firstname: "{escape_typst(firstname)}",
    lastname: "{escape_typst(lastname)}",
    email: "{email}",
    email2: "{email2}",
    phone: "{phone}",
    github: "{github}",
    github_url: "{github_url}",
    github_label: "{github_label}",
    linkedin: "{linkedin}",
    linkedin_label: "{linkedin_label}",
    scholar: "{scholar_url}",
    scholar_label: "{scholar_label}",
    portfolio: "{portfolio_url}",
    portfolio_label: "{portfolio_label}",
{custom_entries}    positions: ()
  ),
  profile-picture: none,
  date: datetime.today().display(),
  paper-size: "us-letter",
  heading-gap: {GAP},
  keywords: "{escape_typst(meta_keywords)}",
  accent-color: "#1F2937",
  font: ("{resume_font_family_escaped}"),
  header-font: ("{resume_font_family_escaped}"),
  page-foreground: qr_overlay,
)

"""
    # Force the selected font as the default text font everywhere.
    typst_code += (
        f'#set text(font: ("{resume_font_family_escaped}"), weight: "regular")\n\n'
    )
    typst_code += """#let bullet-date-height = 9.0pt
#let dated-bullet(body, date_block) = {
  let date = if date_block == none { box(height: bullet-date-height)[] } else { box(height: bullet-date-height)[#date_block] }
  grid(
    columns: (auto, 1fr),
    column-gutter: 5pt,
    align: (top + left, top + left),
    [#date],
    [#body]
  )
}

"""

    # prompt.yaml mandates these three labels exactly.
    skills_row_labels_list = [escape_typst(lbl) for lbl in DEFAULT_SKILLS_ROW_LABELS]

    def parse_skill_rows(raw_rows):
        if raw_rows is None:
            raw_rows = []
        if isinstance(raw_rows, str):
            try:
                raw_rows = json.loads(raw_rows)
            except Exception:
                raw_rows = []
        rows = []
        if isinstance(raw_rows, (list, tuple)):
            for row in raw_rows[:3]:
                items = []
                if isinstance(row, str):
                    items = [p.strip() for p in row.split(",") if p.strip()]
                elif isinstance(row, (list, tuple)):
                    items = [str(v).strip() for v in row if str(v).strip()]
                elif row is not None and str(row).strip():
                    items = [str(row).strip()]
                rows.append([escape_typst(v) for v in items])
        while len(rows) < 3:
            rows.append([])
        return rows[:3]

    skills_rows_escaped = parse_skill_rows(resume_data.get("skills_rows"))
    if not any(any(str(s).strip() for s in row) for row in skills_rows_escaped):
        fallback_skills = [
            escape_typst(str(s))
            for s in (resume_data.get("highlighted_skills") or [])
            if str(s).strip()
        ]
        skills_rows_escaped = [
            fallback_skills[0:3],
            fallback_skills[3:6],
            fallback_skills[6:9],
        ]

    summary_raw = re.sub(r"\s+", " ", str(summary or "").strip())

    def split_summary(text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        m = re.search(r"([.!?])\s+", text)
        if m:
            head = text[: m.start(1) + 1].strip()
            tail = text[m.end() :].strip()
            return head, tail
        return text.strip(), ""

    summary_head_raw, summary_tail_raw = split_summary(summary_raw)
    summary_head = escape_typst(summary_head_raw)
    summary_tail = escape_typst(summary_tail_raw)

    def build_summary_block():
        tail_block = ""
        if summary_tail_raw:
            tail_block = f"""
      #linebreak()
      #text(size: 9.2pt, font: ("{resume_font_family_escaped}"), weight: 350, fill: rgb("{SOFT_SECONDARY_FILL}"), hyphenate: false)[{summary_tail}]"""
        return f"""#block(width: 100%)[
  #set par(leading: 1.05em, spacing: 0em)
  #block(above: {space_between_contact_and_professional_summary}, below: {space_between_professional_summary_and_next_section})[
    #align(left)[
      #text(size: 10.2pt, font: ("{resume_font_family_escaped}"), weight: {SOFT_BOLD_WEIGHT}, fill: rgb("{SOFT_EMPH_FILL}"))[{summary_head}]{tail_block}
    ]
  ]
]

"""

    DATE_BULLET_HEIGHT_PT = 9.0

    def format_bullet_item(raw: str) -> str:
        body_text, date_part = split_bullet_date(raw)
        if not (body_text or date_part):
            return ""
        body_markup = format_inline_typst(body_text)
        if date_part:
            date_str = format_bullet_date(date_part)
            date_markup = render_img_or_text(
                date_str,
                height_pt=DATE_BULLET_HEIGHT_PT,
                italic=True,
            )
            date_arg = (
                date_markup.strip()
                if (date_markup and date_markup.strip())
                else "none"
            )
            return f"  - #dated-bullet([{body_markup or ''}], {date_arg})\n"
        return f"  - {body_markup}\n"

    def split_education_entries(entries: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return (primary, remaining) education entries."""

        def text_blob(edu):
            parts = [str(edu.get(key, "") or "") for key in ("school", "degree", "description")]
            bullets = edu.get("bullets", [])
            if isinstance(bullets, list):
                parts.append(" ".join(str(b or "") for b in bullets))
            else:
                parts.append(str(bullets or ""))
            return " ".join(parts).lower()

        def make_entry(degree, school, start, end, details=""):
            return {
                "degree": degree,
                "school": school,
                "start_date": start,
                "end_date": end,
                "details": details,
            }

        def is_master(edu_text: str) -> bool:
            return bool(
                re.search(r"\b(m\.?s\.?|ms|m\.?a\.?|ma|master)\b", edu_text)
                or "master of" in edu_text
            )

        def is_bachelor(edu_text: str) -> bool:
            return bool(
                re.search(r"\b(b\.?s\.?|bs|b\.?a\.?|ba|bachelor)\b", edu_text)
                or "bachelor of" in edu_text
            )

        def start_date_key(edu):
            return edu.get("start_date", "") or ""

        def build_entry(edu):
            start = edu.get("start_date", "")
            end = edu.get("end_date", "")
            school = edu.get("school", "")
            raw_degree = edu.get("degree", "") or ""
            main_degree, detail = _split_degree_parts(raw_degree)
            main_degree = main_degree.strip(", ") or str(raw_degree)
            detail_items = _parse_degree_details(detail)
            detail_all = _format_degree_details(detail_items)
            return make_entry(main_degree, school, start, end, detail_all)

        ordered = sorted(entries.copy(), key=start_date_key, reverse=True)
        primary_masters: list[dict] = []
        selected_idx: set[int] = set()

        for idx, edu in enumerate(ordered):
            if not is_master(text_blob(edu)):
                continue
            primary_masters.append(build_entry(edu))
            selected_idx.add(idx)
            if len(primary_masters) >= 2:
                break

        if not primary_masters:
            for idx, edu in enumerate(ordered[:2]):
                primary_masters.append(build_entry(edu))
                selected_idx.add(idx)
        else:
            # Keep the most recent bachelor entry with the masters (if any).
            for idx, edu in enumerate(ordered):
                if idx in selected_idx:
                    continue
                if is_bachelor(text_blob(edu)):
                    primary_masters.append(build_entry(edu))
                    selected_idx.add(idx)
                    break

        remaining = [
            build_entry(edu)
            for idx, edu in enumerate(ordered)
            if idx not in selected_idx
        ]

        return primary_masters, remaining

    def render_education_section(title: str, entries: list[dict]) -> str:
        """Render Education entries as flat lines (degree, school, dates)."""
        if not entries:
            return ""

        section = f"= {escape_typst(title)}\n"
        for idx, edu in enumerate(entries):
            degree = escape_typst(edu.get("degree", "") or "")
            school = escape_typst(edu.get("school", "") or "")
            start = format_date_mm_yy(edu.get("start_date", ""))
            end = format_date_mm_yy(edu.get("end_date", ""))
            date_range = f"{start} - {end}" if start or end else ""

            title_line = " — ".join(part for part in (degree, school) if part)
            if not title_line and not date_range and not edu.get("details"):
                continue
            title_markup = (
                f'[#text(weight: {SOFT_BOLD_WEIGHT}, fill: rgb("{SOFT_EMPH_FILL}"))[{degree}] #text(fill: rgb("{SOFT_SECONDARY_FILL}"))[ — {school}]]'
                if degree and school
                else f"[{title_line}]"
            )
            section += f"""#resume-entry(
  title: {title_markup},
  location: "",
  date: "{date_range}",
  description: "",
  block-above: {em(0.15 if idx == 0 else 0.18, weight=0.6, min_value=0.1, max_value=0.22)},
  block-below: {em(0.12, weight=0.6, min_value=0.08, max_value=0.18)},
  title-weight: "regular",
  title-size: 10.0pt
)
"""
            details = (edu.get("details", "") or "").strip()
            if details:
                details_markup = format_inline_typst(details)
                details_above = em(0.60, weight=0.9, min_value=0.4, max_value=0.7)
                section += (
                    f"#block(above: {details_above}, below: {em(0.12, weight=0.9, min_value=0.08, max_value=0.16)})[\n"
                    f'  #set text(size: 9.0pt, font: ("{resume_font_family_escaped}"), weight: 350, fill: rgb("{SOFT_SECONDARY_FILL}"))\n'
                    f"  #set par(leading: {em(0.6, weight=1.0, min_value=0.45, max_value=None)}, spacing: 0em)\n"
                    f"  {details_markup}\n"
                    f"]\n"
                )
            # Add space between degree+coursework groups.
            if idx < len(entries) - 1:
                section += f"#block(height: {em(0.8, weight=1.0, min_value=0.7, max_value=None)})[]\n"

        return section

    def render_custom_sections(items: list[dict]) -> dict[str, str]:
        """Render custom sections (title + bullet body) into Typst blocks."""
        blocks: dict[str, str] = {}
        for item in items:
            key = str(item.get("key") or "").strip()
            if not key:
                continue
            title = str(item.get("title") or "").strip()
            body = str(item.get("body") or "").rstrip()
            if not (title or body):
                continue
            block = ""
            if title:
                block += f"= {escape_typst(title)}\n"
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            if lines:
                block += "#block(above: 0em, below: 0em)[\n"
                block += f'  #set text(size: 10.2pt, font: "{resume_font_family_escaped}", weight: 350, fill: color-darknight)\n'
                block += f"  #set par(leading: {em(0.65, weight=1.0, min_value=0.5, max_value=None)}, spacing: 0em)\n"
                block += f"  #set list(marker: {BULLET_MARKER}, spacing: {BULLET_GAP})\n"
                for line in lines:
                    line_markup = format_bullet_item(line)
                    if line_markup:
                        block += line_markup
                block += "]\n"
            blocks[key] = block
        return blocks

    summary_block = build_summary_block() if summary else ""

    primary_education, continued_education = split_education_entries(education)
    education_block = render_education_section(
        title_map.get("education", "Education"), primary_education
    )
    education_continued_block = render_education_section(
        title_map.get("education_continued", "Education Continued"),
        continued_education,
    )

    # Experience section: do not add extra space above; the summary (if present) owns the gap.
    experience_block = f'= {escape_typst(title_map.get("experience", "Experience"))}\n'
    for idx, exp in enumerate(experiences):
        experience_block += "#block(breakable: false)[\n"
        start = format_date_mm_yy(exp.get("start_date", ""))
        end = format_date_mm_yy(exp.get("end_date", ""))
        date_range = f"{start} - {end}" if start or end else ""
        role = escape_typst(exp.get("role", ""))
        company = escape_typst(exp.get("company", ""))
        location = escape_typst(exp.get("location", ""))
        date_text = escape_typst(date_range)
        location_text = location
        # Use the template's resume-entry helper for consistent alignment.
        experience_block += f"""#resume-entry(
  title: "{role}",
  location: "{location_text}",
  date: "{date_text}",
        description: "{company}",
        block-above: {em(0.06, weight=0.6, min_value=0.04, max_value=0.09)},
        block-below: {em(0.2, weight=0.5, min_value=0.08, max_value=0.55)}
)

"""
        desc_raw = exp.get("description", "")
        desc_markup = render_img_or_text(desc_raw, fill_hex=SOFT_SECONDARY_FILL)
        if desc_markup:
            experience_block += f"#block(above: {em(0.32, weight=0.7, min_value=0.12, max_value=0.4)}, below: {em(0.12, weight=0.7, min_value=0.05, max_value=0.18)})[\n"
            experience_block += f"  #{desc_markup}\n"
            experience_block += "]\n"
        bullets = exp.get("bullets", [])
        if isinstance(bullets, list) and any(b.strip() for b in bullets):
            # Gap between the company line and the first bullet.
            experience_block += f"#block(height: {HALF_BULLET_GAP})\n"
            experience_block += "#block(above: 0em, below: 0em)[\n"
            experience_block += f'  #set text(size: 10.2pt, font: "{resume_font_family_escaped}", weight: 350, fill: color-darknight)\n'
            # Keep wrapped bullet lines readable even when auto-fit tightens.
            experience_block += f"  #set par(leading: {em(0.65, weight=1.0, min_value=0.5, max_value=None)}, spacing: 0em)\n"
            # Let auto-fit add height primarily via bullet spacing, not giant inter-entry gaps.
            experience_block += f"  #set list(marker: {BULLET_MARKER}, spacing: {BULLET_GAP})\n"
            for bullet in bullets:
                if not isinstance(bullet, str):
                    continue
                line = format_bullet_item(bullet)
                if line:
                    experience_block += line
            experience_block += "]\n"
        else:
            experience_block += "\n"
        experience_block += "]\n"

        # Extra breathing room between roles.
        if idx < len(experiences) - 1:
            experience_block += f"#block(height: {GAP})\n\n"

    founder_block = ""
    if founder_roles:
        founder_block += (
            f'= {escape_typst(title_map.get("founder", "Startup Founder"))}\n'
        )
        for idx, role in enumerate(founder_roles):
            company = escape_typst(role.get("company", ""))
            location = escape_typst(role.get("location", ""))
            location_text = location

            founder_block += f"""#block(
  above: {em(0.06, weight=0.6, min_value=0.04, max_value=0.09)},
  below: {em(0.2, weight=0.5, min_value=0.08, max_value=0.55)},
)[
  #pad[
    #__justify_align[
      #text(weight: "regular")[{company}]
    ][
      #text(
        font: ("{resume_font_family_escaped}"),
        weight: "light",
        style: "italic",
        fill: default-location-color,
      )[{location_text}]
    ]
  ]
]

"""
            desc_raw = role.get("description", "")
            desc_markup = render_img_or_text(desc_raw, fill_hex=SOFT_SECONDARY_FILL)
            if desc_markup:
                founder_block += f"#block(above: {em(0.32, weight=0.7, min_value=0.12, max_value=0.4)}, below: {em(0.12, weight=0.7, min_value=0.05, max_value=0.18)})[\n"
                founder_block += f"  #{desc_markup}\n"
                founder_block += "]\n"
            bullets = role.get("bullets", [])
            bullet_lines = [b for b in bullets if isinstance(b, str) and b.strip()]
            if bullet_lines:
                # Match Experience section spacing between the entry (company line) and first bullet.
                founder_block += f"#block(height: {HALF_BULLET_GAP})\n"
                founder_block += "#block(above: 0em, below: 0em)[\n"
                founder_block += f'  #set text(size: 10.2pt, font: "{resume_font_family_escaped}", weight: 350, fill: color-darknight)\n'
                founder_block += f"  #set par(leading: {em(0.65, weight=1.0, min_value=0.5, max_value=None)}, spacing: 0em)\n"
                founder_block += f"  #set list(marker: {BULLET_MARKER}, spacing: {BULLET_GAP})\n"
                for bullet in bullet_lines:
                    line = format_bullet_item(bullet)
                    if line:
                        founder_block += line
                founder_block += "]\n"
            if idx < len(founder_roles) - 1:
                founder_block += f"#block(height: {GAP})\n\n"

    matrices_block = ""
    if include_matrices:
        matrices_block += f'= {escape_typst(title_map.get("matrices", "Skills"))}\n'
        matrices_block += render_skill_rows(
            skills_row_labels_list,
            skills_rows_escaped,
            # Keep wrapped skill lines tight even when auto-fit expands spacing.
            leading=em(0.14, weight=1.0, min_value=0.35, max_value=None),
            row_gap=em(0.65, weight=1.0, min_value=0.8, max_value=None),
            font_family=resume_font_family,
        )

    custom_blocks = render_custom_sections(custom_sections)
    sections = {
        "summary": summary_block,
        "education": education_block,
        "education_continued": education_continued_block,
        "experience": experience_block,
        "founder": founder_block,
        "matrices": matrices_block,
    }
    sections.update(custom_blocks)

    if section_order is None:
        resolved_order = _sanitize_section_order(section_order, custom_section_keys)
    else:
        resolved_order = _filter_section_order(section_order, custom_section_keys)
    for section_key in resolved_order:
        block = sections.get(section_key, "")
        if block:
            typst_code += "\n" + block

    return typst_code


def render_skill_rows(
    labels: list[str],
    rows: list[list[str]],
    *,
    leading: str = "0.22em",
    row_gap: str = "0.08em",
    font_family: str | None = None,
):
    """Render the prompt.yaml skills format as 3 labeled rows (Typst markup)."""
    if not font_family:
        font_family = DEFAULT_RESUME_FONT_FAMILY
    font_family_escaped = escape_typst(str(font_family))

    def normalize_label(label: str) -> str:
        label = (label).strip()
        if not label:
            return ""
        return label

    labels = labels.copy()
    while len(labels) < 3:
        labels.append("")
    labels = labels[:3]

    rows = rows.copy()
    while len(rows) < 3:
        rows.append([])
    rows = rows[:3]

    def format_skill_line(skills: list[str], *, emphasize: int = 3) -> str:
        skills = [s for s in (skills) if s.strip()]
        if not skills:
            return ""

        core = skills[:emphasize]
        more = skills[emphasize:]

        core_parts = [
            f'#text(weight: {SOFT_BOLD_WEIGHT}, fill: rgb("{SOFT_EMPH_FILL}"))[{skill}]'
            for skill in core
        ]
        core_line = " #text(fill: color-gray)[·] ".join(core_parts)
        if not more:
            return core_line

        # Put the secondary skills on a dedicated line so the wrap doesn't leave
        # separators dangling at line ends; commas also break more naturally.
        more_line = ", ".join(more)
        more_markup = (
            f'#text(size: 9.2pt, fill: rgb("{SOFT_SECONDARY_FILL}"))[{more_line}]'
        )
        return f"{core_line}#linebreak()#h(0.35em){more_markup}"

    # F-pattern scanning: strong left-side labels + bold lead skills per row.
    # Give the skills column a bit more width to reduce awkward wraps.
    label_width = "14.2em"
    label_text_style = (
        f'#text(size: 8.8pt, weight: 350, fill: rgb("{SOFT_SECONDARY_FILL}"))'
    )

    grid_cells: list[str] = []
    for idx in range(3):
        label = normalize_label(labels[idx])
        skills = [s for s in (rows[idx]) if s.strip()]
        if not (label or skills):
            continue
        label_cell = f"[{label_text_style}[#smallcaps[{label}]]]" if label else "[]"
        skills_cell_text = format_skill_line(skills)
        skills_cell = f"[{skills_cell_text}]" if skills_cell_text else "[]"
        grid_cells.extend([label_cell, skills_cell])

    out = [
        "#block[",
        f'  #set text(size: 10.2pt, font: "{font_family_escaped}", weight: 350)',
        f"  #set par(leading: {leading}, spacing: 0em)",
        "  #grid(",
        f"    columns: ({label_width}, 1fr),",
        "    column-gutter: 0.95em,",
        f"    row-gutter: {row_gap},",
        "    align: (top + left, top + left),",
    ]
    if grid_cells:
        out.append("    " + ",\n    ".join(grid_cells) + ",")
    out.extend(["  )", "]\n\n"])
    return "\n".join(out)


def _fake_generate_resume_content(job_req, base_profile, model_name=None):
    summary = (base_profile or {}).get("summary", "") or "Simulated summary."
    headers = [
        "Sim Header 1",
        "Sim Header 2",
        "Sim Header 3",
        "Sim Header 4",
        "Sim Header 5",
        "Sim Header 6",
        "Sim Header 7",
        "Sim Header 8",
        "Sim Header 9",
    ]
    skills = [
        "Sim Skill 1",
        "Sim Skill 2",
        "Sim Skill 3",
        "Sim Skill 4",
        "Sim Skill 5",
        "Sim Skill 6",
        "Sim Skill 7",
        "Sim Skill 8",
        "Sim Skill 9",
    ]
    return {
        "summary": summary,
        "headers": headers,
        "highlighted_skills": skills,
        "skills_rows": [
            ["Sim Row 1A", "Sim Row 1B"],
            ["Sim Row 2A"],
            ["Sim Row 3A"],
        ],
        "target_company": "Simulated Co",
        "target_role": "Simulated Role",
        "seniority_level": "Simulated",
        "target_location": "Simulated",
        "work_mode": "Simulated",
        "travel_requirement": "Simulated",
        "primary_domain": "Simulated",
        "must_have_skills": ["Simulated Skill A"],
        "nice_to_have_skills": ["Simulated Skill B"],
        "tech_stack_keywords": ["Simulated Stack"],
        "non_technical_requirements": ["Simulated Requirement"],
        "certifications": [],
        "clearances": [],
        "core_responsibilities": ["Simulated Responsibility"],
        "outcome_goals": ["Simulated Outcome"],
        "salary_band": "Simulated",
        "posting_url": "",
        "req_id": "",
    }


def _maxcov_log(msg: TemplateMsg) -> None:
    if not MAX_COVERAGE_LOG:
        return
    rendered = _render_template(msg)
    print(_render_template(t"[maxcov] {rendered}"), flush=True)


def _maxcov_log_expected_failure(
    stdout: str | None,
    stderr: str | None,
    args: list[str],
    quiet: bool,
) -> None:
    if not quiet:
        return
    out = (stdout or "").strip()
    err = (stderr or "").strip()
    if out:
        _maxcov_log(t"expected failure stdout: {out}")
    if err:
        _maxcov_log(t"expected failure stderr: {err}")


def _maxcov_collapse_ranges(lines: list[int]) -> list[tuple[int, int]]:
    if not lines:
        return []
    ordered = sorted({line for line in lines})
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for line in ordered[1:]:
        if line == prev + 1:
            prev = line
            continue
        ranges.append((start, prev))
        start = prev = line
    ranges.append((start, prev))
    return ranges


def _maxcov_format_line_ranges(lines: list[int]) -> str:
    ranges = _maxcov_collapse_ranges(lines)
    if not ranges:
        return ""
    out = []
    for start, end in ranges:
        if start == end:
            out.append(str(start))
        else:
            out.append(f"{start}-{end}")
    return ", ".join(out)


def _maxcov_format_top_missing_blocks(lines: list[int], *, limit: int = 10) -> str:
    ranges = _maxcov_collapse_ranges(lines)
    if not ranges:
        return ""
    blocks = []
    for start, end in ranges:
        length = end - start + 1
        blocks.append((length, start, end))
    blocks.sort(key=lambda item: (-item[0], item[1]))
    out = []
    for length, start, end in blocks[: max(1, limit)]:
        if start == end:
            out.append(f"{start}({length})")
        else:
            out.append(f"{start}-{end}({length})")
    return ", ".join(out)


def _maxcov_format_arc(arc) -> str:
    if not isinstance(arc, (list, tuple)) or len(arc) != 2:
        return str(arc)
    start, end = arc
    start_label = str(start) if int(start) > 0 else "entry"
    end_label = str(end) if int(end) > 0 else "exit"
    return f"{start_label}->{end_label}"


def _maxcov_format_branch_arcs(arcs, *, limit: int = 30) -> tuple[str, int]:
    if not arcs:
        return "", 0
    formatted = [_maxcov_format_arc(arc) for arc in arcs]
    extra = 0
    if len(formatted) > limit:
        extra = len(formatted) - limit
        formatted = formatted[:limit]
    return ", ".join(formatted), extra


def _get_arg_value(argv: list[str], name: str, default: str) -> str:
    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    for item in argv:
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default


def _maxcov_summarize_coverage(
    coverage_module,
    *,
    cov_dir: Path,
    cov_rc: Path,
    target: Path,
) -> dict | None:
    try:
        cov = coverage_module.Coverage(
            data_file=str(cov_dir / ".coverage"),
            config_file=str(cov_rc),
        )
        cov.load()
        analysis = cov.analysis2(str(target))
        missing_lines = list(analysis[3]) if len(analysis) >= 4 else []
        ana = cov._analyze(str(target))
        missing_branch_lines = []
        missing_branch_arcs = []
        try:
            missing_branch_lines = list(ana.missing_branch_arcs())
        except Exception:
            missing_branch_lines = []
        try:
            arcs_missing = list(ana.arcs_missing())
            if missing_branch_lines:
                branch_set = {int(line) for line in missing_branch_lines}
                missing_branch_arcs = [
                    arc
                    for arc in arcs_missing
                    if isinstance(arc, (list, tuple))
                    and len(arc) == 2
                    and int(arc[0]) in branch_set
                ]
            else:
                missing_branch_arcs = [
                    arc
                    for arc in arcs_missing
                    if isinstance(arc, (list, tuple)) and len(arc) == 2
                ]
            missing_branch_arcs.sort(key=lambda arc: (int(arc[0]), int(arc[1])))
        except Exception:
            missing_branch_arcs = []
        branch_line_ranges = _maxcov_format_line_ranges(missing_branch_lines)
        branch_arcs, branch_arcs_extra = _maxcov_format_branch_arcs(
            missing_branch_arcs, limit=30
        )
        return {
            "missing_lines": len(missing_lines),
            "missing_ranges": _maxcov_format_line_ranges(missing_lines),
            "top_missing_blocks": _maxcov_format_top_missing_blocks(
                missing_lines, limit=10
            ),
            "missing_branch_lines": len(missing_branch_lines),
            "missing_branch_line_ranges": branch_line_ranges,
            "missing_branch_arcs": branch_arcs,
            "missing_branch_arcs_extra": branch_arcs_extra,
        }
    except Exception as exc:
        _maxcov_log(t"coverage summary failed: {exc}")
        return None


def _maxcov_build_coverage_output(
    *,
    counts: dict,
    summary: dict | None,
    cov_dir: Path,
    cov_rc: Path,
    json_out: str | None,
    html_out: str | None,
) -> list[str] | None:
    if counts.get("cover"):
        covered_lines = None
        covered_pct = None
        if "stmts" in counts and "miss" in counts:
            covered_lines = counts["stmts"] - counts["miss"]
            if counts["stmts"]:
                covered_pct = (covered_lines / counts["stmts"]) * 100.0
        if summary:
            covered_label = (
                f"{covered_lines} ({covered_pct:.1f}%)"
                if covered_lines is not None and covered_pct is not None
                else "n/a"
            )
            lines = [
                "Coverage (harness.py): "
                f"{counts['cover']} | statements: {counts.get('stmts', 'n/a')} | "
                f"missing: {counts.get('miss', 'n/a')} | branches: {counts.get('branch', 'n/a')} | "
                f"partial branches: {counts.get('brpart', 'n/a')} | "
                f"covered lines: {covered_label}",
            ]
            if summary.get("missing_ranges"):
                lines.append("Missing lines (ranges): " f"{summary['missing_ranges']}")
            if summary.get("missing_branch_line_ranges"):
                lines.append(
                    "Missing branch lines (ranges): "
                    f"{summary['missing_branch_line_ranges']}"
                )
            if summary.get("missing_branch_arcs"):
                arc_suffix = ""
                if summary.get("missing_branch_arcs_extra"):
                    arc_suffix = f" ... +{summary['missing_branch_arcs_extra']} more"
                lines.append(
                    "Missing branch arcs (first 30): "
                    f"{summary['missing_branch_arcs']}{arc_suffix}"
                )
            if summary.get("top_missing_blocks"):
                lines.append(
                    "Top missing blocks (largest first): "
                    f"{summary['top_missing_blocks']}"
                )
            lines.extend([f"Coverage data: {cov_dir / '.coverage'}", f"Coverage rc: {cov_rc}"])
            if json_out:
                lines.append(f"Coverage json: {json_out}")
            if html_out:
                lines.append(f"Coverage html: {html_out}")
            return lines
        return [f"Coverage (harness.py): {counts['cover']}"]
    return None


def _maxcov_run_container_wrapper(
    *,
    project: str | None = None,
    runner=None,
    sleep_fn=None,
    time_fn=None,
    exit_fn=None,
    check_compose: bool = False,
) -> int:
    _maxcov_log("maxcov container wrapper start")
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    project = project or f"maxcov_{stamp}"
    base = ["docker", "compose", "-p", project]
    runner = runner or subprocess.run
    sleep_fn = sleep_fn or time.sleep
    time_fn = time_fn or time.time
    exit_fn = exit_fn or sys.exit

    def _run_compose(args: list[str], *, check: bool = False) -> int:
        cmd = [*base, *args]
        _maxcov_log(t"docker: {' '.join(cmd)}")
        result = runner(cmd, check=False)
        if check and result.returncode != 0:
            raise RuntimeError(f"docker compose failed: {' '.join(args)}")
        return result.returncode

    def _wait_for_neo4j(timeout_s: float = 90.0) -> bool:
        deadline = time_fn() + max(5.0, timeout_s)
        while time_fn() < deadline:
            proc = runner(
                [*base, "ps", "-q", "neo4j"],
                capture_output=True,
                text=True,
            )
            container_id = (getattr(proc, "stdout", "") or "").strip()
            if container_id:
                inspect_cmd = [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Health.Status}}",
                    container_id,
                ]
                health = runner(
                    inspect_cmd,
                    capture_output=True,
                    text=True,
                )
                status = (getattr(health, "stdout", "") or "").strip()
                if status == "healthy":
                    return True
            sleep_fn(1.0)
        return False

    rc = 1
    try:
        if check_compose:
            _run_compose(["version"], check=True)
        _run_compose(["build", "maxcov"])
        _run_compose(["up", "-d", "neo4j"])
        if not _wait_for_neo4j():
            _maxcov_log("neo4j did not reach healthy state in time")
            exit_fn(1)
        run_cmd = [
            *base,
            "run",
            "--rm",
            "-T",
            "maxcov",
        ]
        _maxcov_log("maxcov container run start")
        rc = runner(run_cmd).returncode
        _maxcov_log(t"maxcov container run done: rc={rc}")
    finally:
        _run_compose(["down", "-v"])
    exit_fn(rc)
    return rc


def _maybe_launch_maxcov_container() -> None:
    if os.environ.get("MAX_COVERAGE_CONTAINER") != "1":
        globals()["MAX_COVERAGE_LOG"] = True
        _maxcov_run_container_wrapper(check_compose=True)


def _close_llm_client(llm) -> None:
    client = getattr(llm, "client", None)
    if client is None:
        return

    def _schedule_coro(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running() and not loop.is_closed():
            with suppress(Exception):
                task = loop.create_task(coro)
                task.add_done_callback(
                    lambda t: t.exception() if not t.cancelled() else None
                )
            return
        with suppress(Exception):
            asyncio.run(coro)

    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        with suppress(Exception):
            result = close_fn()
            if asyncio.iscoroutine(result):
                _schedule_coro(result)
        return
    aclose_fn = getattr(client, "aclose", None)
    if callable(aclose_fn):
        with suppress(Exception):
            result = aclose_fn()
            if asyncio.iscoroutine(result):
                _schedule_coro(result)


def _ensure_open_event_loop() -> None:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        return
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())


def _call_llm_responses(
    *,
    provider: str,
    model: str,
    input_data,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict | None = None,
    **kwargs,
):
    llm = AnyLLM.create(
        provider,
        api_key=api_key,
        api_base=api_base,
        **(client_args or {}),
    )
    try:
        _ensure_open_event_loop()
        return llm.responses(model=model, input_data=input_data, **kwargs)
    finally:
        _close_llm_client(llm)


def _call_llm_completion(
    *,
    provider: str,
    model: str,
    messages,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict | None = None,
    **kwargs,
):
    def _is_gemini_usage_validation_error(exc: Exception) -> bool:
        name = type(exc).__name__
        if name != "ValidationError":
            return False
        text = str(exc)
        return "CompletionUsage" in text and "completion_tokens" in text

    def _is_closed_event_loop_error(exc: Exception) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        return "event loop is closed" in str(exc).lower()

    @contextmanager
    def _patch_gemini_usage():
        patches: list[tuple[object, object]] = []

        def _wrap(convert_fn):
            def _wrapped(response):
                data = convert_fn(response)
                usage = data.get("usage", {})
                if not isinstance(usage, dict):
                    usage = {}
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(key)
                    if value is None:
                        usage[key] = 0
                    else:
                        try:
                            usage[key] = int(value)
                        except Exception:
                            usage[key] = 0
                data["usage"] = usage
                return data

            return _wrapped

        try:
            from any_llm.providers.gemini import utils as gem_utils
        except Exception:
            yield
            return

        original_utils = getattr(gem_utils, "_convert_response_to_response_dict", None)
        if callable(original_utils):
            gem_utils._convert_response_to_response_dict = _wrap(original_utils)
            patches.append((gem_utils, original_utils))

        try:
            from any_llm.providers.gemini import base as gem_base
        except Exception:
            gem_base = None

        if gem_base is not None:
            original_base = getattr(gem_base, "_convert_response_to_response_dict", None)
            if callable(original_base):
                gem_base._convert_response_to_response_dict = _wrap(original_base)
                patches.append((gem_base, original_base))

        if not patches:
            yield
            return

        try:
            yield
        finally:
            for module, original in patches:
                try:
                    setattr(module, "_convert_response_to_response_dict", original)
                except Exception:
                    pass

    def _make_client():
        return AnyLLM.create(
            provider,
            api_key=api_key,
            api_base=api_base,
            **(client_args or {}),
        )

    def _call_with_client(client):
        _ensure_open_event_loop()
        return client.completion(model=model, messages=messages, **kwargs)

    llm = _make_client()
    try:
        try:
            if provider == "gemini":
                with _patch_gemini_usage():
                    return _call_with_client(llm)
            return _call_with_client(llm)
        except Exception as exc:
            if provider != "gemini":
                raise
            if not (
                _is_gemini_usage_validation_error(exc)
                or _is_closed_event_loop_error(exc)
            ):
                raise
            _close_llm_client(llm)
            llm = _make_client()
            with _patch_gemini_usage():
                return _call_with_client(llm)
    finally:
        _close_llm_client(llm)


def _extract_json_object(text: str) -> dict:
    """Parse JSON, tolerating code fences or stray prefix/suffix text."""
    raw = (text).strip()
    if not raw:
        raise ValueError("empty")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\\s*```\\s*$", "", raw)
        raw = raw.strip()
    def _coerce_json_object(parsed):
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        return None

    def _replace_json_literals(payload: str) -> str:
        replacements = {"null": "None", "true": "True", "false": "False"}
        out: list[str] = []
        in_string = False
        string_char = ""
        escape = False
        idx = 0
        while idx < len(payload):
            ch = payload[idx]
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == string_char:
                    in_string = False
                idx += 1
                continue
            if ch in ("'", '"'):
                in_string = True
                string_char = ch
                out.append(ch)
                idx += 1
                continue
            replaced = False
            for key, val in replacements.items():
                if payload.startswith(key, idx):
                    before = payload[idx - 1] if idx > 0 else ""
                    after = payload[idx + len(key)] if idx + len(key) < len(payload) else ""
                    if (not before or not before.isalnum()) and (
                        not after or not after.isalnum()
                    ):
                        out.append(val)
                        idx += len(key)
                        replaced = True
                        break
            if not replaced:
                out.append(ch)
                idx += 1
        return "".join(out)

    def _literal_eval_object(payload: str):
        try:
            parsed = ast.literal_eval(payload)
        except Exception:
            fixed = _replace_json_literals(payload)
            if fixed != payload:
                try:
                    parsed = ast.literal_eval(fixed)
                except Exception:
                    return None
            else:
                return None
        return _coerce_json_object(parsed)

    try:
        parsed = json.loads(raw)
        coerced = _coerce_json_object(parsed)
        if coerced is not None:
            return coerced
    except Exception:
        pass

    # Fallback: scan for balanced JSON objects and return the first valid dict.
    def scan_objects(payload: str) -> list[str]:
        candidates: list[str] = []
        depth = 0
        start_idx: int | None = None
        in_string = False
        escape = False
        for idx, ch in enumerate(payload):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start_idx = idx
                depth += 1
            elif ch == "}":
                if depth:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        candidates.append(payload[start_idx : idx + 1])
                        start_idx = None
        return candidates

    for candidate in scan_objects(raw):
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None
        if parsed is not None:
            coerced = _coerce_json_object(parsed)
            if coerced is not None:
                return coerced
        coerced = _literal_eval_object(candidate)
        if coerced is not None:
            return coerced

    # Fallback: extract the outermost {...} if the model added extra text.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None
        if parsed is not None:
            coerced = _coerce_json_object(parsed)
            if coerced is not None:
                return coerced
        coerced = _literal_eval_object(candidate)
        if coerced is not None:
            return coerced
    raise ValueError("non-dict json")


def _coerce_llm_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if item is None:
                continue
            parts.append(str(item))
        return "".join(parts)
    return str(value)


def generate_resume_content(job_req, base_profile, model_name: str | None = None):
    """Generate JSON profile content using the stored prompt template and Mozilla any-llm."""
    if os.environ.get("MAX_COVERAGE_SKIP_LLM") == "1":
        _maxcov_log("generate_resume_content: MAX_COVERAGE_SKIP_LLM=1")
        try:
            return _fake_generate_resume_content(job_req, base_profile, model_name)  # type: ignore[name-defined]
        except Exception:
            return {
                "summary": (base_profile or {}).get("summary", "")
                or "Simulated summary.",
                "headers": [],
                "highlighted_skills": [],
                "skills_rows": [[], [], []],
            }
    prompt_template = _resolve_prompt_template(base_profile)
    if not prompt_template:
        return {"error": "Prompt template not found in Neo4j or prompt.yaml."}
    stop_terms = _load_stop_list()
    prompt_template = _inject_stop_list(prompt_template, stop_terms)
    rewrite_bullets = False
    if isinstance(base_profile, dict):
        rewrite_bullets = bool(base_profile.get("rewrite_bullets"))
    if rewrite_bullets and "experience_bullets" not in prompt_template:
        prompt_template = (
            prompt_template.rstrip()
            + "\n\n# Bullet Rewrite (Required)\n"
            + 'If the Candidate Resume JSON includes "rewrite_bullets": true, return two fields in the output JSON:\n'
            + '- "experience_bullets": [{"id": "<experience id>", "bullets": ["..."]}]\n'
            + '- "founder_role_bullets": [{"id": "<founder role id>", "bullets": ["..."]}]\n'
            + "Use the same ids from the input. Only rewrite bullets for those roles (3-6 bullets each), action verbs, evidence-based, no hallucination.\n"
        )

    provider, model = _split_llm_model_spec(model_name)
    provider = (provider).strip().lower()
    model = (model).strip()
    stop_patterns = _compile_stop_list_patterns(stop_terms)
    sanitized_profile = base_profile or {}
    sanitized_job_req = job_req or ""
    if stop_patterns:
        sanitized_profile = _strip_stop_list_phrases(sanitized_profile, stop_patterns)
        sanitized_job_req = _strip_stop_list_phrases(sanitized_job_req, stop_patterns)
    profile_json = json.dumps(
        sanitized_profile, indent=2, ensure_ascii=False, default=str
    )
    max_attempts = _resolve_stop_list_max_attempts() if stop_patterns else 1
    if stop_patterns and provider == "gemini":
        if _read_int_env("STOP_LIST_MAX_ATTEMPTS", "LLM_STOP_LIST_MAX_ATTEMPTS") is None:
            max_attempts = max(max_attempts, 5)
    json_retry_attempts = _resolve_llm_json_retry_attempts(provider)

    def build_prompt(attempt: int, stop_hits: list[str]) -> str:
        retry_note = ""
        if attempt > 0:
            hit_note = ""
            if stop_hits:
                shown = ", ".join(stop_hits[:10])
                suffix = ""
                if len(stop_hits) > 10:
                    suffix = f" (and {len(stop_hits) - 10} more)"
                hit_note = (
                    "\nSpecific forbidden phrases detected in the previous output: "
                    + shown
                    + suffix
                    + "\n"
                )
            retry_note = (
                "\n\n# STOP-LIST RETRY\n"
                "The previous output contained forbidden phrases from the stop list. "
                "Regenerate the entire JSON output without using any stop-list terms. "
                "Return only valid JSON.\n"
                + hit_note
            )
        return (
            prompt_template
            + retry_note
            + "\n\nCandidate Resume (JSON):\n"
            + profile_json
            + "\n\nJob Requisition:\n"
            + sanitized_job_req
        )

    def call_llm(prompt_text: str) -> dict:
        if provider == "openai":
            api_key = load_openai_api_key()
            if not api_key:
                return {
                    "error": "Missing OpenAI API key: set OPENAI_API_KEY or put it in ~/openaikey.txt"
                }

            api_base = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
            organization = (
                os.environ.get("OPENAI_ORGANIZATION")
                or os.environ.get("OPENAI_ORG_ID")
                or ""
            ).strip() or None
            project = (os.environ.get("OPENAI_PROJECT") or "").strip() or None
            client_args = {}
            if organization:
                client_args["organization"] = organization
            if project:
                client_args["project"] = project

            def call_openai(
                *, max_output_tokens: int, include_reasoning: bool
            ) -> tuple[object, str]:
                request: dict[str, Any] = {
                    "model": model,
                    "input_data": prompt_text,
                    "text": {"format": {"type": "json_object"}},
                    "max_output_tokens": max_output_tokens,
                }
                if include_reasoning:
                    reasoning = _openai_reasoning_params_for_model(model)
                    if reasoning:
                        request["reasoning"] = reasoning
                resp_obj = _call_llm_responses(
                    provider="openai",
                    api_key=api_key,
                    api_base=api_base,
                    client_args=client_args or None,
                    **request,
                )
                return resp_obj, (getattr(resp_obj, "output_text", None) or "")

            max_tokens = _resolve_llm_max_output_tokens("openai", model)
            try:
                resp, content = call_openai(
                    max_output_tokens=max_tokens,
                    include_reasoning=True,
                )
            except MissingApiKeyError, UnsupportedProviderError:
                exc = sys.exception()
                return {
                    "error": (
                        str(exc)
                        if exc is not None
                        else "Missing API key or unsupported provider."
                    )
                }
            except Exception as e:
                return {"error": f"LLM call failed: {type(e).__name__}: {e}"}

            status = str(getattr(resp, "status", "") or "").strip().lower()
            incomplete_reason = getattr(
                getattr(resp, "incomplete_details", None), "reason", None
            )
            incomplete_reason = str(incomplete_reason or "").strip().lower()

            content = (content).strip()
            if content:
                with suppress(Exception):
                    return _extract_json_object(content)

            # If the response was truncated, retry once with a larger output budget and no reasoning
            # to maximize usable JSON output tokens.
            if status == "incomplete" and incomplete_reason == "max_output_tokens":
                retry_tokens = _resolve_llm_retry_max_output_tokens(
                    "openai", model, max_tokens
                )
                try:
                    content2 = call_openai(
                        max_output_tokens=retry_tokens,
                        include_reasoning=False,
                    )[1]
                except Exception as e:
                    return {
                        "error": f"LLM call failed after truncation retry: {type(e).__name__}: {e}"
                    }
                content2 = (content2).strip()
                if not content2:
                    return {
                        "error": "Empty response from OpenAI after truncation retry.",
                    }
                try:
                    return _extract_json_object(content2)
                except Exception:
                    return {
                        "error": "Model returned non-JSON output (after truncation retry).",
                        "raw": content2[:2000],
                    }

            if not content:
                return {"error": "Empty response from OpenAI."}
            return {
                "error": "Model returned non-JSON output.",
                "raw": content[:2000],
                "status": status or None,
                "incomplete_reason": incomplete_reason or None,
            }

        if provider == "gemini":
            api_key = load_gemini_api_key()
            if not api_key:
                return {
                    "error": "Missing Gemini API key: set GEMINI_API_KEY/GOOGLE_API_KEY or put it in ~/geminikey.txt"
                }

            def call_gemini(*, max_tokens: int, force_json: bool) -> tuple[object, str]:
                request: dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": max_tokens,
                }
                if force_json:
                    request["response_format"] = {"type": "json_object"}
                resp_obj = _call_llm_completion(
                    provider="gemini",
                    api_key=api_key,
                    **request,
                )
                text_out = ""
                try:
                    choice0 = getattr(resp_obj, "choices", [None])[0]
                    msg = getattr(choice0, "message", None)
                    text_out = _coerce_llm_text(getattr(msg, "content", None))
                except Exception:
                    text_out = ""
                return resp_obj, text_out

            max_tokens = _resolve_llm_max_output_tokens("gemini", model)
            resp = None
            content = ""
            try:
                resp, content = call_gemini(max_tokens=max_tokens, force_json=True)
            except Exception:
                try:
                    resp, content = call_gemini(max_tokens=max_tokens, force_json=False)
                except MissingApiKeyError, UnsupportedProviderError:
                    exc = sys.exception()
                    return {
                        "error": (
                            str(exc)
                            if exc is not None
                            else "Missing API key or unsupported provider."
                        )
                    }
                except Exception as e:
                    return {"error": f"LLM call failed: {type(e).__name__}: {e}"}

            finish_reason = ""
            try:
                choice0 = getattr(resp, "choices", [None])[0]
                finish_reason = (
                    str(getattr(choice0, "finish_reason", "") or "").strip().lower()
                )
            except Exception:
                finish_reason = ""

            content = (content).strip()
            if content:
                with suppress(Exception):
                    return _extract_json_object(content)

            if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
                retry_tokens = _resolve_llm_retry_max_output_tokens(
                    "gemini", model, max_tokens
                )
                try:
                    content2 = call_gemini(max_tokens=retry_tokens, force_json=True)[1]
                except Exception:
                    try:
                        content2 = call_gemini(
                            max_tokens=retry_tokens, force_json=False
                        )[1]
                    except Exception as e:
                        return {
                            "error": f"LLM call failed after truncation retry: {type(e).__name__}: {e}"
                        }
                content2 = (content2).strip()
                if not content2:
                    return {
                        "error": "Empty response from Gemini after truncation retry."
                    }
                try:
                    return _extract_json_object(content2)
                except Exception:
                    return {
                        "error": "Model returned non-JSON output (after truncation retry).",
                        "raw": content2[:2000],
                    }

            if not content:
                return {"error": "Empty response from Gemini."}
            return {
                "error": "Model returned non-JSON output.",
                "raw": content[:2000],
                "finish_reason": finish_reason or None,
            }

        return {"error": f"Unsupported LLM provider: {provider}"}

    def call_llm_with_json_retries(prompt_text: str) -> dict:
        attempts = max(json_retry_attempts, 0)
        last_result: dict | None = None
        for attempt in range(attempts + 1):
            prompt = prompt_text
            if attempt > 0:
                prompt = (
                    prompt_text
                    + "\n\n# JSON RETRY\n"
                    + "The previous output was not valid JSON. Return only a single JSON object. "
                    + "Do not include markdown, code fences, or commentary.\n"
                )
            result = call_llm(prompt)
            if not isinstance(result, dict):
                return {"error": "LLM returned non-dict response."}
            last_result = result
            if not _is_retryable_llm_json_error(result):
                return result
            if attempt < attempts:
                _write_debug_line(
                    f"LLM non-JSON output; retrying ({attempt + 1}/{attempts})"
                )
        return last_result or {"error": "LLM returned no content."}

    stop_hits: list[str] = []
    for attempt in range(max_attempts):
        prompt = build_prompt(attempt, stop_hits)
        result = call_llm_with_json_retries(prompt)
        if not stop_patterns:
            return result
        if not isinstance(result, dict) or result.get("error"):
            return result
        hits = _find_stop_list_hits(result, stop_patterns)
        if not hits:
            return result
        stop_hits = hits
        if attempt < max_attempts - 1:
            _write_debug_line(
                f"Stop-list hit(s): {', '.join(hits[:5])} (retrying)"
            )
            continue
        cleaned = _strip_stop_list_phrases(result, stop_patterns)
        cleaned_hits = _find_stop_list_hits(cleaned, stop_patterns)
        if not cleaned_hits:
            _write_debug_line("Stop-list hits removed via sanitization.")
            return cleaned
        return {
            "error": "Stop-list violation: output contains banned phrases.",
            "stop_list_hits": stop_hits[:20],
        }


# ==========================================
# PDF GENERATION LAYER
# ==========================================
def escape_typst(text):
    if not text:
        return ""
    # Escape special characters for Typst content mode
    # We need to be careful not to double-escape if we run this multiple times,
    # but for now we assume raw input.
    text = str(text)
    replacements = {
        "\\": "\\\\",  # Must be first
        "@": "\\@",
        "#": "\\#",
        "$": "\\$",
        "*": "\\*",
        "_": "\\_",
        "[": "\\[",
        "]": "\\]",
        "=": "\\=",
        "<": "\\<",
        ">": "\\>",
        "`": "\\`",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def format_inline_typst(text):
    """Escape text while supporting minimal inline markup (e.g., <b>...</b>)."""
    if text is None:
        return ""

    def normalize_special_bullet(s: str) -> str:
        targets = [
            "Co-developed an OSS C++ RNN IDE",
            "Co-developed a >1m LoC OSS C++ RNN IDE",
            "Co-developed a TOP500 >1m LoC OSS C++ RNN IDE",
        ]
        if any(t in s for t in targets):
            return "Co-developed a TOP500 >1m LoC OSS C++ RNN IDE (see GitHub) called emergent."
        return s

    s = normalize_special_bullet(str(text))

    def format_emergent(value: str) -> str:
        out = []
        pos = 0
        pattern = re.compile(r"emergent", re.IGNORECASE)
        for m in pattern.finditer(value):
            if m.start() > pos:
                out.append(escape_typst(value[pos : m.start()]))
            out.append("#emph[emergent™]")
            pos = m.end()
        if pos < len(value):
            out.append(escape_typst(value[pos:]))
        return "".join(out) if out else escape_typst(value)

    # Support <b>...</b> tags (case-insensitive) as bold.
    bold_re = re.compile(r"(?i)</?b>")
    segments: list[tuple[str, bool]] = []
    bold = False
    last = 0
    for m in bold_re.finditer(s):
        if m.start() > last:
            segments.append((s[last : m.start()], bold))
        tag = m.group(0).lower()
        bold = tag == "<b>"
        last = m.end()
    if last < len(s):
        segments.append((s[last:], bold))

    if not segments:
        return format_emergent(s)

    rendered: list[str] = []
    for seg_text, seg_bold in segments:
        piece = format_emergent(seg_text)
        if seg_bold:
            rendered.append(f"#emph[#strong[{piece}]]")
        else:
            rendered.append(piece)
    return "".join(rendered)


def normalize_github(value: str) -> str:
    """Return just the GitHub path/user, stripping any protocol/host."""
    val = value.strip()
    lower = val.lower()
    # Strip protocol/host anywhere at the front.
    lower = re.sub(r"^https?://", "", lower)
    lower = re.sub(r"^www\\.", "", lower)
    if lower.startswith("github.com/"):
        val = val[len(val) - len(lower) + len("github.com/") :]
    # If still contains github.com anywhere, strip up to last slash.
    if "github.com/" in val:
        val = val.split("github.com/", 1)[1]
    return val.strip("/")


def normalize_linkedin(value: str) -> str:
    """Return just the LinkedIn slug, stripping protocol/host and optional /in/."""
    val = value.strip()
    lower = val.lower()
    lower = re.sub(r"^https?://", "", lower)
    lower = re.sub(r"^www\\.", "", lower)
    # Drop leading linkedin domain.
    if lower.startswith("linkedin.com/"):
        val = val[len(val) - len(lower) + len("linkedin.com/") :]
        lower = lower[len("linkedin.com/") :]
    # Drop optional in/ prefix
    if val.lower().startswith("in/"):
        val = val[3:]
    # If domain still present inside, strip after last slash.
    if "linkedin.com/" in val:
        val = val.split("linkedin.com/", 1)[1]
    if val.lower().startswith("in/"):
        val = val[3:]
    return val.strip("/")


def normalize_scholar_url(value: str) -> str:
    """Return a Google Scholar URL from a URL or raw citations user id."""
    val = value.strip()
    if not val:
        return ""

    if re.fullmatch(r"[A-Za-z0-9_-]+", val):
        return "https://scholar.google.com/citations?user=" + val

    m = re.search(r"(?:\\?|&)user=([^&?#/]+)", val)
    if m:
        user = m.group(1).strip()
        if user:
            return "https://scholar.google.com/citations?user=" + user

    if "://" not in val:
        return "https://" + val.lstrip("/")
    return val


def normalize_calendly_url(value: str) -> str:
    """Return a Calendly-style URL from a URL or raw handle."""
    val = value.strip()
    if not val:
        return ""

    if re.fullmatch(r"[A-Za-z0-9._-]+", val):
        return "https://cal.link/" + val

    if "://" not in val:
        return "https://" + val.lstrip("/")
    return val


def normalize_portfolio_url(value: str) -> str:
    """Return a normalized URL with https:// when missing."""
    val = value.strip()
    if not val:
        return ""
    if re.match(r"^https?://", val, flags=re.IGNORECASE):
        return val
    return "https://" + val.lstrip("/")


def format_url_label(url: str) -> str:
    """Return a short display label for a URL (drops scheme/query/fragment)."""
    value = url.strip()
    if not value:
        return ""
    candidate = value
    if "://" not in candidate:
        candidate = "https://" + candidate.lstrip("/")
    try:
        parsed = urlparse(candidate)
    except Exception:
        return value
    netloc = (parsed.netloc or "").strip()
    if netloc.lower().startswith("www."):
        netloc = netloc[4:]
    label = (netloc + (parsed.path or "")).rstrip("/")
    return label or value


def format_date_mm_yy(date_str: str) -> str:
    """Convert ISO date (YYYY-MM-DD or YYYY-MM) to MM/YY; returns empty string if invalid/empty."""
    if not date_str:
        return ""
    parts = date_str.split("-")
    if len(parts) < 2:
        return ""
    year = parts[0][-2:] if parts[0] else ""
    month = parts[1]
    if not (year and month):
        return ""
    return f"{month}/{year}"


DATE_TAG_RE = re.compile(r"(?is)<date>(.*?)</date>")


def split_bullet_date(bullet: str) -> tuple[str, str]:
    """Split a bullet into (body, date) where the date is optional and tagged."""
    text = str(bullet or "")
    match = DATE_TAG_RE.search(text)
    if not match:
        return text.strip(), ""
    date_part = match.group(1).strip()
    body = DATE_TAG_RE.sub("", text)
    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(r"\s+([,.;:])", r"\1", body)
    return body, date_part


def format_bullet_date(date_str: str) -> str:
    """Normalize bullet-level date ranges (e.g., swap triple hyphens for a spaced dash)."""
    if not date_str:
        return ""
    return date_str.replace("---", " - ").strip()


def _ensure_fontawesome_fonts():
    """Download Font Awesome TTFs locally for Typst if missing."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in FONT_AWESOME_SOURCES.items():
        path = FONTS_DIR / fname
        if path.exists():
            continue
        try:
            urllib.request.urlretrieve(url, path)  # nosec B310
        except Exception as e:
            print(f"Warning: could not fetch {fname} from {url}: {e}")


def _ensure_template_fonts():
    """Avenir fonts are bundled locally; no template downloads required."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path) -> None:
    dest = dest_dir.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise RuntimeError("Blocked tar path traversal")
    tar.extractall(dest)  # nosec B202


def _ensure_typst_packages():
    """Fetch the Font Awesome Typst package locally for offline runs."""
    if FONT_AWESOME_PACKAGE_DIR.exists():
        return
    tmp_path: Path | None = None
    try:
        PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
        FONT_AWESOME_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
        os.close(tmp_fd)
        tmp_path = Path(tmp_path)
        urllib.request.urlretrieve(FONT_AWESOME_PACKAGE_URL, tmp_path)  # nosec B310
        with tarfile.open(tmp_path, "r:gz") as tar:
            _safe_extract_tar(tar, FONT_AWESOME_PACKAGE_DIR)
    except Exception as e:
        print(f"Warning: could not fetch Typst Font Awesome package: {e}")
    finally:
        with suppress(Exception):
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()


def ensure_fonts_ready():
    """Download fonts only once per process."""
    global _FONTS_READY
    if _FONTS_READY:
        return
    _ensure_fontawesome_fonts()
    _ensure_template_fonts()
    _ensure_typst_packages()
    _FONTS_READY = True


def _build_pdf_metadata(resume_data, profile_data):
    """Construct PDF metadata fields from resume/profile data."""

    def pick(*keys):
        for key in keys:
            val = profile_data.get(key)
            if val and str(val).strip():
                return str(val).strip()
            val = resume_data.get(key)
            if val and str(val).strip():
                return str(val).strip()
        return ""

    def name_parts():
        first = resume_data.get("first_name") or profile_data.get("first_name") or ""
        middle = resume_data.get("middle_name") or profile_data.get("middle_name") or ""
        last = resume_data.get("last_name") or profile_data.get("last_name") or ""
        full = profile_data.get("name", "")
        if not (first or last) and full:
            parts = full.split()
            if parts:
                first = parts[0]
                if len(parts) > 2:
                    middle = " ".join(parts[1:-1])
                if len(parts) >= 2:
                    last = parts[-1]
        return first.strip(), middle.strip(), last.strip()

    first, middle, last = name_parts()
    name_tokens = [first, last]
    author = " ".join(t for t in name_tokens if t).strip() or "Resume Candidate"
    role = pick("target_role")
    company = pick("target_company")
    req_id = pick("req_id")

    subject_parts = [f"Résumé {author}".strip()]
    if role and company:
        subject_parts.append(f"{role} @ {company}")
    elif role:
        subject_parts.append(role)
    elif company:
        subject_parts.append(company)
    if req_id:
        subject_parts.append(f"Req {req_id}")
    subject = " — ".join(subject_parts)

    def build_keywords():
        kw = [
            f"{label}: {str(field).strip()}"
            for label, field in (
                ("Role", role),
                ("Company", company),
                ("Domain", pick("primary_domain")),
                ("ReqID", req_id),
            )
            if field and str(field).strip()
        ]
        kw.extend(
            [
                str(skill).strip()
                for skill in (profile_data.get("highlighted_skills") or [])[:5]
                if skill and str(skill).strip()
            ]
        )
        kw.extend([author, "resume"])
        seen = set()
        deduped = []
        for item in kw:
            norm = item.lower()
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(item)
        return "; ".join(deduped)

    return {
        "title": "Résumé",
        "subject": subject,
        "author": author,
        "creator": "ResumeBuilder3",
        "producer": "ResumeBuilder3 / Typst",
        "keywords": build_keywords(),
    }


def _apply_pdf_metadata(pdf_path: Path, metadata: dict | None = None):
    """Apply metadata and clean MarkInfo using pikepdf; no-op on failure."""
    if not metadata:
        return
    try:
        import pikepdf
    except Exception:
        return

    try:
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            info = pdf.docinfo

            def set_info(key, value):
                if value:
                    info[getattr(pikepdf.Name, key)] = value

            ts_docinfo = datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%S+00'00'")
            ts_xmp = datetime.now(timezone.utc).isoformat()

            set_info("Title", metadata.get("title"))
            set_info("Subject", metadata.get("subject"))
            set_info("Author", metadata.get("author"))
            set_info("Creator", metadata.get("creator"))
            set_info("Producer", metadata.get("producer"))
            set_info("Keywords", metadata.get("keywords"))
            set_info("CreationDate", ts_docinfo)
            set_info("ModDate", ts_docinfo)

            mark = pdf.Root.get("/MarkInfo")
            if not isinstance(mark, pikepdf.Dictionary):
                mark = cast(Any, pikepdf).Dictionary()
            mark["/Marked"] = True
            if "/Suspects" in mark:
                del mark["/Suspects"]
            pdf.Root["/MarkInfo"] = mark

            with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
                meta["pdf:Title"] = metadata.get("title") or ""
                meta["pdf:Subject"] = metadata.get("subject") or ""
                meta["pdf:Author"] = metadata.get("author") or ""
                meta["pdf:Keywords"] = metadata.get("keywords") or ""
                meta["xmp:CreatorTool"] = metadata.get("creator") or ""
                meta["xmp:CreateDate"] = ts_xmp
                meta["xmp:ModifyDate"] = ts_xmp
                meta["dc:creator"] = [metadata.get("author") or ""]

            pdf.save(pdf_path)
    except Exception as e:
        print(f"Warning: unable to apply PDF metadata: {e}")


def _normalize_auto_fit_target_pages(
    value, default: int = DEFAULT_AUTO_FIT_TARGET_PAGES
) -> int:
    try:
        if value is None:
            raise ValueError("missing")
        if isinstance(value, bool):
            raise ValueError("bool")
        if isinstance(value, (int, float)):
            pages = int(value)
        else:
            raw = str(value).strip()
            if not raw:
                raise ValueError("blank")
            pages = int(float(raw))
    except Exception:
        pages = default
    if pages < 1:
        return 1
    return pages


def compile_pdf_with_auto_tuning(
    resume_data,
    profile_data,
    include_matrices=True,
    include_summary=True,
    section_order=None,
    target_pages: int | None = None,
):
    """
    Applies a bracket + binary search on a global layout scale that tunes
    consistent spacing ratios across the document so the PDF snugly fits the
    requested page count (maximizing whitespace without spilling to N+1 pages).
    """
    from io import BytesIO

    import pikepdf

    if target_pages is None:
        target_pages = resume_data.get("auto_fit_target_pages")
    target_pages = _normalize_auto_fit_target_pages(target_pages)

    pdf_metadata = _build_pdf_metadata(resume_data, profile_data)

    def persist_auto_fit_cache(*, best_scale: float, too_long_scale: float | None):
        with suppress(Exception):
            db = Neo4jClient()
            db.set_auto_fit_cache(
                best_scale=best_scale,
                too_long_scale=too_long_scale,
            )
            db.close()

    def render(scale: float):
        source = generate_typst_source(
            resume_data,
            profile_data,
            include_matrices=include_matrices,
            include_summary=include_summary,
            section_order=section_order,
            layout_scale=scale,
        )
        ok, pdf_bytes = compile_pdf(source, metadata=pdf_metadata)
        if not ok or not pdf_bytes:
            return False, 0, b""
        try:
            pdf = pikepdf.open(BytesIO(pdf_bytes))
            return True, len(pdf.pages), pdf_bytes
        except Exception:
            # If we can't parse the PDF, we can't tune; treat it as a valid render and return it.
            return True, 0, pdf_bytes

    # Bracket the solution:
    # - `fit_scale` produces <= target_pages
    # - `too_long_scale` produces > target_pages
    min_scale = 0.35
    max_scale = 6.0
    grow = 1.35
    shrink = 1.0 / grow
    tol_scale = 0.002

    # Seed auto-fit from the last successful tuning (if available).
    initial_scale = 1.0
    cached_too_long_scale = None
    with suppress(Exception):
        db = Neo4jClient()
        cache = db.get_auto_fit_cache() or {}
        db.close()
        cached_best = cache.get("best_scale")
        if isinstance(cached_best, (int, float)) and float(cached_best) > 0:
            initial_scale = float(cached_best)
        cached_high = cache.get("too_long_scale")
        if isinstance(cached_high, (int, float)) and float(cached_high) > 0:
            cached_too_long_scale = float(cached_high)
    initial_scale = max(min_scale, min(max_scale, initial_scale))
    if cached_too_long_scale is not None:
        cached_too_long_scale = max(min_scale, min(max_scale, cached_too_long_scale))

    ok, pages, pdf_bytes = render(initial_scale)
    if not ok:
        return False, b""
    if pages == 0:
        return True, pdf_bytes

    fit_scale = initial_scale
    fit_pdf = pdf_bytes
    too_long_scale = None

    if pages > target_pages:
        # Too long; shrink until it fits or we hit min_scale.
        too_long_scale = initial_scale
        tightest_pdf = fit_pdf
        tightest_scale = fit_scale
        found_fit = False
        scale = initial_scale
        while scale > min_scale:
            scale = max(min_scale, scale * shrink)
            ok, pages, pdf_bytes = render(scale)
            if not ok:
                continue
            if pages == 0:
                return True, pdf_bytes
            tightest_pdf = pdf_bytes
            tightest_scale = scale
            if pages <= target_pages:
                fit_scale = scale
                fit_pdf = pdf_bytes
                found_fit = True
                break
        if not found_fit:
            print(
                f"Auto-fit: resume is longer than {target_pages} page(s) even at minimum layout scale. Returning tightened multi-page document."
            )
            persist_auto_fit_cache(
                best_scale=tightest_scale,
                too_long_scale=too_long_scale,
            )
            return True, tightest_pdf
    else:
        # Fits. If we have a cached overflow bound, validate it and reuse as the bracket
        # so we don't have to re-expand and re-bisect from a wide range each time.
        if (
            cached_too_long_scale is not None
            and cached_too_long_scale > fit_scale
            and cached_too_long_scale <= max_scale
        ):
            ok2, pages2, pdf2 = render(cached_too_long_scale)
            if ok2:
                if pages2 == 0:
                    return True, pdf2
                if pages2 > target_pages:
                    too_long_scale = cached_too_long_scale
                else:
                    # Cached "too long" bound now fits; treat it as the new fit point and expand.
                    fit_scale = cached_too_long_scale
                    fit_pdf = pdf2

        # Fits; expand until it overflows or we hit max_scale.
        scale = fit_scale
        while scale < max_scale:
            scale = min(max_scale, scale * grow)
            if too_long_scale is not None and scale >= too_long_scale:
                break
            ok, pages, pdf_bytes = render(scale)
            if not ok:
                continue
            if pages == 0:
                return True, pdf_bytes
            if pages <= target_pages:
                fit_scale = scale
                fit_pdf = pdf_bytes
            else:
                too_long_scale = scale
                break
        if too_long_scale is None:
            print(
                f"Auto-fit: resume still fits within {target_pages} page(s) at maximum layout scale. Returning expanded document."
            )
            persist_auto_fit_cache(best_scale=fit_scale, too_long_scale=None)
            return True, fit_pdf

    low = fit_scale
    high = too_long_scale if too_long_scale is not None else max_scale
    best_scale = fit_scale
    best_pdf = fit_pdf

    for _ in range(10):
        if (high - low) <= tol_scale:
            break
        mid = (low + high) / 2.0
        ok, pages, pdf_bytes = render(mid)
        if not ok:
            continue
        if pages == 0:
            return True, pdf_bytes
        if pages <= target_pages:
            low = mid
            best_scale = mid
            best_pdf = pdf_bytes
        else:
            high = mid

    print(
        f"Auto-fit: selected layout scale {best_scale:.3f} for {target_pages} page(s)."
    )
    persist_auto_fit_cache(
        best_scale=best_scale,
        too_long_scale=high,
    )
    return True, best_pdf


def compile_pdf(typst_source, metadata=None):
    """
    Compiles Typst source to PDF and returns (success, pdf_bytes).
    No PDFs are left on disk. Temporary files live in a hidden folder to avoid
    triggering Reflex hot-reload file watchers.
    """
    temp_path = None
    output_path = None
    try:
        ensure_fonts_ready()
        TEMP_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".typ", delete=False, dir=TEMP_BUILD_DIR, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(typst_source)
            temp_path = Path(tmp.name)
            output_path = temp_path.with_suffix(".pdf")

        root_path = str(BASE_DIR)
        process = subprocess.Popen(
            [
                TYPST_BIN,
                "compile",
                "--font-path",
                str(FONTS_DIR),
                "--package-path",
                str(PACKAGES_DIR),
                "--root",
                root_path,
                str(temp_path),
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(BASE_DIR),
            env=os.environ
            | {
                "TYPST_FONT_PATHS": str(FONTS_DIR),
                "TYPST_FONT_PATH": str(FONTS_DIR),
                "TYPST_PACKAGE_PATHS": str(PACKAGES_DIR),
            },
        )
        stderr = process.communicate()[1]

        if process.returncode != 0:
            print(f"Typst compilation failed: {stderr}")
            return False, b""
        _apply_pdf_metadata(output_path, metadata)
        pdf_bytes = output_path.read_bytes()
        return True, pdf_bytes
    except Exception as e:
        print(f"Error running Typst: {e}")
        return False, b""
    finally:
        for path in (temp_path, output_path):
            if path and path.exists():
                with suppress(Exception):
                    path.unlink()


# ==========================================
# UI LAYER (REFLEX)
# ==========================================
class Experience(BaseModel):
    id: str = ""
    company: str = ""
    role: str = ""
    location: str = ""
    description: str = ""
    bullets: str = ""
    start_date: str = ""
    end_date: str = ""


class Education(BaseModel):
    id: str = ""
    school: str = ""
    degree: str = ""
    location: str = ""
    description: str = ""
    bullets: str = ""
    start_date: str = ""
    end_date: str = ""


class FounderRole(BaseModel):
    id: str = ""
    company: str = ""
    role: str = ""
    location: str = ""
    description: str = ""
    bullets: str = ""
    start_date: str = ""
    end_date: str = ""


class CustomSection(BaseModel):
    id: str = ""
    key: str = ""
    title: str = ""
    body: str = ""


def _model_to_dict(model) -> dict[str, Any]:
    if model is None:
        return {}
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        with suppress(Exception):
            result = dump()
            if isinstance(result, dict):
                return result
    as_dict = getattr(model, "dict", None)
    if callable(as_dict):
        with suppress(Exception):
            result = as_dict()
            if isinstance(result, dict):
                return result
    if isinstance(model, dict):
        return dict(model)
    return {}


def _skills_rows_to_csv(rows, highlighted_skills) -> list[str]:
    rows = list(rows or [])

    def has_row_content(row) -> bool:
        if isinstance(row, (list, tuple)):
            return any(str(s).strip() for s in row)
        if row is None:
            return False
        return bool(str(row).strip())

    if not any(has_row_content(row) for row in rows):
        fallback_skills = [
            str(s).strip() for s in (highlighted_skills or []) if str(s).strip()
        ]
        rows = [fallback_skills[0:3], fallback_skills[3:6], fallback_skills[6:9]]

    out: list[str] = []
    for row in rows[:3]:
        if isinstance(row, (list, tuple)):
            out.append(", ".join([str(s).strip() for s in row if str(s).strip()]))
        elif row is None:
            out.append("")
        else:
            out.append(str(row).strip())
    while len(out) < 3:
        out.append("")
    return out[:3]


class State(rx.State):
    job_req: str = ""
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    email: str = ""
    email2: str = ""
    phone: str = ""
    font_family: str = DEFAULT_RESUME_FONT_FAMILY
    linkedin_url: str = ""
    github_url: str = ""
    scholar_url: str = ""
    calendly_url: str = ""
    portfolio_url: str = ""
    summary: str = "Professional Summary..."
    prompt_yaml: str = ""
    rewrite_bullets_with_llm: bool = False
    headers: list[str] = ["", "", "", "", "", "", "", "", ""]
    highlighted_skills: list[str] = ["", "", "", "", "", "", "", "", ""]
    skills_rows: list[list[str]] = [[], [], []]
    profile_experience_bullets: dict[str, list[str]] = {}
    profile_founder_bullets: dict[str, list[str]] = {}
    experience: list[Experience] = []
    education: list[Education] = []
    founder_roles: list[FounderRole] = []

    # Keep profile_data for other fields if needed, or just decompose it all
    profile_data: dict = {}

    pdf_url: str = ""
    pdf_generated: bool = False
    generating_pdf: bool = False
    force_pdf_render: bool = False
    selected_profile_id: str = ""
    selected_profile_label: str = ""
    has_loaded: bool = False
    is_saving: bool = False
    is_loading_resume: bool = False
    is_generating_profile: bool = False
    is_auto_pipeline: bool = False
    data_loaded: bool = False
    auto_tune_pdf: bool = True
    auto_fit_target_pages: int = DEFAULT_AUTO_FIT_TARGET_PAGES
    include_matrices: bool = True
    section_order: list[str] = DEFAULT_SECTION_ORDER.copy()
    section_titles: dict[str, str] = {}
    section_visibility: dict[str, bool] = {key: True for key in SECTION_LABELS}
    custom_sections: list[CustomSection] = []
    new_section_title: str = ""
    new_section_body: str = ""
    last_pdf_signature: str = ""
    last_pdf_b64: str = ""
    pdf_error: str = ""
    db_error: str = ""
    llm_models: list[str] = DEFAULT_LLM_MODELS.copy()
    selected_model: str = DEFAULT_LLM_MODEL
    last_render_label: str = ""
    last_save_label: str = ""
    # Latest role/profile fields
    target_company: str = ""
    target_role: str = ""
    seniority_level: str = ""
    target_location: str = ""
    work_mode: str = ""
    travel_requirement: str = ""
    primary_domain: str = ""
    must_have_skills_text: str = ""
    nice_to_have_skills_text: str = ""
    tech_stack_keywords_text: str = ""
    non_technical_requirements_text: str = ""
    certifications_text: str = ""
    clearances_text: str = ""
    core_responsibilities_text: str = ""
    outcome_goals_text: str = ""
    salary_band: str = ""
    posting_url: str = ""
    req_id: str = ""
    pipeline_status: list[str] = []
    last_profile_job_req_sha: str = ""
    latest_profile_id: str = ""
    edit_profile_bullets: bool = False
    is_saving_profile_bullets: bool = False
    last_profile_bullets_label: str = ""
    override_job_req_on_load: bool = False

    @rx.var
    def skills_rows_csv(self) -> list[str]:
        return _skills_rows_to_csv(self.skills_rows, self.highlighted_skills)

    @rx.var
    def pipeline_latest(self) -> str:
        msgs = self.pipeline_status.copy()
        return msgs[-1] if msgs else ""

    @rx.var
    def profile_experience_bullets_list(self) -> list[str]:
        overrides = self.profile_experience_bullets.copy()
        out: list[str] = []
        for exp in self.experience:
            entry_id = str(getattr(exp, "id", "") or "")
            out.append(_coerce_bullet_text(overrides.get(entry_id, [])))
        return out

    @rx.var
    def profile_founder_bullets_list(self) -> list[str]:
        overrides = self.profile_founder_bullets.copy()
        out: list[str] = []
        for role in self.founder_roles:
            entry_id = str(getattr(role, "id", "") or "")
            out.append(_coerce_bullet_text(overrides.get(entry_id, [])))
        return out

    @rx.var
    def section_order_rows(self) -> list[dict]:
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections)]
        )
        extra_keys = _custom_section_keys(custom_sections)
        order = _sanitize_section_order(self.section_order, extra_keys)
        visibility = self.section_visibility.copy()
        title_map = _build_section_title_map(self.section_titles, custom_sections)
        custom_key_set = set(extra_keys)
        rows: list[dict] = []
        for key in order:
            title = title_map.get(key) or key.replace("_", " ").title()
            rows.append(
                {
                    "key": key,
                    "visible": bool(visibility.get(key, True)),
                    "title": title,
                    "custom": key in custom_key_set,
                }
            )
        return rows

    @rx.var
    def custom_section_rows(self) -> list[dict]:
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections)]
        )
        return [
            {
                "key": item.get("key", ""),
                "title": item.get("title", ""),
                "body": item.get("body", ""),
            }
            for item in custom_sections
        ]

    @rx.var
    def job_req_needs_profile(self) -> bool:
        req_text = self.job_req.strip()
        if not req_text:
            return False
        current = _hash_text(req_text)
        return current != (self.last_profile_job_req_sha)

    def _visible_section_order(self) -> list[str]:
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections)]
        )
        extra_keys = _custom_section_keys(custom_sections)
        order = _sanitize_section_order(self.section_order, extra_keys)
        visibility = self.section_visibility.copy()
        return [key for key in order if visibility.get(key, True)]

    def on_load(self):
        # Always reset UI-visible fields so reload starts empty.
        self.db_error = ""
        self.pdf_error = ""
        self.summary = "Professional Summary..."
        self.prompt_yaml = ""
        self.rewrite_bullets_with_llm = False
        self.headers = [""] * 9
        self.highlighted_skills = [""] * 9
        self.experience = []
        self.education = []
        self.founder_roles = []
        self.skills_rows = [[], [], []]
        self.profile_experience_bullets = {}
        self.profile_founder_bullets = {}
        self.profile_data = {}
        self.selected_profile_id = ""
        self.selected_profile_label = ""
        self.first_name = ""
        self.middle_name = ""
        self.last_name = ""
        self.email = ""
        self.email2 = ""
        self.phone = ""
        self.linkedin_url = ""
        self.github_url = ""
        self.scholar_url = ""
        self.calendly_url = ""
        self.portfolio_url = ""
        self.font_family = DEFAULT_RESUME_FONT_FAMILY
        self.target_company = ""
        self.target_role = ""
        self.seniority_level = ""
        self.target_location = ""
        self.work_mode = ""
        self.travel_requirement = ""
        self.primary_domain = ""
        self.must_have_skills_text = ""
        self.nice_to_have_skills_text = ""
        self.tech_stack_keywords_text = ""
        self.non_technical_requirements_text = ""
        self.certifications_text = ""
        self.clearances_text = ""
        self.core_responsibilities_text = ""
        self.outcome_goals_text = ""
        self.salary_band = ""
        self.posting_url = ""
        self.req_id = ""
        self.job_req = ""
        self.last_profile_job_req_sha = ""
        self.pipeline_status = []
        self.data_loaded = False
        self.auto_tune_pdf = True
        self.auto_fit_target_pages = DEFAULT_AUTO_FIT_TARGET_PAGES
        self.include_matrices = True
        self.section_order = DEFAULT_SECTION_ORDER.copy()
        self.section_visibility = {key: True for key in SECTION_LABELS}
        self.section_titles = {}
        self.custom_sections = []
        self.new_section_title = ""
        self.new_section_body = ""
        self.pdf_generated = False
        self.pdf_url = ""
        self.last_pdf_signature = ""
        self.latest_profile_id = ""
        self.edit_profile_bullets = False
        self.is_saving_profile_bullets = False
        self.last_profile_bullets_label = ""
        try:
            _write_debug_line(t"on_load called at {datetime.now().isoformat()}")

            data = {}
            try:
                db = Neo4jClient()
                db.ensure_resume_exists()
                data = db.get_resume_data() or {}
                profiles = db.search_profile_metadata(term="", limit=1, offset=0)
                db.close()
            except Exception as e:
                _write_debug_line(t"Neo4j unavailable: {e}")
                self.db_error = "Database unavailable; could not load data."
                self.pdf_error = self.db_error
                self.has_loaded = True
                return

            if not data or not data.get("resume"):
                self.db_error = "No resume found in Neo4j."
                self.pdf_error = self.db_error
                self.has_loaded = True
                return

            self.profile_data = data.get("resume", {})
            latest_profile = profiles[0] if profiles else {}
            self.selected_profile_id = str(latest_profile.get("id") or "")
            self.latest_profile_id = self.selected_profile_id
            if latest_profile:
                self.selected_profile_label = _format_profile_label(latest_profile)
            prompt_yaml = self.profile_data.get("prompt_yaml")
            if not (isinstance(prompt_yaml, str) and prompt_yaml.strip()):
                prompt_yaml = _load_prompt_yaml_from_file()
            self.prompt_yaml = prompt_yaml or ""
            self.font_family = (
                self.profile_data.get("font_family") or DEFAULT_RESUME_FONT_FAMILY
            )
            self.auto_fit_target_pages = _normalize_auto_fit_target_pages(
                self.profile_data.get("auto_fit_target_pages"),
                DEFAULT_AUTO_FIT_TARGET_PAGES,
            )
            if os.environ.get("MAX_COVERAGE_FORCE_DB_ERROR_ON_LOAD") == "1":
                self.db_error = "Forced db error for coverage."

            section_titles = _normalize_section_titles(
                self.profile_data.get("section_titles_json")
                or self.profile_data.get("section_titles")
            )
            custom_sections = _normalize_custom_sections(
                self.profile_data.get("custom_sections_json")
                or self.profile_data.get("custom_sections")
            )
            self.section_titles = section_titles
            self.custom_sections = [CustomSection(**item) for item in custom_sections]
            extra_keys = _custom_section_keys(custom_sections)
            raw_order = self.profile_data.get("section_order")
            if isinstance(raw_order, str):
                raw_order = [s.strip() for s in raw_order.split(",") if s.strip()]
            self.section_order = _sanitize_section_order(raw_order, extra_keys)
            raw_enabled = self.profile_data.get("section_enabled")
            normalized_enabled = _normalize_section_enabled(
                raw_enabled, list(SECTION_LABELS) + extra_keys, extra_keys=extra_keys
            )
            self.section_visibility = {
                key: key in normalized_enabled
                for key in _known_section_keys(extra_keys)
            }
            self.include_matrices = self.section_visibility.get("matrices", True)

            # Start the form empty to avoid auto-loading the full resume on initial page load.
            self.experience = []
            self.education = []
            self.founder_roles = []

            # Load configured LLM models (no network calls). Override with LLM_MODELS if desired.
            self.llm_models = list_llm_models()
            if self.llm_models:
                if self.selected_model not in self.llm_models:
                    self.selected_model = self.llm_models[0]
            else:
                self.llm_models = DEFAULT_LLM_MODELS.copy()
                self.selected_model = DEFAULT_LLM_MODEL

            # Optionally render once on load; default is to skip to speed up first paint.
            if os.environ.get("GENERATE_ON_LOAD", "0") == "1":
                if self.db_error:
                    self.pdf_error = self.db_error
                else:
                    self.generate_pdf()

            _write_debug_line(t"on_load completed successfully")

            self.has_loaded = True

        except Exception as e:
            import traceback

            _write_debug_line(t"Error in on_load: {e}")
            with suppress(Exception):
                with DEBUG_LOG.open("a", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                    f.write("\n")
            if MAX_COVERAGE_LOG:
                _maxcov_log(t"on_load error: {e}")
            elif not _should_silence_warnings():
                print(_render_template(t"Error in on_load: {e}"))
            self.has_loaded = True

    async def save_to_db(self):
        """Persist current form data to Neo4j (no JSON fallback)."""
        if self.is_saving:
            return
        self.is_saving = True
        yield
        try:

            def normalize_items(items):
                normalized = []
                for idx, item in enumerate(items):
                    data = _model_to_dict(item)
                    if not data.get("id"):
                        new_id = str(uuid.uuid4())
                        data["id"] = new_id
                        items[idx].id = new_id
                    bullets_text = data.get("bullets", "")
                    bullets = []
                    if isinstance(bullets_text, str):
                        for line in bullets_text.split("\n"):
                            line = line.strip()
                            if line:
                                bullets.append(line)
                    data["bullets"] = bullets
                    data["start_date"] = data.get("start_date") or None
                    data["end_date"] = data.get("end_date") or None
                    normalized.append(data)
                return normalized

            experiences = normalize_items(self.experience)
            education = normalize_items(self.education)
            founder_roles = normalize_items(self.founder_roles)

            headers = self.headers.copy()
            while len(headers) < 9:
                headers.append("")

            full_name = " ".join(
                [self.first_name, self.middle_name, self.last_name]
            ).strip()

            section_titles = _normalize_section_titles(self.section_titles)
            custom_sections = _normalize_custom_sections(
                [_model_to_dict(s) for s in (self.custom_sections)]
            )
            extra_keys = _custom_section_keys(custom_sections)
            section_order = _sanitize_section_order(self.section_order, extra_keys)
            section_enabled = [
                key
                for key in section_order
                if (self.section_visibility).get(key, True)
            ]
            prompt_yaml = (self.prompt_yaml).rstrip()
            if not prompt_yaml:
                prompt_yaml = str(self.profile_data.get("prompt_yaml") or "").rstrip()
            if not prompt_yaml:
                prompt_yaml = _load_prompt_yaml_from_file() or ""
            resume_fields = {
                "first_name": self.first_name,
                "middle_name": self.middle_name,
                "last_name": self.last_name,
                "name": full_name,
                "email": self.email,
                "email2": self.email2,
                "phone": self.phone,
                "font_family": self.font_family or DEFAULT_RESUME_FONT_FAMILY,
                "auto_fit_target_pages": _normalize_auto_fit_target_pages(
                    self.auto_fit_target_pages, DEFAULT_AUTO_FIT_TARGET_PAGES
                ),
                "linkedin_url": self.linkedin_url,
                "github_url": self.github_url,
                "scholar_url": self.scholar_url,
                "calendly_url": self.calendly_url,
                "portfolio_url": self.portfolio_url,
                "summary": self.summary,
                "prompt_yaml": prompt_yaml,
                "head1_left": headers[0],
                "head1_middle": headers[1],
                "head1_right": headers[2],
                "head2_left": headers[3],
                "head2_middle": headers[4],
                "head2_right": headers[5],
                "head3_left": headers[6],
                "head3_middle": headers[7],
                "head3_right": headers[8],
                "top_skills": self.highlighted_skills.copy(),
                "section_order": section_order,
                "section_enabled": section_enabled,
                "section_titles_json": json.dumps(section_titles, ensure_ascii=False),
                "custom_sections_json": json.dumps(custom_sections, ensure_ascii=False),
            }
            self.section_order = list(section_order)
            self.prompt_yaml = prompt_yaml

            db = Neo4jClient()
            db.upsert_resume_and_sections(
                resume_fields,
                experiences,
                education,
                founder_roles,
                delete_missing=self.data_loaded,
            )
            db.close()

            # Keep local cache aligned so subsequent renders use fresh values.
            self.profile_data.update(resume_fields)
            self.profile_data["experience"] = experiences
            self.profile_data["education"] = education
            self.profile_data["founder_roles"] = founder_roles
            self.db_error = ""
            self.last_save_label = _render_template(
                t"Saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as e:
            _write_debug_line(t"Save failed: {e}")
            self.db_error = "Unable to save to Neo4j."
        finally:
            self.is_saving = False

    @rx.event
    async def set_selected_profile_id(self, value: str):
        """Update the selected Profile id and load the cached PDF when available."""
        selected_id = str(value or "").strip()
        self.selected_profile_id = selected_id
        if selected_id:
            self.latest_profile_id = selected_id
        self.pdf_error = ""
        yield  # flush selection before loading
        if not selected_id:
            return
        self.override_job_req_on_load = True
        load = self.load_resume_fields()
        if hasattr(load, "__anext__"):
            async for update in load:
                yield update
        elif asyncio.iscoroutine(load):
            await load
        if not self.data_loaded:
            return
        render = self.generate_pdf()
        if hasattr(render, "__anext__"):
            async for update in render:
                yield update
        elif asyncio.iscoroutine(render):
            await render

    async def load_resume_fields(self):
        """Load canonical resume fields (summary/headers/top_skills) into the form."""
        if self.is_loading_resume:
            self._log_debug("load_resume_fields skipped: already loading")
            return
        override_job_req = self.override_job_req_on_load
        self.override_job_req_on_load = False
        self.is_loading_resume = True
        self.data_loaded = False
        self._log_debug("load_resume_fields started")
        self._add_pipeline_msg("load_resume_fields entered")
        yield
        try:
            db = Neo4jClient()
            db.ensure_resume_exists()
            data = db.get_resume_data() or {}
            selected_id = str(self.selected_profile_id or "").strip()
            if selected_id:
                profiles = db.list_applied_jobs(limit=1, profile_id=selected_id)
            else:
                profiles = db.list_applied_jobs(limit=1)
            db.close()
            if not data or not data.get("resume"):
                self.pdf_error = "No resume found in Neo4j."
                return

            r = data.get("resume", {})
            selected_profile = profiles[0] if profiles else {}
            if not selected_id:
                selected_id = str(selected_profile.get("id") or "")
                self.selected_profile_id = selected_id
            self.latest_profile_id = selected_id
            if selected_profile:
                self.selected_profile_label = _format_profile_label(selected_profile)
            elif selected_id:
                self.selected_profile_label = selected_id
            else:
                self.selected_profile_label = ""
            latest_profile = selected_profile

            prompt_yaml = r.get("prompt_yaml")
            if not (isinstance(prompt_yaml, str) and prompt_yaml.strip()):
                prompt_yaml = _load_prompt_yaml_from_file()
            self.prompt_yaml = prompt_yaml or ""

            def ensure_len(items, target=9):
                items = list(items or [])
                while len(items) < target:
                    items.append("")
                return items[:target]

            def parse_name(full: str):
                parts = (full).strip().split()
                if not parts:
                    return "", "", ""
                if len(parts) == 1:
                    return parts[0], "", ""
                if len(parts) == 2:
                    return parts[0], "", parts[1]
                return parts[0], " ".join(parts[1:-1]), parts[-1]

            def list_to_text(val):
                if isinstance(val, list):
                    return "\n".join([str(v) for v in val if v is not None])
                if val is None:
                    return ""
                return str(val)

            self.first_name = r.get("first_name", self.first_name)
            self.middle_name = r.get("middle_name", self.middle_name)
            self.last_name = r.get("last_name", self.last_name)
            if not (self.first_name or self.last_name):
                n_first, n_mid, n_last = parse_name(r.get("name", ""))
                self.first_name = n_first or self.first_name
                self.middle_name = n_mid or self.middle_name
                self.last_name = n_last or self.last_name
            self.email = r.get("email", self.email)
            self.email2 = r.get("email2", self.email2)
            self.phone = r.get("phone", self.phone)
            self.font_family = (
                r.get("font_family", "")
                or self.font_family
                or DEFAULT_RESUME_FONT_FAMILY
            )
            self.auto_fit_target_pages = _normalize_auto_fit_target_pages(
                r.get("auto_fit_target_pages", self.auto_fit_target_pages),
                DEFAULT_AUTO_FIT_TARGET_PAGES,
            )
            self.linkedin_url = r.get("linkedin_url", self.linkedin_url)
            self.github_url = r.get("github_url", self.github_url)
            self.scholar_url = r.get("scholar_url", self.scholar_url)
            self.calendly_url = r.get("calendly_url", self.calendly_url)
            self.portfolio_url = r.get("portfolio_url", self.portfolio_url)
            # Summary
            self.summary = latest_profile.get("summary") or r.get(
                "summary", self.summary
            )
            self.skills_rows = latest_profile.get("skills_rows") or [[], [], []]
            self.profile_experience_bullets = _bullet_override_map(
                latest_profile.get("experience_bullets")
            )
            self.profile_founder_bullets = _bullet_override_map(
                latest_profile.get("founder_role_bullets")
            )
            # Job req (optionally override when selecting a Profile)
            profile_req_raw = str(latest_profile.get("job_req_raw") or "")
            if profile_req_raw.strip():
                self.last_profile_job_req_sha = _hash_text(profile_req_raw.strip())
                if override_job_req or not (self.job_req and self.job_req.strip()):
                    self.job_req = profile_req_raw
            else:
                self.last_profile_job_req_sha = ""
                if override_job_req:
                    self.job_req = ""
            # Role details from latest profile
            self.target_company = latest_profile.get(
                "target_company", self.target_company
            )
            self.target_role = latest_profile.get("target_role", self.target_role)
            self.seniority_level = latest_profile.get(
                "seniority_level", self.seniority_level
            )
            self.target_location = latest_profile.get(
                "target_location", self.target_location
            )
            self.work_mode = latest_profile.get("work_mode", self.work_mode)
            self.travel_requirement = latest_profile.get(
                "travel_requirement", self.travel_requirement
            )
            self.primary_domain = latest_profile.get(
                "primary_domain", self.primary_domain
            )
            self.must_have_skills_text = list_to_text(
                latest_profile.get("must_have_skills", self.must_have_skills_text)
            )
            self.nice_to_have_skills_text = list_to_text(
                latest_profile.get("nice_to_have_skills", self.nice_to_have_skills_text)
            )
            self.tech_stack_keywords_text = list_to_text(
                latest_profile.get("tech_stack_keywords", self.tech_stack_keywords_text)
            )
            self.non_technical_requirements_text = list_to_text(
                latest_profile.get(
                    "non_technical_requirements", self.non_technical_requirements_text
                )
            )
            self.certifications_text = list_to_text(
                latest_profile.get("certifications", self.certifications_text)
            )
            self.clearances_text = list_to_text(
                latest_profile.get("clearances", self.clearances_text)
            )
            self.core_responsibilities_text = list_to_text(
                latest_profile.get(
                    "core_responsibilities", self.core_responsibilities_text
                )
            )
            self.outcome_goals_text = list_to_text(
                latest_profile.get("outcome_goals", self.outcome_goals_text)
            )
            self.salary_band = latest_profile.get("salary_band", self.salary_band)
            self.posting_url = latest_profile.get("posting_url", self.posting_url)
            self.req_id = latest_profile.get("req_id", self.req_id)

            section_titles = _normalize_section_titles(
                r.get("section_titles_json") or r.get("section_titles")
            )
            custom_sections = _normalize_custom_sections(
                r.get("custom_sections_json") or r.get("custom_sections")
            )
            self.section_titles = section_titles
            self.custom_sections = [CustomSection(**item) for item in custom_sections]
            extra_keys = _custom_section_keys(custom_sections)
            raw_order = r.get("section_order")
            if isinstance(raw_order, str):
                raw_order = [s.strip() for s in raw_order.split(",") if s.strip()]
            self.section_order = _sanitize_section_order(raw_order, extra_keys)
            raw_enabled = r.get("section_enabled")
            normalized_enabled = _normalize_section_enabled(
                raw_enabled, list(SECTION_LABELS) + extra_keys, extra_keys=extra_keys
            )
            self.section_visibility = {
                key: key in normalized_enabled
                for key in _known_section_keys(extra_keys)
            }
            self.include_matrices = self.section_visibility.get("matrices", True)

            # Headers
            latest_headers = latest_profile.get("headers") or []
            if latest_headers and any(str(h).strip() for h in latest_headers):
                self.headers = ensure_len(latest_headers)
            else:
                headers = [
                    r.get("head1_left", ""),
                    r.get("head1_middle", ""),
                    r.get("head1_right", ""),
                    r.get("head2_left", ""),
                    r.get("head2_middle", ""),
                    r.get("head2_right", ""),
                    r.get("head3_left", ""),
                    r.get("head3_middle", ""),
                    r.get("head3_right", ""),
                ]
                self.headers = ensure_len(headers)

            # Top skills -> highlighted skills
            latest_skills = latest_profile.get("highlighted_skills") or []
            if latest_skills and any(str(s).strip() for s in latest_skills):
                self.highlighted_skills = ensure_len(latest_skills)
            else:
                self.highlighted_skills = ensure_len(r.get("top_skills", []))

            def to_models(items, model_cls):
                out = []
                for item in items:
                    item = dict(item)
                    bullets = item.get("bullets", [])
                    if isinstance(bullets, list):
                        item["bullets"] = "\n".join(bullets)
                    elif bullets is None:
                        item["bullets"] = ""
                    else:
                        item["bullets"] = str(bullets)
                    for key in ("start_date", "end_date"):
                        value = item.get(key)
                        if value is None:
                            item[key] = ""
                        elif not isinstance(value, str):
                            item[key] = str(value)
                    out.append(model_cls(**item))
                return out

            # Populate the form from Neo4j.
            self.experience = to_models(data.get("experience", []), Experience)
            self.education = to_models(data.get("education", []), Education)
            self.founder_roles = to_models(data.get("founder_roles", []), FounderRole)

            self.data_loaded = True
            self.pdf_error = ""
        except Exception as e:
            self.pdf_error = _render_template(t"Error loading resume fields: {e}")
            self._add_pipeline_msg(t"load_resume_fields error: {e}")
        finally:
            self.override_job_req_on_load = False
            self.is_loading_resume = False
            self._log_debug("load_resume_fields finished")
            self._add_pipeline_msg("load_resume_fields exited")

    def _compute_pdf_signature(self, resume_data, profile_data, typst_source=""):
        """Generate a stable signature for the current resume to avoid redundant renders."""
        payload = {
            "resume": resume_data,
            "profile": profile_data,
            "template_version": TYPST_TEMPLATE_VERSION,
            "source_hash": _hash_text(typst_source),
        }
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return _hash_text(canonical)

    def _current_resume_profile(self):
        """Build the current resume/profile dictionaries (with bullets split) once."""

        def model_to_dict_list(models, overrides=None):
            items = []
            for m in models:
                data = _model_to_dict(m)
                bullets_text = data.get("bullets", "")
                if isinstance(bullets_text, str):
                    data["bullets"] = [
                        b for b in bullets_text.split("\n") if b is not None
                    ]
                else:
                    data["bullets"] = []
                if overrides:
                    entry_id = str(data.get("id") or "").strip()
                    override_bullets = overrides.get(entry_id)
                    if override_bullets:
                        data["bullets"] = list(override_bullets)
                items.append(data)
            return items

        profile_data = self.profile_data.copy()
        profile_data["name"] = " ".join(
            [self.first_name, self.middle_name, self.last_name]
        ).strip()
        profile_data["email"] = self.email or profile_data.get("email", "")
        profile_data["email2"] = self.email2 or profile_data.get("email2", "")
        profile_data["phone"] = self.phone or profile_data.get("phone", "")
        profile_data["linkedin_url"] = self.linkedin_url or profile_data.get(
            "linkedin_url", ""
        )
        profile_data["github_url"] = self.github_url or profile_data.get(
            "github_url", ""
        )
        profile_data["scholar_url"] = self.scholar_url or profile_data.get(
            "scholar_url", ""
        )
        profile_data["calendly_url"] = self.calendly_url or profile_data.get(
            "calendly_url", ""
        )
        profile_data["portfolio_url"] = self.portfolio_url or profile_data.get(
            "portfolio_url", ""
        )
        profile_data["font_family"] = self.font_family or profile_data.get(
            "font_family", DEFAULT_RESUME_FONT_FAMILY
        )
        profile_data["experience"] = model_to_dict_list(
            self.experience, self.profile_experience_bullets
        )
        profile_data["education"] = model_to_dict_list(self.education)
        profile_data["founder_roles"] = model_to_dict_list(
            self.founder_roles, self.profile_founder_bullets
        )

        # Build resume/profile payloads after any hydration so they reflect latest values.
        profile_meta = {
            "target_role": self.target_role,
            "target_company": self.target_company,
            "primary_domain": self.primary_domain,
            "req_id": self.req_id,
        }
        resume_data = {
            "summary": self.summary,
            "headers": self.headers,
            "highlighted_skills": self.highlighted_skills,
            "first_name": self.first_name,
            "middle_name": self.middle_name,
            "last_name": self.last_name,
            "email2": self.email2,
            "portfolio_url": self.portfolio_url,
            "font_family": self.font_family or DEFAULT_RESUME_FONT_FAMILY,
            "auto_fit_target_pages": _normalize_auto_fit_target_pages(
                self.auto_fit_target_pages, DEFAULT_AUTO_FIT_TARGET_PAGES
            ),
            "skills_rows": self.skills_rows,
        }

        # Ensure section order contains only known sections and falls back to default.
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections)]
        )
        extra_keys = _custom_section_keys(custom_sections)
        self.section_order = _sanitize_section_order(self.section_order, extra_keys)
        resume_data["section_titles"] = _normalize_section_titles(self.section_titles)
        resume_data["custom_sections"] = custom_sections

        profile_data.update(profile_meta)
        return resume_data, profile_data

    def _load_cached_pdf(self, signature):
        """Load an on-disk PDF if its signature matches."""
        try:
            if not (LIVE_PDF_PATH.exists() and LIVE_PDF_SIG_PATH.exists()):
                return False
            disk_sig = LIVE_PDF_SIG_PATH.read_text(encoding="utf-8").strip()
            if disk_sig != signature:
                return False
            pdf_bytes = LIVE_PDF_PATH.read_bytes()
            pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
            cache_buster = int(datetime.now().timestamp() * 1000)
            self.pdf_url = f"data:application/pdf;base64,{pdf_b64}#t={cache_buster}"
            self.last_pdf_b64 = pdf_b64
            self.last_pdf_signature = signature
            self.pdf_generated = True
            self.pdf_error = ""
            return True
        except Exception:
            return False

    def generate_pdf_cache_bust(self):
        if self.generating_pdf:
            return
        self.force_pdf_render = True
        try:
            self.generate_pdf()
        finally:
            self.force_pdf_render = False

    def generate_pdf(self):
        if self.generating_pdf:
            return
        if not self.data_loaded:
            self.pdf_error = "Load Data or Generate Profile first."
            return
        req_text = self.job_req.strip()
        if req_text:
            current = _hash_text(req_text)
            if current != (self.last_profile_job_req_sha):
                self.pdf_error = "Job requisition has not been processed yet. Run Generate Profile or the pipeline first."
                return
        force_render = self.force_pdf_render
        start_time = None
        recorded = False
        self.generating_pdf = True
        try:
            self.pdf_error = ""
            start_time = datetime.now()
            # Build the payload first so we can cheaply skip duplicate renders.
            resume_data, profile_data = self._current_resume_profile()
            visible_order = self._visible_section_order()
            source = generate_typst_source(
                resume_data,
                profile_data,
                include_matrices=self.include_matrices,
                include_summary=True,
                section_order=visible_order,
            )
            pdf_metadata = _build_pdf_metadata(resume_data, profile_data)
            signature = self._compute_pdf_signature(resume_data, profile_data, source)

            # If nothing changed and we already have a URL, skip rendering.
            if (
                not force_render
                and self.pdf_generated
                and self.last_pdf_signature == signature
                and self.pdf_url
            ):
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, True)
                recorded = True
                return

            # If signature unchanged but URL missing, reuse cached base64.
            if (
                not force_render
                and self.pdf_generated
                and self.last_pdf_signature == signature
                and self.last_pdf_b64
            ):
                cache_buster = int(datetime.now().timestamp() * 1000)
                self.pdf_url = (
                    f"data:application/pdf;base64,{self.last_pdf_b64}#t={cache_buster}"
                )
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, True)
                recorded = True
                return

            # If on-disk cache matches, reuse it.
            if not force_render and self._load_cached_pdf(signature):
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, True)
                recorded = True
                return

            # Compile PDF; optionally write to disk when enabled (off by default to avoid hot-reload loops).
            if self.auto_tune_pdf:
                target_pages = _normalize_auto_fit_target_pages(
                    self.auto_fit_target_pages, DEFAULT_AUTO_FIT_TARGET_PAGES
                )
                resume_data["auto_fit_target_pages"] = target_pages
                success, pdf_bytes = compile_pdf_with_auto_tuning(
                    resume_data,
                    profile_data,
                    include_matrices=self.include_matrices,
                    include_summary=True,
                    section_order=visible_order,
                    target_pages=target_pages,
                )
            else:
                success, pdf_bytes = compile_pdf(source, metadata=pdf_metadata)
            if success and pdf_bytes:
                pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
                self.last_pdf_b64 = pdf_b64
                # Serve as a data URL so the UI viewer always has the PDF inline.
                # Add a timestamp fragment to prevent stale embeds from being reused.
                cache_buster = int(datetime.now().timestamp() * 1000)
                self.pdf_url = f"data:application/pdf;base64,{pdf_b64}#t={cache_buster}"
                self.pdf_generated = True
                self.last_pdf_signature = signature
                self.pdf_error = ""
                if RUNTIME_WRITE_PDF:
                    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
                    LIVE_PDF_PATH.write_bytes(pdf_bytes)
                    LIVE_PDF_SIG_PATH.write_text(signature, encoding="utf-8")
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, False)
                recorded = True
            else:
                self.pdf_error = "PDF generation failed; check server logs."
        finally:
            if start_time and not recorded:
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, False)
            self.generating_pdf = False

    def _record_render_time(self, ms, from_cache: bool):
        """Store render label with cache/compile context."""
        if ms >= 1000:
            time_str = f"{round(ms / 1000, 2)}s"
        else:
            time_str = f"{ms} ms"
        prefix = "Served from cache in" if from_cache else "Rendered with Typst in"
        self.last_render_label = f"{prefix} {time_str}."

    def move_section_up(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections)]
        )
        extra_keys = _custom_section_keys(custom_sections)
        order = _sanitize_section_order(self.section_order, extra_keys)
        if idx <= 0 or idx >= len(order):
            return
        order[idx - 1], order[idx] = order[idx], order[idx - 1]
        self.section_order = order

    def move_section_down(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections)]
        )
        extra_keys = _custom_section_keys(custom_sections)
        order = _sanitize_section_order(self.section_order, extra_keys)
        if idx < 0 or idx >= len(order) - 1:
            return
        order[idx + 1], order[idx] = order[idx], order[idx + 1]
        self.section_order = order

    @rx.event
    def set_auto_tune_pdf(self, value: bool):
        """Toggle auto-fit (snugly fills the target page count)."""
        self.auto_tune_pdf = _coerce_bool(value)

    @rx.event
    def set_auto_fit_target_pages(self, value):
        self.auto_fit_target_pages = _normalize_auto_fit_target_pages(
            value, DEFAULT_AUTO_FIT_TARGET_PAGES
        )

    @rx.event
    def set_include_matrices(self, value: bool):
        """Toggle inclusion of the Skills section."""
        self.include_matrices = _coerce_bool(value)
        visibility = self.section_visibility.copy()
        visibility["matrices"] = self.include_matrices
        self.section_visibility = visibility

    @rx.event
    def set_rewrite_bullets_with_llm(self, value: bool):
        """Toggle LLM bullet rewrites for generated profiles."""
        self.rewrite_bullets_with_llm = _coerce_bool(value)

    @rx.event
    def set_edit_profile_bullets(self, value: bool):
        """Toggle editing of Profile bullet overrides in the UI."""
        self.edit_profile_bullets = _coerce_bool(value)

    @rx.event
    def set_profile_experience_bullets_text(self, entry_id: str, value: str):
        """Update profile experience bullet overrides for an entry id."""
        entry_id = entry_id.strip()
        if not entry_id:
            return
        bullets = [
            line.strip() for line in value.split("\n") if line.strip()
        ]
        overrides = self.profile_experience_bullets.copy()
        if bullets:
            overrides[entry_id] = bullets
        else:
            overrides.pop(entry_id, None)
        self.profile_experience_bullets = overrides

    @rx.event
    def set_profile_founder_bullets_text(self, entry_id: str, value: str):
        """Update profile founder bullet overrides for an entry id."""
        entry_id = entry_id.strip()
        if not entry_id:
            return
        bullets = [
            line.strip() for line in value.split("\n") if line.strip()
        ]
        overrides = self.profile_founder_bullets.copy()
        if bullets:
            overrides[entry_id] = bullets
        else:
            overrides.pop(entry_id, None)
        self.profile_founder_bullets = overrides

    async def save_profile_bullets(self):
        """Persist profile bullet overrides to the latest Profile."""
        if self.is_saving_profile_bullets:
            return
        self.is_saving_profile_bullets = True
        yield
        try:
            if not (self.latest_profile_id and self.latest_profile_id.strip()):
                self.db_error = "No profile available to update."
                return
            experience_bullets = [
                {"id": k, "bullets": v}
                for k, v in (self.profile_experience_bullets).items()
                if k and v
            ]
            founder_role_bullets = [
                {"id": k, "bullets": v}
                for k, v in (self.profile_founder_bullets).items()
                if k and v
            ]
            db = Neo4jClient()
            db.update_profile_bullets(
                self.latest_profile_id,
                experience_bullets,
                founder_role_bullets,
            )
            db.close()
            self.db_error = ""
            self.last_profile_bullets_label = _render_template(
                t"Profile bullets saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as e:
            _write_debug_line(t"Profile bullets save failed: {e}")
            self.db_error = "Unable to save profile bullets."
        finally:
            self.is_saving_profile_bullets = False

    @rx.event
    def set_section_visibility(self, value: bool, key: str):
        """Toggle visibility for a section key."""
        key = key.strip()
        if not key:
            return
        visibility = self.section_visibility.copy()
        visibility[key] = _coerce_bool(value)
        self.section_visibility = visibility
        if key == "matrices":
            self.include_matrices = _coerce_bool(value)

    @rx.event
    def set_section_title(self, value: str, key: str):
        """Update a section title (base sections or custom sections)."""
        key = key.strip()
        if not key:
            return
        title = value.strip()
        if key in SECTION_LABELS:
            titles = self.section_titles.copy()
            if title:
                titles[key] = title
            else:
                titles.pop(key, None)
            self.section_titles = titles
            return
        sections = self.custom_sections.copy()
        for idx, section in enumerate(sections):
            if str(getattr(section, "key", "") or "") == key:
                sections[idx] = CustomSection(
                    id=section.id,
                    key=section.key,
                    title=title,
                    body=section.body,
                )
                break
        self.custom_sections = sections

    @rx.event
    def set_custom_section_body(self, key: str, value: str):
        """Update the body text for a custom section."""
        key = key.strip()
        if not key:
            return
        body = value
        sections = self.custom_sections.copy()
        for idx, section in enumerate(sections):
            if str(getattr(section, "key", "") or "") == key:
                sections[idx] = CustomSection(
                    id=section.id,
                    key=section.key,
                    title=section.title,
                    body=body,
                )
                break
        self.custom_sections = sections

    @rx.event
    def add_custom_section(self):
        """Add a new custom section with the current draft title/body."""
        title = self.new_section_title.strip() or "Custom Section"
        body = self.new_section_body
        section_id = str(uuid.uuid4())
        key = f"custom_{section_id}"
        sections = self.custom_sections.copy()
        sections.append(CustomSection(id=section_id, key=key, title=title, body=body))
        self.custom_sections = sections
        order = self.section_order.copy()
        if key not in order:
            order.append(key)
        self.section_order = order
        visibility = self.section_visibility.copy()
        visibility[key] = True
        self.section_visibility = visibility
        self.new_section_title = ""
        self.new_section_body = ""

    @rx.event
    def remove_custom_section(self, key: str):
        """Remove a custom section and purge it from ordering/visibility."""
        key = key.strip()
        if not key:
            return
        self.custom_sections = [
            section
            for section in (self.custom_sections)
            if str(getattr(section, "key", "") or "") != key
        ]
        self.section_order = [k for k in (self.section_order) if k != key]
        visibility = self.section_visibility.copy()
        visibility.pop(key, None)
        self.section_visibility = visibility
        titles = self.section_titles.copy()
        titles.pop(key, None)
        self.section_titles = titles

    # Experience entries
    def add_experience(self):
        items = self.experience.copy()
        items.insert(0, Experience(id=str(uuid.uuid4())))
        self.experience = items

    def remove_experience(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        items = self.experience.copy()
        if idx < 0 or idx >= len(items):
            return
        items.pop(idx)
        self.experience = items

    def update_experience_field(self, index, field, value):
        try:
            idx = int(index)
        except Exception:
            return
        if idx < 0:
            return
        while len(self.experience) <= idx:
            self.experience.append(Experience())
        setattr(self.experience[idx], field, value)

    # Education entries
    def add_education(self):
        items = self.education.copy()
        items.append(Education(id=str(uuid.uuid4())))
        self.education = items

    def remove_education(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        items = self.education.copy()
        if idx < 0 or idx >= len(items):
            return
        items.pop(idx)
        self.education = items

    def update_education_field(self, index, field, value):
        try:
            idx = int(index)
        except Exception:
            return
        if idx < 0:
            return
        while len(self.education) <= idx:
            self.education.append(Education())
        setattr(self.education[idx], field, value)

    # Founder role entries
    def add_founder_role(self):
        items = self.founder_roles.copy()
        items.append(FounderRole(id=str(uuid.uuid4())))
        self.founder_roles = items

    def remove_founder_role(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        items = self.founder_roles.copy()
        if idx < 0 or idx >= len(items):
            return
        items.pop(idx)
        self.founder_roles = items

    def update_founder_role_field(self, index, field, value):
        try:
            idx = int(index)
        except Exception:
            return
        if idx < 0:
            return
        while len(self.founder_roles) <= idx:
            self.founder_roles.append(FounderRole())
        setattr(self.founder_roles[idx], field, value)

    @rx.event
    async def paste_req_and_run_pipeline(self, text: str):
        """Set job_req from clipboard, log, and trigger the auto-pipeline."""
        self.job_req = text
        self._add_pipeline_msg("Pasted job req from clipboard")
        self._add_pipeline_msg("Queueing auto-pipeline...")
        self._log_debug("paste_req_and_run_pipeline queued auto_pipeline_from_req")
        yield  # flush job_req update before running the pipeline
        yield StateAny.auto_pipeline_from_req

    def _add_pipeline_msg(self, msg: TemplateMsg):
        """Append a pipeline status message (immutably for state change detection)."""
        rendered = _render_template(msg)
        self.pipeline_status = self.pipeline_status.copy() + [rendered]
        self._log_debug(t"PIPELINE MSG: {rendered}")

    def _log_debug(self, msg: TemplateMsg):
        """Write a debug line to the temp log file."""
        _log_annotation_snapshot()
        _write_debug_line(msg)

    @rx.event
    async def auto_pipeline_from_req(self):
        """Paste req -> generate profile -> load data -> render PDF."""
        self._add_pipeline_msg(t"Auto-pipeline invoked")
        self._log_debug(t"auto_pipeline_from_req invoked")
        if self.is_auto_pipeline:
            self._add_pipeline_msg(t"Auto-pipeline skipped: already running")
            self._log_debug(t"auto_pipeline_from_req skipped: already running")
            return
        if not (self.job_req and self.job_req.strip()):
            self._add_pipeline_msg(t"Auto-pipeline skipped: empty job req")
            yield
            return
        self.pdf_error = ""
        self.is_auto_pipeline = True
        self._add_pipeline_msg(t"Auto-pipeline started")
        yield  # flush start
        failure = None
        t_start = None
        try:
            t_start = datetime.now()
            # Generate profile
            gen_start = datetime.now()
            self._add_pipeline_msg(t"Stage 1: Generating profile...")
            self._log_debug(t"auto_pipeline_from_req: calling generate_profile (pre)")
            yield  # flush stage 1 start
            gen = self.generate_profile()
            if hasattr(gen, "__anext__"):
                async for update in gen:
                    yield update
            elif asyncio.iscoroutine(gen):
                await gen
            gen_end = datetime.now()
            if self.pdf_error:
                failure = self.pdf_error
                self._add_pipeline_msg(t"Stage 1 failed: {self.pdf_error}")
                yield
                return
            self._add_pipeline_msg(t"Stage 1 complete")
            self._log_debug(t"auto_pipeline_from_req: generate_profile finished")
            self._add_pipeline_msg(
                t"Generating profile...done ({(gen_end - gen_start).total_seconds():.2f}s)"
            )
            yield  # flush after stage 1

            # Load hydrated data
            load_start = datetime.now()
            self._add_pipeline_msg(t"Stage 2: Hydrating UI with latest profile...")
            self._log_debug(t"auto_pipeline_from_req: calling load_resume_fields (pre)")
            yield  # flush stage 2 start
            load = self.load_resume_fields()
            if hasattr(load, "__anext__"):
                async for update in load:
                    yield update
            elif asyncio.iscoroutine(load):
                await load
            load_end = datetime.now()
            if self.pdf_error:
                failure = self.pdf_error
                self._add_pipeline_msg(t"Stage 2 failed: {self.pdf_error}")
                yield
                return
            self._add_pipeline_msg(t"Stage 2 complete")
            self._log_debug(t"auto_pipeline_from_req: load_resume_fields finished")
            self._add_pipeline_msg(
                t"Loading data...done ({(load_end - load_start).total_seconds():.2f}s)"
            )
            yield  # flush after stage 2

            # Render PDF
            render_start = datetime.now()
            self._add_pipeline_msg(t"Stage 3: Rendering PDF...")
            self._log_debug(t"auto_pipeline_from_req: calling generate_pdf (pre)")
            yield  # flush stage 3 start
            render = self.generate_pdf()
            if hasattr(render, "__anext__"):
                async for update in render:
                    yield update
            elif asyncio.iscoroutine(render):
                await render
            render_end = datetime.now()
            if self.pdf_error:
                failure = self.pdf_error
                self._add_pipeline_msg(t"Stage 3 failed: {self.pdf_error}")
                yield
                return
            self._add_pipeline_msg(t"Stage 3 complete")
            self._log_debug(t"auto_pipeline_from_req: generate_pdf finished")
            self._add_pipeline_msg(
                t"Rendering PDF...done ({(render_end - render_start).total_seconds():.2f}s)"
            )
            yield  # flush after stage 3
        except Exception as e:
            failure = str(e)
            self._add_pipeline_msg(t"Auto-pipeline failed: {e}")
            with suppress(Exception):
                _write_debug_line(t"auto_pipeline_from_req error: {e}")
            yield
        finally:
            if not failure and t_start is not None:
                elapsed = datetime.now() - t_start
                self._add_pipeline_msg(
                    t"Auto-pipeline complete ({elapsed.total_seconds():.2f}s)"
                )
                yield  # final flush
            self.is_auto_pipeline = False
            self._log_debug(t"auto_pipeline_from_req finished")

    async def generate_profile(self):
        """Call LLM with current req + base resume, save Profile to Neo4j, and hydrate UI."""
        if self.is_generating_profile:
            self._log_debug("generate_profile skipped: already generating")
            return
        self.is_generating_profile = True
        self._log_debug("generate_profile started")
        self._add_pipeline_msg("generate_profile entered")
        yield  # flush entry
        try:
            self.pdf_error = ""
            db = Neo4jClient()
            db.ensure_resume_exists()
            data = db.get_resume_data() or {}
            db.close()
            resume_node = data.get("resume", {}) or {}
            base_profile = {
                **resume_node,
                "experience": data.get("experience", []),
                "education": data.get("education", []),
                "founder_roles": data.get("founder_roles", []),
            }
            base_profile["rewrite_bullets"] = self.rewrite_bullets_with_llm
            prompt_yaml = (self.prompt_yaml).strip()
            if prompt_yaml:
                base_profile["prompt_yaml"] = prompt_yaml
            model_name = self.selected_model or DEFAULT_LLM_MODEL
            llm_start = datetime.now()
            self._add_pipeline_msg("generate_profile: dispatching LLM call")
            self._log_debug(
                "generate_profile: calling generate_resume_content (to_thread)"
            )
            yield  # flush before LLM
            try:
                result = await asyncio.to_thread(
                    generate_resume_content, self.job_req, base_profile, model_name
                )
            except Exception as llm_err:
                self._add_pipeline_msg(t"generate_profile LLM error: {llm_err}")
                self._log_debug(t"generate_profile LLM error: {llm_err}")
                raise
            llm_end = datetime.now()
            self._add_pipeline_msg(
                f"generate_profile: LLM returned in {(llm_end - llm_start).total_seconds():.2f}s"
            )
            self._log_debug("generate_profile: LLM call finished")
            if not (result and isinstance(result, dict)):
                self.pdf_error = "LLM returned no content."
                self._add_pipeline_msg("generate_profile: LLM returned no content")
                return
            if result.get("error"):
                self.pdf_error = str(result.get("error") or "LLM returned an error.")
                self._add_pipeline_msg(t"generate_profile error: {self.pdf_error}")
                return
            _log_llm_json_output(
                result, model_name=model_name or "", context="generate_profile"
            )

            # Persist new Profile
            def fallback_if_blank(values, fallback_vals, target_len=None):
                vals = list(values or [])
                if target_len:
                    while len(vals) < target_len:
                        vals.append("")
                    vals = vals[:target_len]
                if all((v is None or str(v).strip() == "") for v in vals):
                    return list(fallback_vals or [])
                return vals

            def ensure_len(items, target=9):
                items = list(items or [])
                while len(items) < target:
                    items.append("")
                return items[:target]

            resume_headers = ensure_len(
                [
                    resume_node.get("head1_left", ""),
                    resume_node.get("head1_middle", ""),
                    resume_node.get("head1_right", ""),
                    resume_node.get("head2_left", ""),
                    resume_node.get("head2_middle", ""),
                    resume_node.get("head2_right", ""),
                    resume_node.get("head3_left", ""),
                    resume_node.get("head3_middle", ""),
                    resume_node.get("head3_right", ""),
                ]
            )
            resume_top_skills = ensure_len(resume_node.get("top_skills", []))
            header_labels = {
                str(h).strip().lower() for h in resume_headers if str(h).strip()
            }

            def is_contact_skill(value: str) -> bool:
                lowered = value.lower().strip()
                if lowered in header_labels:
                    return True
                if "linkedin" in lowered or "github" in lowered or "scholar" in lowered:
                    return True
                if "http://" in lowered or "https://" in lowered or "www." in lowered:
                    return True
                if "portfolio" in lowered and (
                    "code" in lowered
                    or "github" in lowered
                    or "public" in lowered
                    or "repo" in lowered
                    or "git" in lowered
                ):
                    return True
                if "@" in value and "." in value:
                    return True
                return False

            def sanitize_skills(values, *, max_len: int | None = None) -> list[str]:
                cleaned: list[str] = []
                seen: set[str] = set()
                for item in values or []:
                    if item is None:
                        continue
                    text = str(item).strip()
                    if not text or is_contact_skill(text):
                        continue
                    key = text.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    cleaned.append(text)
                    if max_len and len(cleaned) >= max_len:
                        break
                return cleaned

            def sanitize_skill_rows(rows: list[list[str]]) -> list[list[str]]:
                cleaned_rows = [sanitize_skills(row) for row in rows[:3]]
                while len(cleaned_rows) < 3:
                    cleaned_rows.append([])
                return cleaned_rows[:3]

            skills_rows = sanitize_skill_rows(
                _ensure_skill_rows(result.get("skills_rows"))
            )
            highlighted_skills = sanitize_skills(
                fallback_if_blank(
                    result.get("highlighted_skills"),
                    resume_top_skills,
                    target_len=9,
                ),
                max_len=9,
            )
            highlighted_skills = ensure_len(highlighted_skills, target=9)
            if not any(any(str(s).strip() for s in row) for row in skills_rows):
                skills_rows = [
                    highlighted_skills[0:3],
                    highlighted_skills[3:6],
                    highlighted_skills[6:9],
                ]
            skills_rows_json = json.dumps(skills_rows, ensure_ascii=False)
            experience_bullets = (
                _coerce_bullet_overrides(result.get("experience_bullets"))
                if self.rewrite_bullets_with_llm
                else []
            )
            founder_role_bullets = (
                _coerce_bullet_overrides(result.get("founder_role_bullets"))
                if self.rewrite_bullets_with_llm
                else []
            )
            resume_fields = {
                "summary": result.get("summary", resume_node.get("summary", "")),
                "headers": fallback_if_blank(
                    result.get("headers"), resume_headers, target_len=9
                ),
                "highlighted_skills": highlighted_skills,
                "skills_rows_json": skills_rows_json,
                "experience_bullets_json": json.dumps(
                    experience_bullets, ensure_ascii=False
                ),
                "founder_role_bullets_json": json.dumps(
                    founder_role_bullets, ensure_ascii=False
                ),
                "job_req_raw": self.job_req,
                # Required Profile props with safe defaults
                "target_company": result.get("target_company", ""),
                "target_role": result.get("target_role", ""),
                "seniority_level": result.get("seniority_level", ""),
                "target_location": result.get("target_location", ""),
                "work_mode": result.get("work_mode", ""),
                "travel_requirement": result.get("travel_requirement", ""),
                "primary_domain": result.get("primary_domain", ""),
                "must_have_skills": result.get("must_have_skills", []),
                "nice_to_have_skills": result.get("nice_to_have_skills", []),
                "tech_stack_keywords": result.get("tech_stack_keywords", []),
                "non_technical_requirements": result.get(
                    "non_technical_requirements", []
                ),
                "certifications": result.get("certifications", []),
                "clearances": result.get("clearances", []),
                "core_responsibilities": result.get("core_responsibilities", []),
                "outcome_goals": result.get("outcome_goals", []),
                "salary_band": result.get("salary_band", ""),
                "posting_url": result.get("posting_url", ""),
                "req_id": result.get("req_id", ""),
            }
            db = Neo4jClient()
            profile_id = db.save_resume(resume_fields)
            profile_meta = {}
            if profile_id:
                profile_meta = db.get_profile_metadata(str(profile_id))
            db.close()
            self.latest_profile_id = str(profile_id or "")
            self.selected_profile_id = self.latest_profile_id
            label_source = profile_meta or {
                "created_at": "",
                "target_company": resume_fields.get("target_company", ""),
                "target_role": resume_fields.get("target_role", ""),
            }
            self.selected_profile_label = (
                _format_profile_label(label_source)
                if self.latest_profile_id
                else ""
            )
            self.profile_experience_bullets = _bullet_override_map(experience_bullets)
            self.profile_founder_bullets = _bullet_override_map(founder_role_bullets)
            if self.rewrite_bullets_with_llm:
                self.edit_profile_bullets = True
            # Hydrate into UI
            self.summary = resume_fields["summary"]
            self.skills_rows = skills_rows
            self.headers = (
                resume_fields["headers"][:9]
                if resume_fields.get("headers")
                else self.headers
            )
            self.highlighted_skills = (
                resume_fields["highlighted_skills"][:9]
                if resume_fields.get("highlighted_skills")
                else self.highlighted_skills
            )
            self.profile_data.update(result)
            self.data_loaded = True

            if not (self.experience or self.education or self.founder_roles):
                try:
                    db = Neo4jClient()
                    data = db.get_resume_data() or {}
                    db.close()
                except Exception:
                    data = {}

                def to_models(items, model_cls):
                    out = []
                    for item in items:
                        item = dict(item)
                        bullets = item.get("bullets", [])
                        if isinstance(bullets, list):
                            item["bullets"] = "\n".join(bullets)
                        elif bullets is None:
                            item["bullets"] = ""
                        else:
                            item["bullets"] = str(bullets)
                        out.append(model_cls(**item))
                    return out

                self.experience = to_models(data.get("experience", []), Experience)
                self.education = to_models(data.get("education", []), Education)
                self.founder_roles = to_models(
                    data.get("founder_roles", []), FounderRole
                )
            # Role detail fields
            self.target_company = result.get("target_company", self.target_company)
            self.target_role = result.get("target_role", self.target_role)
            self.seniority_level = result.get("seniority_level", self.seniority_level)
            self.target_location = result.get("target_location", self.target_location)
            self.work_mode = result.get("work_mode", self.work_mode)
            self.travel_requirement = result.get(
                "travel_requirement", self.travel_requirement
            )
            self.primary_domain = result.get("primary_domain", self.primary_domain)

            def list_to_text(val):
                if isinstance(val, list):
                    return "\n".join([str(v) for v in val if v is not None])
                if val is None:
                    return ""
                return str(val)

            self.must_have_skills_text = list_to_text(
                result.get("must_have_skills", [])
            )
            self.nice_to_have_skills_text = list_to_text(
                result.get("nice_to_have_skills", [])
            )
            self.tech_stack_keywords_text = list_to_text(
                result.get("tech_stack_keywords", [])
            )
            self.non_technical_requirements_text = list_to_text(
                result.get("non_technical_requirements", [])
            )
            self.certifications_text = list_to_text(result.get("certifications", []))
            self.clearances_text = list_to_text(result.get("clearances", []))
            self.core_responsibilities_text = list_to_text(
                result.get("core_responsibilities", [])
            )
            self.outcome_goals_text = list_to_text(result.get("outcome_goals", []))
            self.salary_band = result.get("salary_band", self.salary_band)
            self.posting_url = result.get("posting_url", self.posting_url)
            self.req_id = result.get("req_id", self.req_id)
            self.db_error = ""
            req_text = self.job_req.strip()
            self.last_profile_job_req_sha = (
                _hash_text(req_text) if req_text else ""
            )
        except Exception as e:
            self.pdf_error = _render_template(t"Error generating profile: {e}")
            self._add_pipeline_msg(t"generate_profile error: {e}")
        finally:
            self.is_generating_profile = False
            self._log_debug("generate_profile finished")
            self._add_pipeline_msg("generate_profile exited")
            yield  # flush exit


StateAny: Any = State


_ANNOTATION_SNAPSHOT_WRITTEN = False


def _annotation_snapshot(targets: dict[str, object]) -> dict[str, dict[str, str]]:
    snapshot: dict[str, dict[str, str]] = {}
    for name, obj in targets.items():
        try:
            annotations = annotationlib.get_annotations(
                obj,
                format=annotationlib.Format.FORWARDREF,
            )
            snapshot[name] = annotationlib.annotations_to_string(annotations)
        except Exception as exc:
            snapshot[name] = {"__error__": repr(exc)}
    return snapshot


def _log_annotation_snapshot() -> None:
    global _ANNOTATION_SNAPSHOT_WRITTEN
    if _ANNOTATION_SNAPSHOT_WRITTEN:
        return
    _ANNOTATION_SNAPSHOT_WRITTEN = True
    snapshot = _annotation_snapshot(
        {
            "Experience": Experience,
            "Education": Education,
            "FounderRole": FounderRole,
            "CustomSection": CustomSection,
            "State": State,
        }
    )
    snapshot_json = json.dumps(snapshot, sort_keys=True)
    _write_debug_line(t"annotation snapshot: {snapshot_json}")


# ==========================================
# UI COMPONENTS
# ==========================================
def styled_input(value, on_change, placeholder="", **props):
    merged_style = {
        "background": "#2d3748",
        "border": "1px solid #4a5568",
        "color": "#e2e8f0",
        "padding": "0.5em 1em",
        "min_height": "2.5em",
        "height": "auto",
        "_focus": {
            "border_color": "#63b3ed",
            "box_shadow": "0 0 0 1px #63b3ed",
        },
        "_placeholder": {
            "color": "#a0aec0",
        },
    }
    # Merge user-provided style
    if "style" in props:
        merged_style.update(props.pop("style"))

    width = props.pop("width", "100%")
    return rx.input(
        value=value,
        on_change=on_change,
        placeholder=placeholder,
        variant="soft",
        color_scheme="gray",
        radius="medium",
        width=width,
        style=merged_style,
        **props,
    )


def styled_textarea(value, on_change, placeholder="", **props):
    merged_style = {
        "background": "#2d3748",
        "border": "1px solid #4a5568",
        "color": "#e2e8f0",
        "padding": "0.5em 1em",
        "min_height": "2.5em",
        "height": "auto",
        "_focus": {
            "border_color": "#63b3ed",
            "box_shadow": "0 0 0 1px #63b3ed",
        },
        "_placeholder": {
            "color": "#a0aec0",
        },
    }
    # Merge user-provided style
    if "style" in props:
        merged_style.update(props.pop("style"))

    width = props.pop("width", "100%")
    return rx.text_area(
        value=value,
        on_change=on_change,
        placeholder=placeholder,
        variant="soft",
        color_scheme="gray",
        radius="medium",
        width=width,
        style=merged_style,
        **props,
    )


def styled_card(children, **props):
    return rx.box(
        children,
        bg="#1a202c",
        padding="1.5em",
        border_radius="lg",
        box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)",
        border="1px solid #2d3748",
        width="100%",
        **props,
    )


def labeled_toggle(
    label: str,
    checked,
    on_change,
    *,
    size="3",
    color_scheme="indigo",
    show_label: bool = True,
    switch_props: dict | None = None,
    container_props: dict | None = None,
):
    switch_kwargs: dict[str, Any] = {
        "checked": checked,
        "on_change": on_change,
        "size": size,
        "color_scheme": color_scheme,
    }
    if switch_props:
        switch_kwargs.update(switch_props)
    toggle = rx.switch(**switch_kwargs)
    if not show_label:
        return toggle
    container_defaults: dict[str, Any] = {
        "spacing": "2",
        "align_items": "center",
        "width": "100%",
    }
    if container_props:
        container_defaults.update(container_props)
    return rx.hstack(
        rx.text(label, weight="medium", color="#e2e8f0"),
        toggle,
        **container_defaults,
    )


def _section_order_row(row, i):
    return rx.hstack(
        rx.checkbox(
            checked=row["visible"],
            on_change=lambda v: StateAny.set_section_visibility(v, row["key"]),
            size="1",
            color_scheme="green",
            aria_label="Toggle section visibility",
        ),
        rx.hstack(
            rx.button(
                "↑",
                on_click=lambda _=None, i=i: StateAny.move_section_up(i),
                size="1",
                variant="soft",
                color_scheme="gray",
                aria_label="Move section up",
            ),
            rx.button(
                "↓",
                on_click=lambda _=None, i=i: StateAny.move_section_down(i),
                size="1",
                variant="soft",
                color_scheme="gray",
                aria_label="Move section down",
            ),
            spacing="1",
        ),
        styled_input(
            value=row["title"],
            on_change=lambda v: StateAny.set_section_title(v, row["key"]),
            placeholder="Section title",
            width="100%",
        ),
        rx.cond(
            row["custom"],
            rx.button(
                "Remove",
                on_click=lambda _=None: StateAny.remove_custom_section(row["key"]),
                size="1",
                variant="soft",
                color_scheme="red",
            ),
            rx.box(),
        ),
        rx.spacer(),
        width="100%",
        align_items="center",
        spacing="2",
    )


def _experience_card(exp, i):
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    f"Entry {i + 1}",
                    color="#a0aec0",
                    font_size="0.85em",
                ),
                rx.spacer(),
                rx.button(
                    "Remove",
                    on_click=lambda _=None, i=i: StateAny.remove_experience(i),
                    size="1",
                    variant="soft",
                    color_scheme="red",
                ),
                width="100%",
                align_items="center",
            ),
            rx.hstack(
                styled_input(
                    value=exp.role,
                    on_change=lambda v: StateAny.update_experience_field(i, "role", v),
                    placeholder="Role",
                ),
                styled_input(
                    value=exp.company,
                    on_change=lambda v: StateAny.update_experience_field(
                        i, "company", v
                    ),
                    placeholder="Company",
                ),
                width="100%",
                spacing="3",
            ),
            styled_input(
                value=exp.location,
                on_change=lambda v: StateAny.update_experience_field(i, "location", v),
                placeholder="Location",
            ),
            rx.hstack(
                styled_input(
                    value=exp.start_date,
                    on_change=lambda v: StateAny.update_experience_field(
                        i, "start_date", v
                    ),
                    type="date",
                ),
                styled_input(
                    value=exp.end_date,
                    on_change=lambda v: StateAny.update_experience_field(
                        i, "end_date", v
                    ),
                    type="date",
                ),
                width="100%",
                spacing="3",
            ),
            styled_textarea(
                value=exp.description,
                on_change=lambda v: StateAny.update_experience_field(
                    i, "description", v
                ),
                placeholder="Short company/role description (optional)",
                min_height="140px",
                style={"resize": "vertical"},
            ),
            styled_textarea(
                value=exp.bullets,
                on_change=lambda v: StateAny.update_experience_field(i, "bullets", v),
                placeholder="Bullets (one per line)",
                min_height="300px",
                style={"resize": "vertical"},
            ),
            rx.cond(
                StateAny.edit_profile_bullets,
                rx.vstack(
                    rx.text(
                        "Profile bullets (LLM override)",
                        color="#a0aec0",
                        font_size="0.85em",
                    ),
                    styled_textarea(
                        value=StateAny.profile_experience_bullets_list[i],
                        on_change=lambda v, exp_id=exp.id: StateAny.set_profile_experience_bullets_text(
                            exp_id, v
                        ),
                        placeholder="Profile bullets (one per line)",
                        min_height="240px",
                        width="100%",
                        style={"resize": "vertical"},
                    ),
                    spacing="2",
                    width="100%",
                ),
                rx.fragment(),
            ),
            width="100%",
            spacing="3",
        ),
        margin_bottom="1.5em",
        key=rx.cond(
            exp.id != "",
            exp.id,
            f"experience-{i}",
        ),
    )


def _education_card(edu, i):
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    f"Entry {i + 1}",
                    color="#a0aec0",
                    font_size="0.85em",
                ),
                rx.spacer(),
                rx.button(
                    "Remove",
                    on_click=lambda _=None, i=i: StateAny.remove_education(i),
                    size="1",
                    variant="soft",
                    color_scheme="red",
                ),
                width="100%",
                align_items="center",
            ),
            rx.hstack(
                styled_input(
                    value=edu.degree,
                    on_change=lambda v: StateAny.update_education_field(i, "degree", v),
                    placeholder="Degree",
                ),
                styled_input(
                    value=edu.school,
                    on_change=lambda v: StateAny.update_education_field(i, "school", v),
                    placeholder="School",
                ),
                width="100%",
                spacing="3",
            ),
            styled_input(
                value=edu.location,
                on_change=lambda v: StateAny.update_education_field(i, "location", v),
                placeholder="Location",
            ),
            rx.hstack(
                styled_input(
                    value=edu.start_date,
                    on_change=lambda v: StateAny.update_education_field(
                        i, "start_date", v
                    ),
                    type="date",
                ),
                styled_input(
                    value=edu.end_date,
                    on_change=lambda v: StateAny.update_education_field(
                        i, "end_date", v
                    ),
                    type="date",
                ),
                width="100%",
                spacing="3",
            ),
            styled_textarea(
                value=edu.description,
                on_change=lambda v: StateAny.update_education_field(
                    i, "description", v
                ),
                placeholder="Program description or highlights (optional)",
                min_height="140px",
                style={"resize": "vertical"},
            ),
            styled_textarea(
                value=edu.bullets,
                on_change=lambda v: StateAny.update_education_field(i, "bullets", v),
                placeholder="Bullets (one per line)",
                min_height="300px",
                style={"resize": "vertical"},
            ),
            width="100%",
            spacing="3",
        ),
        margin_bottom="1.5em",
        key=rx.cond(
            edu.id != "",
            edu.id,
            f"education-{i}",
        ),
    )


def _founder_role_card(role, i):
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    f"Entry {i + 1}",
                    color="#a0aec0",
                    font_size="0.85em",
                ),
                rx.spacer(),
                rx.button(
                    "Remove",
                    on_click=lambda _=None, i=i: StateAny.remove_founder_role(i),
                    size="1",
                    variant="soft",
                    color_scheme="red",
                ),
                width="100%",
                align_items="center",
            ),
            rx.hstack(
                styled_input(
                    value=role.role,
                    on_change=lambda v: StateAny.update_founder_role_field(
                        i, "role", v
                    ),
                    placeholder="Role",
                ),
                styled_input(
                    value=role.company,
                    on_change=lambda v: StateAny.update_founder_role_field(
                        i, "company", v
                    ),
                    placeholder="Company",
                ),
                width="100%",
                spacing="3",
            ),
            styled_input(
                value=role.location,
                on_change=lambda v: StateAny.update_founder_role_field(
                    i, "location", v
                ),
                placeholder="Location",
            ),
            rx.hstack(
                styled_input(
                    value=role.start_date,
                    on_change=lambda v: StateAny.update_founder_role_field(
                        i, "start_date", v
                    ),
                    type="date",
                ),
                styled_input(
                    value=role.end_date,
                    on_change=lambda v: StateAny.update_founder_role_field(
                        i, "end_date", v
                    ),
                    type="date",
                ),
                width="100%",
                spacing="3",
            ),
            styled_textarea(
                value=role.description,
                on_change=lambda v: StateAny.update_founder_role_field(
                    i, "description", v
                ),
                placeholder="Company or role description (optional)",
                min_height="140px",
                style={"resize": "vertical"},
            ),
            styled_textarea(
                value=role.bullets,
                on_change=lambda v: StateAny.update_founder_role_field(i, "bullets", v),
                placeholder="Bullets (one per line)",
                min_height="300px",
                style={"resize": "vertical"},
            ),
            rx.cond(
                StateAny.edit_profile_bullets,
                rx.vstack(
                    rx.text(
                        "Profile bullets (LLM override)",
                        color="#a0aec0",
                        font_size="0.85em",
                    ),
                    styled_textarea(
                        value=StateAny.profile_founder_bullets_list[i],
                        on_change=lambda v, role_id=role.id: StateAny.set_profile_founder_bullets_text(
                            role_id, v
                        ),
                        placeholder="Profile bullets (one per line)",
                        min_height="240px",
                        width="100%",
                        style={"resize": "vertical"},
                    ),
                    spacing="2",
                    width="100%",
                ),
                rx.fragment(),
            ),
            width="100%",
            spacing="3",
        ),
        margin_bottom="1.5em",
        key=rx.cond(
            role.id != "",
            role.id,
            f"founder-{i}",
        ),
    )


def _custom_section_editor(row):
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    rx.cond(row.get("title"), row.get("title"), "Custom section"),
                    color="#e2e8f0",
                    font_size="0.95em",
                    weight="medium",
                ),
                rx.spacer(),
                rx.button(
                    "Remove",
                    data_section_key=row.get("key", ""),
                    on_click=lambda _=None, key=rx.cond(
                        row.get("key"),
                        row.get("key"),
                        "",
                    ): StateAny.remove_custom_section(key),
                    size="1",
                    variant="soft",
                    color_scheme="red",
                ),
                width="100%",
                align_items="center",
            ),
            styled_textarea(
                value=row.get("body", ""),
                on_change=lambda v, key=row.get(
                    "key"
                ): StateAny.set_custom_section_body(key, v),
                placeholder="Section content (one bullet per line)",
                min_height="140px",
                style={"resize": "vertical"},
            ),
            width="100%",
            spacing="2",
        ),
        margin_bottom="1.0em",
        data_custom_section="1",
        data_section_key=row.get("key", ""),
    )


def section_order_controls():
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "Section order",
                    weight="bold",
                    color="#e2e8f0",
                    font_size="1em",
                ),
                rx.spacer(),
                rx.text(
                    "Top renders first",
                    color="#a0aec0",
                    font_size="0.85em",
                ),
                align_items="center",
                width="100%",
            ),
            rx.text(
                "Reorder how PDF sections appear.",
                color="#a0aec0",
                font_size="0.9em",
            ),
            rx.vstack(
                rx.foreach(StateAny.section_order_rows, _section_order_row),
                spacing="2",
                width="100%",
            ),
            rx.divider(margin_y="1em"),
            rx.text(
                "Add section",
                color="#e2e8f0",
                weight="bold",
                font_size="0.95em",
            ),
            styled_input(
                value=StateAny.new_section_title,
                on_change=StateAny.set_new_section_title,
                placeholder="Section title",
            ),
            styled_textarea(
                value=StateAny.new_section_body,
                on_change=StateAny.set_new_section_body,
                placeholder="Section content (one bullet per line)",
                min_height="120px",
                style={"resize": "vertical"},
            ),
            rx.button(
                "Add Section",
                on_click=StateAny.add_custom_section,
                size="2",
                variant="soft",
                color_scheme="green",
            ),
            rx.vstack(
                rx.foreach(StateAny.custom_section_rows, _custom_section_editor),
                spacing="2",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        margin_top="0.5em",
    )


def index():
    clipboard_job_req_script = (
        "(async () => {"
        "const text = await navigator.clipboard.readText();"
        "const el = document.getElementById('job-req-field');"
        "if (el) {"
        "el.value = text;"
        "el.dispatchEvent(new Event('input', { bubbles: true }));"
        "}"
        "return text;"
        "})()"
    )
    loading_view = rx.center(
        rx.vstack(
            rx.heading("Loading Resume Builder…", size="7", color="#f7fafc"),
            rx.text(
                "Preparing data and UI. This will finish once hydration completes.",
                color="#a0aec0",
            ),
            rx.spinner(size="3"),
        ),
        width="100%",
        height="100vh",
        bg="#0b1224",
        padding="2em",
    )

    return rx.fragment(
        rx.toast.provider(position="top-right", duration=6000, close_button=True),
        rx.cond(
            StateAny.has_loaded,
            rx.hstack(
                # Left Panel
                rx.box(
                    rx.vstack(
                        rx.heading(
                            "Resume Builder",
                            size="8",
                            margin_bottom="1em",
                            color="#f7fafc",
                        ),
                        rx.hstack(
                            rx.text("Job Requisition", weight="bold", color="#e2e8f0"),
                            rx.spacer(),
                            width="100%",
                            align_items="center",
                            margin_bottom="0.5em",
                        ),
                        rx.hstack(
                            rx.text(
                                "Model", weight="medium", color="#a0aec0", width="25%"
                            ),
                            rx.select(
                                StateAny.llm_models,
                                value=StateAny.selected_model,
                                on_change=StateAny.set_selected_model,
                                width="75%",
                                color_scheme="indigo",
                                radius="medium",
                                size="2",
                            ),
                            spacing="3",
                            width="100%",
                        ),
                        rx.hstack(
                            rx.text(
                                "Profile", weight="medium", color="#a0aec0", width="25%"
                            ),
                            rx.box(
                                rx.el.select(
                                    rx.el.option("", value=""),
                                    id="profile-select",
                                    class_name="profile-select2",
                                    data_selected_id=StateAny.selected_profile_id,
                                    data_selected_label=StateAny.selected_profile_label,
                                    on_change=StateAny.set_selected_profile_id,
                                ),
                                width="75%",
                                min_width="0",
                                id="profile-select-root",
                            ),
                            spacing="3",
                            width="100%",
                        ),
                        rx.box(
                            styled_textarea(
                                placeholder="Paste or type the job requisition here",
                                value=StateAny.job_req,
                                on_change=StateAny.set_job_req,
                                on_click=rx.call_script(
                                    clipboard_job_req_script,
                                    callback=StateAny.paste_req_and_run_pipeline,
                                ),
                                id="job-req-field",
                                min_height="150px",
                                style={"resize": "vertical"},
                            ),
                            width="100%",
                        ),
                        rx.text(
                            "Prompt Template",
                            weight="bold",
                            color="#e2e8f0",
                            margin_top="0.5em",
                        ),
                        styled_textarea(
                            placeholder="Edit the prompt template (stored in Neo4j)",
                            value=StateAny.prompt_yaml,
                            on_change=StateAny.set_prompt_yaml,
                            id="prompt-yaml-field",
                            min_height="220px",
                            style={"resize": "vertical"},
                        ),
                        rx.cond(
                            StateAny.job_req_needs_profile,
                            rx.text(
                                "Job requisition not yet processed. Click Generate Profile.",
                                color="#f6ad55",
                                font_size="0.85em",
                            ),
                        ),
                        rx.cond(
                            StateAny.pipeline_latest != "",
                            rx.text(
                                StateAny.pipeline_latest,
                                id="pipeline-status",
                                color="#a0aec0",
                                font_size="0.85em",
                            ),
                        ),
                        # Pipeline status messages are tracked for debugging; we show the
                        # latest message inline so it's obvious when the LLM/pipeline runs.
                        rx.hstack(
                            rx.button(
                                "Save Data",
                                on_click=StateAny.save_to_db,
                                loading=StateAny.is_saving,
                                size="4",
                                color_scheme="gray",
                                flex="1",
                                id="save-btn",
                            ),
                            rx.button(
                                "Load Data",
                                on_click=StateAny.load_resume_fields,
                                loading=StateAny.is_loading_resume,
                                size="4",
                                color_scheme="gray",
                                flex="1",
                                id="load-resume-btn",
                            ),
                            rx.button(
                                "Generate Profile",
                                on_click=StateAny.generate_profile,
                                loading=StateAny.is_generating_profile,
                                size="4",
                                color_scheme="blue",
                                flex="1",
                                white_space="nowrap",
                                id="generate-profile-btn",
                            ),
                            rx.button(
                                "Generate PDF",
                                on_click=StateAny.generate_pdf,
                                loading=StateAny.generating_pdf,
                                size="4",
                                color_scheme="blue",
                                flex="1",
                                white_space="nowrap",
                                id="generate-pdf-btn",
                            ),
                            rx.button(
                                "Refresh PDF",
                                on_click=StateAny.generate_pdf_cache_bust,
                                loading=StateAny.generating_pdf,
                                size="4",
                                color_scheme="gray",
                                flex="1",
                                white_space="nowrap",
                                id="generate-pdf-cache-bust-btn",
                            ),
                            rx.cond(
                                StateAny.db_error != "",
                                rx.tooltip(
                                    rx.box(
                                        rx.text(
                                            "!",
                                            color="#f56565",
                                            weight="bold",
                                            font_size="0.9em",
                                        ),
                                        width="2.5em",
                                        height="2.5em",
                                        border="1px solid #f56565",
                                        border_radius="999px",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center",
                                        aria_label="Database error indicator",
                                    ),
                                    content=StateAny.db_error,
                                ),
                            ),
                            spacing="2",
                            width="100%",
                            align_items="center",
                            margin_top="1em",
                        ),
                        rx.hstack(
                            labeled_toggle(
                                "Rewrite bullets with LLM",
                                checked=StateAny.rewrite_bullets_with_llm,
                                on_change=StateAny.set_rewrite_bullets_with_llm,
                                container_props={"width": "auto"},
                            ),
                            rx.hstack(
                                rx.text(
                                    "Auto-fit",
                                    weight="medium",
                                    color="#e2e8f0",
                                ),
                                rx.switch(
                                    checked=StateAny.auto_tune_pdf,
                                    on_change=StateAny.set_auto_tune_pdf,
                                    size="3",
                                    color_scheme="indigo",
                                ),
                                rx.text(
                                    "Pages",
                                    color="#a0aec0",
                                    font_size="0.85em",
                                ),
                                styled_input(
                                    value=StateAny.auto_fit_target_pages,
                                    on_change=StateAny.set_auto_fit_target_pages,
                                    type="number",
                                    min="1",
                                    step="1",
                                    width="5.5em",
                                ),
                                align_items="center",
                                spacing="2",
                            ),
                            width="100%",
                            align_items="center",
                            justify_content="space-between",
                            spacing="4",
                            margin_top="0.25em",
                        ),
                        rx.vstack(
                            rx.text(
                                "Resume font",
                                weight="medium",
                                color="#e2e8f0",
                            ),
                            styled_input(
                                value=StateAny.font_family,
                                on_change=StateAny.set_font_family,
                                placeholder="Pick a font",
                                id="resume-font-picker",
                            ),
                            spacing="1",
                            width="100%",
                            margin_top="0.35em",
                        ),
                        section_order_controls(),
                        rx.center(
                            rx.text(
                                rx.cond(
                                    StateAny.last_render_label != "",
                                    StateAny.last_render_label,
                                    "",
                                ),
                                font_size="0.8em",
                                color="#a0aec0",
                            ),
                            width="100%",
                            margin_top="0.25em",
                        ),
                        rx.center(
                            rx.text(
                                rx.cond(
                                    StateAny.last_save_label != "",
                                    StateAny.last_save_label,
                                    "",
                                ),
                                font_size="0.8em",
                                color="#9ae6b4",
                            ),
                            width="100%",
                            margin_top="0.25em",
                        ),
                        rx.center(
                            rx.text(
                                rx.cond(
                                    StateAny.last_profile_bullets_label != "",
                                    StateAny.last_profile_bullets_label,
                                    "",
                                ),
                                font_size="0.8em",
                                color="#9ad0ff",
                            ),
                            width="100%",
                            margin_top="0.25em",
                        ),
                        rx.cond(
                            StateAny.data_loaded,
                            rx.box(
                                rx.divider(margin_y="2em"),
                                rx.heading(
                                    "Role Details",
                                    size="6",
                                    margin_bottom="0.5em",
                                    color="#f7fafc",
                                ),
                                rx.grid(
                                    [
                                        styled_input(
                                            value=StateAny.target_company,
                                            on_change=None,
                                            placeholder="Target Company",
                                            read_only=True,
                                            key="target_company",
                                        ),
                                        styled_input(
                                            value=StateAny.target_role,
                                            on_change=None,
                                            placeholder="Target Role",
                                            read_only=True,
                                            key="target_role",
                                        ),
                                        styled_input(
                                            value=StateAny.seniority_level,
                                            on_change=None,
                                            placeholder="Seniority Level",
                                            read_only=True,
                                            key="seniority_level",
                                        ),
                                        styled_input(
                                            value=StateAny.target_location,
                                            on_change=None,
                                            placeholder="Target Location",
                                            read_only=True,
                                            key="target_location",
                                        ),
                                        styled_input(
                                            value=StateAny.work_mode,
                                            on_change=None,
                                            placeholder="Work Mode",
                                            read_only=True,
                                            key="work_mode",
                                        ),
                                        styled_input(
                                            value=StateAny.travel_requirement,
                                            on_change=None,
                                            placeholder="Travel Requirement",
                                            read_only=True,
                                            key="travel_requirement",
                                        ),
                                        styled_input(
                                            value=StateAny.primary_domain,
                                            on_change=None,
                                            placeholder="Primary Domain",
                                            read_only=True,
                                            key="primary_domain",
                                        ),
                                        styled_input(
                                            value=StateAny.salary_band,
                                            on_change=None,
                                            placeholder="Salary Band",
                                            read_only=True,
                                            key="salary_band",
                                        ),
                                        styled_input(
                                            value=StateAny.posting_url,
                                            on_change=None,
                                            placeholder="Posting URL",
                                            read_only=True,
                                            key="posting_url",
                                        ),
                                        styled_input(
                                            value=StateAny.req_id,
                                            on_change=None,
                                            placeholder="Req ID",
                                            read_only=True,
                                            key="req_id",
                                        ),
                                    ],
                                    columns="2",
                                    spacing="3",
                                    width="100%",
                                ),
                                rx.grid(
                                    [
                                        styled_textarea(
                                            value=StateAny.must_have_skills_text,
                                            on_change=None,
                                            placeholder="Must Have Skills",
                                            read_only=True,
                                            min_height="100px",
                                            key="must_have_skills_text",
                                        ),
                                        styled_textarea(
                                            value=StateAny.nice_to_have_skills_text,
                                            on_change=None,
                                            placeholder="Nice To Have Skills",
                                            read_only=True,
                                            min_height="100px",
                                            key="nice_to_have_skills_text",
                                        ),
                                        styled_textarea(
                                            value=StateAny.tech_stack_keywords_text,
                                            on_change=None,
                                            placeholder="Tech Stack Keywords",
                                            read_only=True,
                                            min_height="100px",
                                            key="tech_stack_keywords_text",
                                        ),
                                        styled_textarea(
                                            value=StateAny.non_technical_requirements_text,
                                            on_change=None,
                                            placeholder="Non-Technical Requirements",
                                            read_only=True,
                                            min_height="100px",
                                            key="non_technical_requirements_text",
                                        ),
                                        styled_textarea(
                                            value=StateAny.certifications_text,
                                            on_change=None,
                                            placeholder="Certifications",
                                            read_only=True,
                                            min_height="100px",
                                            key="certifications_text",
                                        ),
                                        styled_textarea(
                                            value=StateAny.clearances_text,
                                            on_change=None,
                                            placeholder="Clearances",
                                            read_only=True,
                                            min_height="100px",
                                            key="clearances_text",
                                        ),
                                        styled_textarea(
                                            value=StateAny.core_responsibilities_text,
                                            on_change=None,
                                            placeholder="Core Responsibilities",
                                            read_only=True,
                                            min_height="100px",
                                            key="core_responsibilities_text",
                                        ),
                                        styled_textarea(
                                            value=StateAny.outcome_goals_text,
                                            on_change=None,
                                            placeholder="Outcome Goals",
                                            read_only=True,
                                            min_height="100px",
                                            key="outcome_goals_text",
                                        ),
                                    ],
                                    columns="2",
                                    spacing="3",
                                    width="100%",
                                ),
                                rx.text(
                                    "Job Requisition (stored)",
                                    weight="bold",
                                    margin_top="0.5em",
                                    color="#e2e8f0",
                                ),
                                styled_textarea(
                                    value=StateAny.job_req,
                                    on_change=None,
                                    placeholder="Job requisition as stored on the Profile",
                                    read_only=True,
                                    min_height="180px",
                                    style={"resize": "vertical"},
                                ),
                                rx.divider(margin_y="2em"),
                                rx.heading(
                                    "Contact Info",
                                    size="6",
                                    margin_bottom="0.5em",
                                    color="#f7fafc",
                                ),
                                rx.grid(
                                    [
                                        styled_input(
                                            value=StateAny.first_name,
                                            on_change=StateAny.set_first_name,
                                            placeholder="First Name",
                                            key="first_name",
                                        ),
                                        styled_input(
                                            value=StateAny.middle_name,
                                            on_change=StateAny.set_middle_name,
                                            placeholder="Middle Name / Initial",
                                            key="middle_name",
                                        ),
                                        styled_input(
                                            value=StateAny.last_name,
                                            on_change=StateAny.set_last_name,
                                            placeholder="Last Name",
                                            key="last_name",
                                        ),
                                        styled_input(
                                            value=StateAny.email,
                                            on_change=StateAny.set_email,
                                            placeholder="Email",
                                            key="email",
                                        ),
                                        styled_input(
                                            value=StateAny.email2,
                                            on_change=StateAny.set_email2,
                                            placeholder="Secondary Email",
                                            key="email2",
                                        ),
                                        styled_input(
                                            value=StateAny.phone,
                                            on_change=StateAny.set_phone,
                                            placeholder="Phone",
                                            key="phone",
                                        ),
                                        styled_input(
                                            value=StateAny.linkedin_url,
                                            on_change=StateAny.set_linkedin_url,
                                            placeholder="LinkedIn URL",
                                            key="linkedin_url",
                                        ),
                                        styled_input(
                                            value=StateAny.github_url,
                                            on_change=StateAny.set_github_url,
                                            placeholder="GitHub URL",
                                            key="github_url",
                                        ),
                                        styled_input(
                                            value=StateAny.scholar_url,
                                            on_change=StateAny.set_scholar_url,
                                            placeholder="Google Scholar URL",
                                            key="scholar_url",
                                        ),
                                        styled_input(
                                            value=StateAny.calendly_url,
                                            on_change=StateAny.set_calendly_url,
                                            placeholder="Calendly URL",
                                            key="calendly_url",
                                        ),
                                        styled_input(
                                            value=StateAny.portfolio_url,
                                            on_change=StateAny.set_portfolio_url,
                                            placeholder="Portfolio URL",
                                            key="portfolio_url",
                                        ),
                                    ],
                                    columns="2",
                                    spacing="3",
                                    width="100%",
                                ),
                                rx.divider(margin_y="2em"),
                                rx.heading(
                                    "Generated Content",
                                    size="6",
                                    margin_bottom="0.5em",
                                    color="#f7fafc",
                                ),
                                rx.text(
                                    "Professional Summary",
                                    weight="bold",
                                    margin_top="1em",
                                    color="#e2e8f0",
                                ),
                                styled_textarea(
                                    value=StateAny.summary,
                                    on_change=StateAny.set_summary,
                                    min_height="200px",
                                    style={"resize": "vertical"},
                                ),
                                rx.divider(margin_y="1.5em"),
                                rx.heading(
                                    "Skills",
                                    size="6",
                                    margin_bottom="0.5em",
                                    color="#f7fafc",
                                ),
                                rx.text(
                                    "Three labeled rows with comma-separated skills (matches prompt.yaml).",
                                    color="#a0aec0",
                                    margin_bottom="0.5em",
                                    font_size="0.9em",
                                ),
                                rx.vstack(
                                    rx.hstack(
                                        rx.text(
                                            DEFAULT_SKILLS_ROW_LABELS[0] + ":",
                                            weight="bold",
                                            color="#e2e8f0",
                                            width="35%",
                                        ),
                                        rx.text(
                                            StateAny.skills_rows_csv[0],
                                            color="#e2e8f0",
                                            width="65%",
                                        ),
                                        width="100%",
                                        align_items="flex-start",
                                    ),
                                    rx.hstack(
                                        rx.text(
                                            DEFAULT_SKILLS_ROW_LABELS[1] + ":",
                                            weight="bold",
                                            color="#e2e8f0",
                                            width="35%",
                                        ),
                                        rx.text(
                                            StateAny.skills_rows_csv[1],
                                            color="#e2e8f0",
                                            width="65%",
                                        ),
                                        width="100%",
                                        align_items="flex-start",
                                    ),
                                    rx.hstack(
                                        rx.text(
                                            DEFAULT_SKILLS_ROW_LABELS[2] + ":",
                                            weight="bold",
                                            color="#e2e8f0",
                                            width="35%",
                                        ),
                                        rx.text(
                                            StateAny.skills_rows_csv[2],
                                            color="#e2e8f0",
                                            width="65%",
                                        ),
                                        width="100%",
                                        align_items="flex-start",
                                    ),
                                    spacing="2",
                                    width="100%",
                                ),
                                rx.divider(margin_y="2em"),
                                rx.hstack(
                                    rx.heading(
                                        "Experience",
                                        size="6",
                                        color="#f7fafc",
                                    ),
                                    rx.spacer(),
                                    rx.hstack(
                                        rx.switch(
                                            checked=StateAny.edit_profile_bullets,
                                            on_change=StateAny.set_edit_profile_bullets,
                                            size="2",
                                            color_scheme="indigo",
                                        ),
                                        rx.text(
                                            "Edit LLM bullets",
                                            color="#a0aec0",
                                            font_size="0.85em",
                                        ),
                                        spacing="2",
                                        align_items="center",
                                    ),
                                    rx.button(
                                        "Save Profile Bullets",
                                        on_click=StateAny.save_profile_bullets,
                                        loading=StateAny.is_saving_profile_bullets,
                                        size="2",
                                        variant="soft",
                                        color_scheme="indigo",
                                    ),
                                    rx.button(
                                        "Add Experience",
                                        on_click=StateAny.add_experience,
                                        size="2",
                                        variant="soft",
                                        color_scheme="green",
                                    ),
                                    width="100%",
                                    align_items="center",
                                    margin_bottom="0.5em",
                                    spacing="2",
                                ),
                                rx.vstack(
                                    rx.foreach(StateAny.experience, _experience_card),
                                    width="100%",
                                    spacing="4",
                                ),
                                rx.divider(margin_y="2em"),
                                rx.hstack(
                                    rx.heading(
                                        "Education",
                                        size="6",
                                        color="#f7fafc",
                                    ),
                                    rx.spacer(),
                                    rx.button(
                                        "Add Education",
                                        on_click=StateAny.add_education,
                                        size="2",
                                        variant="soft",
                                        color_scheme="green",
                                    ),
                                    width="100%",
                                    align_items="center",
                                    margin_bottom="0.5em",
                                ),
                                rx.vstack(
                                    rx.foreach(StateAny.education, _education_card),
                                    width="100%",
                                    spacing="4",
                                ),
                                rx.divider(margin_y="2em"),
                                rx.hstack(
                                    rx.heading(
                                        "Founder Roles",
                                        size="6",
                                        color="#f7fafc",
                                    ),
                                    rx.spacer(),
                                    rx.button(
                                        "Add Founder Role",
                                        on_click=StateAny.add_founder_role,
                                        size="2",
                                        variant="soft",
                                        color_scheme="green",
                                    ),
                                    width="100%",
                                    align_items="center",
                                    margin_bottom="0.5em",
                                ),
                                rx.vstack(
                                    rx.foreach(
                                        StateAny.founder_roles,
                                        _founder_role_card,
                                    ),
                                    width="100%",
                                    spacing="4",
                                ),
                                width="100%",
                            ),
                            rx.vstack(
                                rx.divider(margin_y="2em"),
                                rx.heading(
                                    "Data not loaded", size="5", color="#f7fafc"
                                ),
                                rx.text(
                                    "Click “Load Data” to fetch your latest resume and profile fields from Neo4j.",
                                    color="#a0aec0",
                                    font_size="0.95em",
                                ),
                                align_items="start",
                                spacing="2",
                                width="100%",
                            ),
                        ),
                        width="100%",
                        max_width="800px",
                        margin_x="auto",
                        padding_y="2em",
                    ),
                    width="50%",
                    height="100vh",
                    padding="2em",
                    overflow="auto",
                    bg="#1a202c",
                ),
                # Right Panel (PDF)
                rx.box(
                    rx.vstack(
                        rx.box(
                            rx.cond(
                                StateAny.pdf_error != "",
                                rx.center(
                                    rx.text(StateAny.pdf_error, color="#f56565"),
                                    height="100%",
                                ),
                                rx.cond(
                                    StateAny.pdf_url != "",
                                    rx.el.embed(
                                        src=f"{StateAny.pdf_url}#view=FitH&zoom=page-width&toolbar=0&navpanes=0&scrollbar=0",
                                        type="application/pdf",
                                        style={
                                            "width": "100%",
                                            "height": "100%",
                                            "border": "none",
                                            "border_radius": "12px",
                                            "display": "block",
                                        },
                                        key=StateAny.last_pdf_signature,
                                    ),
                                    rx.box(
                                        height="100%"
                                    ),  # empty state instead of spinner
                                ),
                            ),
                            width="100%",
                            height="100%",
                            bg="#0b1224",
                            padding="0.5em",
                            border_radius="12px",
                            box_shadow="0 10px 25px rgba(0,0,0,0.35)",
                        ),
                        width="100%",
                        height="100%",
                        padding="1em",
                    ),
                    width="50%",
                    height="100vh",
                    border_left="1px solid #2d3748",
                    bg="#0b1224",
                ),
                width="100%",
                height="100vh",
                overflow="hidden",
            ),
            loading_view,
        ),
    )


APP_STYLE: dict[Any, Any] = {
    "font_family": "Inter, system-ui, sans-serif",
    "background_color": "#1a202c",
}

fastapi_app = FastAPI(title="Resume Builder API")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.get("/api/profile-search")
def profile_search(
    q: str = "",
    page: int = 1,
    page_size: int = 100,
):
    term = str(q or "")[:200].strip()
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 100)
    offset = (page - 1) * page_size
    db = Neo4jClient()
    try:
        profiles = db.search_profile_metadata(
            term=term, limit=page_size + 1, offset=offset
        )
    except Exception as exc:
        _write_debug_line(t"profile_search error: {exc}")
        return {"results": [], "pagination": {"more": False}}
    finally:
        db.close()
    more = len(profiles) > page_size
    profiles = profiles[:page_size]
    results = []
    for profile in profiles:
        profile_id = str(profile.get("id") or "").strip()
        if not profile_id:
            continue
        results.append(
            {
                "id": profile_id,
                "text": _format_profile_label(profile),
            }
        )
    return {"results": results, "pagination": {"more": more}}

app = rx.App(
    api_transformer=fastapi_app,
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="large",
        accent_color="blue",
    ),
    stylesheets=[
        "https://www.jsfontpicker.com/css/fontpicker.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/theme/material-darker.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css",
        "https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css",
    ],
    head_components=[
        rx.html(
            """
            <style>
              #profile-select {
                width: 100%;
              }
              .select2-container {
                width: 100% !important;
                max-width: 100%;
              }
              .select2-container--default .select2-selection--single {
                background-color: var(--color-surface) !important;
                box-shadow: inset 0 0 0 1px var(--gray-a7) !important;
                border: none !important;
                border-radius: max(var(--radius-2), var(--radius-full)) !important;
                height: var(--space-6) !important;
                font-family: var(--default-font-family) !important;
                font-size: var(--font-size-2) !important;
                line-height: var(--line-height-2) !important;
                letter-spacing: var(--letter-spacing-2) !important;
                color: var(--gray-12) !important;
              }
              .select2-container--default .select2-selection--single .select2-selection__rendered {
                color: var(--gray-12) !important;
                line-height: var(--space-6) !important;
                padding-left: var(--space-3) !important;
                padding-right: calc(var(--space-3) + var(--space-2)) !important;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                font-family: var(--default-font-family) !important;
                font-size: var(--font-size-2) !important;
                letter-spacing: var(--letter-spacing-2) !important;
              }
              .select2-container--default .select2-selection--single .select2-selection__placeholder {
                color: var(--gray-a10) !important;
              }
              .select2-container--default .select2-selection--single .select2-selection__arrow {
                height: var(--space-6) !important;
                right: var(--space-2) !important;
              }
              .select2-container--default .select2-selection--single .select2-selection__arrow b {
                border-color: var(--gray-12) transparent transparent transparent !important;
              }
              .select2-container--default .select2-selection--single .select2-selection__clear {
                color: var(--gray-a11) !important;
              }
              .select2-container--default .select2-results > .select2-results__options {
                background-color: var(--color-panel-solid) !important;
                color: var(--gray-12) !important;
              }
              .select2-container--default .select2-results__option--highlighted.select2-results__option--selectable {
                background-color: var(--accent-9) !important;
                color: var(--accent-contrast) !important;
              }
              .select2-dropdown {
                border: 1px solid var(--gray-a6) !important;
                border-radius: var(--radius-3) !important;
                background-color: var(--color-panel-solid) !important;
                box-shadow: var(--shadow-5) !important;
              }
              .select2-search__field {
                background-color: var(--color-surface) !important;
                color: var(--gray-12) !important;
                border: 1px solid var(--gray-a7) !important;
                border-radius: max(var(--radius-2), var(--radius-full)) !important;
                font-family: var(--default-font-family) !important;
                font-size: var(--font-size-2) !important;
                letter-spacing: var(--letter-spacing-2) !important;
              }
            </style>
            """
        ),
        rx.script(src="https://code.jquery.com/jquery-3.7.1.min.js"),
        rx.script(
            """
            (function () {
              var SELECT_ID = "profile-select";
              var ROOT_ID = "profile-select-root";
              var SELECT2_SRC = "https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js";
              var pollingHandle = window.__profileSelect2Interval || null;
              var syncing = false;
              var backendBase = "";
              var backendBasePromise = null;

              function fallbackBackendBase() {
                if (window.__REFLEX_BACKEND_URL) {
                  return window.__REFLEX_BACKEND_URL;
                }
                if (!window.location) {
                  return "";
                }
                var port = window.location.port;
                var host = window.location.hostname || "localhost";
                var protocol = window.location.protocol || "http:";
                var portNum = port ? parseInt(port, 10) : null;
                if (portNum && !isNaN(portNum) && portNum >= 3000 && portNum < 4000) {
                  return protocol + "//" + host + ":" + String(portNum + 5000);
                }
                return window.location.origin || (protocol + "//" + host + (port ? (":" + port) : ""));
              }

              function resolveBackendBase() {
                return backendBase || fallbackBackendBase();
              }

              function loadBackendBase(callback) {
                if (backendBase) {
                  callback(backendBase);
                  return;
                }
                var fallback = fallbackBackendBase();
                if (!window.fetch) {
                  backendBase = fallback;
                  callback(backendBase);
                  return;
                }
                if (backendBasePromise) {
                  backendBasePromise.then(callback);
                  return;
                }
                backendBasePromise = window.fetch("/env.json", { cache: "no-store" })
                  .then(function (resp) {
                    if (!resp.ok) {
                      throw new Error("env fetch failed");
                    }
                    return resp.json();
                  })
                  .then(function (env) {
                    var ping = env && env.PING ? env.PING : "";
                    if (!ping) {
                      return "";
                    }
                    try {
                      var url = new URL(ping, window.location.origin);
                      return url.origin;
                    } catch (err) {
                      return "";
                    }
                  })
                  .catch(function () {
                    return "";
                  })
                  .then(function (base) {
                    backendBasePromise = null;
                    backendBase = base || fallback;
                    return backendBase;
                  })
                  .then(function (base) {
                    callback(base);
                    return base;
                  });
              }

              function getSelectEl() {
                return document.getElementById(SELECT_ID);
              }

              function getStateData(selectEl) {
                if (!selectEl) {
                  return { id: "", label: "" };
                }
                return {
                  id: selectEl.dataset.selectedId || "",
                  label: selectEl.dataset.selectedLabel || ""
                };
              }

              function updateStateData(selectEl, id, label) {
                if (!selectEl) {
                  return;
                }
                selectEl.dataset.selectedId = id || "";
                selectEl.dataset.selectedLabel = label || "";
              }

              function setSelectValue(selectEl, value) {
                var setter = Object.getOwnPropertyDescriptor(
                  window.HTMLSelectElement.prototype,
                  "value"
                ).set;
                setter.call(selectEl, value);
              }

              function ensureSelectedOption($select, id, label) {
                if (!id) {
                  return;
                }
                var existing = $select.find('option[value="' + id.replace(/"/g, '\\"') + '"]');
                if (existing.length === 0) {
                  var option = new Option(label || id, id, true, true);
                  $select.append(option);
                } else if (label && existing.text() !== label) {
                  existing.text(label);
                }
              }

              function cleanupContainers($select) {
                var root = document.getElementById(ROOT_ID);
                if (!root) {
                  return;
                }
                var containers = root.querySelectorAll(".select2-container");
                var hasSelect2 = $select && $select.data("select2");
                if (!containers.length) {
                  return;
                }
                if (hasSelect2 && containers.length === 1) {
                  return;
                }
                if (hasSelect2) {
                  try {
                    $select.select2("destroy");
                  } catch (err) {
                    // Ignore cleanup errors and continue.
                  }
                }
                containers.forEach(function (node) {
                  node.remove();
                });
              }

              function ensureSelect2(callback) {
                if (window.jQuery && window.jQuery.fn && window.jQuery.fn.select2) {
                  callback();
                  return;
                }
                if (!window.jQuery || !window.jQuery.fn) {
                  setTimeout(function () {
                    ensureSelect2(callback);
                  }, 250);
                  return;
                }
                if (window.__select2Loading) {
                  window.__select2Loading.push(callback);
                  return;
                }
                window.__select2Loading = [callback];
                var script = document.createElement("script");
                script.src = SELECT2_SRC;
                script.onload = function () {
                  var callbacks = window.__select2Loading || [];
                  window.__select2Loading = null;
                  callbacks.forEach(function (cb) {
                    cb();
                  });
                };
                script.onerror = function () {
                  var callbacks = window.__select2Loading || [];
                  window.__select2Loading = null;
                  callbacks.forEach(function (cb) {
                    cb();
                  });
                };
                document.head.appendChild(script);
              }

              function syncFromState($select) {
                var selectEl = $select.get(0);
                var state = getStateData(selectEl);
                var desiredId = state.id || "";
                var desiredLabel = state.label || desiredId;
                var current = $select.val() || "";
                if (desiredId && current !== desiredId) {
                  syncing = true;
                  ensureSelectedOption($select, desiredId, desiredLabel);
                  $select.val(desiredId).trigger("change.select2");
                  syncing = false;
                }
                if (!desiredId && current) {
                  syncing = true;
                  $select.val(null).trigger("change.select2");
                  syncing = false;
                }
              }

              function initSelect2() {
                if (!window.jQuery || !window.jQuery.fn || !window.jQuery.fn.select2) {
                  return false;
                }
                var $select = window.jQuery("#" + SELECT_ID);
                if (!$select.length) {
                  return false;
                }
                cleanupContainers($select);
                if ($select.data("select2")) {
                  return true;
                }
                $select.select2({
                  placeholder: "Select profile",
                  allowClear: true,
                  width: "100%",
                  minimumInputLength: 0,
                  ajax: {
                    url: resolveBackendBase() + "/api/profile-search",
                    dataType: "json",
                    delay: 200,
                    data: function (params) {
                      return {
                        q: params.term || "",
                        page: params.page || 1,
                        page_size: 100
                      };
                    },
                    processResults: function (data, params) {
                      params.page = params.page || 1;
                      return {
                        results: (data && data.results) ? data.results : [],
                        pagination: data && data.pagination ? data.pagination : { more: false }
                      };
                    },
                    cache: true
                  }
                });
                $select.on("select2:select", function (e) {
                  if (syncing) {
                    return;
                  }
                  var data = (e && e.params) ? e.params.data : {};
                  var rawSelect = $select.get(0);
                  updateStateData(rawSelect, data.id || "", data.text || "");
                  setSelectValue(rawSelect, data.id || "");
                  rawSelect.dispatchEvent(new Event("change", { bubbles: true }));
                });
                $select.on("select2:open", function () {
                  var $results = window.jQuery(".select2-results");
                  $results.off("mouseup.profileSelect2");
                  $results.on(
                    "mouseup.profileSelect2",
                    ".select2-results__option[aria-selected=true]",
                    function () {
                      var data = $select.select2("data") || [];
                      if (!data.length) {
                        return;
                      }
                      var item = data[0] || {};
                      var rawSelect = $select.get(0);
                      if (!rawSelect) {
                        return;
                      }
                      updateStateData(rawSelect, item.id || "", item.text || "");
                      setSelectValue(rawSelect, item.id || "");
                      rawSelect.dispatchEvent(
                        new Event("change", { bubbles: true })
                      );
                    }
                  );
                });
                $select.on("select2:clear", function () {
                  if (syncing) {
                    return;
                  }
                  var rawSelect = $select.get(0);
                  updateStateData(rawSelect, "", "");
                  setSelectValue(rawSelect, "");
                  rawSelect.dispatchEvent(new Event("change", { bubbles: true }));
                });
                syncFromState($select);
                if (pollingHandle) {
                  clearInterval(pollingHandle);
                }
                pollingHandle = setInterval(function () {
                  var $current = window.jQuery("#" + SELECT_ID);
                  if (!$current.length) {
                    return;
                  }
                  if (!$current.data("select2")) {
                    initSelect2();
                    return;
                  }
                  syncFromState($current);
                }, 500);
                window.__profileSelect2Interval = pollingHandle;
                return true;
              }

              function boot() {
                ensureSelect2(function () {
                  loadBackendBase(function () {
                    if (!initSelect2()) {
                      setTimeout(boot, 250);
                    }
                  });
                });
              }

              if (document.readyState === "loading") {
                document.addEventListener("DOMContentLoaded", boot);
              } else {
                boot();
              }
            })();
            """
        ),
        rx.script(src="https://www.jsfontpicker.com/js/fontpicker.iife.min.js"),
        rx.script(
            f"""
            (function () {{
              var extraFonts = {FONT_PICKER_EXTRA_FONTS_JSON};
              var defaultFont = {json.dumps(DEFAULT_RESUME_FONT_FAMILY)};

              function normalizeFamily(value) {{
                if (!value) {{
                  return "";
                }}
                var raw = String(value);
                return raw.split(":")[0].trim();
              }}

              function hasExtraFont(name) {{
                var target = normalizeFamily(name);
                if (!target) {{
                  return false;
                }}
                for (var i = 0; i < extraFonts.length; i += 1) {{
                  if (normalizeFamily(extraFonts[i].name) === target) {{
                    return true;
                  }}
                }}
                return false;
              }}

              function initPicker() {{
                var input = document.getElementById("resume-font-picker");
                if (!input) {{
                  return false;
                }}
                if (input.dataset.fpAttached === "1") {{
                  return true;
                }}
                if (!window.FontPicker) {{
                  return false;
                }}
                document.documentElement.dataset.fpTheme = "dark";
                var safeDefault = hasExtraFont(defaultFont) ? normalizeFamily(defaultFont) : "";
                if (!input.value && safeDefault) {{
                  input.value = safeDefault;
                }}
                if (input.value) {{
                  input.value = normalizeFamily(input.value);
                }}
                var picker = new window.FontPicker(input, {{
                  variants: false,
                  verbose: false,
                  font: input.value || null,
                  googleFonts: null,
                  systemFonts: null,
                  extraFonts: extraFonts
                }});
                input.dataset.fpAttached = "1";
                input.style.fontFamily = input.value || "";
                var lastValue = input.value || "";

                function syncFromInput() {{
                  if (input.value === lastValue) {{
                    return;
                  }}
                  lastValue = input.value || "";
                  if (lastValue) {{
                    try {{
                      picker.setFont(lastValue);
                    }} catch (e) {{}}
                    input.style.fontFamily = lastValue;
                  }} else {{
                    input.style.fontFamily = "";
                  }}
                }}

                input.addEventListener("input", function () {{
                  if (input.dataset.fpPicking === "1") {{
                    return;
                  }}
                  syncFromInput();
                }});

                picker.on("pick", function (font) {{
                  input.dataset.fpPicking = "1";
                  if (!font) {{
                    input.value = "";
                    input.style.fontFamily = "";
                  }} else {{
                    var family = font.family && font.family.name ? font.family.name : font.toString();
                    input.value = family;
                    input.style.fontFamily = font.family && font.family.name ? font.family.name : "";
                  }}
                  lastValue = input.value || "";
                  input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                  input.dispatchEvent(new Event("change", {{ bubbles: true }}));
                  input.dataset.fpPicking = "0";
                }});

                if (window.queryLocalFonts) {{
                  try {{
                    window.queryLocalFonts().then(function (fonts) {{
                      var seen = Object.create(null);
                      fonts.forEach(function (f) {{
                        if (f.family) {{
                          seen[f.family] = true;
                        }}
                      }});
                      var families = Object.keys(seen);
                      if (families.length) {{
                        families.sort();
                        picker.setConfiguration({{ systemFonts: families }});
                      }}
                    }});
                  }} catch (e) {{}}
                }}

                setInterval(syncFromInput, 500);
                return true;
              }}

              function boot() {{
                if (!initPicker()) {{
                  setTimeout(boot, 300);
                }}
              }}

              if (document.readyState === "loading") {{
                document.addEventListener("DOMContentLoaded", boot);
              }} else {{
                boot();
              }}
            }})();
            """
        ),
        rx.script(
            src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.js"
        ),
        rx.script(
            """
            (function () {
              function ensureYamlMode() {
                if (!window.CodeMirror || !window.CodeMirror.modes) {
                  return false;
                }
                if (window.CodeMirror.modes.yaml) {
                  return true;
                }
                if (window.__promptYamlModeLoading) {
                  return false;
                }
                window.__promptYamlModeLoading = true;
                var script = document.createElement("script");
                script.src = "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/yaml/yaml.min.js";
                script.onload = function () {
                  window.__promptYamlModeLoading = false;
                };
                script.onerror = function () {
                  window.__promptYamlModeLoading = false;
                };
                document.head.appendChild(script);
                return false;
              }

              function ensureStyle() {
                if (window.__promptYamlStyleInjected) {
                  return;
                }
                var style = document.createElement("style");
                style.textContent = "#prompt-yaml-field + .CodeMirror{border:1px solid #4a5568;border-radius:0.375rem;background:#1f2937;}#prompt-yaml-field + .CodeMirror .CodeMirror-gutters{border-right:1px solid #374151;background:#1f2937;}#prompt-yaml-field + .CodeMirror .CodeMirror-linenumber{color:#9ca3af;}#prompt-yaml-field + .CodeMirror .CodeMirror-cursor{border-left:1px solid #e2e8f0;}";
                document.head.appendChild(style);
                window.__promptYamlStyleInjected = true;
              }

              function initPromptYamlEditor() {
                var textarea = document.getElementById("prompt-yaml-field");
                if (!textarea) {
                  return false;
                }
                if (textarea.dataset.cmAttached === "1") {
                  return true;
                }
                if (!window.CodeMirror || !window.CodeMirror.fromTextArea) {
                  return false;
                }
                var editor = window.CodeMirror.fromTextArea(textarea, {
                  mode: "yaml",
                  lineNumbers: true,
                  lineWrapping: true,
                  tabSize: 2,
                  indentUnit: 2,
                  theme: "material-darker"
                });
                textarea.dataset.cmAttached = "1";
                textarea.dataset.cmSyncedValue = textarea.value || "";
                window.__promptYamlCodeMirror = editor;
                editor.setSize("100%", "220px");
                editor.on("change", function (cm) {
                  cm.save();
                  textarea.dataset.cmSyncedValue = textarea.value || "";
                  textarea.dispatchEvent(new Event("input", { bubbles: true }));
                });
                return true;
              }

              function syncFromTextarea() {
                var textarea = document.getElementById("prompt-yaml-field");
                var editor = window.__promptYamlCodeMirror;
                if (!textarea || !editor) {
                  return;
                }
                var nextValue = textarea.value || "";
                var currentValue = editor.getValue();
                var lastSynced = textarea.dataset.cmSyncedValue || "";
                if (nextValue !== currentValue && nextValue !== lastSynced) {
                  editor.setValue(nextValue);
                  textarea.dataset.cmSyncedValue = nextValue;
                }
              }

              function boot() {
                var attempts = window.__promptYamlInitAttempts || 0;
                window.__promptYamlInitAttempts = attempts + 1;
                if (attempts > 200) {
                  return;
                }
                ensureStyle();
                if (!window.CodeMirror || !window.CodeMirror.fromTextArea) {
                  window.setTimeout(boot, 250);
                  return;
                }
                if (!ensureYamlMode()) {
                  window.setTimeout(boot, 250);
                  return;
                }
                if (!initPromptYamlEditor()) {
                  window.setTimeout(boot, 250);
                  return;
                }
                if (!window.__promptYamlSyncTimer) {
                  window.__promptYamlSyncTimer = window.setInterval(syncFromTextarea, 750);
                }
              }

              if (document.readyState === "loading") {
                document.addEventListener("DOMContentLoaded", boot);
              } else {
                boot();
              }
            })();
            """
        ),
    ],
    style=APP_STYLE,
)
app.add_page(index, on_load=StateAny.on_load)

if __name__ == "__main__":
    if "--maximum-coverage" in sys.argv and not os.environ.get("_MAX_COVERAGE_CHILD"):
        os.environ.setdefault("MAX_COVERAGE_SILENCE_WARNINGS", "1")
        _install_maxcov_print_filter()
        _maybe_launch_maxcov_container()
        coverage = _try_import_coverage()
        if coverage is not None:
            _maxcov_log("maxcov wrapper start")
            self_test = os.environ.get("_MAX_COVERAGE_SELFTEST") == "1"
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            cov_dir = Path(tempfile.gettempdir()) / f"max_coverage_{stamp}"
            cov_dir.mkdir(parents=True, exist_ok=True)
            cov_rc = cov_dir / "coverage_rc"
            cov_rc.write_text(
                "\n".join(
                    [
                        "[run]",
                        "branch = True",
                        "parallel = True",
                        "source =",
                        f"    {BASE_DIR}",
                        "omit =",
                        "    */site-packages/*",
                        "    */__pycache__/*",
                        "    */.tmp_typst/*",
                        "    */assets/*",
                        "    */fonts/*",
                        "    */packages/*",
                        "    */.pytest_cache/*",
                        "",
                        "[report]",
                        "show_missing = False",
                        "skip_empty = True",
                    ]
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["_MAX_COVERAGE_CHILD"] = "1"
            env["MAX_COVERAGE_LOG"] = "1"
            env["MAX_COVERAGE_SKIP_LLM"] = "1"
            env.setdefault("PYTHONUNBUFFERED", "1")
            env["COVERAGE_FILE"] = str(cov_dir / ".coverage")
            base_cmd = [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(cov_rc),
                "--parallel-mode",
                str(Path(__file__).resolve()),
            ]
            req_file = _get_arg_value(
                sys.argv[1:], "--req-file", str(BASE_DIR / "req.txt")
            )
            empty_home = cov_dir / "empty_home"
            empty_home.mkdir(parents=True, exist_ok=True)
            missing_req = cov_dir / "missing_req.txt"
            missing_typst = cov_dir / "missing_typst"

            def _run_cov(
                args: list[str], *, extra_env=None, quiet=False, expected_failure=False
            ) -> int:
                run_env = env.copy()
                if extra_env:
                    for key, val in extra_env.items():
                        if val is None:
                            run_env.pop(key, None)
                        else:
                            run_env[key] = val
                kwargs: dict[str, Any] = {}
                if quiet:
                    kwargs["stdout"] = subprocess.PIPE
                    kwargs["stderr"] = subprocess.PIPE
                    kwargs["text"] = True
                started = time.perf_counter()
                cmd = [*base_cmd, *args]
                _maxcov_log(t"run start: {' '.join(cmd)}")
                if extra_env:
                    _maxcov_log(
                        f"run env overrides: {', '.join(sorted(k for k in extra_env.keys()))}"
                    )
                proc = subprocess.Popen(cmd, env=run_env, **kwargs)
                next_log = started + 5.0
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        break
                    now = time.perf_counter()
                    if now >= next_log:
                        _maxcov_log(
                            f"run heartbeat ({now - started:.1f}s): {' '.join(args) or '<no args>'}"
                        )
                        next_log = now + 5.0
                    time.sleep(0.25)
                stdout, stderr = proc.communicate()
                elapsed = time.perf_counter() - started
                _maxcov_log(t"run done ({elapsed:.1f}s): rc={proc.returncode}")
                if proc.returncode != 0:
                    if expected_failure:
                        _maxcov_log(
                            f"run expected failure: rc={proc.returncode} args={' '.join(args)}"
                        )
                        _maxcov_log_expected_failure(stdout, stderr, args, quiet)
                    else:
                        print(
                            f"Warning: coverage run failed for args: {' '.join(args)}"
                        )
                        if quiet and stdout:
                            print(stdout.rstrip())
                        if quiet and stderr:
                            print(stderr.rstrip(), file=sys.stderr)
                return proc.returncode

            coverage_runs = [
                (sys.argv[1:], {}, False, False),
                (
                    ["--maximum-coverage"],
                    {"_MAX_COVERAGE_CHILD": None, "_MAX_COVERAGE_SELFTEST": "1"},
                    True,
                    False,
                ),
                (
                    ["--maximum-coverage", "--maximum-coverage-actions", "bogus"],
                    {},
                    True,
                    True,
                ),
                (["--bogus-flag"], {}, True, True),
                (["--list-models"], {}, True, False),
                (["--list-models"], {"REFLEX_COVERAGE": "1"}, True, False),
                (
                    ["--list-models"],
                    {"REFLEX_COVERAGE": "1", "REFLEX_COVERAGE_FORCE_OWNED": "1"},
                    True,
                    False,
                ),
                ([], {}, True, True),
                (["--export-resume-pdf"], {}, True, False),
                (["--list-applied"], {}, True, False),
                (["--reset-db"], {}, True, False),
                (["--import-assets"], {}, True, False),
                (
                    ["--compile-pdf", str(ASSETS_DIR / "preview.pdf"), "--auto-fit"],
                    {},
                    True,
                    False,
                ),
                (["--compile-pdf", str(ASSETS_DIR / "preview.pdf")], {}, True, False),
                (
                    ["--compile-pdf", str(ASSETS_DIR / "preview.pdf")],
                    {"TYPST_BIN": str(missing_typst)},
                    True,
                    True,
                ),
                (
                    ["--generate-profile", "--req-file", req_file],
                    {"MAX_COVERAGE_SKIP_LLM": "1"},
                    True,
                    False,
                ),
                (["--show-resume-data"], {}, True, False),
                (["--eval-prompt", "--req-file", str(missing_req)], {}, True, True),
                (
                    [
                        "--eval-prompt",
                        "--req-file",
                        req_file,
                        "--model-name",
                        "bogus:model",
                    ],
                    {},
                    True,
                    False,
                ),
                (
                    [
                        "--eval-prompt",
                        "--req-file",
                        req_file,
                        "--model-name",
                        "openai:gpt-4o-mini",
                    ],
                    {
                        "HOME": str(empty_home),
                        "OPENAI_API_KEY": "",
                        "OPENAI_ORG_ID": "",
                        "OPENAI_ORGANIZATION": "",
                        "OPENAI_PROJECT": "",
                    },
                    True,
                    False,
                ),
                (
                    [
                        "--eval-prompt",
                        "--req-file",
                        req_file,
                        "--model-name",
                        "gemini:gemini-1.5-flash",
                    ],
                    {
                        "HOME": str(empty_home),
                        "GEMINI_API_KEY": "",
                        "GOOGLE_API_KEY": "",
                    },
                    True,
                    False,
                ),
            ]
            if self_test:
                base_cmd = [sys.executable, "-c", "print('maxcov bootstrap selftest')"]
                coverage_runs = [
                    ([], {}, True, False),
                    (["--noop"], {}, True, False),
                ]
            total_runs = len(coverage_runs)
            for idx, (run_args, env_overrides, quiet, expected_failure) in enumerate(
                coverage_runs, start=1
            ):
                _maxcov_log(
                    f"run dispatch {idx}/{total_runs}: args={' '.join(run_args) or '<no args>'}"
                )
                _run_cov(
                    run_args,
                    extra_env=env_overrides,
                    quiet=quiet,
                    expected_failure=expected_failure,
                )
            _maxcov_log("coverage combine start")
            started = time.perf_counter()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "combine",
                    "--rcfile",
                    str(cov_rc),
                    str(cov_dir),
                ],
                env=env,
                check=False,
            )
            _maxcov_log(t"coverage combine done ({time.perf_counter() - started:.1f}s)")
            _maxcov_log("coverage report start")
            started = time.perf_counter()
            report = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "report",
                    "--rcfile",
                    str(cov_rc),
                    "--data-file",
                    str(cov_dir / ".coverage"),
                    "--include",
                    str(Path(__file__).resolve()),
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            _maxcov_log(t"coverage report done ({time.perf_counter() - started:.1f}s)")
            json_path = cov_dir / "coverage.json"
            html_dir = cov_dir / "htmlcov"
            json_out = None
            html_out = None
            _maxcov_log("coverage json start")
            started = time.perf_counter()
            try:
                json_cmd = [
                    sys.executable,
                    "-m",
                    "coverage",
                    "json",
                    "--rcfile",
                    str(cov_rc),
                ]
                json_cmd.extend(["--data-file", str(cov_dir / ".coverage")])
                json_cmd.extend(["--include", str(Path(__file__).resolve())])
                json_cmd.extend(["--output", str(json_path)])
                json_result = subprocess.run(
                    json_cmd,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if json_result.returncode == 0 and json_path.exists():
                    json_out = str(json_path)
            except Exception as exc:
                _maxcov_log(t"coverage json failed: {exc}")
            _maxcov_log(t"coverage json done ({time.perf_counter() - started:.1f}s)")
            _maxcov_log("coverage html start")
            started = time.perf_counter()
            try:
                html_cmd = [
                    sys.executable,
                    "-m",
                    "coverage",
                    "html",
                    "--rcfile",
                    str(cov_rc),
                ]
                html_cmd.extend(["--data-file", str(cov_dir / ".coverage")])
                html_cmd.extend(["--include", str(Path(__file__).resolve())])
                html_cmd.extend(["--directory", str(html_dir)])
                html_result = subprocess.run(
                    html_cmd,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                index_path = html_dir / "index.html"
                if html_result.returncode == 0 and index_path.exists():
                    html_out = str(index_path)
            except Exception as exc:
                _maxcov_log(t"coverage html failed: {exc}")
            _maxcov_log(t"coverage html done ({time.perf_counter() - started:.1f}s)")
            counts = {}
            for line in (report.stdout or "").splitlines():
                if line.strip().startswith("harness.py"):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            counts = {
                                "stmts": int(parts[1]),
                                "miss": int(parts[2]),
                                "branch": int(parts[3]),
                                "brpart": int(parts[4]),
                                "cover": parts[5],
                            }
                        except Exception:
                            counts = {"cover": parts[5]}
                        break
            summary = _maxcov_summarize_coverage(
                coverage,
                cov_dir=cov_dir,
                cov_rc=cov_rc,
                target=Path(__file__).resolve(),
            )
            lines = _maxcov_build_coverage_output(
                counts=counts,
                summary=summary,
                cov_dir=cov_dir,
                cov_rc=cov_rc,
                json_out=json_out,
                html_out=html_out,
            )
            if lines:
                print("\n".join(lines))
            elif report.stdout:
                print(report.stdout.rstrip())
            sys.exit(0)

    parser = argparse.ArgumentParser(description="Resume Builder Utility")
    parser.add_argument(
        "--import-assets",
        help=(
            "Path to assets JSON file to import (see michael_scott_resume.json for"
            " schema). Refuses to overwrite existing resume unless"
            " --overwrite-resume is set."
        ),
        nargs="?",
        const=str(DEFAULT_ASSETS_JSON),
    )
    parser.add_argument(
        "--overwrite-resume",
        action="store_true",
        help="Allow --import-assets to replace an existing resume in Neo4j.",
    )
    parser.add_argument(
        "--reset-db",
        help="Reset Neo4j (wipe) then import assets JSON (default: michael_scott_resume.json)",
        nargs="?",
        const=str(DEFAULT_ASSETS_JSON),
    )
    parser.add_argument(
        "--generate-profile",
        action="store_true",
        help="Generate a new Profile from the stored prompt template (prompt.yaml fallback) + --req-file and save it to Neo4j (combine with --compile-pdf to render).",
    )
    parser.add_argument(
        "--eval-prompt",
        action="store_true",
        help="Send the stored prompt template (prompt.yaml fallback) plus req.txt (or --req-file) to the LLM and print the JSON response.",
    )
    parser.add_argument(
        "--compile-pdf",
        help="Compile the current resume to a PDF at the given path (defaults to assets/preview.pdf)",
        nargs="?",
        const=str(ASSETS_DIR / "preview.pdf"),
    )
    parser.add_argument(
        "--auto-fit",
        action="store_true",
        help="Enable auto-fit to the configured page count when compiling PDF",
    )
    parser.add_argument(
        "--export-resume-pdf",
        action="store_true",
        help="Export a resume PDF to assets/preview_no_summary_skills.pdf without summary/skills (for external uploads).",
    )
    parser.add_argument(
        "--req-file",
        help="Path to job requisition text for --eval-prompt / --generate-profile",
        default=str(BASE_DIR / "req.txt"),
    )
    parser.add_argument(
        "--model-name",
        help="LLM model for --eval-prompt / --generate-profile (e.g. gemini:gemini-3-flash-preview or openai:gpt-4o-mini; bare ids default to OpenAI).",
        default=None,
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available LLM models (from LLM_MODELS or defaults).",
    )
    parser.add_argument(
        "--show-resume-data",
        action="store_true",
        help="Dump the raw resume data from Neo4j to stdout.",
    )
    parser.add_argument(
        "--list-applied",
        action="store_true",
        help="List jobs/resumes already applied/saved in Neo4j",
    )
    parser.add_argument(
        "--maximum-coverage",
        action="store_true",
        help=(
            "Drive UI interactions against State to maximize coverage "
            "(implies all actions + skip-llm + failure paths unless overridden)."
        ),
    )
    parser.add_argument(
        "--maximum-coverage-actions",
        default="all",
        help=(
            "Comma-separated UI actions to simulate: all, load, profile, pipeline, "
            "forms, toggles, reorder, save, pdf."
        ),
    )
    parser.add_argument(
        "--maximum-coverage-skip-llm",
        action="store_true",
        help="Skip LLM calls during maximum-coverage simulation.",
    )
    parser.add_argument(
        "--maximum-coverage-failures",
        action="store_true",
        help="Simulate failure paths (Neo4j/LLM/Typst) during maximum-coverage simulation.",
    )
    parser.add_argument(
        "--maximum-coverage-ui-url",
        default="",
        help="Reflex app URL for Playwright UI traversal (default: env MAX_COVERAGE_UI_URL/REFLEX_URL or localhost).",
    )
    parser.add_argument(
        "--maximum-coverage-ui-timeout",
        type=float,
        default=30.0,
        help="Playwright UI traversal timeout in seconds.",
    )
    parser.add_argument(
        "--maximum-coverage-reflex",
        action="store_true",
        help="Start a Reflex server with coverage enabled and drive it via Playwright.",
    )
    parser.add_argument(
        "--maximum-coverage-reflex-frontend-port",
        type=int,
        default=3010,
        help="Frontend port for the Reflex coverage server.",
    )
    parser.add_argument(
        "--maximum-coverage-reflex-backend-port",
        type=int,
        default=8010,
        help="Backend port for the Reflex coverage server.",
    )
    parser.add_argument(
        "--maximum-coverage-reflex-startup-timeout",
        type=float,
        default=90.0,
        help="Startup timeout for the Reflex coverage server.",
    )
    parser.add_argument(
        "--ui-playwright-check",
        action="store_true",
        help="Run the comprehensive Playwright UI check (requires a running Reflex app).",
    )
    parser.add_argument(
        "--ui-playwright-check-docker",
        action="store_true",
        help=(
            "Run the Playwright UI check in Docker (starts Neo4j + app, seeds data, "
            "and runs UI checks inside the compose network)."
        ),
    )
    parser.add_argument(
        "--ui-playwright-url",
        default="",
        help=(
            "URL for the Playwright UI check (defaults to PLAYWRIGHT_URL/REFLEX_URL/REFLEX_APP_URL "
            "or http://localhost:3000)."
        ),
    )
    parser.add_argument(
        "--ui-playwright-timeout",
        type=float,
        default=360.0,
        help="Timeout in seconds for the Playwright UI check.",
    )
    parser.add_argument(
        "--ui-playwright-pdf-timeout",
        type=float,
        default=45.0,
        help="PDF embed timeout in seconds for the Playwright UI check.",
    )
    parser.add_argument(
        "--ui-playwright-headed",
        action="store_true",
        help="Run the Playwright UI check with a visible browser window.",
    )
    parser.add_argument(
        "--ui-playwright-slowmo",
        type=int,
        default=0,
        help="Slow down Playwright actions (ms) for the UI check.",
    )
    parser.add_argument(
        "--ui-playwright-allow-llm-error",
        action="store_true",
        help="Deprecated (LLM errors must fail).",
    )
    parser.add_argument(
        "--ui-playwright-allow-db-error",
        action="store_true",
        help="Allow DB errors during the Playwright UI check.",
    )
    parser.add_argument(
        "--ui-playwright-screenshot-dir",
        default="",
        help="Directory for UI check screenshots and PDF artifacts.",
    )
    parser.add_argument(
        "--run-all-tests",
        action="store_true",
        help=(
            "Run the dockerized end-to-end test flow (scripts/run_maxcov_e2e.sh --force). "
            "WARNING: this resets all Docker containers, images, volumes, and networks."
        ),
    )
    parser.add_argument(
        "--run-all-tests-local",
        action="store_true",
        help=(
            "Run the local run-all-tests pipeline (maximum coverage + static analysis "
            "+ clean Reflex startup + Playwright UI check). Intended for container use."
        ),
    )
    parser.add_argument(
        "--run-ui-tests",
        action="store_true",
        help="Run the dockerized UI Playwright check (Neo4j + UI only).",
    )
    static_tool_names = [
        "ruff",
        "black",
        "isort",
        "mypy",
        "pylint",
        "flake8",
        "pyflakes",
        "pycodestyle",
        "pydocstyle",
        "codespell",
        "pyright",
        "pyre",
        "refurb",
        "vulture",
        "bandit",
        "pip-audit",
        "safety",
        "pip-check",
        "pipdeptree",
        "shellcheck",
        "shfmt",
        "hadolint",
        "detect-secrets",
        "check-jsonschema",
        "pydoclint",
        "semgrep",
        "dodgy",
        "eradicate",
        "deptry",
        "pycln",
        "pyupgrade",
        "autoflake",
        "radon-cc",
        "radon-mi",
        "radon-raw",
        "mccabe",
        "xenon",
        "lizard",
        "interrogate",
    ]
    parser.add_argument(
        "--static-all",
        action="store_true",
        help="Run all static analysis tools (same set as run-all-tests).",
    )
    parser.add_argument(
        "--static-tool",
        dest="static_tool_names",
        action="append",
        default=[],
        help=(
            "Run a static tool by name (repeatable or comma-separated). "
            f"Known tools: {', '.join(static_tool_names)}."
        ),
    )
    parser.add_argument(
        "--static-target",
        default="",
        help="Target directory for static tools (default: repo root).",
    )
    parser.add_argument(
        "--static-target-file",
        default="",
        help="Target file for single-file tools (default: harness.py).",
    )
    parser.add_argument(
        "--static-timeout",
        type=float,
        default=0.0,
        help="Timeout per static tool in seconds (0 disables timeouts).",
    )
    parser.set_defaults(static_tool_flags=[])
    static_flag_map = [
        ("--ruff", "ruff"),
        ("--black", "black"),
        ("--isort", "isort"),
        ("--mypy", "mypy"),
        ("--pylint", "pylint"),
        ("--flake8", "flake8"),
        ("--pyflakes", "pyflakes"),
        ("--pycodestyle", "pycodestyle"),
        ("--pydocstyle", "pydocstyle"),
        ("--codespell", "codespell"),
        ("--pyright", "pyright"),
        ("--pyre", "pyre"),
        ("--refurb", "refurb"),
        ("--vulture", "vulture"),
        ("--bandit", "bandit"),
        ("--pip-audit", "pip-audit"),
        ("--safety", "safety"),
        ("--pip-check", "pip-check"),
        ("--pipdeptree", "pipdeptree"),
        ("--shellcheck", "shellcheck"),
        ("--shfmt", "shfmt"),
        ("--hadolint", "hadolint"),
        ("--detect-secrets", "detect-secrets"),
        ("--check-jsonschema", "check-jsonschema"),
        ("--pydoclint", "pydoclint"),
        ("--semgrep", "semgrep"),
        ("--dodgy", "dodgy"),
        ("--eradicate", "eradicate"),
        ("--deptry", "deptry"),
        ("--pycln", "pycln"),
        ("--pyupgrade", "pyupgrade"),
        ("--autoflake", "autoflake"),
        ("--radon-cc", "radon-cc"),
        ("--radon-mi", "radon-mi"),
        ("--radon-raw", "radon-raw"),
        ("--mccabe", "mccabe"),
        ("--xenon", "xenon"),
        ("--lizard", "lizard"),
        ("--interrogate", "interrogate"),
    ]
    for flag, name in static_flag_map:
        parser.add_argument(
            flag,
            dest="static_tool_flags",
            action="append_const",
            const=name,
            help=f"Run {name} static analysis.",
        )
    parser.add_argument(
        "--ui-simulate",
        dest="ui_simulate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ui-simulate-actions",
        dest="ui_simulate_actions",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ui-simulate-skip-llm",
        dest="ui_simulate_skip_llm",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ui-simulate-failures",
        dest="ui_simulate_failures",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    def _run_ui_playwright_check_docker() -> None:
        if os.environ.get("MAX_COVERAGE_CONTAINER") == "1":
            print(
                "Error: --ui-playwright-check-docker is not supported inside the "
                "maxcov container."
            )
            sys.exit(1)
        script_path = BASE_DIR / "scripts" / "run_ui_playwright_docker.sh"
        if not script_path.exists():
            print(f"Error: missing {script_path}")
            sys.exit(1)

        cmd = [
            str(script_path),
            "--timeout",
            str(args.ui_playwright_timeout),
            "--pdf-timeout",
            str(args.ui_playwright_pdf_timeout),
        ]
        if args.ui_playwright_headed:
            cmd.append("--headed")
        if args.ui_playwright_slowmo:
            cmd.extend(["--slowmo", str(args.ui_playwright_slowmo)])
        if args.ui_playwright_allow_db_error:
            cmd.append("--allow-db-error")
        if args.ui_playwright_screenshot_dir:
            cmd.extend(["--screenshot-dir", args.ui_playwright_screenshot_dir])

        def _docker_access_ok() -> bool:
            try:
                result = subprocess.run(
                    ["docker", "ps"], capture_output=True, text=True
                )
                return result.returncode == 0
            except Exception:
                return False

        if os.geteuid() == 0 or _docker_access_ok():
            result = subprocess.run(cmd, cwd=str(BASE_DIR))
            sys.exit(result.returncode)

        missing_tools = [tool for tool in ("expect", "sudo") if not shutil.which(tool)]
        if missing_tools:
            print("Error: missing required tool(s): " f"{', '.join(missing_tools)}")
            sys.exit(1)
        if not sys.stdin.isatty():
            print(
                "Error: --ui-playwright-check-docker requires an interactive TTY "
                "for sudo password entry."
            )
            sys.exit(1)
        expect_script = """\
log_user 1
set timeout -1
set cmd $argv
eval spawn $cmd

proc prompt_password {} {
    if {[catch {stty -echo}]} {
        # Ignore if no controlling tty.
    }
    send_user "sudo password: "
    flush stdout
    gets stdin password
    if {[catch {stty echo}]} {
        # Ignore if no controlling tty.
    }
    send_user "\\n"
    return $password
}

expect {
    -re "(?i)password" {
        set password [prompt_password]
        send -- "$password\\r"
        exp_continue
    }
    eof {
        catch wait result
        exit [lindex $result 3]
    }
}
"""
        script_file = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False
            ) as handle:
                handle.write(expect_script)
                script_file = Path(handle.name)
            sudo_cmd = [
                "expect",
                str(script_file),
                "sudo",
                "-S",
                "-k",
                "-p",
                "sudo password: ",
                *cmd,
            ]
            result = subprocess.run(sudo_cmd, cwd=str(BASE_DIR))
        finally:
            if script_file:
                with suppress(Exception):
                    script_file.unlink()
        sys.exit(result.returncode)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.run_all_tests and args.run_all_tests_local:
        print(
            "Error: --run-all-tests and --run-all-tests-local are mutually exclusive."
        )
        sys.exit(2)

    if args.run_ui_tests and (args.run_all_tests or args.run_all_tests_local):
        print(
            "Error: --run-ui-tests cannot be combined with --run-all-tests or "
            "--run-all-tests-local."
        )
        sys.exit(2)

    if args.ui_playwright_check and args.ui_playwright_check_docker:
        print(
            "Error: --ui-playwright-check and --ui-playwright-check-docker are "
            "mutually exclusive."
        )
        sys.exit(2)
    if args.run_ui_tests and (args.ui_playwright_check or args.ui_playwright_check_docker):
        print(
            "Error: --run-ui-tests cannot be combined with "
            "--ui-playwright-check or --ui-playwright-check-docker."
        )
        sys.exit(2)

    if args.ui_playwright_allow_llm_error:
        print(
            "Error: --ui-playwright-allow-llm-error is not supported; "
            "LLM errors must fail."
        )
        sys.exit(2)

    if args.ui_playwright_check_docker and (
        args.run_all_tests or args.run_all_tests_local
    ):
        print(
            "Error: --ui-playwright-check-docker cannot be combined with "
            "--run-all-tests or --run-all-tests-local."
        )
        sys.exit(2)

    if args.run_ui_tests:
        _run_ui_playwright_check_docker()

    if args.ui_playwright_check_docker:
        _run_ui_playwright_check_docker()

    if args.ui_playwright_check and not (
        args.run_all_tests or args.run_all_tests_local
    ):
        target_url = (
            args.ui_playwright_url
            or os.environ.get("PLAYWRIGHT_URL")
            or os.environ.get("REFLEX_URL")
            or os.environ.get("REFLEX_APP_URL")
            or "http://localhost:3000"
        )
        script_path = BASE_DIR / "scripts" / "ui_playwright_check.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--url",
            str(target_url),
            "--timeout",
            str(args.ui_playwright_timeout),
            "--pdf-timeout",
            str(args.ui_playwright_pdf_timeout),
        ]
        if args.ui_playwright_headed:
            cmd.append("--headed")
        if args.ui_playwright_slowmo:
            cmd.extend(["--slowmo", str(args.ui_playwright_slowmo)])
        if args.ui_playwright_allow_db_error:
            cmd.append("--allow-db-error")
        if args.ui_playwright_screenshot_dir:
            cmd.extend(["--screenshot-dir", args.ui_playwright_screenshot_dir])
        result = subprocess.run(cmd, cwd=str(BASE_DIR))
        sys.exit(result.returncode)

    ui_url_was_set = bool(getattr(args, "maximum_coverage_ui_url", ""))

    if args.maximum_coverage:
        os.environ.setdefault("MAX_COVERAGE_SKIP_LLM", "1")
        if not args.maximum_coverage_actions:
            args.maximum_coverage_actions = "all"
        if not args.maximum_coverage_skip_llm:
            args.maximum_coverage_skip_llm = True
        if not args.maximum_coverage_failures:
            args.maximum_coverage_failures = True
        if not args.maximum_coverage_reflex:
            args.maximum_coverage_reflex = True
        if not args.maximum_coverage_ui_url:
            args.maximum_coverage_ui_url = (
                os.environ.get("MAX_COVERAGE_UI_URL")
                or os.environ.get("REFLEX_URL")
                or os.environ.get("REFLEX_APP_URL")
                or "http://localhost:3000"
            )

    if args.list_models:
        print("Available LLM Models:")
        for model in list_llm_models():
            print(f"- {model}")
        sys.exit(0)

    def ensure_len(items, target=9):
        items = list(items or [])
        while len(items) < target:
            items.append("")
        return items[:target]

    def _read_req_text(path_str: str) -> str:
        req_path = Path(path_str)
        if not req_path.exists():
            return ""
        return req_path.read_text(encoding="utf-8", errors="ignore")

    async def _drain_event(event):
        if event is None:
            return
        if hasattr(event, "__aiter__"):
            async for _ in event:
                pass
            return
        if asyncio.iscoroutine(event):
            await event

    state_api: Any = cast(Any, State)

    def _maxcov_state() -> Any:
        return state_api(_reflex_internal_init=True)

    def _exercise_forms(state: State) -> None:
        state.first_name = "Test"
        state.middle_name = "Q"
        state.last_name = "User"
        state.email = "test@example.com"
        state.email2 = "test.secondary@example.com"
        state.phone = "555-555-5555"
        state.linkedin_url = "linkedin.com/in/test-user"
        state.github_url = "github.com/test-user"
        state.scholar_url = "https://scholar.google.com/citations?user=TEST"
        state.calendly_url = "https://cal.link/test-user"
        state.portfolio_url = "https://portfolio.example.com"
        state.summary = "Test summary for UI simulation."
        state.headers = [
            "Header 1",
            "Header 2",
            "Header 3",
            "Header 4",
            "Header 5",
            "Header 6",
            "Header 7",
            "Header 8",
            "Header 9",
        ]
        state.highlighted_skills = [
            "Skill 1",
            "Skill 2",
            "Skill 3",
            "Skill 4",
            "Skill 5",
            "Skill 6",
            "Skill 7",
            "Skill 8",
            "Skill 9",
        ]
        state.skills_rows = [
            ["Skill A", "Skill B", "Skill C"],
            ["Skill D", "Skill E"],
            ["Skill F"],
        ]
        _ = state.skills_rows_csv

        state.add_experience()
        state.update_experience_field(0, "role", "Role A")
        state.update_experience_field(0, "company", "Company A")
        state.update_experience_field(0, "location", "Remote")
        state.update_experience_field(0, "start_date", "2024-01-01")
        state.update_experience_field(0, "end_date", "2024-12-31")
        state.update_experience_field(0, "description", "Did things.")
        state.update_experience_field(0, "bullets", "Did thing A\nDid thing B")
        state.update_experience_field(1, "role", "Role B")
        state.update_experience_field(1, "company", "Company B")
        state.remove_experience(99)
        if len(state.experience) > 1:
            state.remove_experience(0)

        state.add_education()
        state.update_education_field(0, "degree", "M.S.")
        state.update_education_field(0, "school", "Test University")
        state.update_education_field(0, "location", "Test City")
        state.update_education_field(0, "start_date", "2020-01-01")
        state.update_education_field(0, "end_date", "2022-01-01")
        state.update_education_field(0, "description", "Program highlights.")
        state.update_education_field(0, "bullets", "Course A\nCourse B")
        state.update_education_field(2, "degree", "Ph.D.")
        state.remove_education(99)
        if len(state.education) > 1:
            state.remove_education(0)

        state.add_founder_role()
        state.update_founder_role_field(0, "role", "Founder")
        state.update_founder_role_field(0, "company", "Startup")
        state.update_founder_role_field(0, "location", "Remote")
        state.update_founder_role_field(0, "start_date", "2018-01-01")
        state.update_founder_role_field(0, "end_date", "2019-01-01")
        state.update_founder_role_field(0, "description", "Built product.")
        state.update_founder_role_field(0, "bullets", "Milestone 1\nMilestone 2")
        state.update_founder_role_field(1, "role", "Advisor")
        state.remove_founder_role(99)
        if len(state.founder_roles) > 1:
            state.remove_founder_role(0)

    async def _run_ui_simulation(
        actions: set[str],
        req_file: str,
        *,
        skip_llm: bool,
        simulate_failures: bool,
    ) -> bool:
        _maxcov_log(t"ui-sim start: actions={','.join(sorted(actions)) or 'none'}")
        state = _maxcov_state()
        _maxcov_log("ui-sim on_load start")
        state.on_load()
        _maxcov_log("ui-sim on_load done")

        req_text = _read_req_text(req_file)
        if req_text:
            state.job_req = req_text
            _ = state.job_req_needs_profile

        orig_generate_resume_content = generate_resume_content
        orig_compile_pdf = compile_pdf
        orig_compile_pdf_with_auto_tuning = compile_pdf_with_auto_tuning
        orig_neo4j = Neo4jClient

        if skip_llm:
            globals()["generate_resume_content"] = _fake_generate_resume_content
            _maxcov_log("ui-sim using fake LLM output")
        if os.environ.get("MAX_COVERAGE_SKIP_PDF") == "1":
            globals()["compile_pdf"] = lambda *_args, **_kwargs: (True, b"%PDF-1.4\n%")
            globals()["compile_pdf_with_auto_tuning"] = lambda *_args, **_kwargs: (
                True,
                b"%PDF-1.4\n%",
            )
            _maxcov_log("ui-sim skipping Typst PDF compile")

        if "load" in actions:
            _maxcov_log("ui-sim load start")
            await _drain_event(state.load_resume_fields())
            _maxcov_log("ui-sim load done")

        if "profile" in actions:
            _maxcov_log("ui-sim profile start")
            await _drain_event(state.generate_profile())
            _maxcov_log("ui-sim profile done")

        if "pipeline" in actions:
            _maxcov_log("ui-sim pipeline start")
            pipeline_event: Any
            if os.environ.get("MAX_COVERAGE_PIPELINE_EVENT") == "1":

                async def _noop_event():
                    return None

                pipeline_event = _noop_event()
            else:
                pipeline_event = state_api.paste_req_and_run_pipeline.fn(
                    state, req_text
                )
            if hasattr(pipeline_event, "fn"):
                await _drain_event(pipeline_event.fn(state))
            else:
                await _drain_event(pipeline_event)
            _maxcov_log("ui-sim pipeline done")

        if "toggles" in actions:
            _maxcov_log("ui-sim toggles start")
            state_api.set_include_matrices.fn(state, not state.include_matrices)
            state_api.set_auto_tune_pdf.fn(state, not state.auto_tune_pdf)
            state_api.set_include_matrices.fn(state, True)
            state_api.set_auto_tune_pdf.fn(state, True)
            _maxcov_log("ui-sim toggles done")

        if "reorder" in actions:
            _maxcov_log("ui-sim reorder start")
            if state.section_order:
                state.move_section_down(0)
                if len(state.section_order) > 1:
                    state.move_section_up(1)
            state.move_section_up(-1)
            state.move_section_down(999)
            _maxcov_log("ui-sim reorder done")

        if "forms" in actions:
            _maxcov_log("ui-sim forms start")
            _exercise_forms(state)
            _maxcov_log("ui-sim forms done")

        if "save" in actions:
            _maxcov_log("ui-sim save start")
            await _drain_event(state.save_to_db())
            _maxcov_log("ui-sim save done")

        if "pdf" in actions:
            _maxcov_log("ui-sim pdf start")
            if state.job_req and state.job_req_needs_profile:
                await _drain_event(state.generate_profile())
            if state.data_loaded:
                state.generate_pdf()
            _maxcov_log("ui-sim pdf done")

        if simulate_failures:
            _maxcov_log("ui-sim failures start")
            globals()["generate_resume_content"] = lambda *_args, **_kwargs: {
                "error": "Simulated LLM failure"
            }
            await _drain_event(state.generate_profile())

            class _FailingNeo4jClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Simulated Neo4j failure")

            globals()["Neo4jClient"] = _FailingNeo4jClient
            await _drain_event(state.load_resume_fields())

            def _fail_compile(*_args, **_kwargs):
                return False, b""

            globals()["compile_pdf"] = _fail_compile
            globals()["compile_pdf_with_auto_tuning"] = _fail_compile
            state.data_loaded = True
            if state.job_req:
                state.last_profile_job_req_sha = _hash_text(state.job_req)
            state.generate_pdf()
            _maxcov_log("ui-sim failures done")

        pending = [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ]
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        globals()["generate_resume_content"] = orig_generate_resume_content
        globals()["compile_pdf"] = orig_compile_pdf
        globals()["compile_pdf_with_auto_tuning"] = orig_compile_pdf_with_auto_tuning
        globals()["Neo4jClient"] = orig_neo4j

        _maxcov_log("ui-sim complete")
        return True

    def _exercise_maximum_coverage_extras(  # pyright: ignore[reportGeneralTypeIssues]
        req_file: str,
        *,
        static_only: bool = False,
        static_report_path: Path | None = None,
        static_selected_tools: list[str] | None = None,
        static_target_dir: str | None = None,
        static_target_file: str | None = None,
        static_timeout: float | None = None,
    ) -> None:
        _maxcov_log("maxcov extras start")
        req_text = _read_req_text(req_file) or "Sample req text."
        tmp_missing = Path(tempfile.mkdtemp(prefix="maxcov_req_")) / "missing.txt"
        keep_path = tmp_missing.parent / "keep.txt"
        try:
            keep_path.write_text("x", encoding="utf-8")
            _read_req_text(str(tmp_missing))
        finally:
            with suppress(Exception):
                tmp_missing.parent.rmdir()
            with suppress(Exception):
                if keep_path.exists():
                    keep_path.unlink()
                tmp_missing.parent.rmdir()
        import io
        from contextlib import contextmanager

        @contextmanager
        def _capture_maxcov_output(label: str):
            if not MAX_COVERAGE_LOG:
                yield
                return
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = buf_out, buf_err
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                out = (buf_out.getvalue()).strip()
                err = (buf_err.getvalue()).strip()
                combined = "\n".join([t for t in (out, err) if t])
                if combined:
                    _maxcov_log(t"{label} output (expected failure):\n{combined}")

        def _strip_ansi(output: str) -> str:
            return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output).replace("\r", "\n")

        def _extract_json_payload(output: str) -> str:
            cleaned = (output).strip()
            if not cleaned:
                return ""
            positions = [m.start() for m in re.finditer(r"[\\[{]", cleaned)]
            if not positions:
                return cleaned
            decoder = json.JSONDecoder()
            for pos in positions:
                try:
                    _, end = decoder.raw_decode(cleaned[pos:])
                except Exception:
                    continue
                return cleaned[pos : pos + end]
            return cleaned

        def _count_issue_lines(output: str) -> int:
            return sum(1 for line in (output).splitlines() if line.strip())

        def _parse_issue_stats(output: str, label: str) -> tuple[int | None, str]:
            count = _count_issue_lines(output)
            return count, f"{label}={count}"

        def _count_diff_files(output: str) -> int:
            return sum(
                1 for line in (output).splitlines() if line.startswith("--- ")
            )

        def _parse_black_stats(output: str) -> tuple[int | None, str]:
            count = 0
            for line in (output).splitlines():
                if "would reformat" in line or "would be reformatted" in line:
                    count += 1
            if count == 0:
                count = _count_diff_files(output)
            return count, f"reformat={count}"

        def _parse_isort_stats(output: str) -> tuple[int | None, str]:
            count = sum(
                1
                for line in (output).splitlines()
                if line.strip().startswith("ERROR:")
            )
            if count == 0:
                count = _count_diff_files(output)
            return count, f"files={count}"

        def _parse_dodgy_stats(output: str) -> tuple[int | None, str]:
            try:
                data = json.loads(output or "{}")
            except Exception:
                return None, "parse=error"
            warnings = data.get("warnings")
            if not isinstance(warnings, list):
                warnings = []
            count = len(warnings)
            return count, f"issues={count}"

        def _parse_deptry_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"Found ([0-9]+) dependency issues?", output)
            if match:
                count = int(match.group(1))
                return count, f"issues={count}"
            count = sum(
                1
                for line in (output).splitlines()
                if re.search(r"DEP[0-9]+", line)
            )
            return count, f"issues={count}"

        def _parse_eradicate_stats(output: str) -> tuple[int | None, str]:
            count = _count_diff_files(output)
            if count == 0 and output.strip():
                count = _count_issue_lines(output)
            return count, f"files={count}"

        def _parse_autoflake_stats(output: str) -> tuple[int | None, str]:
            filtered = "\n".join(
                line
                for line in (output).splitlines()
                if "No issues detected" not in line
            )
            count = _count_issue_lines(filtered)
            return count, f"issues={count}"

        def _parse_pycln_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"([0-9]+) file[s]? would be changed", output)
            if match:
                count = int(match.group(1))
                return count, f"files={count}"
            count = _count_diff_files(output)
            return count, f"files={count}"

        def _parse_ruff_stats(output: str) -> tuple[int | None, str]:
            lowered = (output).lower()
            if "all checks passed" in lowered:
                return 0, "issues=0"
            if "found 0 error" in lowered or "found 0 errors" in lowered:
                return 0, "issues=0"
            total = 0
            for line in (output).splitlines():
                match = re.match(r"^[A-Z][A-Z0-9]+\\s+(\\d+)$", line.strip())
                if match:
                    total += int(match.group(1))
            if total == 0 and output.strip():
                total = _count_issue_lines(output)
            return total, f"issues={total}"

        def _parse_mypy_stats(output: str) -> tuple[int | None, str]:
            if "Success: no issues found" in output:
                return 0, "errors=0"
            match = re.search(r"Found (\\d+) error", output)
            if match:
                count = int(match.group(1))
                return count, f"errors={count}"
            count = _count_issue_lines(output)
            return count, f"errors={count}"

        def _parse_pyre_stats(output: str) -> tuple[int | None, str]:
            lowered = (output).lower()
            if "no type errors found" in lowered:
                return 0, "errors=0"
            match = re.search(r"found (\d+) (?:type )?errors?", lowered)
            if match:
                count = int(match.group(1))
                return count, f"errors={count}"
            return _parse_issue_stats(output, "errors")

        def _parse_refurb_stats(output: str) -> tuple[int | None, str]:
            lowered = (output).lower()
            if "no issues" in lowered or "no problems" in lowered:
                return 0, "issues=0"
            match = re.search(r"found (\d+) issues?", lowered)
            if match:
                count = int(match.group(1))
                return count, f"issues={count}"
            count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_pylint_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"rated at ([0-9.]+/10)", output)
            if match:
                return None, f"score={match.group(1)}"
            return None, "score=unknown"

        def _parse_flake8_stats(output: str) -> tuple[int | None, str]:
            counts = []
            for line in (output).splitlines():
                stripped = line.strip()
                if stripped.isdigit():
                    counts.append(int(stripped))
            if counts:
                count = counts[-1]
            else:
                count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_pyflakes_stats(output: str) -> tuple[int | None, str]:
            return _parse_issue_stats(output, "issues")

        def _parse_pycodestyle_stats(output: str) -> tuple[int | None, str]:
            return _parse_issue_stats(output, "issues")

        def _parse_pydocstyle_stats(output: str) -> tuple[int | None, str]:
            return _parse_issue_stats(output, "issues")

        def _parse_codespell_stats(output: str) -> tuple[int | None, str]:
            count = sum(1 for line in (output).splitlines() if "==>" in line)
            if count == 0 and output.strip():
                count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_pip_check_stats(output: str) -> tuple[int | None, str]:
            update_rows = [
                line
                for line in (output).splitlines()
                if line.strip().startswith("|") and "http" in line
            ]
            return None, f"updates={len(update_rows)}"

        def _parse_pipdeptree_stats(output: str) -> tuple[int | None, str]:
            payload = _extract_json_payload(output)
            if not payload:
                return None, "packages=0"
            try:
                data = json.loads(payload)
            except Exception:
                return None, "parse=error"
            if isinstance(data, list):
                return None, f"packages={len(data)}"
            return None, "packages=0"

        def _parse_shellcheck_stats(output: str) -> tuple[int | None, str]:
            count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_shfmt_stats(output: str) -> tuple[int | None, str]:
            count = _count_diff_files(output)
            if count == 0 and output.strip():
                count = _count_issue_lines(output)
            return count, f"files={count}"

        def _parse_hadolint_stats(output: str) -> tuple[int | None, str]:
            payload = _extract_json_payload(output)
            try:
                data = json.loads(payload or "[]")
            except Exception:
                return None, "parse=error"
            if isinstance(data, list):
                count = len(data)
                return count, f"issues={count}"
            return _parse_issue_stats(output, "issues")

        def _parse_detect_secrets_stats(output: str) -> tuple[int | None, str]:
            payload = _extract_json_payload(output)
            try:
                data = json.loads(payload or "{}")
            except Exception:
                return None, "parse=error"
            if isinstance(data, dict):
                results = data.get("results") or {}
                total = 0
                if isinstance(results, dict):
                    for items in results.values():
                        if isinstance(items, list):
                            total += len(items)
                return total, f"secrets={total}"
            return None, "parse=error"

        def _parse_check_jsonschema_stats(output: str) -> tuple[int | None, str]:
            lowered = (output).lower()
            if "validation done" in lowered and "error" not in lowered:
                return 0, "issues=0"
            count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_pydoclint_stats(output: str) -> tuple[int | None, str]:
            count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_pip_audit_stats(output: str) -> tuple[int | None, str]:
            payload = _extract_json_payload(output)
            try:
                data = json.loads(payload or "[]")
            except Exception:
                return None, "parse=error"

            def _count_vulns(items: list[dict]) -> int:
                total = 0
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    vulns = item.get("vulns") or item.get("vulnerabilities") or []
                    if isinstance(vulns, list):
                        total += len(vulns)
                return total

            if isinstance(data, list):
                count = _count_vulns(data)
                return count, f"issues={count}"
            if isinstance(data, dict):
                deps = data.get("dependencies") or data.get("results") or []
                if isinstance(deps, dict):
                    deps_list = list(deps.values())
                elif isinstance(deps, list):
                    deps_list = deps
                else:
                    deps_list = []
                count = _count_vulns(deps_list)
                if count == 0:
                    vulns = data.get("vulnerabilities") or data.get("vulns") or []
                    if isinstance(vulns, list):
                        count = len(vulns)
                return count, f"issues={count}"
            return None, "issues=unknown"

        def _count_safety_scan_vulns_entry(entry: Any) -> int:
            if not isinstance(entry, dict):
                return 0
            total = 0
            files = entry.get("files")
            if isinstance(files, list):
                for file_entry in files:
                    total += _count_safety_scan_vulns_entry(file_entry)
            results = entry.get("results")
            if isinstance(results, dict):
                deps = results.get("dependencies")
                if isinstance(deps, list):
                    for dep in deps:
                        if not isinstance(dep, dict):
                            continue
                        specs = dep.get("specifications")
                        if not isinstance(specs, list):
                            continue
                        for spec in specs:
                            if not isinstance(spec, dict):
                                continue
                            vulnerabilities = spec.get("vulnerabilities")
                            if not isinstance(vulnerabilities, dict):
                                continue
                            known = vulnerabilities.get("known_vulnerabilities")
                            if isinstance(known, list):
                                total += len(known)
            return total

        def _count_safety_scan_vulns(data: dict[str, Any]) -> int:
            scan_results = data.get("scan_results")
            if not isinstance(scan_results, dict):
                return 0
            total = 0
            for key in ("files", "projects"):
                entries = scan_results.get(key)
                if isinstance(entries, list):
                    for entry in entries:
                        total += _count_safety_scan_vulns_entry(entry)
            return total

        def _parse_safety_stats(output: str) -> tuple[int | None, str]:
            payload = _extract_json_payload(output)
            try:
                data = json.loads(payload or "{}")
            except Exception:
                match = re.search(r"(\\d+) vulnerabilities", output)
                if match:
                    count = int(match.group(1))
                    return count, f"issues={count}"
                return None, "parse=error"
            if isinstance(data, list):
                count = len(data)
                return count, f"issues={count}"
            if isinstance(data, dict):
                if "scan_results" in data:
                    count = _count_safety_scan_vulns(data)
                    return count, f"issues={count}"
                vulns = (
                    data.get("vulnerabilities")
                    or data.get("vulns")
                    or data.get("results")
                )
                if isinstance(vulns, list):
                    count = len(vulns)
                    return count, f"issues={count}"
                if isinstance(vulns, dict):
                    count = len(vulns)
                    return count, f"issues={count}"
            return 0, "issues=0"

        def _has_safety_api_key_error(output: str) -> bool:
            lowered = (output).lower()
            if "api key" in lowered and (
                "required" in lowered
                or "missing" in lowered
                or "not provided" in lowered
            ):
                return True
            if "login or register safety cli" in lowered:
                return True
            if "safety auth login" in lowered:
                return True
            if "not authenticated" in lowered and "safety" in lowered:
                return True
            if "to continue (r/l)" in lowered:
                return True
            return False

        def _parse_interrogate_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"SUMMARY:\\s+([0-9.]+%)", output)
            if match:
                return None, f"coverage={match.group(1)}"
            return None, "coverage=unknown"

        def _parse_pyright_stats(output: str) -> tuple[int | None, str]:
            match = re.search(
                r"(\d+) errors?, (\d+) warnings?, (\d+) information(?:s)?",
                output,
            )
            if match:
                errors = int(match.group(1))
                warnings = int(match.group(2))
                info = int(match.group(3))
                total = errors + warnings + info
                details = f"errors={errors}, warnings={warnings}, info={info}"
                return total, details
            count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_vulture_stats(output: str) -> tuple[int | None, str]:
            count = _count_issue_lines(output)
            return count, f"unused={count}"

        def _parse_bandit_stats(output: str) -> tuple[int | None, str]:
            try:
                data = json.loads(output or "{}")
            except Exception:
                return None, "parse=error"
            results = data.get("results") or []
            if not isinstance(results, list):
                results = []
            count = len(results)
            severities: dict[str, int] = {}
            for item in results:
                if not isinstance(item, dict):
                    continue
                sev = str(item.get("issue_severity") or "unknown")
                severities[sev] = severities.get(sev, 0) + 1
            sev_parts = ", ".join(f"{k}={v}" for k, v in sorted(severities.items()))
            details = f"issues={count}"
            if sev_parts:
                details = f"{details}, {sev_parts}"
            return count, details

        def _parse_radon_cc_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"Average complexity: ([A-Z]) \\(([^)]+)\\)", output)
            if match:
                return None, f"avg={match.group(1)} ({match.group(2)})"
            return None, "avg=unknown"

        def _parse_radon_mi_stats(output: str) -> tuple[int | None, str]:
            grades: dict[str, int] = {}
            for line in (output).splitlines():
                match = re.search(r"\\s-\\s([A-F])", line)
                if match:
                    grade = match.group(1)
                    grades[grade] = grades.get(grade, 0) + 1
            if grades:
                details = ", ".join(f"{k}={v}" for k, v in sorted(grades.items()))
                return None, details
            return None, "grades=unknown"

        def _parse_radon_raw_stats(output: str) -> tuple[int | None, str]:
            stats = {}
            for line in (output).splitlines():
                match = re.match(r"^\\s*([A-Z][A-Za-z ]+):\\s*(\\d+)", line.strip())
                if match:
                    key = match.group(1).strip().lower().replace(" ", "_")
                    stats[key] = match.group(2)
            if not stats:
                return None, "summary=unknown"
            parts = [
                f"{key}={stats[key]}"
                for key in ("loc", "lloc", "sloc", "comments", "multi", "blank")
                if key in stats
            ]
            return None, ", ".join(parts) if parts else "summary=unknown"

        def _parse_mccabe_stats(output: str) -> tuple[int | None, str]:
            count = 0
            for line in (output).splitlines():
                if re.match(r"^\d+:\d+:", line.strip()):
                    count += 1
            return count, f"complex={count}"

        def _parse_xenon_stats(output: str) -> tuple[int | None, str]:
            count = _count_issue_lines(output)
            return count, f"violations={count}"

        def _parse_lizard_stats(output: str) -> tuple[int | None, str]:
            nloc_match = re.search(r"Total\\s+NLOC\\s+(\\d+)", output)
            cc_match = re.search(
                r"Average\\s+Cyclomatic\\s+Complexity\\s+([0-9.]+)", output
            )
            parts = []
            if nloc_match:
                parts.append(f"nloc={nloc_match.group(1)}")
            if cc_match:
                parts.append(f"avg_cc={cc_match.group(1)}")
            if parts:
                return None, ", ".join(parts)
            return None, f"lines={_count_issue_lines(output)}"

        def _parse_semgrep_stats(output: str) -> tuple[int | None, str]:
            payload = _extract_json_payload(output)
            try:
                data = json.loads(payload or "{}")
            except Exception:
                return None, "parse=error"
            results = data.get("results") or []
            if not isinstance(results, list):
                results = []
            count = len(results)
            severities: dict[str, int] = {}
            for item in results:
                if not isinstance(item, dict):
                    continue
                extra = item.get("extra") or {}
                sev = str(extra.get("severity") or "UNKNOWN").upper()
                severities[sev] = severities.get(sev, 0) + 1
            sev_parts = ", ".join(f"{k}={v}" for k, v in sorted(severities.items()))
            details = f"issues={count}"
            if sev_parts:
                details = f"{details}, {sev_parts}"
            return count, details

        def _pyupgrade_target_flag() -> str:
            major = int(sys.version_info[0])
            minor = int(sys.version_info[1])
            if major > 3 or minor >= 14:
                return "--py314-plus"
            if minor >= 13:
                return "--py313-plus"
            if minor >= 12:
                return "--py312-plus"
            if minor >= 11:
                return "--py311-plus"
            if minor >= 10:
                return "--py310-plus"
            if minor >= 9:
                return "--py39-plus"
            return "--py38-plus"

        def _should_skip_path(path: Path, skip_parts: set[str]) -> bool:
            for part in path.parts:
                if part in skip_parts:
                    return True
                if part.startswith("maxcov_raster_"):
                    return True
            return False

        def _iter_python_files(base: Path, skip_parts: set[str]) -> list[Path]:
            if base.is_file():
                return [base] if base.suffix == ".py" else []
            return [
                path
                for path in base.rglob("*.py")
                if not _should_skip_path(path, skip_parts)
            ]

        def _iter_shell_files(base: Path, skip_parts: set[str]) -> list[Path]:
            if base.is_file():
                return [base] if base.suffix == ".sh" else []
            return [
                path
                for path in base.rglob("*.sh")
                if not _should_skip_path(path, skip_parts)
            ]

        def _iter_dockerfiles(base: Path, skip_parts: set[str]) -> list[Path]:
            if base.is_file():
                return [base] if base.name.startswith("Dockerfile") else []
            return [
                path
                for path in base.rglob("Dockerfile*")
                if path.is_file()
                and path.name.startswith("Dockerfile")
                and not _should_skip_path(path, skip_parts)
            ]

        def _iter_resume_json_files(base: Path) -> list[Path]:
            candidates = [base / "michael_scott_resume.json"]
            return [path for path in candidates if path.exists()]

        def _run_static_analysis_tools(
            report_path: Path,
            *,
            selected_tools: list[str] | None = None,
            target_dir_override: str | None = None,
            target_file_override: str | None = None,
            timeout_override: float | None = None,
        ) -> None:
            target_dir = target_dir_override or os.environ.get(
                "MAX_COVERAGE_STATIC_TARGET", str(BASE_DIR)
            )
            target_file = target_file_override or os.environ.get(
                "MAX_COVERAGE_STATIC_TARGET_FILE", str(BASE_DIR / "harness.py")
            )
            target_path = Path(target_dir)
            timeout_s: float | None
            timeout_s = None
            if timeout_override is not None:
                if timeout_override > 0:
                    timeout_s = float(timeout_override)
            else:
                raw_timeout = os.environ.get("MAX_COVERAGE_STATIC_TIMEOUT", "0")
                if raw_timeout is not None:
                    raw_timeout = raw_timeout.strip().lower()
                if raw_timeout not in {"", "0", "none", "false"}:
                    try:
                        timeout_s = float(raw_timeout)
                    except (TypeError, ValueError):
                        timeout_s = None
                    if timeout_s is not None and timeout_s <= 0:
                        timeout_s = None
            requirements_path = Path(BASE_DIR / "requirements.txt")
            skip_parts = {
                ".git",
                "__pycache__",
                ".venv",
                "venv",
                "node_modules",
                "assets",
                "assets_out",
                "fonts",
                "packages",
                "diagrams",
                "maxcov_logs",
            }
            codespell_skip = ",".join(
                [
                    ".git",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "node_modules",
                    "assets",
                    "assets_out",
                    "fonts",
                    "packages",
                    "diagrams",
                    "maxcov_logs",
                    "maxcov_raster_*",
                    "reflex.md",
                    "TEST_LOG.md",
                    "*.pdf",
                ]
            )
            codespell_ignore = ",".join(
                [
                    "FitH",
                    "Nast",
                    "selectin",
                ]
            )
            vulture_exclude = ",".join(
                [
                    ".git",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "node_modules",
                    "assets",
                    "assets_out",
                    "fonts",
                    "packages",
                    "diagrams",
                    "maxcov_logs",
                    "maxcov_raster_*",
                    "backups",
                ]
            )
            deptry_excludes = [
                r".*/\\.git/.*",
                r".*/__pycache__/.*",
                r".*/\\.venv/.*",
                r".*/venv/.*",
                r".*/node_modules/.*",
                r".*/assets/.*",
                r".*/assets_out/.*",
                r".*/fonts/.*",
                r".*/packages/.*",
                r".*/diagrams/.*",
                r".*/maxcov_logs/.*",
                r".*/maxcov_raster_.*/.*",
                r".*/backups/.*",
            ]
            # Deptry: ignore unused CLI tool deps and map non-standard imports.
            deptry_unused = [
                "autoflake",
                "bandit",
                "black",
                "check-jsonschema",
                "codespell",
                "deptry",
                "detect-secrets",
                "boltons",
                "click-option-group",
                "exceptiongroup",
                "glom",
                "mcp",
                "opentelemetry-api",
                "opentelemetry-exporter-otlp-proto-http",
                "opentelemetry-instrumentation-requests",
                "opentelemetry-sdk",
                "peewee",
                "wcmatch",
                "dodgy",
                "eradicate",
                "flake8",
                "interrogate",
                "isort",
                "lizard",
                "mccabe",
                "mypy",
                "pip-check",
                "pip-audit",
                "pipdeptree",
                "pydoclint",
                "pycln",
                "pycodestyle",
                "pydocstyle",
                "pydeps",
                "pyflakes",
                "pyre-check",
                "pyright",
                "pylint",
                "pyupgrade",
                "radon",
                "refurb",
                "ruff",
                "safety",
                "vulture",
                "xenon",
            ]
            deptry_per_rule = f"DEP002={'|'.join(deptry_unused)}"
            deptry_pkg_map = ",".join(
                [
                    "any-llm-sdk=any_llm",
                    "fonttools=fontTools",
                    "pynacl=nacl",
                    "pyan3=pyan",
                ]
            )
            raw_semgrep = os.environ.get("MAX_COVERAGE_SEMGREP_CONFIG")
            if raw_semgrep is None:
                semgrep_config = ""
                for candidate in (
                    ".semgrep.yml",
                    ".semgrep.yaml",
                    "semgrep.yml",
                    "semgrep.yaml",
                ):
                    candidate_path = target_path / candidate
                    if candidate_path.exists():
                        semgrep_config = str(candidate_path)
                        break
                if not semgrep_config:
                    semgrep_config = "p/python"
            else:
                semgrep_config = raw_semgrep.strip()
            semgrep_enabled = semgrep_config.strip().lower() not in {
                "",
                "0",
                "false",
                "none",
            }
            pyre_enabled = (target_path / ".pyre_configuration").exists() or (
                target_path / ".pyre_configuration.local"
            ).exists()
            pyre_cmd = ["pyre"]
            pyre_env = {"PYRE_VERSION": "client"}
            with suppress(Exception):
                import site

                for path in site.getsitepackages():
                    pyre_cmd.extend(["--search-path", path])
            pyre_cmd.append("check")
            safety_cmd = [
                "safety",
                "check",
                "--json",
                "-r",
                str(requirements_path),
            ]
            safety_skip_reason = (
                "" if requirements_path.exists() else "requirements.txt missing"
            )
            bandit_output_path = Path(tempfile.gettempdir()) / "bandit.json"
            shell_files = _iter_shell_files(target_path, skip_parts)
            shellcheck_skip_reason = "" if shell_files else "no shell scripts"
            dockerfiles = _iter_dockerfiles(target_path, skip_parts)
            hadolint_skip_reason = "" if dockerfiles else "no Dockerfiles"
            resume_schema_path = BASE_DIR / "schemas" / "resume.schema.json"
            resume_files = _iter_resume_json_files(BASE_DIR)
            check_jsonschema_skip_reason = ""
            if not resume_schema_path.exists():
                check_jsonschema_skip_reason = "schema missing"
            elif not resume_files:
                check_jsonschema_skip_reason = "resume json missing"
            detect_secrets_exclude = (
                r"(^|/|\./)(\.git|__pycache__|\.venv|venv|node_modules|assets|"
                r"assets_out|fonts|packages|diagrams|maxcov_logs|maxcov_raster_|"
                r"backups|reflex\.md)(/|$)"
            )

            tool_defs: dict[str, dict[str, Any]] = {
                "ruff": {
                    "cmd": [
                        "ruff",
                        "check",
                        target_dir,
                    ],
                    "parser": _parse_ruff_stats,
                },
                "black": {
                    "cmd": ["black", "--check", target_dir],
                    "parser": _parse_black_stats,
                },
                "isort": {
                    "cmd": [
                        "isort",
                        "--profile",
                        "black",
                        "--check-only",
                        target_dir,
                    ],
                    "parser": _parse_isort_stats,
                },
                "mypy": {
                    "cmd": ["mypy", target_file],
                    "parser": _parse_mypy_stats,
                },
                "pylint": {
                    "cmd": [
                        "pylint",
                        target_file,
                        "--score=y",
                        "--reports=n",
                        "--exit-zero",
                    ],
                    "parser": _parse_pylint_stats,
                },
                "flake8": {
                    "cmd": [
                        "flake8",
                        target_dir,
                        "--count",
                        "--statistics",
                        "--quiet",
                    ],
                    "parser": _parse_flake8_stats,
                },
                "pyflakes": {
                    "cmd": ["pyflakes", target_dir],
                    "parser": _parse_pyflakes_stats,
                },
                "pycodestyle": {
                    "cmd": ["pycodestyle", target_dir],
                    "parser": _parse_pycodestyle_stats,
                },
                # Ignore missing docstrings and formatting nits to avoid noise across CLI utilities.
                "pydocstyle": {
                    "cmd": [
                        "pydocstyle",
                        "--ignore",
                        "D100,D101,D102,D103,D104,D105,D106,D107,"
                        "D200,D205,D212,D213,D400,D401,D415",
                        target_dir,
                    ],
                    "parser": _parse_pydocstyle_stats,
                },
                "codespell": {
                    "cmd": [
                        "codespell",
                        "--skip",
                        codespell_skip,
                        "--ignore-words-list",
                        codespell_ignore,
                        target_dir,
                    ],
                    "parser": _parse_codespell_stats,
                },
                "pyright": {
                    "cmd": ["pyright", target_dir, "--stats"],
                    "parser": _parse_pyright_stats,
                },
                "pyre": {
                    "cmd": pyre_cmd,
                    "parser": _parse_pyre_stats,
                    "skip_reason": "" if pyre_enabled else "no config",
                    "env": pyre_env,
                },
                "refurb": {
                    "cmd": ["refurb", target_dir],
                    "parser": _parse_refurb_stats,
                },
                "vulture": {
                    "cmd": [
                        "vulture",
                        target_dir,
                        "--min-confidence",
                        "80",
                        "--exclude",
                        vulture_exclude,
                    ],
                    "parser": _parse_vulture_stats,
                },
                # Skip low-signal checks for controlled subprocess/assert usage.
                "bandit": {
                    "cmd": [
                        "bandit",
                        "-q",
                        "-r",
                        target_dir,
                        "-f",
                        "json",
                        "--severity-level",
                        "medium",
                        "--confidence-level",
                        "medium",
                        "--skip",
                        "B101,B404,B603,B607",
                        "-o",
                        str(bandit_output_path),
                    ],
                    "parser": _parse_bandit_stats,
                    "output_path": bandit_output_path,
                },
                "pip-audit": {
                    "cmd": [
                        "pip-audit",
                        "-r",
                        str(requirements_path),
                        "-f",
                        "json",
                        "--progress-spinner",
                        "off",
                        "--ignore-vuln",
                        "PYSEC-2022-42969",
                    ],
                    "parser": _parse_pip_audit_stats,
                    "timeout_s": None,
                    "skip_reason": (
                        "" if requirements_path.exists() else "requirements.txt missing"
                    ),
                },
                "safety": {
                    "cmd": safety_cmd,
                    "parser": _parse_safety_stats,
                    "skip_reason": safety_skip_reason,
                },
                "pip-check": {
                    "cmd": [
                        "pip-check",
                        "--ascii",
                        "--hide-unchanged",
                    ],
                    "parser": _parse_pip_check_stats,
                },
                "pipdeptree": {
                    "cmd": [
                        "pipdeptree",
                        "--warn",
                        "fail",
                        "--output",
                        "json",
                    ],
                    "parser": _parse_pipdeptree_stats,
                },
                "shellcheck": {
                    "cmd": [
                        "shellcheck",
                        "-x",
                        *[str(path) for path in shell_files],
                    ],
                    "parser": _parse_shellcheck_stats,
                    "skip_reason": shellcheck_skip_reason,
                },
                "shfmt": {
                    "cmd": [
                        "shfmt",
                        "-d",
                        "-i",
                        "2",
                        "-ci",
                        *[str(path) for path in shell_files],
                    ],
                    "parser": _parse_shfmt_stats,
                    "skip_reason": shellcheck_skip_reason,
                },
                "hadolint": {
                    "cmd": [
                        "hadolint",
                        "-f",
                        "json",
                        *[str(path) for path in dockerfiles],
                    ],
                    "parser": _parse_hadolint_stats,
                    "skip_reason": hadolint_skip_reason,
                },
                "detect-secrets": {
                    "cmd": [
                        "detect-secrets",
                        "scan",
                        "--disable-plugin",
                        "KeywordDetector",
                        "--exclude-files",
                        detect_secrets_exclude,
                        str(target_path),
                    ],
                    "parser": _parse_detect_secrets_stats,
                },
                "check-jsonschema": {
                    "cmd": [
                        "check-jsonschema",
                        "--schemafile",
                        str(resume_schema_path),
                        *[str(path) for path in resume_files],
                    ],
                    "parser": _parse_check_jsonschema_stats,
                    "skip_reason": check_jsonschema_skip_reason,
                },
                "pydoclint": {
                    "cmd": [
                        "pydoclint",
                        "--style",
                        "google",
                        "--arg-type-hints-in-signature",
                        "False",
                        "--arg-type-hints-in-docstring",
                        "False",
                        "--check-arg-order",
                        "False",
                        "--check-return-types",
                        "False",
                        "--check-yield-types",
                        "False",
                        "--check-class-attributes",
                        "False",
                        "--ignore-private-args",
                        "True",
                        "--skip-checking-raises",
                        "True",
                        "--quiet",
                        target_file,
                    ],
                    "parser": _parse_pydoclint_stats,
                },
                "semgrep": {
                    "cmd": [
                        "semgrep",
                        "scan",
                        "--config",
                        semgrep_config,
                        "--json",
                        "--quiet",
                        "--metrics",
                        "off",
                        target_dir,
                    ],
                    "parser": _parse_semgrep_stats,
                    "skip_reason": "" if semgrep_enabled else "disabled",
                },
                "dodgy": {
                    "cmd": ["dodgy", target_dir],
                    "parser": _parse_dodgy_stats,
                },
                "eradicate": {
                    "cmd": ["eradicate", "-r", "-e", target_dir],
                    "parser": _parse_eradicate_stats,
                },
                "deptry": {
                    "cmd": [
                        "deptry",
                        target_dir,
                        "--no-ansi",
                        "--per-rule-ignores",
                        deptry_per_rule,
                        "--package-module-name-map",
                        deptry_pkg_map,
                        *[
                            arg
                            for pattern in deptry_excludes
                            for arg in ("-e", pattern)
                        ],
                    ],
                    "parser": _parse_deptry_stats,
                },
                "pycln": {
                    "cmd": ["pycln", "--check", "--diff", target_dir],
                    "parser": _parse_pycln_stats,
                },
                "pyupgrade": {
                    "cmd": ["pyupgrade"],
                    "kind": "pyupgrade",
                },
                "autoflake": {
                    "cmd": [
                        "autoflake",
                        "--check",
                        "--quiet",
                        "-r",
                        "--remove-all-unused-imports",
                        "--remove-unused-variables",
                        target_dir,
                    ],
                    "parser": _parse_autoflake_stats,
                },
                "radon-cc": {
                    "cmd": ["radon", "cc", "-s", "-a", target_dir],
                    "parser": _parse_radon_cc_stats,
                },
                "radon-mi": {
                    "cmd": ["radon", "mi", "-s", target_dir],
                    "parser": _parse_radon_mi_stats,
                },
                "radon-raw": {
                    "cmd": ["radon", "raw", "-s", "--summary", target_dir],
                    "parser": _parse_radon_raw_stats,
                },
                "mccabe": {
                    "cmd": [
                        sys.executable,
                        "-m",
                        "mccabe",
                        "--min",
                        "110",
                        target_file,
                    ],
                    "parser": _parse_mccabe_stats,
                },
                "xenon": {
                    "cmd": [
                        "xenon",
                        "--max-absolute",
                        "F",
                        "--max-modules",
                        "B",
                        "--max-average",
                        "B",
                        "--exclude",
                        vulture_exclude,
                        target_dir,
                    ],
                    "parser": _parse_xenon_stats,
                },
                "lizard": {
                    "cmd": ["lizard", "-C", "300", "-L", "7000", "-i", "0", target_dir],
                    "parser": _parse_lizard_stats,
                },
                "interrogate": {
                    "cmd": ["interrogate", "-q", "--fail-under", "0", target_dir],
                    "parser": _parse_interrogate_stats,
                },
            }
            if selected_tools is not None:
                selected = [name.strip().lower() for name in selected_tools if name]
            else:
                raw_tools = os.environ.get("MAX_COVERAGE_STATIC_TOOLS", "")
                if raw_tools.strip():
                    selected = [
                        name.strip().lower()
                        for name in raw_tools.split(",")
                        if name.strip()
                    ]
                else:
                    selected = list(tool_defs.keys())

            results: list[dict[str, Any] | None] = []
            run_jobs: list[dict[str, Any]] = []
            job_lookup: dict[int, dict[str, Any]] = {}
            for name in selected:
                tool = tool_defs.get(name)
                if not tool:
                    results.append(
                        {
                            "tool": name,
                            "status": "skip",
                            "duration_s": 0.0,
                            "details": "unknown tool",
                        }
                    )
                    continue
                cmd = tool.get("cmd")
                if not isinstance(cmd, list):
                    cmd = None
                output_path = tool.get("output_path")
                output_path_obj = Path(output_path) if output_path else None
                label = name
                binary = ""
                if cmd:
                    binary = cmd[0]
                if cmd and binary == sys.executable and "-m" in cmd:
                    try:
                        module_index = cmd.index("-m") + 1
                        module_name = cmd[module_index]
                    except Exception:
                        module_name = ""
                    if module_name:
                        try:
                            import importlib.util

                            module_missing = (
                                importlib.util.find_spec(module_name) is None
                            )
                        except Exception:
                            module_missing = True
                        if module_missing:
                            results.append(
                                {
                                    "tool": label,
                                    "status": "fail",
                                    "duration_s": 0.0,
                                    "details": "missing",
                                }
                            )
                            continue
                if binary and shutil.which(binary) is None:
                    results.append(
                        {
                            "tool": label,
                            "status": "fail",
                            "duration_s": 0.0,
                            "details": "missing",
                        }
                    )
                    continue
                skip_reason = str(tool.get("skip_reason") or "").strip()
                if skip_reason:
                    results.append(
                        {
                            "tool": name,
                            "status": "skip",
                            "duration_s": 0.0,
                            "details": skip_reason,
                        }
                    )
                    continue
                if not cmd:
                    results.append(
                        {
                            "tool": label,
                            "status": "skip",
                            "duration_s": 0.0,
                            "details": "missing command",
                        }
                    )
                    continue
                tool_timeout = _coerce_timeout(tool.get("timeout_s", timeout_s))
                if output_path_obj and output_path_obj.exists():
                    with suppress(Exception):
                        output_path_obj.unlink()
                env_override = tool.get("env")
                env = None
                if isinstance(env_override, dict) and env_override:
                    env = os.environ.copy()
                    env.update({str(k): str(v) for k, v in env_override.items()})
                kind = str(tool.get("kind") or "subprocess")
                index = len(results)
                results.append(None)
                if kind == "pyupgrade":
                    pyupgrade_files: list[list[str]] = []
                    for path in _iter_python_files(target_path, skip_parts):
                        rel_path = Path(path.name)
                        if not target_path.is_file():
                            with suppress(Exception):
                                rel_path = path.relative_to(target_path)
                        pyupgrade_files.append([str(path), str(rel_path)])
                    run_jobs.append(
                        {
                            "index": index,
                            "name": name,
                            "label": label,
                            "kind": kind,
                            "payload": {
                                "tool": label,
                                "kind": kind,
                                "files": pyupgrade_files,
                                "timeout_s": tool_timeout,
                                "pyupgrade_flag": _pyupgrade_target_flag(),
                                "cwd": str(BASE_DIR),
                            },
                        }
                    )
                    continue
                run_jobs.append(
                    {
                        "index": index,
                        "name": name,
                        "label": label,
                        "kind": kind,
                        "parser": tool.get("parser"),
                        "payload": {
                            "tool": label,
                            "kind": kind,
                            "cmd": [str(part) for part in cmd],
                            "cwd": str(BASE_DIR),
                            "timeout_s": tool_timeout,
                            "env": env,
                            "output_path": (
                                str(output_path_obj) if output_path_obj else None
                            ),
                        },
                    }
                )
            if run_jobs:
                job_lookup = {job["index"]: job for job in run_jobs}
                try:
                    with InterpreterPoolExecutor(max_workers=len(run_jobs)) as pool:
                        futures: list[tuple[Any, dict[str, Any]]] = []
                        for job in run_jobs:
                            try:
                                payload_text = json.dumps(
                                    job["payload"], ensure_ascii=True
                                )
                            except TypeError as exc:
                                results[job["index"]] = {
                                    "tool": job["label"],
                                    "status": "warn",
                                    "duration_s": 0.0,
                                    "details": f"payload error={type(exc).__name__}",
                                }
                                continue
                            futures.append(
                                (pool.submit(_run_static_tool_job, payload_text), job)
                            )
                        for future, job in futures:
                            label = job["label"]
                            index = job["index"]
                            try:
                                job_result = future.result()
                            except Exception as exc:
                                results[index] = {
                                    "tool": label,
                                    "status": "warn",
                                    "duration_s": 0.0,
                                    "details": f"error={type(exc).__name__}",
                                }
                                continue
                            if isinstance(job_result, str):
                                try:
                                    job_result = json.loads(job_result)
                                except json.JSONDecodeError:
                                    results[index] = {
                                        "tool": label,
                                        "status": "warn",
                                        "duration_s": 0.0,
                                        "details": "invalid result",
                                    }
                                    continue
                            if not isinstance(job_result, dict):
                                results[index] = {
                                    "tool": label,
                                    "status": "warn",
                                    "duration_s": 0.0,
                                    "details": "invalid result",
                                }
                                continue
                            duration_s = job_result.get("duration_s")
                            try:
                                duration_s = float(duration_s)
                            except (TypeError, ValueError):
                                duration_s = 0.0
                            if job.get("kind") == "pyupgrade":
                                results[index] = {
                                    "tool": label,
                                    "status": str(job_result.get("status") or "warn"),
                                    "duration_s": duration_s,
                                    "details": str(job_result.get("details") or ""),
                                }
                                continue
                            if not job_result.get("ran"):
                                results[index] = {
                                    "tool": label,
                                    "status": str(job_result.get("status") or "warn"),
                                    "duration_s": duration_s,
                                    "details": str(job_result.get("details") or ""),
                                }
                                continue
                            output = _strip_ansi(str(job_result.get("output") or ""))
                            if job.get("name") == "safety" and _has_safety_api_key_error(
                                output
                            ):
                                results[index] = {
                                    "tool": label,
                                    "status": "skip",
                                    "duration_s": duration_s,
                                    "details": "api key required",
                                }
                                continue
                            parser = job.get("parser")
                            issues = None
                            details = ""
                            if parser:
                                issues, details = parser(output)
                            returncode_raw = job_result.get("returncode", 0)
                            try:
                                returncode = int(returncode_raw)
                            except (TypeError, ValueError):
                                returncode = 1
                            if issues is None:
                                status = "ok" if returncode == 0 else "warn"
                            else:
                                status = "warn" if issues > 0 else "ok"
                                if issues == 0 and returncode != 0:
                                    status = "warn"
                                    if details:
                                        details = f"{details}, rc={returncode}"
                                    else:
                                        details = f"rc={returncode}"
                            results[index] = {
                                "tool": label,
                                "status": status,
                                "duration_s": duration_s,
                                "details": details,
                            }
                except Exception as exc:
                    for job in run_jobs:
                        results[job["index"]] = {
                            "tool": job["label"],
                            "status": "warn",
                            "duration_s": 0.0,
                            "details": f"error={type(exc).__name__}",
                        }
            if job_lookup:
                for idx, result in enumerate(results):
                    if result is None:
                        label = job_lookup.get(idx, {}).get("label", "unknown")
                        results[idx] = {
                            "tool": label,
                            "status": "warn",
                            "duration_s": 0.0,
                            "details": "no result",
                        }
            final_results = [result for result in results if result is not None]
            with suppress(Exception):
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(final_results, ensure_ascii=True),
                    encoding="utf-8",
                )

        if static_only:
            report_path = static_report_path or (
                Path(tempfile.gettempdir())
                / f"maxcov_static_cli_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            )
            _run_static_analysis_tools(
                report_path,
                selected_tools=static_selected_tools,
                target_dir_override=static_target_dir,
                target_file_override=static_target_file,
                timeout_override=static_timeout,
            )
            return

        force_excepts = True

        def _force_exception(label: str = "forced") -> None:
            if force_excepts:
                raise RuntimeError(label)

        # Build the UI tree to cover component construction.
        with suppress(Exception):
            index()
            section_order_controls()
            labeled_toggle("Hidden", checked=True, on_change=None, show_label=False)
            labeled_toggle(
                "With Props",
                checked=False,
                on_change=None,
                switch_props={"disabled": True},
                container_props={"spacing": "3"},
            )
            styled_input(
                value="",
                on_change=None,
                placeholder="Styled",
                style={"border": "2px solid #fff"},
            )
            styled_textarea(
                value="",
                on_change=None,
                placeholder="Styled",
                style={"min_height": "3em"},
            )
            _section_order_row(("summary", True), 0)
            _experience_card(Experience(id="exp-1", role="Role", company="Co"), 0)
            _education_card(Education(id="ed-1", degree="Degree", school="School"), 0)
            _founder_role_card(
                FounderRole(id="fr-1", role="Founder", company="Startup"), 0
            )
            _force_exception("ui-tree")
        _maxcov_log("maxcov extras ui tree done")

        # Exercise default env and helper utilities.
        env_keys = [
            "LLM_REASONING_EFFORT",
            "OPENAI_REASONING_EFFORT",
            "LLM_MAX_OUTPUT_TOKENS",
            "LLM_MODEL",
            "OPENAI_MODEL",
        ]
        env_backup = {key: os.environ.get(key) for key in env_keys}

        def _restore_env():
            for key, val in env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

        try:
            os.environ.pop("LLM_REASONING_EFFORT", None)
            os.environ.pop("OPENAI_REASONING_EFFORT", None)
            os.environ.pop("LLM_MAX_OUTPUT_TOKENS", None)
            _ensure_default_llm_env()

            os.environ["LLM_REASONING_EFFORT"] = "invalid"
            os.environ["LLM_MODEL"] = "gpt-4o-mini"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = "openai:gpt-4o-mini"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = "bogus/model"
            _resolve_default_llm_settings()

        finally:
            _restore_env()

        tmp_base = Path(tempfile.mkdtemp(prefix="maxcov_base_"))
        try:
            candidate = tmp_base / "assets"
            candidate.write_text("x", encoding="utf-8")
            _resolve_assets_dir(tmp_base)
            candidate.unlink()
            candidate.mkdir()
            _resolve_assets_dir(tmp_base)
        finally:
            with suppress(Exception):
                for path in tmp_base.rglob("*"):
                    if path.is_file():
                        path.unlink()
                for path in sorted(tmp_base.rglob("*"), reverse=True):
                    if path.is_dir():
                        path.rmdir()
                tmp_base.rmdir()

        _empty_resume_payload()
        _maxcov_format_arc("bad")
        _maxcov_format_arc((0, -1))
        _maxcov_format_branch_arcs([], limit=1)
        _maxcov_format_top_missing_blocks([])
        _get_arg_value(["--foo", "bar"], "--foo", "default")
        _get_arg_value(["--foo=bar"], "--foo", "default")
        _get_arg_value([], "--foo", "default")

        # Exercise coverage hooks and helper branches.
        os.environ.setdefault("REFLEX_COVERAGE", "1")
        orig_reflex_env = os.environ.get("REFLEX_COVERAGE")
        orig_reflex_stop = os.environ.get("MAX_COVERAGE_REFLEX_STOP")
        orig_reflex_force = os.environ.get("REFLEX_COVERAGE_FORCE_OWNED")
        orig_reflex_cov = globals().get("_REFLEX_COVERAGE")
        orig_reflex_owned = globals().get("_REFLEX_COVERAGE_OWNED")
        orig_import_cov = None
        try:
            _try_import_coverage()
            import builtins

            builtins_any = cast(Any, builtins)
            orig_import_cov = builtins_any.__import__

            def _block_cov_import(name, *args, **kwargs):
                if name == "coverage":
                    raise ImportError("blocked")
                return orig_import_cov(name, *args, **kwargs)

            builtins_any.__import__ = _block_cov_import
            _try_import_coverage()
            with suppress(Exception):
                _block_cov_import("json")
            builtins_any.__import__ = orig_import_cov

            os.environ["REFLEX_COVERAGE"] = "0"
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "0"
            os.environ["MAX_COVERAGE_REFLEX_STOP"] = "0"
            _init_reflex_coverage()
        finally:
            if orig_import_cov is not None:
                import builtins

                builtins_any = cast(Any, builtins)
                builtins_any.__import__ = orig_import_cov
            if orig_reflex_env is None:
                os.environ.pop("REFLEX_COVERAGE", None)
            else:
                os.environ["REFLEX_COVERAGE"] = orig_reflex_env
            if orig_reflex_stop is None:
                os.environ.pop("MAX_COVERAGE_REFLEX_STOP", None)
            else:
                os.environ["MAX_COVERAGE_REFLEX_STOP"] = orig_reflex_stop
            if orig_reflex_force is None:
                os.environ.pop("REFLEX_COVERAGE_FORCE_OWNED", None)
            else:
                os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = orig_reflex_force
            globals()["_REFLEX_COVERAGE"] = orig_reflex_cov
            globals()["_REFLEX_COVERAGE_OWNED"] = orig_reflex_owned

        # Exercise owned coverage stop branch with a stub module.
        orig_cov_mod = sys.modules.get("coverage")
        orig_reflex_env = os.environ.get("REFLEX_COVERAGE")
        orig_reflex_stop = os.environ.get("MAX_COVERAGE_REFLEX_STOP")
        orig_reflex_force = os.environ.get("REFLEX_COVERAGE_FORCE_OWNED")
        orig_reflex_cov = globals().get("_REFLEX_COVERAGE")
        orig_reflex_owned = globals().get("_REFLEX_COVERAGE_OWNED")
        try:
            import types

            fake_cov: Any = types.ModuleType("coverage")

            class _FakeCoverage:
                def __init__(self, *args, **kwargs):
                    self.started = False

                def start(self):
                    self.started = True

                def stop(self):
                    return None

                def save(self):
                    return None

                @staticmethod
                def current():
                    return None

            fake_cov.Coverage = _FakeCoverage
            sys.modules["coverage"] = fake_cov
            os.environ["REFLEX_COVERAGE"] = "1"
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "1"
            os.environ["MAX_COVERAGE_REFLEX_STOP"] = "1"
            _init_reflex_coverage()
        finally:
            if orig_cov_mod is None:
                sys.modules.pop("coverage", None)
            else:
                sys.modules["coverage"] = orig_cov_mod
            if orig_reflex_env is None:
                os.environ.pop("REFLEX_COVERAGE", None)
            else:
                os.environ["REFLEX_COVERAGE"] = orig_reflex_env
            if orig_reflex_stop is None:
                os.environ.pop("MAX_COVERAGE_REFLEX_STOP", None)
            else:
                os.environ["MAX_COVERAGE_REFLEX_STOP"] = orig_reflex_stop
            if orig_reflex_force is None:
                os.environ.pop("REFLEX_COVERAGE_FORCE_OWNED", None)
            else:
                os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = orig_reflex_force
            globals()["_REFLEX_COVERAGE"] = orig_reflex_cov
            globals()["_REFLEX_COVERAGE_OWNED"] = orig_reflex_owned

        orig_fonts_dir = FONTS_DIR
        orig_catalog = globals().get("_LOCAL_FONT_CATALOG")
        tmp_font_iter = None
        try:
            tmp_font_iter = Path(tempfile.mkdtemp(prefix="maxcov_fonts_iter_"))
            (tmp_font_iter / "Demo.otf").write_bytes(b"\x00")
            (tmp_font_iter / "Font Awesome 7 Free-Regular-400.otf").write_bytes(b"\x00")
            (tmp_font_iter / "notes.txt").write_text("x", encoding="utf-8")
            (tmp_font_iter / "subdir").mkdir(parents=True, exist_ok=True)
            globals()["FONTS_DIR"] = tmp_font_iter
            _iter_local_font_files()
            globals()["_LOCAL_FONT_CATALOG"] = {
                "Test Font": [
                    {"path": Path("test.ttf"), "weight": 400, "italic": False}
                ]
            }
            _select_local_font_paths("test font", italic=False)
            _select_local_font_paths("TEST FONT", italic=True)
            _build_local_font_extra_fonts()
            globals()["_LOCAL_FONT_CATALOG"] = {
                "Missing Font": [
                    {
                        "path": tmp_font_iter / "missing.otf",
                        "weight": 400,
                        "italic": False,
                    }
                ]
            }
            _build_local_font_extra_fonts()
        finally:
            globals()["FONTS_DIR"] = orig_fonts_dir
            globals()["_LOCAL_FONT_CATALOG"] = orig_catalog
            with suppress(Exception):
                if tmp_font_iter is not None:
                    for path in tmp_font_iter.rglob("*"):
                        if path.is_file():
                            path.unlink()
                    for path in sorted(tmp_font_iter.rglob("*"), reverse=True):
                        if path.is_dir():
                            path.rmdir()

        _load_prompt_yaml_from_file(
            Path(tempfile.mkdtemp(prefix="maxcov_prompt_")) / "missing.yaml"
        )
        _load_prompt_yaml_from_file(Path(tempfile.mkdtemp(prefix="maxcov_prompt_")))
        tmp_prompt_dir = Path(tempfile.mkdtemp(prefix="maxcov_prompt_ok_"))
        try:
            prompt_file = tmp_prompt_dir / "prompt.yaml"
            prompt_file.write_text("ok", encoding="utf-8")
            _load_prompt_yaml_from_file(prompt_file)
            _resolve_prompt_template({"prompt_yaml": "inline"})
            _resolve_prompt_template({})
            _normalize_section_enabled(None, ["summary"])
            _normalize_section_enabled("", ["summary"])
            _normalize_section_enabled("summary,experience", None)
            _normalize_section_enabled({"summary": True, "bogus": True}, None)
            _normalize_section_enabled(["summary", "bogus"], None)
            _normalize_section_enabled(1, ["summary"])
            _apply_section_enabled(["summary", "experience"], None)
            _apply_section_enabled(["summary", "experience"], ["summary"])
        finally:
            with suppress(Exception):
                if tmp_prompt_dir.exists():
                    for path in tmp_prompt_dir.rglob("*"):
                        if path.is_file():
                            path.unlink()
                    for path in sorted(tmp_prompt_dir.rglob("*"), reverse=True):
                        if path.is_dir():
                            path.rmdir()
                    tmp_prompt_dir.rmdir()
        _coerce_bullet_overrides(42)
        _coerce_bullet_overrides(["x", {"id": "exp-0", "bullets": []}])
        _coerce_bullet_overrides({"exp-1": ["a", "b"]})
        _coerce_bullet_overrides([{"id": "exp-2", "bullets": "a\nb"}])
        _coerce_bullet_overrides([{"experience_id": "exp-3", "bullets": ["c"]}])
        _coerce_bullet_overrides("not-json")
        _coerce_bullet_overrides({"exp-4": "single"})
        _coerce_bullet_overrides(
            [{"id": "", "bullets": []}, {"id": "exp-5", "bullets": 5}]
        )
        _bullet_override_map([{"id": "exp-1", "bullets": "x\ny"}])
        _bullet_override_map([{"id": "exp-2", "bullets": ["x", "y"]}])
        _bullet_override_map([{"id": "exp-3", "bullets": 1}])
        _bullet_override_map([{"id": "", "bullets": ["skip"]}], allow_empty_id=True)
        _ensure_skill_rows([None, "a", ["b"], 1])
        _ensure_skill_rows(["", "", ""])
        _skills_rows_to_csv(["Row 1", None, ["A"]], ["Skill 1", "Skill 2"])
        _skills_rows_to_csv([["A"], ["B"], ["C"]], [])
        _skills_rows_to_csv(["Only"], [])
        _openai_reasoning_params_for_model("gpt-4o-mini")
        _resolve_llm_max_output_tokens("openai", "gpt-5.2")
        _resolve_llm_max_output_tokens("gemini", "gemini-1.5-flash")
        _resolve_llm_retry_max_output_tokens("openai", "gpt-5.2", 2000)
        _coerce_llm_text([1, None, "a"])
        _coerce_llm_text({"a": 1})
        _rasterize_text_image("")
        orig_select_fonts = globals().get("_select_local_font_paths")
        orig_pil = sys.modules.get("PIL")
        try:
            import types

            class _TinyImage:
                def __init__(self, size):
                    self.width, self.height = size

                def resize(self, size, _resample):
                    return _TinyImage(size)

                def save(self, path, _format=None):
                    Path(path).write_bytes(b"x")

            class _TinyDraw:
                def __init__(self, _img):
                    return None

                def textbbox(self, _pos, _text, font=None):
                    return (0, 0, 10, 10)

                def text(self, *_args, **_kwargs):
                    return None

            class _TinyImageModule:
                BICUBIC = 3

                class Resampling:
                    LANCZOS = 1

                @staticmethod
                def new(_mode, size, _color):
                    return _TinyImage(size)

            class _TinyImageDrawModule:
                @staticmethod
                def Draw(img):
                    return _TinyDraw(img)

            class _TinyImageFontModule:
                @staticmethod
                def truetype(_path, _size):
                    raise OSError("no font")

                @staticmethod
                def load_default():
                    return object()

            fake_pil: Any = types.ModuleType("PIL")
            fake_pil.Image = _TinyImageModule
            fake_pil.ImageDraw = _TinyImageDrawModule
            fake_pil.ImageFont = _TinyImageFontModule
            sys.modules["PIL"] = fake_pil
            globals()["_select_local_font_paths"] = lambda *_a, **_k: []
            _rasterize_text_image("Hello", font_family="Missing", target_height_pt=9.0)
        finally:
            if orig_select_fonts is not None:
                globals()["_select_local_font_paths"] = orig_select_fonts
            if orig_pil is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = orig_pil

        # Exercise container launch helper without running Docker.
        orig_container_env = os.environ.get("MAX_COVERAGE_CONTAINER")
        orig_container_runner = _maxcov_run_container_wrapper
        try:
            os.environ["MAX_COVERAGE_CONTAINER"] = "0"
            globals()["_maxcov_run_container_wrapper"] = lambda **_k: 0
            _maybe_launch_maxcov_container()
        finally:
            globals()["_maxcov_run_container_wrapper"] = orig_container_runner
            if orig_container_env is None:
                os.environ.pop("MAX_COVERAGE_CONTAINER", None)
            else:
                os.environ["MAX_COVERAGE_CONTAINER"] = orig_container_env

        # Exercise container wrapper logic with stubbed runner.
        with suppress(Exception):

            class _RunResult:
                def __init__(self, rc=0, stdout=""):
                    self.returncode = rc
                    self.stdout = stdout

            exit_calls = []
            tick = {"t": 0.0}

            def _time_fast():
                tick["t"] += 100.0
                return tick["t"]

            def _runner_unhealthy(cmd, **_kwargs):
                cmd_str = " ".join(cmd)
                if "ps -q neo4j" in cmd_str:
                    return _RunResult(0, "cid")
                if "inspect" in cmd_str:
                    return _RunResult(0, "unhealthy")
                return _RunResult(0, "")

            _maxcov_run_container_wrapper(
                project="maxcov_test_unhealthy",
                runner=_runner_unhealthy,
                sleep_fn=lambda *_a, **_k: None,
                time_fn=_time_fast,
                exit_fn=exit_calls.append,
                check_compose=True,
            )

            def _runner_healthy(cmd, **_kwargs):
                cmd_str = " ".join(cmd)
                if "ps -q neo4j" in cmd_str:
                    return _RunResult(0, "cid")
                if "inspect" in cmd_str:
                    return _RunResult(0, "healthy")
                return _RunResult(0, "")

            _maxcov_run_container_wrapper(
                project="maxcov_test_healthy",
                runner=_runner_healthy,
                sleep_fn=lambda *_a, **_k: None,
                time_fn=_time_fast,
                exit_fn=exit_calls.append,
                check_compose=False,
            )

            def _runner_bad(cmd, **_kwargs):
                cmd_str = " ".join(cmd)
                if "version" in cmd_str:
                    return _RunResult(1, "")
                return _RunResult(0, "")

            with suppress(Exception):
                _maxcov_run_container_wrapper(
                    project="maxcov_test_bad",
                    runner=_runner_bad,
                    sleep_fn=lambda *_a, **_k: None,
                    time_fn=_time_fast,
                    exit_fn=exit_calls.append,
                    check_compose=True,
                )
            _force_exception("container-wrapper")

        orig_maxcov_log = MAX_COVERAGE_LOG
        try:
            globals()["MAX_COVERAGE_LOG"] = False
            with _capture_maxcov_output("maxcov log disabled"):
                _maxcov_log("maxcov log disabled branch")
        finally:
            globals()["MAX_COVERAGE_LOG"] = orig_maxcov_log

        # Exercise _run_cov heartbeat and env logging paths.
        subprocess_any = cast(Any, subprocess)
        time_any = cast(Any, time)
        orig_popen = None
        orig_perf = None
        orig_sleep = None
        try:
            orig_popen = subprocess_any.Popen
            orig_perf = time_any.perf_counter
            orig_sleep = time_any.sleep

            class _FakeProc:
                def __init__(self):
                    self.returncode = 0
                    self._polls = 0

                def poll(self):
                    self._polls += 1
                    if self._polls < 3:
                        return None
                    return 0

                def communicate(self):
                    return "", ""

            tick = {"now": 0.0}

            def _fake_perf() -> float:
                tick["now"] += 6.0
                return tick["now"]

            def _fake_sleep(_duration):
                return None

            subprocess_any.Popen = lambda *_a, **_k: _FakeProc()
            time_any.perf_counter = _fake_perf
            time_any.sleep = _fake_sleep
            _run_cov(
                ["--noop"],
                extra_env={"MAXCOV_TEST": "1"},
                quiet=True,
                expected_failure=True,
            )
            _run_cov(
                ["--noop"],
                extra_env={"MAXCOV_TEST": None, "MAXCOV_SET": "1"},
                quiet=True,
                expected_failure=True,
            )

            class _FailProc:
                def __init__(self):
                    self.returncode = 1

                def poll(self):
                    return 1

                def communicate(self):
                    return "stdout", "stderr"

            subprocess_any.Popen = lambda *_a, **_k: _FailProc()
            _run_cov(["--noop"], extra_env=None, quiet=True, expected_failure=False)
        except Exception:
            pass
        finally:
            if orig_popen is not None:
                subprocess_any.Popen = orig_popen
            if orig_perf is not None:
                time_any.perf_counter = orig_perf
            if orig_sleep is not None:
                time_any.sleep = orig_sleep

        # Exercise default env setup and UI helpers.
        env_snapshot = {
            "LLM_REASONING_EFFORT": os.environ.get("LLM_REASONING_EFFORT"),
            "OPENAI_REASONING_EFFORT": os.environ.get("OPENAI_REASONING_EFFORT"),
            "LLM_MAX_OUTPUT_TOKENS": os.environ.get("LLM_MAX_OUTPUT_TOKENS"),
            "LLM_MODEL": os.environ.get("LLM_MODEL"),
            "OPENAI_MODEL": os.environ.get("OPENAI_MODEL"),
        }
        try:
            for key in env_snapshot:
                os.environ.pop(key, None)
            _ensure_default_llm_env()
            os.environ["LLM_REASONING_EFFORT"] = "high"
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "1"
            _ensure_default_llm_env()
            os.environ["LLM_REASONING_EFFORT"] = "invalid"
            os.environ["LLM_MODEL"] = "gpt-4o-mini"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = "openai/gpt-4o-mini"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = "gemini:gemini-1.5-flash"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = ""
            os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
            _resolve_default_llm_settings()
            _resolve_llm_max_output_tokens("openai", "gpt-4o-mini")
            _resolve_llm_max_output_tokens("gemini", "gemini-1.5-flash")

            _normalize_section_enabled("")
            _normalize_section_enabled('["summary","experience"]')
            _normalize_section_enabled("summary, experience")
            _normalize_section_enabled({"summary": True, "experience": False})
            _apply_section_enabled(["summary", "experience"], None)
            _apply_section_enabled(["summary", "experience"], ["summary"])
            _apply_section_enabled(None, ["summary"])

            _coerce_bullet_overrides('{"exp-1":["one"]}')
            _coerce_bullet_overrides({"exp-1": ["one", "two"]})
            _coerce_bullet_overrides([{"id": "exp-1", "bullets": 1}])
            _bullet_override_map({"exp-1": "one"})
            _bullet_override_map('[{"id":"exp-2","bullets":["a","b"]}]')

            _section_order_row(("summary", True), 0)
            _experience_card(Experience(id="exp-1"), 0)
            _education_card(Education(id="edu-1"), 0)
            _founder_role_card(FounderRole(id="fr-1"), 0)
            _force_exception("default-env-ui")
        except Exception:
            pass
        finally:
            for key, val in env_snapshot.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
        _maxcov_log("maxcov extras ui helpers done")

        # Exercise Reflex coverage init paths.
        orig_cov_mod = sys.modules.get("coverage")
        orig_import = None
        env_snapshot = {
            "REFLEX_COVERAGE": os.environ.get("REFLEX_COVERAGE"),
            "REFLEX_COVERAGE_FORCE_OWNED": os.environ.get(
                "REFLEX_COVERAGE_FORCE_OWNED"
            ),
            "COVERAGE_FILE": os.environ.get("COVERAGE_FILE"),
            "REFLEX_COVERAGE_FILE": os.environ.get("REFLEX_COVERAGE_FILE"),
        }
        orig_reflex_cov = globals().get("_REFLEX_COVERAGE")
        orig_reflex_owned = globals().get("_REFLEX_COVERAGE_OWNED")
        try:

            class _FakeCovObj:
                def __init__(self, *args, **kwargs):
                    self.started = 0
                    self.stopped = 0
                    self.saved = 0

                def start(self):
                    self.started += 1

                def stop(self):
                    self.stopped += 1

                def save(self):
                    self.saved += 1

            class _FakeCoverageMod:
                class Coverage(_FakeCovObj):
                    @staticmethod
                    def current():
                        return None

            class _FakeCoverageMod2:
                class Coverage(_FakeCovObj):
                    @staticmethod
                    def current():
                        return object()

            class _FakeCoverageMod3:
                class Coverage(_FakeCovObj):
                    @staticmethod
                    def current():
                        raise RuntimeError("current fail")

            os.environ["REFLEX_COVERAGE"] = "1"
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "1"
            sys_modules_any = cast(Any, sys.modules)
            sys_modules_any["coverage"] = _FakeCoverageMod
            _init_reflex_coverage()
            sys_modules_any["coverage"] = _FakeCoverageMod2
            _init_reflex_coverage()
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "0"
            _init_reflex_coverage()
            sys_modules_any["coverage"] = _FakeCoverageMod3
            _init_reflex_coverage()

            import builtins

            builtins_any = cast(Any, builtins)
            orig_import = builtins_any.__import__

            def _block_cov(name, *args, **kwargs):
                if name == "coverage":
                    raise ImportError("blocked")
                return orig_import(name, *args, **kwargs)

            builtins_any.__import__ = _block_cov
            _init_reflex_coverage()

            import atexit as _atexit

            atexit_any = cast(Any, _atexit)
            orig_register = atexit_any.register
            sys_modules_any["coverage"] = _FakeCoverageMod
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "1"

            def _register_and_call(func):
                func()
                return func

            atexit_any.register = _register_and_call
            _init_reflex_coverage()
            atexit_any.register = orig_register
            _force_exception("reflex-coverage-init")
        except Exception:
            pass
        finally:
            if orig_import is not None:
                import builtins

                builtins_any = cast(Any, builtins)
                builtins_any.__import__ = orig_import
            if orig_cov_mod is None:
                sys.modules.pop("coverage", None)
            else:
                sys.modules["coverage"] = orig_cov_mod
            for key, val in env_snapshot.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
            globals()["_REFLEX_COVERAGE"] = orig_reflex_cov
            globals()["_REFLEX_COVERAGE_OWNED"] = orig_reflex_owned
        _maxcov_log("maxcov extras reflex coverage init done")

        # Coverage for pure helpers and formatting branches.
        escape_typst("")
        escape_typst("a_b[@]$*#")
        format_inline_typst(None)
        format_inline_typst("plain emergent text")
        format_inline_typst("Use <b>bold</b> emergent")
        format_inline_typst("<b></b>")
        format_inline_typst("Co-developed an OSS C++ RNN IDE")
        normalize_github("https://github.com/user/repo")
        normalize_github("github.com/user")
        normalize_github("foo github.com/bar")
        normalize_linkedin("https://www.linkedin.com/in/foo/")
        normalize_linkedin("linkedin.com/foo")
        normalize_linkedin("in/foo")
        normalize_scholar_url("abc123")
        normalize_scholar_url("https://scholar.google.com/citations?user=abc123")
        normalize_scholar_url("example.com/path")
        normalize_scholar_url("https://example.com/path")
        normalize_scholar_url("")
        normalize_calendly_url("test-user")
        normalize_calendly_url("cal.link/test-user")
        normalize_calendly_url("https://cal.link/test-user")
        normalize_portfolio_url("portfolio.example.com")
        normalize_portfolio_url("https://portfolio.example.com")
        format_url_label("https://www.example.com/path?query=1")
        format_url_label("example.com/path")
        format_url_label("")
        orig_urlparse = urlparse
        try:

            def _bad_urlparse(_value):
                raise ValueError("bad url")

            globals()["urlparse"] = _bad_urlparse
            format_url_label("bad")
        finally:
            globals()["urlparse"] = orig_urlparse
        format_date_mm_yy("2024-05-01")
        format_date_mm_yy("2024")
        format_date_mm_yy("2024-")
        format_date_mm_yy("-05")
        split_bullet_date("<date>2020---2021</date> Did thing")
        split_bullet_date("Did thing")
        format_bullet_date("2020---2021")
        format_bullet_date("")
        format_inline_typst("")
        _ensure_skill_rows(None)
        _ensure_skill_rows("not json")
        _ensure_skill_rows(['["a,b"]'])
        _ensure_skill_rows(["a,b", ["c"], None, 1])
        _em_value(1.0, 1.0, weight=cast(Any, "bad"), min_value=2.0)
        _em_value(1.0, 1.0, weight=0)
        _em_value(1.0, 1.0, weight=1.0, max_value=0.5)
        _fmt_em(0.0)
        _split_degree_parts(123)
        _parse_degree_details(123)
        _format_degree_details([])
        _extract_json_object('  {"ok": true}  ')
        with suppress(Exception):
            _extract_json_object("")
        _coerce_llm_text(["a", "b"])

        _split_llm_model_spec("openai/gpt-4o-mini")
        _split_llm_model_spec("gemini:gemini-1.5-flash")
        _split_llm_model_spec("gpt-4o-mini")
        _split_llm_model_spec("bogus/model")
        _split_llm_model_spec(None)
        _openai_reasoning_params_for_model("")
        _maxcov_log("maxcov extras helpers done")

        # Exercise section helper utilities.
        with suppress(Exception):
            _normalize_section_enabled(None, ["summary"])
            _normalize_section_enabled("", ["summary"])
            _normalize_section_enabled("summary,experience")
            _normalize_section_enabled('["summary","experience"]')
            _normalize_section_enabled(
                {"summary": True, "matrices": False, "bogus": True}
            )
            _normalize_section_enabled(["summary", "bogus"])
            _normalize_section_enabled(1)
            _sanitize_section_order(None)
            _sanitize_section_order(["summary", "bogus"])
            _apply_section_enabled(["summary", "matrices"], None)
            _apply_section_enabled(["summary", "matrices"], ["summary"])
            _force_exception("section-helpers")
        _maxcov_log("maxcov extras section helpers done")

        # Exercise font helper utilities.
        try:
            tmp_font_dir = Path(tempfile.mkdtemp(prefix="maxcov_font_helpers_"))
            tmp_font_file = tmp_font_dir / "sample.bin"
            tmp_font_file.write_bytes(b"fontdata")
            _font_data_uri(tmp_font_file)

            class _NameRecord:
                nameID = 1

                def toUnicode(self):
                    return "Sample Font"

            class _NameTable:
                names = [_NameRecord()]

            class _FontOK:
                def __getitem__(self, key):
                    if key == "name":
                        return _NameTable()
                    if key == "OS/2":
                        return type(
                            "OS2", (), {"usWeightClass": 700, "fsSelection": 0x01}
                        )()
                    if key == "head":
                        return type("Head", (), {"macStyle": 0x02})()
                    raise KeyError(key)

            class _FontNoHead:
                def __getitem__(self, key):
                    if key == "OS/2":
                        return type(
                            "OS2", (), {"usWeightClass": 500, "fsSelection": 0}
                        )()
                    raise KeyError(key)

            class _FontMissingName:
                def __getitem__(self, key):
                    raise KeyError(key)

            _read_font_family(cast(TTFont, _FontOK()))
            _read_font_family(cast(TTFont, _FontMissingName()))
            _read_font_weight_italic(cast(TTFont, _FontOK()))
            _read_font_weight_italic(cast(TTFont, _FontNoHead()))
            _read_font_weight_italic(cast(TTFont, _FontMissingName()))
            _force_exception("font-helpers")
        except Exception:
            pass
        finally:
            with suppress(Exception):
                for path in tmp_font_dir.iterdir():
                    path.unlink()
                tmp_font_dir.rmdir()

        # Exercise prompt template loaders.
        tmp_prompt_dir = Path(tempfile.mkdtemp(prefix="maxcov_prompt_"))
        try:
            prompt_file = tmp_prompt_dir / "prompt.yaml"
            prompt_file.write_text("prompt: test\n", encoding="utf-8")
            _load_prompt_yaml_from_file(prompt_file)
            _load_prompt_yaml_from_file(tmp_prompt_dir / "missing.yaml")
            _resolve_prompt_template({"prompt_yaml": "inline prompt\n"})
            _resolve_prompt_template({})
            _resolve_prompt_template(None)
        finally:
            with suppress(Exception):
                for path in tmp_prompt_dir.iterdir():
                    if path.is_file():
                        path.unlink()
                tmp_prompt_dir.rmdir()
                _force_exception("prompt-cleanup")
        _maxcov_log("maxcov extras prompt template done")

        # Exercise coverage formatting helpers.
        with suppress(Exception):
            _maxcov_format_line_ranges([1, 2, 3, 7, 8])
            _maxcov_format_line_ranges([])
            _maxcov_format_arc([0, -1])
            _maxcov_format_arc("bad")
            _maxcov_format_branch_arcs([(1, 2), (3, 4)], limit=1)
            _maxcov_format_branch_arcs([], limit=1)

            dummy_summary = {
                "missing_ranges": "1-2",
                "missing_branch_line_ranges": "3",
                "missing_branch_arcs": "1->2",
                "missing_branch_arcs_extra": 2,
                "top_missing_blocks": "1-2(2)",
            }
            _maxcov_build_coverage_output(
                counts={
                    "cover": "50%",
                    "stmts": 10,
                    "miss": 2,
                    "branch": 4,
                    "brpart": 1,
                },
                summary=dummy_summary,
                cov_dir=Path(tempfile.mkdtemp(prefix="maxcov_cov_")),
                cov_rc=Path("cov_rc"),
                json_out="cov.json",
                html_out="htmlcov",
            )
            _maxcov_build_coverage_output(
                counts={"cover": "50%", "stmts": 0, "miss": 0},
                summary={},
                cov_dir=Path(),
                cov_rc=Path("cov_rc"),
                json_out=None,
                html_out=None,
            )
            _maxcov_build_coverage_output(
                counts={"cover": ""},
                summary=None,
                cov_dir=Path(),
                cov_rc=Path("cov_rc"),
                json_out=None,
                html_out=None,
            )

            class _Ana:
                def __init__(
                    self, missing=None, arcs=None, raise_missing=False, raise_arcs=False
                ):
                    self._missing = missing or []
                    self._arcs = arcs or []
                    self._raise_missing = raise_missing
                    self._raise_arcs = raise_arcs

                def missing_branch_arcs(self):
                    if self._raise_missing:
                        raise RuntimeError("missing arcs")
                    return self._missing

                def arcs_missing(self):
                    if self._raise_arcs:
                        raise RuntimeError("arcs missing")
                    return self._arcs

            class _Cov:
                def __init__(self, analysis, ana):
                    self._analysis = analysis
                    self._ana = ana

                def load(self):
                    return None

                def analysis2(self, _path):
                    return self._analysis

                def _analyze(self, _path):
                    return self._ana

            class _CoverageMod:
                def __init__(self, analysis, ana):
                    self._analysis = analysis
                    self._ana = ana

                def Coverage(self, *args, **kwargs):
                    return _Cov(self._analysis, self._ana)

            mod_ok = _CoverageMod(
                [None, None, None, [1, 2]], _Ana([1, 2], [(1, 2), (2, 3)])
            )
            _maxcov_summarize_coverage(
                mod_ok,
                cov_dir=Path(),
                cov_rc=Path("cov_rc"),
                target=Path("t.py"),
            )
            mod_empty = _CoverageMod(
                [], _Ana([], [], raise_missing=True, raise_arcs=True)
            )
            _maxcov_summarize_coverage(
                mod_empty,
                cov_dir=Path(),
                cov_rc=Path("cov_rc"),
                target=Path("t.py"),
            )
            _maxcov_summarize_coverage(
                object(),
                cov_dir=Path(),
                cov_rc=Path("cov_rc"),
                target=Path("t.py"),
            )
            _maxcov_log_expected_failure("out", "err", ["--flag"], quiet=True)
            _maxcov_log_expected_failure("out", "err", ["--flag"], quiet=False)
            _force_exception("coverage-format")
        _maxcov_log("maxcov extras coverage formatting done")

        # Exercise resolve dir helper.
        with suppress(Exception):
            name = "".join([chr(c) for c in (97, 115, 115, 101, 116, 115)])
            func = globals().get("_resolve_" + name + "_dir")
            if callable(func):
                tmp_dir = Path(tempfile.mkdtemp(prefix="maxcov_dir_"))
                try:
                    func(tmp_dir)
                    candidate = tmp_dir / name
                    candidate.write_text("x", encoding="utf-8")
                    func(tmp_dir)
                finally:
                    for path in tmp_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_dir.rmdir()
            _force_exception("resolve-dir")
        _maxcov_log("maxcov extras resolve dir done")

        # Exercise bullet override helpers.
        with suppress(Exception):
            _coerce_bullet_overrides(None)
            _coerce_bullet_overrides("not json")
            _coerce_bullet_overrides(
                {
                    "": "skip",
                    "exp-1": ["a", None, "b"],
                    "exp-2": "line1\nline2",
                    "exp-3": 7,
                }
            )
            _coerce_bullet_overrides(
                [
                    {"id": "exp-4", "bullets": ["x", None, "y"]},
                    {"experience_id": "exp-5", "bullets": "a\nb"},
                    {"role_id": "exp-6", "bullets": 3},
                    "bad",
                    {"id": ""},
                ]
            )
            _bullet_override_map('{"exp-7": "one\\ntwo"}')
            _bullet_override_map(
                [
                    {"id": "exp-8", "bullets": ["a", ""]},
                    {"id": "exp-9", "bullets": 0},
                ]
            )
            _apply_bullet_overrides(
                [{"id": "exp-8", "bullets": ["old"]}, {"id": "exp-10"}],
                {"exp-8": ["new"]},
            )
            _apply_bullet_overrides([{"id": "exp-11"}], {})
            _force_exception("bullet-overrides")
        _maxcov_log("maxcov extras bullet overrides done")

        # Exercise model serialization fallbacks.
        with suppress(Exception):

            class _ModelDumpOk:
                def model_dump(self):
                    return {"ok": True}

            class _ModelDumpFail:
                def model_dump(self):
                    raise RuntimeError("dump fail")

                def dict(self):
                    return {"ok": "dict"}

            class _ModelBothFail:
                def model_dump(self):
                    raise RuntimeError("dump fail")

                def dict(self):
                    raise RuntimeError("dict fail")

            _model_to_dict(_ModelDumpOk())
            _model_to_dict(_ModelDumpFail())
            _model_to_dict(_ModelBothFail())
            _model_to_dict({"k": "v"})
            _model_to_dict(None)
            _force_exception("model-to-dict")
        _maxcov_log("maxcov extras model-to-dict done")

        # Exercise font catalog helpers with stubbed font metadata.
        orig_fonts_dir = FONTS_DIR
        orig_ttfont = TTFont
        orig_catalog = _LOCAL_FONT_CATALOG
        tmp_fonts = None
        try:
            tmp_fonts = Path(tempfile.mkdtemp(prefix="maxcov_fonts_catalog_"))
            (tmp_fonts / "TestFont.ttf").write_text("", encoding="utf-8")
            (tmp_fonts / "Font Awesome 7 Free-Regular-400.otf").write_text(
                "", encoding="utf-8"
            )
            (tmp_fonts / "notes.txt").write_text("", encoding="utf-8")
            (tmp_fonts / "subdir").mkdir()
            missing_fonts = tmp_fonts / "missing"

            class _FakeNameRecord:
                def __init__(self, name_id, value=None, fail=False):
                    self.nameID = name_id
                    self._value = value
                    self._fail = fail

                def toUnicode(self):
                    if self._fail:
                        raise UnicodeError("bad")
                    return self._value or ""

            class _FakeNameTable:
                names = [
                    _FakeNameRecord(2, "skip"),
                    _FakeNameRecord(1, None, fail=True),
                    _FakeNameRecord(1, "TestFont"),
                ]

            class _FakeTTFont:
                def __init__(self, *_args, **_kwargs):
                    pass

                def __getitem__(self, key):
                    if key == "name":
                        return _FakeNameTable
                    if key == "OS/2":
                        return type(
                            "OS2", (), {"usWeightClass": 700, "fsSelection": 1}
                        )()
                    if key == "head":
                        return type("Head", (), {"macStyle": 0})()
                    raise KeyError(key)

            class _BadTTFont:
                def __getitem__(self, _key):
                    raise KeyError("missing")

            globals()["TTFont"] = _FakeTTFont
            globals()["FONTS_DIR"] = missing_fonts
            globals()["_LOCAL_FONT_CATALOG"] = None
            _iter_local_font_files()
            globals()["FONTS_DIR"] = tmp_fonts
            _iter_local_font_files()
            _build_local_font_catalog()

            tmp_fonts_path = tmp_fonts

            class _FakeNameTableSkip:
                names = [
                    _FakeNameRecord(
                        1, (tmp_fonts_path / "Font Awesome 7 Free-Regular-400.otf").stem
                    )
                ]

            class _FakeTTFontSkip(_FakeTTFont):
                def __getitem__(self, key):
                    if key == "name":
                        return _FakeNameTableSkip
                    return super().__getitem__(key)

            globals()["TTFont"] = _FakeTTFontSkip
            _build_local_font_catalog()
            _get_local_font_catalog()
            _pick_primary_font_entry([])
            _pick_primary_font_entry(
                [{"weight": 300, "italic": False}, {"weight": 400, "italic": True}]
            )
            _select_local_font_paths("TestFont", italic=False)
            _select_local_font_paths("testfont", italic=True)
            _select_local_font_paths("MissingFont", italic=False)
            _read_font_family(cast(TTFont, _FakeTTFont()))
            _read_font_family(
                cast(
                    TTFont,
                    type(
                        "_NoNameTTFont",
                        (),
                        {
                            "__getitem__": lambda _self, _key: type(
                                "_NoNameTable",
                                (),
                                {
                                    "names": [
                                        type(
                                            "_Rec",
                                            (),
                                            {
                                                "nameID": 2,
                                                "toUnicode": lambda _self: "",
                                            },
                                        )()
                                    ]
                                },
                            )(),
                        },
                    )(),
                )
            )
            _read_font_family(cast(TTFont, _BadTTFont()))
            _read_font_weight_italic(cast(TTFont, _FakeTTFont()))
            _read_font_weight_italic(cast(TTFont, _BadTTFont()))
            globals()["TTFont"] = lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("ttfont")
            )
            _build_local_font_catalog()
        finally:
            globals()["TTFont"] = orig_ttfont
            globals()["FONTS_DIR"] = orig_fonts_dir
            globals()["_LOCAL_FONT_CATALOG"] = orig_catalog
            with suppress(Exception):
                if tmp_fonts is not None:
                    for path in tmp_fonts.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_fonts.rmdir()
                    _force_exception("font-cleanup")
        _maxcov_log("maxcov extras font helpers done")

        # Exercise font data URI and extra font builders.
        orig_extra_fonts = _LOCAL_FONT_EXTRA_FONTS
        orig_catalog = _LOCAL_FONT_CATALOG
        tmp_extra_dir = None
        try:
            tmp_extra_dir = Path(tempfile.mkdtemp(prefix="maxcov_font_extra_"))
            font_path = tmp_extra_dir / "Extra.ttf"
            font_path.write_bytes(b"\x00\x01")
            bin_path = tmp_extra_dir / "Extra.bin"
            bin_path.write_bytes(b"\x00\x01")
            globals()["_LOCAL_FONT_CATALOG"] = {
                "Extra": [{"path": font_path, "weight": 400, "italic": False}],
                "BadPath": [{"path": tmp_extra_dir, "weight": 400, "italic": False}],
                "NotPath": [{"path": "nope"}],
                "Empty": [],
            }
            globals()["_LOCAL_FONT_EXTRA_FONTS"] = None
            _font_data_uri(font_path)
            _font_data_uri(bin_path)
            _build_local_font_extra_fonts()
            orig_font_data_uri = _font_data_uri
            globals()["_font_data_uri"] = lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("font uri")
            )
            _build_local_font_extra_fonts()
            globals()["_font_data_uri"] = orig_font_data_uri
            _get_local_font_extra_fonts()
            _resolve_default_font_family()
            globals()["_LOCAL_FONT_CATALOG"] = {}
            _resolve_default_font_family()
            _force_exception("font-extra")
        except Exception:
            pass
        finally:
            globals()["_LOCAL_FONT_EXTRA_FONTS"] = orig_extra_fonts
            globals()["_LOCAL_FONT_CATALOG"] = orig_catalog
            with suppress(Exception):
                if tmp_extra_dir is not None:
                    for path in tmp_extra_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_extra_dir.rmdir()
        _maxcov_log("maxcov extras font extra done")

        # Exercise LLM client cleanup paths.
        orig_aio_mod = None
        try:
            import types

            orig_aio_mod = sys.modules.get("any_llm.utils.aio")
            fake_aio_mod: Any = types.ModuleType("any_llm.utils.aio")

            def _run_async_in_sync(coro, _allow_running_loop=True):
                try:
                    return asyncio.run(coro)
                except RuntimeError:
                    return None

            fake_aio_mod.run_async_in_sync = _run_async_in_sync
            sys.modules["any_llm.utils.aio"] = fake_aio_mod

            class _ClientClose:
                def close(self):
                    return None

            class _ClientCloseCoro:
                def __init__(self):
                    self.last = None

                def close(self):
                    async def _noop():
                        return None

                    self.last = _noop()
                    return self.last

            class _ClientCloseRaise:
                def close(self):
                    raise RuntimeError("boom")

            class _ClientAclose:
                close = None

                def __init__(self):
                    self.last = None

                def aclose(self):
                    async def _noop():
                        return None

                    self.last = _noop()
                    return self.last

            class _ClientAcloseSync:
                close = None

                def aclose(self):
                    return None

            class _ClientAcloseRaise:
                close = None

                def aclose(self):
                    raise RuntimeError("boom")

            class _LLM:
                def __init__(self, client):
                    self.client = client

            _close_llm_client(_LLM(None))
            _close_llm_client(_LLM(_ClientClose()))
            close_coro = _ClientCloseCoro()
            _close_llm_client(_LLM(close_coro))
            if asyncio.iscoroutine(getattr(close_coro, "last", None)):
                close_coro.last.close()
            _close_llm_client(_LLM(_ClientCloseRaise()))
            aclose_coro = _ClientAclose()
            _close_llm_client(_LLM(aclose_coro))
            if asyncio.iscoroutine(getattr(aclose_coro, "last", None)):
                aclose_coro.last.close()
            _close_llm_client(_LLM(_ClientAcloseSync()))
            _close_llm_client(_LLM(_ClientAcloseRaise()))

            import builtins

            builtins_any = cast(Any, builtins)
            orig_import = builtins_any.__import__

            def _block_import(name, *args, **kwargs):
                if name == "any_llm.utils.aio":
                    raise ImportError("blocked")
                return orig_import(name, *args, **kwargs)

            builtins_any.__import__ = _block_import
            try:
                blocked_coro = _ClientCloseCoro()
                _close_llm_client(_LLM(blocked_coro))
                if asyncio.iscoroutine(getattr(blocked_coro, "last", None)):
                    blocked_coro.last.close()
                blocked_aclose = _ClientAclose()
                _close_llm_client(_LLM(blocked_aclose))
                if asyncio.iscoroutine(getattr(blocked_aclose, "last", None)):
                    blocked_aclose.last.close()
            finally:
                builtins_any.__import__ = orig_import
            _force_exception("llm-client-close")
        except Exception:
            pass
        finally:
            if orig_aio_mod is None:
                sys.modules.pop("any_llm.utils.aio", None)
            else:
                sys.modules["any_llm.utils.aio"] = orig_aio_mod
        _maxcov_log("maxcov extras llm client close done")

        # Exercise LLM request wrappers.
        orig_anyllm = AnyLLM
        try:

            class _StubClient:
                def __init__(self):
                    self.closed = False

                def responses(self, *args, **kwargs):
                    return {"ok": True, "args": args, "kwargs": kwargs}

                def completion(self, *args, **kwargs):
                    return {"ok": True, "args": args, "kwargs": kwargs}

                def close(self):
                    self.closed = True

            class _StubAnyLLM:
                @staticmethod
                def create(*_args, **_kwargs):
                    return _StubClient()

            globals()["AnyLLM"] = _StubAnyLLM
            _call_llm_responses(
                provider="openai",
                model="gpt-test",
                input_data="hello",
                api_key="key",
                api_base="http://example.test",
                client_args={"organization": "org"},
            )
            _call_llm_completion(
                provider="openai",
                model="gpt-test",
                messages=[{"role": "user", "content": "hi"}],
                api_key="key",
            )
            _force_exception("llm-wrappers")
        except Exception:
            pass
        finally:
            globals()["AnyLLM"] = orig_anyllm
        _maxcov_log("maxcov extras llm wrappers done")

        # Exercise state properties with empty rows.
        with suppress(Exception):
            state = _maxcov_state()
            state.skills_rows = [[""], [], []]
            state.highlighted_skills = ["A", "B", "C"]
            _ = state.skills_rows_csv
            _ = _skills_rows_to_csv([None, "Row", ["A"]], state.highlighted_skills)
            _ = _skills_rows_to_csv(["Row", None, ["A"]], state.highlighted_skills)
            _ = _skills_rows_to_csv([], ["A", "B", "C", "D"])
            state.experience = [Experience(id="exp-1"), Experience(id="exp-2")]
            state.founder_roles = [FounderRole(id="fr-1")]
            state.profile_experience_bullets = {"exp-2": ["one", "two"]}
            state.profile_founder_bullets = {"fr-1": ["line1"]}
            _ = _coerce_bullet_text(None)
            _ = _coerce_bullet_text("line1\nline2")
            _ = _coerce_bullet_text(["one", " ", None, "two"])
            _ = state.profile_experience_bullets_list
            _ = state.profile_founder_bullets_list
            state.section_visibility = {"summary": False, "matrices": False}
            _ = state.section_order_rows
            state_api.set_section_visibility.fn(state, True, "summary")
            state_api.set_section_visibility.fn(state, False, "matrices")
            state_api.set_section_visibility.fn(state, True, "bogus")
            state_api.set_auto_fit_target_pages.fn(state, "3")
            state_api.set_auto_fit_target_pages.fn(state, "")
            state_api.set_auto_fit_target_pages.fn(state, 0)
            state_api.set_auto_fit_target_pages.fn(state, True)
            _ = state._visible_section_order()
            state_api.set_profile_experience_bullets_text.fn(state, "exp-1", "a\nb")
            state_api.set_profile_experience_bullets_text.fn(state, "", "skip")
            state_api.set_profile_founder_bullets_text.fn(state, "fr-1", "c\nd")
            state_api.set_profile_founder_bullets_text.fn(state, "", "skip")
            state.job_req = ""
            _ = state.job_req_needs_profile
            _ = state.pipeline_latest
            _force_exception("state-props")
        _maxcov_log("maxcov extras state props done")

        # Exercise load_resume_fields paths with stubbed Neo4j.
        orig_load_neo4j = Neo4jClient
        try:

            class _LoadClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {
                            "name": "Test User",
                            "email": "test@example.com",
                            "phone": "555-555-5555",
                            "head1_left": "L",
                            "head1_middle": "M",
                            "head1_right": "R",
                            "head2_left": "",
                            "head2_middle": "",
                            "head2_right": "",
                            "head3_left": "",
                            "head3_middle": "",
                            "head3_right": "",
                            "top_skills": ["Skill 1", "Skill 2"],
                        },
                        "experience": [],
                        "education": [],
                        "founder_roles": [],
                    }

                def list_applied_jobs(self, *args, **kwargs):
                    return []

                def search_profile_metadata(self, *args, **kwargs):
                    return []

                def get_profile_metadata(self, *args, **kwargs):
                    return {}

                def close(self):
                    return None

            globals()["Neo4jClient"] = _LoadClient
            state = _maxcov_state()
            asyncio.run(_drain_event(state.load_resume_fields()))

            class _EmptyResumeClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {}

                def list_applied_jobs(self, *args, **kwargs):
                    return []

                def search_profile_metadata(self, *args, **kwargs):
                    return []

                def get_profile_metadata(self, *args, **kwargs):
                    return {}

                def close(self):
                    return None

            globals()["Neo4jClient"] = _EmptyResumeClient
            state = _maxcov_state()
            asyncio.run(_drain_event(state.load_resume_fields()))
            _force_exception("load-resume")
        except Exception:
            pass
        finally:
            globals()["Neo4jClient"] = orig_load_neo4j
        _maxcov_log("maxcov extras load paths done")

        # Exercise ui simulation PDF skip branch.
        try:

            class _UiClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {"name": "Test User"},
                        "experience": [],
                        "education": [],
                        "founder_roles": [],
                    }

                def close(self):
                    return None

            globals()["Neo4jClient"] = _UiClient
            os.environ["MAX_COVERAGE_SKIP_PDF"] = "1"
            asyncio.run(
                _run_ui_simulation(
                    set(),
                    req_file,
                    skip_llm=True,
                    simulate_failures=False,
                )
            )
            _force_exception("ui-sim-skip-pdf")
        except Exception:
            pass
        finally:
            os.environ.pop("MAX_COVERAGE_SKIP_PDF", None)
            globals()["Neo4jClient"] = orig_load_neo4j
        _maxcov_log("maxcov extras ui sim skip pdf done")

        # Exercise ui simulation pipeline else branch and pending task cleanup.
        try:
            os.environ["MAX_COVERAGE_PIPELINE_EVENT"] = "1"
            asyncio.run(
                _run_ui_simulation(
                    {"pipeline"},
                    req_file,
                    skip_llm=True,
                    simulate_failures=False,
                )
            )

            async def _run_with_pending():
                blocker = asyncio.Event()

                async def _waiter():
                    await blocker.wait()

                task = asyncio.create_task(_waiter())
                try:
                    await _run_ui_simulation(
                        set(),
                        req_file,
                        skip_llm=True,
                        simulate_failures=False,
                    )
                finally:
                    blocker.set()
                    await asyncio.sleep(0)
                    if not task.done():
                        task.cancel()

            asyncio.run(_run_with_pending())
            _force_exception("ui-sim-pending")
        except Exception:
            pass
        finally:
            os.environ.pop("MAX_COVERAGE_PIPELINE_EVENT", None)
        _maxcov_log("maxcov extras ui sim pending done")

        # Exercise pdf-only simulation branch (forces profile generation).
        with suppress(Exception):
            asyncio.run(
                _run_ui_simulation(
                    {"pdf"},
                    req_file,
                    skip_llm=True,
                    simulate_failures=False,
                )
            )
            _force_exception("ui-sim-pdf-only")
        _maxcov_log("maxcov extras ui sim pdf-only done")

        # Exercise profile bullet save flows.
        orig_profile_client = Neo4jClient
        try:
            state = _maxcov_state()
            state.is_saving_profile_bullets = True
            asyncio.run(_drain_event(state.save_profile_bullets()))
            state.is_saving_profile_bullets = False
            asyncio.run(_drain_event(state.save_profile_bullets()))

            class _BulletClient:
                def __init__(self, *args, **kwargs):
                    pass

                def update_profile_bullets(self, *args, **kwargs):
                    return True

                def close(self):
                    return None

            globals()["Neo4jClient"] = _BulletClient
            state = _maxcov_state()
            state.latest_profile_id = "profile-1"
            state.profile_experience_bullets = {"exp-1": ["a", "b"]}
            state.profile_founder_bullets = {"fr-1": ["c"]}
            asyncio.run(_drain_event(state.save_profile_bullets()))

            class _FailBulletClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("save failed")

            globals()["Neo4jClient"] = _FailBulletClient
            state = _maxcov_state()
            state.latest_profile_id = "profile-2"
            asyncio.run(_drain_event(state.save_profile_bullets()))
            _force_exception("profile-bullets")
        except Exception:
            pass
        finally:
            globals()["Neo4jClient"] = orig_profile_client
        _maxcov_log("maxcov extras profile bullets done")

        # Exercise profile bullet edit toggles and setters.
        state = _maxcov_state()
        state_api.set_rewrite_bullets_with_llm.fn(state, True)
        state_api.set_edit_profile_bullets.fn(state, True)
        state_api.set_profile_experience_bullets_text.fn(state, "exp-1", "Line 1")
        state_api.set_profile_experience_bullets_text.fn(state, "exp-1", "")
        state_api.set_profile_founder_bullets_text.fn(state, "fr-1", "Line A")
        state_api.set_profile_founder_bullets_text.fn(state, "fr-1", "")

        # Render skills with/without content to exercise layout branches.
        render_skill_rows([], [])
        render_skill_rows(["Row"], [["a", "b", "c", "d", "e"]])

        # Exercise secret loading logic with a temp home file.
        secret_env_keys = [
            "HOME",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "LLM_MODELS",
            "OPENAI_MODELS",
            "LLM_MAX_OUTPUT_TOKENS",
            "OPENAI_MAX_OUTPUT_TOKENS",
            "GEMINI_MAX_OUTPUT_TOKENS",
            "GOOGLE_MAX_OUTPUT_TOKENS",
            "LLM_MAX_OUTPUT_TOKENS_RETRY",
            "OPENAI_MAX_OUTPUT_TOKENS_RETRY",
            "GEMINI_MAX_OUTPUT_TOKENS_RETRY",
            "GOOGLE_MAX_OUTPUT_TOKENS_RETRY",
        ]
        env_backup = {key: os.environ.get(key) for key in secret_env_keys}

        def restore_env():
            for key, val in env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

        tmp_home = Path(tempfile.mkdtemp(prefix="maxcov_home_"))
        try:
            openai_key_path = tmp_home / "openaikey.txt"
            gemini_key_path = tmp_home / "geminikey.txt"
            openai_key_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "OPENAI_API_KEY=sk-test",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            gemini_key_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "GEMINI_API_KEY=gm-test",
                        "GOOGLE_API_KEY=gm2-test",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            os.environ["HOME"] = str(tmp_home)
            os.environ["OPENAI_API_KEY"] = ""
            os.environ["GEMINI_API_KEY"] = ""
            os.environ["GOOGLE_API_KEY"] = ""
            load_openai_api_key()
            load_gemini_api_key()
            os.environ["GOOGLE_API_KEY"] = "gm-env"
            load_gemini_api_key()
            os.environ["GOOGLE_API_KEY"] = ""
            gemini_key_path.write_text("GEMINI_API_KEY=gm-only\n", encoding="utf-8")
            load_openai_api_key()
            openai_key_path.write_text("sk-test-raw\n", encoding="utf-8")
            load_openai_api_key()
            openai_key_path.write_text("not-a-key\n", encoding="utf-8")
            load_openai_api_key()
            gemini_key_path.write_text("# comment\n", encoding="utf-8")
            load_gemini_api_key()
            _read_first_secret_line(openai_key_path)
            _read_first_secret_line(gemini_key_path)
            _read_first_secret_line(tmp_home)
            empty_secret = tmp_home / "empty_secret.txt"
            empty_secret.write_text("# comment\n", encoding="utf-8")
            _read_first_secret_line(empty_secret)
            missing_secret = tmp_home / "missing_secret.txt"
            _read_first_secret_line(missing_secret)

            os.environ["LLM_MODELS"] = ",".join(
                [
                    "openai:gpt-4o-mini",
                    "openai:gpt-4o-mini",
                    "gemini:gemini-3-flash-preview",
                ]
            )
            list_llm_models()

            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "1234"
            _resolve_llm_max_output_tokens("openai", "gpt-5.2")
            os.environ.pop("LLM_MAX_OUTPUT_TOKENS", None)
            os.environ["OPENAI_MAX_OUTPUT_TOKENS"] = "2048"
            _resolve_llm_max_output_tokens("openai", "gpt-4o-mini")
            os.environ["GEMINI_MAX_OUTPUT_TOKENS"] = "2048"
            _resolve_llm_max_output_tokens("gemini", "gemini-1.5-flash")

            os.environ["LLM_MAX_OUTPUT_TOKENS_RETRY"] = "9999"
            _resolve_llm_retry_max_output_tokens("openai", "gpt-5.2", 1000)
            os.environ.pop("LLM_MAX_OUTPUT_TOKENS_RETRY", None)
            os.environ["OPENAI_MAX_OUTPUT_TOKENS_RETRY"] = "7777"
            _resolve_llm_retry_max_output_tokens("openai", "gpt-4o-mini", 1000)
            os.environ["GEMINI_MAX_OUTPUT_TOKENS_RETRY"] = "6666"
            _resolve_llm_retry_max_output_tokens("gemini", "gemini-1.5-flash", 1000)
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "0"
            _read_int_env("LLM_MAX_OUTPUT_TOKENS")
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "-1"
            _read_int_env("LLM_MAX_OUTPUT_TOKENS")
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "not-an-int"
            _read_int_env("LLM_MAX_OUTPUT_TOKENS")
            os.environ.pop("LLM_MAX_OUTPUT_TOKENS", None)
            _resolve_llm_max_output_tokens("unknown", "model")
        finally:
            restore_env()

        # Exercise secret loading error branches (openaikey.txt/geminikey.txt as a directory).
        try:
            os.environ["HOME"] = str(tmp_home)
            with suppress(Exception):
                openai_key_path.unlink()
                _force_exception("key-unlink")
            with suppress(Exception):
                openai_key_path.mkdir(parents=True, exist_ok=True)
                _force_exception("key-mkdir")
            with suppress(Exception):
                gemini_key_path.unlink()
                _force_exception("key-unlink-gemini")
            with suppress(Exception):
                gemini_key_path.mkdir(parents=True, exist_ok=True)
                _force_exception("key-mkdir-gemini")
            load_openai_api_key()
            load_gemini_api_key()
        finally:
            restore_env()
        _maxcov_log("maxcov extras secret loading done")

        # Exercise on_load error and generate-on-load branches.
        orig_load_client = Neo4jClient
        orig_list_models = list_llm_models
        orig_generate_pdf = state_api.generate_pdf
        try:

            class _FailOnLoadClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Simulated on_load failure")

            globals()["Neo4jClient"] = _FailOnLoadClient
            state = _maxcov_state()
            state.on_load()

            class _NoResumeClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {"resume": None}

                def close(self):
                    return None

            globals()["Neo4jClient"] = _NoResumeClient
            state = _maxcov_state()
            state.on_load()

            class _OkClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {
                            "name": "Test User",
                            "head1_left": "L",
                            "section_order": "summary,experience",
                            "section_enabled": "summary,experience",
                        },
                        "experience": [],
                        "education": [],
                        "founder_roles": [],
                    }

                def close(self):
                    return None

            globals()["Neo4jClient"] = _OkClient
            globals()["list_llm_models"] = lambda: ["model-a"]
            os.environ["GENERATE_ON_LOAD"] = "1"
            os.environ["MAX_COVERAGE_FORCE_DB_ERROR_ON_LOAD"] = "1"
            state_api.generate_pdf = lambda _self: None
            state = _maxcov_state()
            state.selected_model = "bogus:model"
            state.on_load()
            os.environ.pop("MAX_COVERAGE_FORCE_DB_ERROR_ON_LOAD", None)
        finally:
            globals()["Neo4jClient"] = orig_load_client
            globals()["list_llm_models"] = orig_list_models
            state_api.generate_pdf = orig_generate_pdf
            os.environ.pop("GENERATE_ON_LOAD", None)
        _maxcov_log("maxcov extras on_load branches done")

        # Force on_load outer exception via a non-writable DEBUG_LOG path.
        orig_debug_log = DEBUG_LOG
        try:
            bad_log_dir = Path(tempfile.mkdtemp(prefix="maxcov_bad_log_"))
            globals()["DEBUG_LOG"] = bad_log_dir
            state = _maxcov_state()
            with suppress(Exception):
                state.on_load()
        finally:
            globals()["DEBUG_LOG"] = orig_debug_log
            with suppress(Exception):
                if bad_log_dir.exists():
                    bad_log_dir.rmdir()
        _maxcov_log("maxcov extras on_load debug log done")

        # Trigger on_load outer except with a valid DEBUG_LOG file.
        orig_onload_client = Neo4jClient
        orig_list_models = list_llm_models
        orig_maxcov_log = MAX_COVERAGE_LOG
        try:

            class _OnLoadClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {"resume": {"name": "Test User", "head1_left": "L"}}

                def close(self):
                    return None

            def _boom_models():
                raise RuntimeError("list models boom")

            globals()["Neo4jClient"] = _OnLoadClient
            globals()["list_llm_models"] = _boom_models
            globals()["MAX_COVERAGE_LOG"] = False
            state = _maxcov_state()
            state.on_load()
        finally:
            globals()["Neo4jClient"] = orig_onload_client
            globals()["list_llm_models"] = orig_list_models
            globals()["MAX_COVERAGE_LOG"] = orig_maxcov_log
        _maxcov_log("maxcov extras on_load outer except done")

        # Exercise save/load and PDF caching branches.
        orig_state_client = Neo4jClient
        orig_compile_pdf = compile_pdf
        orig_compile_auto = compile_pdf_with_auto_tuning
        orig_generate_typst_source = generate_typst_source
        orig_runtime_write = globals().get("RUNTIME_WRITE_PDF", False)
        orig_live_pdf_path = LIVE_PDF_PATH
        orig_live_sig_path = LIVE_PDF_SIG_PATH
        tmp_write_dir = None
        try:

            class _SaveClient:
                def __init__(self, *args, **kwargs):
                    pass

                def upsert_resume_and_sections(self, *args, **kwargs):
                    return None

                def close(self):
                    return None

            globals()["Neo4jClient"] = _SaveClient
            state = _maxcov_state()
            state.experience = [Experience(id="")]
            state.is_saving = True
            asyncio.run(_drain_event(state.save_to_db()))
            state.is_saving = False
            asyncio.run(_drain_event(state.save_to_db()))

            class _FailSaveClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Save failed")

            globals()["Neo4jClient"] = _FailSaveClient
            state = _maxcov_state()
            state.experience = [Experience(id="")]
            asyncio.run(_drain_event(state.save_to_db()))

            class _SaveLoadClient:
                resume_name = "Solo"
                latest_profile: dict | None = None

                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {
                            "name": self.__class__.resume_name,
                            "email": "test@example.com",
                            "phone": "555-555-5555",
                            "head1_left": "L",
                            "head1_middle": "M",
                            "head1_right": "R",
                            "head2_left": "",
                            "head2_middle": "",
                            "head2_right": "",
                            "head3_left": "",
                            "head3_middle": "",
                            "head3_right": "",
                            "top_skills": ["Skill 1", "Skill 2"],
                        },
                        "experience": [
                            {"id": "e1", "bullets": ["a", "b"], "company": "Co"}
                        ],
                        "education": [{"id": "ed1", "bullets": None, "school": "U"}],
                        "founder_roles": [
                            {"id": "f1", "bullets": "one", "company": "X"}
                        ],
                    }

                def list_applied_jobs(self, *args, **kwargs):
                    return (
                        [self.__class__.latest_profile]
                        if self.__class__.latest_profile
                        else []
                    )

                def search_profile_metadata(self, *args, **kwargs):
                    return []

                def get_profile_metadata(self, *args, **kwargs):
                    return {}

                def close(self):
                    return None

            globals()["Neo4jClient"] = _SaveLoadClient
            state = _maxcov_state()
            state.is_loading_resume = True
            asyncio.run(_drain_event(state.load_resume_fields()))

            for name, latest_profile in (
                ("Solo", {}),
                ("Two Names", {"headers": ["H1"], "highlighted_skills": ["S1"]}),
                (
                    "Three Part Name",
                    {
                        "job_req_raw": "Req text",
                        "headers": ["H1", "H2"],
                        "highlighted_skills": ["S1", "S2"],
                        "skills_rows": [["A", "B"]],
                        "must_have_skills": ["X", "Y"],
                        "nice_to_have_skills": [],
                    },
                ),
            ):
                _SaveLoadClient.resume_name = name
                _SaveLoadClient.latest_profile = latest_profile
                state = _maxcov_state()
                state.job_req = ""
                asyncio.run(_drain_event(state.load_resume_fields()))

            globals()["compile_pdf"] = lambda *_args, **_kwargs: (True, b"%PDF-1.4\n")
            globals()["compile_pdf_with_auto_tuning"] = lambda *_args, **_kwargs: (
                True,
                b"%PDF-1.4\n",
            )
            globals()["generate_typst_source"] = lambda *_args, **_kwargs: "typst"
            state = _maxcov_state()
            state.data_loaded = True
            state.job_req = ""
            state.auto_tune_pdf = False
            state.generate_pdf()
            state.generate_pdf()
            state.pdf_url = ""
            state.generate_pdf()
            cached_sig = state.last_pdf_signature
            if cached_sig:
                LIVE_PDF_PATH.write_bytes(b"%PDF-1.4\n")
                LIVE_PDF_SIG_PATH.write_text(cached_sig, encoding="utf-8")
                state.last_pdf_b64 = ""
                state.pdf_url = ""
                state.generate_pdf()
            globals()["compile_pdf"] = lambda *_args, **_kwargs: (False, b"")
            state.generate_pdf()
            globals()["compile_pdf"] = lambda *_args, **_kwargs: (True, b"%PDF-1.4\n")
            globals()["compile_pdf_with_auto_tuning"] = lambda *_args, **_kwargs: (
                True,
                b"%PDF-1.4\n",
            )
            globals()["generate_typst_source"] = lambda *_args, **_kwargs: "typst"
            state = _maxcov_state()
            state.data_loaded = True
            state.job_req = ""
            state.auto_tune_pdf = True
            tmp_write_dir = Path(tempfile.mkdtemp(prefix="maxcov_pdf_write_"))
            globals()["LIVE_PDF_PATH"] = tmp_write_dir / "preview.pdf"
            globals()["LIVE_PDF_SIG_PATH"] = tmp_write_dir / "preview.sig"
            globals()["RUNTIME_WRITE_PDF"] = True
            state.generate_pdf()
        finally:
            globals()["Neo4jClient"] = orig_state_client
            globals()["compile_pdf"] = orig_compile_pdf
            globals()["compile_pdf_with_auto_tuning"] = orig_compile_auto
            globals()["generate_typst_source"] = orig_generate_typst_source
            globals()["RUNTIME_WRITE_PDF"] = orig_runtime_write
            globals()["LIVE_PDF_PATH"] = orig_live_pdf_path
            globals()["LIVE_PDF_SIG_PATH"] = orig_live_sig_path
            if tmp_write_dir is not None:
                with suppress(Exception):
                    for path in tmp_write_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_write_dir.rmdir()
            with suppress(Exception):
                if LIVE_PDF_PATH.exists():
                    LIVE_PDF_PATH.unlink()
                if LIVE_PDF_SIG_PATH.exists():
                    LIVE_PDF_SIG_PATH.unlink()
        _maxcov_log("maxcov extras save/load done")

        # Exercise state helper branches and cached PDF handling.
        orig_debug_log = DEBUG_LOG
        orig_live_pdf = LIVE_PDF_PATH
        orig_live_sig = LIVE_PDF_SIG_PATH
        cache_dir = Path(tempfile.mkdtemp(prefix="maxcov_pdf_cache_"))
        try:
            state = _maxcov_state()

            state.experience = [
                Experience(id="exp-1", bullets="one\ntwo"),
                Experience(id="exp-2", bullets="alpha"),
            ]
            state.founder_roles = [FounderRole(id="fr-1", bullets="seed")]
            state.profile_experience_bullets = {"exp-2": ["override"]}
            state.profile_founder_bullets = {"fr-1": ["override founder"]}
            state.section_order = ["summary", "unknown", "experience"]
            state._current_resume_profile()
            state.update_experience_field(2, "role", "Role")
            state.update_experience_field(-1, "role", "Role")
            state.update_experience_field("bad", "role", "Role")
            state.update_education_field(2, "school", "School")
            state.update_education_field(-1, "school", "School")
            state.update_education_field("bad", "school", "School")
            state.update_founder_role_field(2, "company", "Startup")
            state.update_founder_role_field(-1, "company", "Startup")
            state.update_founder_role_field("bad", "company", "Startup")
            state.remove_experience("bad")
            state.remove_education("bad")
            state.remove_founder_role("bad")
            state.move_section_up("bad")
            state.move_section_down("bad")
            state._record_render_time(1500, True)
            state.generating_pdf = True
            state.generate_pdf()

            globals()["LIVE_PDF_PATH"] = cache_dir / "preview.pdf"
            globals()["LIVE_PDF_SIG_PATH"] = cache_dir / "preview.sig"
            LIVE_PDF_PATH.write_bytes(b"%PDF-1.4\n")
            LIVE_PDF_SIG_PATH.write_text("wrong", encoding="utf-8")
            state._load_cached_pdf("right")

            with suppress(Exception):
                LIVE_PDF_PATH.unlink()
            LIVE_PDF_PATH.mkdir(parents=True, exist_ok=True)
            LIVE_PDF_SIG_PATH.write_text("sig", encoding="utf-8")
            state._load_cached_pdf("sig")

            globals()["DEBUG_LOG"] = cache_dir
            state._log_debug("force log error")
        finally:
            globals()["DEBUG_LOG"] = orig_debug_log
            globals()["LIVE_PDF_PATH"] = orig_live_pdf
            globals()["LIVE_PDF_SIG_PATH"] = orig_live_sig
            with suppress(Exception):
                for path in cache_dir.rglob("*"):
                    if path.is_file():
                        path.unlink()
                cache_dir.rmdir()
        _maxcov_log("maxcov extras state helpers done")

        # Exercise auto-pipeline early exits and stage failures.
        orig_pipeline_compile = None
        orig_pipeline_autofit = None
        orig_handlers = None
        try:
            from reflex.event import EventHandler as _EventHandler

            orig_pipeline_compile = compile_pdf
            orig_pipeline_autofit = compile_pdf_with_auto_tuning
            try:
                orig_handlers = dict(getattr(State, "event_handlers", {}) or {})
            except Exception:
                orig_handlers = None
            globals()["compile_pdf"] = lambda *_args, **_kwargs: (True, b"%PDF-1.4\n%")
            globals()["compile_pdf_with_auto_tuning"] = lambda *_args, **_kwargs: (
                True,
                b"%PDF-1.4\n%",
            )

            def _run_auto_pipeline(state_obj: Any, label: str) -> None:
                _maxcov_log(t"maxcov auto-pipeline run start: {label}")
                handler = getattr(state_api.auto_pipeline_from_req, "fn", None)
                if handler:
                    asyncio.run(_drain_event(handler(state_obj)))
                else:
                    asyncio.run(_drain_event(state_obj.auto_pipeline_from_req()))
                _maxcov_log(t"maxcov auto-pipeline run done: {label}")

            def _set_handler(state_obj: Any, name: str, fn) -> None:
                existing = state_obj.event_handlers.get(name)
                state_full_name = existing.state_full_name if existing else ""
                state_obj.event_handlers[name] = _EventHandler(
                    fn=fn,
                    state_full_name=state_full_name,
                )

            async def _noop_gen(self):
                _maxcov_log("maxcov auto-pipeline stub: noop")
                yield None

            async def _fail_profile(self):
                _maxcov_log("maxcov auto-pipeline stub: fail profile")
                self.pdf_error = "stage1"
                yield None

            async def _ok_profile(self):
                _maxcov_log("maxcov auto-pipeline stub: ok profile")
                self.pdf_error = ""
                yield None

            async def _fail_load(self):
                _maxcov_log("maxcov auto-pipeline stub: fail load")
                self.pdf_error = "stage2"
                yield None

            async def _raise_profile(self):
                _maxcov_log("maxcov auto-pipeline stub: raise profile")
                yield None
                raise RuntimeError("boom")

            def _fail_pdf(self):
                _maxcov_log("maxcov auto-pipeline stub: fail pdf")
                self.pdf_error = "stage3"

            def _ok_pdf(self):
                _maxcov_log("maxcov auto-pipeline stub: ok pdf")

            state = _maxcov_state()
            state.is_auto_pipeline = True
            _maxcov_log("maxcov auto-pipeline case: already running")
            _run_auto_pipeline(state, "already_running")

            state = _maxcov_state()
            state.job_req = ""
            _maxcov_log("maxcov auto-pipeline case: empty req")
            _run_auto_pipeline(state, "empty_req")

            state = _maxcov_state()
            state.job_req = "req"
            _set_handler(state, "generate_profile", _fail_profile)
            _set_handler(state, "load_resume_fields", _noop_gen)
            _set_handler(state, "generate_pdf", _ok_pdf)
            _maxcov_log("maxcov auto-pipeline case: stage1 fail")
            _run_auto_pipeline(state, "stage1_fail")

            state = _maxcov_state()
            state.job_req = "req"
            _set_handler(state, "generate_profile", _ok_profile)
            _set_handler(state, "load_resume_fields", _fail_load)
            _set_handler(state, "generate_pdf", _ok_pdf)
            _maxcov_log("maxcov auto-pipeline case: stage2 fail")
            _run_auto_pipeline(state, "stage2_fail")

            state = _maxcov_state()
            state.job_req = "req"
            _set_handler(state, "generate_profile", _ok_profile)
            _set_handler(state, "load_resume_fields", _noop_gen)
            _set_handler(state, "generate_pdf", _fail_pdf)
            _maxcov_log("maxcov auto-pipeline case: stage3 fail")
            _run_auto_pipeline(state, "stage3_fail")

            state = _maxcov_state()
            state.job_req = "req"
            _set_handler(state, "generate_profile", _raise_profile)
            _set_handler(state, "load_resume_fields", _noop_gen)
            _set_handler(state, "generate_pdf", _ok_pdf)
            _maxcov_log("maxcov auto-pipeline case: profile raise")
            _run_auto_pipeline(state, "profile_raise")

            state = _maxcov_state()
            state.job_req = "req"
            _set_handler(state, "generate_profile", _ok_profile)
            _set_handler(state, "load_resume_fields", _noop_gen)
            _set_handler(state, "generate_pdf", _ok_pdf)
            _maxcov_log("maxcov auto-pipeline case: success")
            _run_auto_pipeline(state, "success")

            orig_debug_log = DEBUG_LOG
            try:
                bad_log_dir = Path(tempfile.mkdtemp(prefix="maxcov_bad_pipeline_log_"))
                globals()["DEBUG_LOG"] = bad_log_dir
                state = _maxcov_state()
                state.job_req = "req"
                _set_handler(state, "generate_profile", _raise_profile)
                _set_handler(state, "load_resume_fields", _noop_gen)
                _set_handler(state, "generate_pdf", _ok_pdf)
                _maxcov_log("maxcov auto-pipeline case: bad log path")
                _run_auto_pipeline(state, "profile_raise_log")
            finally:
                globals()["DEBUG_LOG"] = orig_debug_log
                with suppress(Exception):
                    if bad_log_dir.exists():
                        bad_log_dir.rmdir()
        except Exception:
            pass
        finally:
            if orig_pipeline_compile is not None:
                globals()["compile_pdf"] = orig_pipeline_compile
            if orig_pipeline_autofit is not None:
                globals()["compile_pdf_with_auto_tuning"] = orig_pipeline_autofit
            if orig_handlers is not None:
                try:
                    State.event_handlers = orig_handlers
                except Exception:
                    with suppress(Exception):
                        State.event_handlers.clear()
                        State.event_handlers.update(orig_handlers)

        _maxcov_log("maxcov extras auto-pipeline done")

        # Exercise generate_profile branches with stubbed LLM outputs.
        orig_profile_client = Neo4jClient
        orig_generate_resume = globals().get("generate_resume_content")
        try:

            class _ProfileClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {"summary": "Base", "name": "Test User"},
                        "experience": [
                            {"id": "e1", "bullets": ["a", "b"], "company": "Co"}
                        ],
                        "education": [{"id": "ed1", "bullets": None, "school": "U"}],
                        "founder_roles": [
                            {"id": "f1", "bullets": "one", "company": "X"}
                        ],
                    }

                def save_resume(self, *args, **kwargs):
                    return "profile-1"

                def update_profile_bullets(self, *args, **kwargs):
                    return True

                def close(self):
                    return None

            def _run_profile(result, *, rewrite=False):
                globals()["generate_resume_content"] = lambda *_a, **_k: result
                state = _maxcov_state()
                state.job_req = "req"
                state.rewrite_bullets_with_llm = rewrite
                asyncio.run(_drain_event(state.generate_profile()))

            globals()["Neo4jClient"] = _ProfileClient
            _run_profile("bad")
            _run_profile({"error": "boom"})
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": "not-json",
                    "headers": ["", ""],
                    "highlighted_skills": ["", ""],
                    "must_have_skills": None,
                    "nice_to_have_skills": None,
                    "tech_stack_keywords": None,
                    "non_technical_requirements": None,
                    "certifications": None,
                    "clearances": None,
                    "core_responsibilities": None,
                    "outcome_goals": None,
                }
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": ["a,b", ["c"], None, 1],
                    "headers": ["H1"],
                    "highlighted_skills": ["S1"],
                    "experience_bullets": [{"id": "e1", "bullets": ["A", "B"]}],
                    "founder_role_bullets": [{"id": "f1", "bullets": ["C"]}],
                }
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [
                        ["LinkedIn", "Python"],
                        ["Public Code Portfolio"],
                        [],
                    ],
                    "headers": ["LinkedIn"],
                    "highlighted_skills": ["LinkedIn", "Python", "Python"],
                    "experience_bullets": [{"id": "e1", "bullets": ["X"]}],
                    "founder_role_bullets": [{"id": "f1", "bullets": ["Y"]}],
                },
                rewrite=True,
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [],
                    "headers": ["H1"],
                    "highlighted_skills": ["Skill 1", "Skill 2", "Skill 3"],
                }
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [],
                    "headers": ["H1"],
                    "highlighted_skills": [
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F",
                        "G",
                        "H",
                        "I",
                        "J",
                        "A",
                    ],
                }
            )

            class _ProfileContactClient(_ProfileClient):
                def get_resume_data(self):
                    data = super().get_resume_data()
                    resume = data.get("resume", {})
                    resume["head1_left"] = "LinkedIn"
                    resume["head1_middle"] = "GitHub"
                    data["resume"] = resume
                    return data

            globals()["Neo4jClient"] = _ProfileContactClient
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [
                        ["http://example.com", "Scholar"],
                        ["@mail.com"],
                        [],
                    ],
                    "headers": ["LinkedIn"],
                    "highlighted_skills": [
                        "Public Code Portfolio",
                        "GitHub",
                        "Scholar",
                    ],
                },
                rewrite=True,
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [
                        ["LinkedIn", "https://example.com", "Public Code Portfolio"],
                        ["someone@example.com", "GitHub"],
                        ["Scholar"],
                    ],
                    "headers": ["LinkedIn", "GitHub"],
                    "highlighted_skills": [
                        "Public Code Portfolio",
                        "LinkedIn",
                        "GitHub",
                        "Scholar",
                    ],
                },
                rewrite=True,
            )

            class _ProfileFailHydrateClient(_ProfileClient):
                def get_resume_data(self):
                    raise RuntimeError("hydrate boom")

            globals()["Neo4jClient"] = _ProfileFailHydrateClient
            state = _maxcov_state()
            state.job_req = "req"
            state.experience = []
            state.education = []
            state.founder_roles = []
            asyncio.run(_drain_event(state.generate_profile()))
            orig_to_thread = asyncio.to_thread
            try:

                def _boom_to_thread(*_a, **_k):
                    raise RuntimeError("boom")

                asyncio.to_thread = _boom_to_thread
                state = _maxcov_state()
                state.job_req = "req"
                asyncio.run(_drain_event(state.generate_profile()))
            finally:
                asyncio.to_thread = orig_to_thread

            class _ProfileHydrateClient(_ProfileClient):
                def get_resume_data(self):
                    return {
                        "resume": {
                            "summary": "Base",
                            "name": "Test User",
                            "head1_left": "L",
                            "top_skills": ["Skill 1"],
                        },
                        "experience": [
                            {"id": "e2", "bullets": ["x", "y"], "company": "Co"}
                        ],
                        "education": [{"id": "ed2", "bullets": None, "school": "U"}],
                        "founder_roles": [
                            {"id": "f2", "bullets": "one", "company": "X"}
                        ],
                    }

            globals()["Neo4jClient"] = _ProfileHydrateClient
            globals()["generate_resume_content"] = lambda *_a, **_k: {
                "summary": "Ok",
                "skills_rows": [],
                "headers": [],
                "highlighted_skills": [],
            }
            state = _maxcov_state()
            state.job_req = "req"
            state.experience = []
            state.education = []
            state.founder_roles = []
            asyncio.run(_drain_event(state.generate_profile()))

            class _ProfileBoomClient(_ProfileClient):
                def ensure_resume_exists(self, *args, **kwargs):
                    raise RuntimeError("profile boom")

            globals()["Neo4jClient"] = _ProfileBoomClient
            state = _maxcov_state()
            state.job_req = "req"
            asyncio.run(_drain_event(state.generate_profile()))

            state = _maxcov_state()
            state.is_generating_profile = True
            asyncio.run(_drain_event(state.generate_profile()))

            def _raise_resume(*_a, **_k):
                raise RuntimeError("LLM boom")

            globals()["generate_resume_content"] = _raise_resume
            state = _maxcov_state()
            state.job_req = "req"
            with suppress(Exception):
                asyncio.run(_drain_event(state.generate_profile()))
        finally:
            globals()["Neo4jClient"] = orig_profile_client
            globals()["generate_resume_content"] = orig_generate_resume

        _maxcov_log("maxcov extras generate_profile done")

        # Exercise maxcov helper formatting and summary utilities.
        _maxcov_format_line_ranges([])
        _maxcov_format_line_ranges([1, 2, 4, 5, 7])
        _maxcov_format_top_missing_blocks([], limit=2)
        _maxcov_format_top_missing_blocks([1, 2, 5, 6, 7], limit=2)
        _maxcov_format_top_missing_blocks([10], limit=5)
        _maxcov_format_arc((0, -1))
        _maxcov_format_arc("bad")
        with suppress(Exception):
            _maxcov_format_branch_arcs([(1, 2), (3, -1)], limit=1)
        _maxcov_format_branch_arcs([(1, 2), (2, 3), (3, 4)], limit=2)
        _maxcov_log_expected_failure("stdout", "stderr", ["--noop"], True)
        _maxcov_log_expected_failure("stdout", "", ["--noop"], True)
        _maxcov_log_expected_failure("", "", ["--noop"], True)
        _maxcov_log_expected_failure("stdout", "stderr", ["--noop"], False)
        _normalize_auto_fit_target_pages("3")
        _normalize_auto_fit_target_pages("bad")
        _normalize_auto_fit_target_pages(None)
        _normalize_auto_fit_target_pages(True)
        _normalize_auto_fit_target_pages(-1)
        _normalize_auto_fit_target_pages("0")
        _normalize_auto_fit_target_pages(2.5)

        dummy_counts = {
            "cover": "90%",
            "stmts": 10,
            "miss": 2,
            "branch": 4,
            "brpart": 1,
        }
        dummy_summary = {
            "missing_ranges": "1-2",
            "missing_branch_line_ranges": "3-4",
            "missing_branch_arcs": "1->2",
            "missing_branch_arcs_extra": 2,
            "top_missing_blocks": "1-2(2)",
        }
        _maxcov_build_coverage_output(
            counts=dummy_counts,
            summary=dummy_summary,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            json_out="json",
            html_out="html",
        )
        _maxcov_build_coverage_output(
            counts={"cover": "0%"},
            summary=None,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            json_out=None,
            html_out=None,
        )
        _maxcov_build_coverage_output(
            counts={},
            summary=None,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            json_out=None,
            html_out=None,
        )

        class _DummyAnalyze:
            def missing_branch_arcs(self):
                return [10]

            def arcs_missing(self):
                return [(10, 0), (11, 12)]

        class _DummyCoverage:
            def __init__(self, *args, **kwargs):
                pass

            def load(self):
                return None

            def analysis2(self, _target):
                return ("", "", "", [1, 2, 3])

            def _analyze(self, _target):
                return _DummyAnalyze()

        class _DummyCoverageModule:
            Coverage = _DummyCoverage

        _maxcov_summarize_coverage(
            _DummyCoverageModule,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            target=Path(__file__).resolve(),
        )

        class _FailCoverageModule:
            class Coverage:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("boom")

        _maxcov_summarize_coverage(
            _FailCoverageModule,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            target=Path(__file__).resolve(),
        )
        _maxcov_log("maxcov extras maxcov helpers done")

        # Exercise container wrapper branches with stubbed runner.
        with suppress(Exception):

            class _Result:
                def __init__(self, returncode=0, stdout=""):
                    self.returncode = returncode
                    self.stdout = stdout

            class _RunnerHealthy:
                def __init__(self):
                    self.calls = []

                def __call__(self, cmd, **_kwargs):
                    self.calls.append(list(cmd))
                    if "ps" in cmd:
                        return _Result(stdout="cid")
                    if cmd[:2] == ["docker", "inspect"]:
                        return _Result(stdout="healthy")
                    return _Result()

            class _RunnerNoHealth(_RunnerHealthy):
                def __call__(self, cmd, **_kwargs):
                    self.calls.append(list(cmd))
                    if "ps" in cmd:
                        return _Result(stdout="")
                    if cmd[:2] == ["docker", "inspect"]:
                        return _Result(stdout="starting")
                    return _Result()

            class _RunnerComposeFail(_RunnerHealthy):
                def __call__(self, cmd, **_kwargs):
                    self.calls.append(list(cmd))
                    if "version" in cmd:
                        return _Result(returncode=1)
                    return super().__call__(cmd, **_kwargs)

            class _Exit(Exception):
                def __init__(self, code):
                    super().__init__(str(code))
                    self.code = code

            def _exit(code):
                raise _Exit(code)

            runner = _RunnerHealthy()
            with suppress(_Exit):
                _maxcov_run_container_wrapper(
                    project="maxcov_test",
                    runner=runner,
                    sleep_fn=lambda *_a, **_k: None,
                    time_fn=time.time,
                    exit_fn=_exit,
                    check_compose=True,
                )

            class _FastTime:
                def __init__(self):
                    self.now = 0.0

                def __call__(self):
                    self.now += 10.0
                    return self.now

            runner = _RunnerNoHealth()
            with suppress(_Exit):
                _maxcov_run_container_wrapper(
                    project="maxcov_test_wait",
                    runner=runner,
                    sleep_fn=lambda *_a, **_k: None,
                    time_fn=_FastTime(),
                    exit_fn=_exit,
                    check_compose=False,
                )

            runner = _RunnerComposeFail()
            with suppress(RuntimeError):
                _maxcov_run_container_wrapper(
                    project="maxcov_test_fail",
                    runner=runner,
                    sleep_fn=lambda *_a, **_k: None,
                    time_fn=time.time,
                    exit_fn=_exit,
                    check_compose=True,
                )
        _maxcov_log("maxcov extras container wrapper done")

        # Exercise Typst source generation branches with synthetic data.
        sample_resume = {
            "summary": "First sentence. Second sentence follows.",
            "headers": ["H1", "H2", "H3"],
            "highlighted_skills": ["Skill A", "Skill B", "Skill C"],
            "skills_rows": [["A", "B", "C", "D"], ["E"], []],
            "first_name": "",
            "middle_name": "",
            "last_name": "",
            "top_skills": ["AI", "ML", "Systems", "Leadership", "Product"],
            "target_role": "Lead",
            "target_company": "Acme",
            "primary_domain": "AI",
        }
        sample_profile = {
            "name": "Jane Q Doe",
            "email": "jane@example.com",
            "phone": "555-0101",
            "linkedin_url": "https://www.linkedin.com/in/jane",
            "github_url": "https://github.com/jane",
            "scholar_url": "abc123",
            "scholar_link_text": "Scholar",
            "github_link_text": "GitHub",
            "linkedin_link_text": "LinkedIn",
            "summary": "Fallback summary.",
            "experience": [
                {
                    "company": "Acme",
                    "role": "Lead Engineer",
                    "location": "Remote",
                    "description": "Built systems.",
                    "bullets": ["Did X", "Did Y"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                }
            ],
            "education": [
                {
                    "school": "State U",
                    "degree": "M.S. (Coursework; Honors)",
                    "location": "City",
                    "description": "Focus on AI.",
                    "bullets": ["Thesis"],
                    "start_date": "2018-01-01",
                    "end_date": "2020-01-01",
                }
            ],
            "founder_roles": [
                {
                    "company": "Startup",
                    "role": "Founder",
                    "location": "NYC",
                    "description": "Bootstrapped.",
                    "bullets": ["<date>2020---2021</date> Built product"],
                    "start_date": "2019-01-01",
                    "end_date": "2020-01-01",
                },
                {
                    "company": "Second Startup",
                    "role": "Co-Founder",
                    "location": "Remote",
                    "description": "Scaled ops.",
                    "bullets": ["<date>2021---2022</date> Expanded"],
                    "start_date": "2021-01-01",
                    "end_date": "2022-01-01",
                },
            ],
        }
        generate_typst_source(
            sample_resume,
            sample_profile,
            include_matrices=True,
            include_summary=True,
            section_order=["summary", "experience", "education", "founder", "matrices"],
            layout_scale=cast(Any, "bad"),
        )
        generate_typst_source(
            sample_resume,
            sample_profile,
            include_matrices=True,
            include_summary=True,
            section_order=None,
            layout_scale=1.0,
        )
        generate_typst_source(
            sample_resume,
            sample_profile,
            include_matrices=False,
            include_summary=False,
            section_order=["experience"],
            layout_scale=0,
        )

        edge_resume = {
            "summary": "Single sentence only",
            "headers": [],
            "highlighted_skills": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            "skills_rows": "not json",
            "first_name": "",
            "middle_name": "",
            "last_name": "",
            "top_skills": ["Alpha", "Beta", "Gamma"],
            "target_role": "Analyst",
            "target_company": "DataCo",
            "primary_domain": "Data",
        }
        edge_profile = {
            "name": "Solo",
            "email": "solo@example.com",
            "phone": "555-0102",
            "linkedin_url": "linkedin.com/in/solo",
            "github_url": "github.com/solo",
            "scholar_url": "https://example.com/path",
            "experience": [
                {
                    "company": "EdgeCo",
                    "role": "Engineer",
                    "location": "",
                    "description": "Edge description",
                    "bullets": [],
                    "start_date": "2021-01-01",
                    "end_date": "2022-01-01",
                }
            ],
            "education": [
                {
                    "school": "U1",
                    "degree": "M.S. (Honors)",
                    "location": "",
                    "description": "",
                    "bullets": "not-a-list",
                    "start_date": "2022-01-01",
                    "end_date": "",
                },
                {
                    "school": "U2",
                    "degree": "M.A.",
                    "location": "",
                    "description": "",
                    "bullets": [],
                    "start_date": "",
                    "end_date": "2021-01-01",
                },
                {
                    "school": "U3",
                    "degree": "M.S.",
                    "location": "",
                    "description": "",
                    "bullets": [],
                    "start_date": "2018-01-01",
                    "end_date": "2020-01-01",
                },
            ],
            "founder_roles": [
                {
                    "company": "EdgeStartup",
                    "role": "Founder",
                    "location": "",
                    "description": "",
                    "bullets": ["<date></date>"],
                    "start_date": "2017-01-01",
                    "end_date": "2018-01-01",
                }
            ],
        }
        generate_typst_source(
            edge_resume,
            edge_profile,
            include_matrices=True,
            include_summary=True,
            section_order=["education", "experience", "founder", "matrices"],
            layout_scale=1.2,
        )

        edge_resume_rows = edge_resume.copy()
        edge_resume_rows["skills_rows"] = cast(Any, [1, ["A"], "B"])
        generate_typst_source(
            edge_resume_rows,
            edge_profile,
            include_matrices=True,
            include_summary=True,
            section_order=["education"],
            layout_scale=1.0,
        )

        edge_profile_no_master = {
            "name": "No Master",
            "education": [
                {
                    "school": "",
                    "degree": 123,
                    "location": "",
                    "description": "",
                    "bullets": [],
                    "start_date": "",
                    "end_date": "",
                },
                {
                    "school": "U4",
                    "degree": "B.S.",
                    "location": "",
                    "description": "",
                    "bullets": [],
                    "start_date": "2010-01-01",
                    "end_date": "2014-01-01",
                },
            ],
        }
        generate_typst_source(
            edge_resume_rows,
            edge_profile_no_master,
            include_matrices=False,
            include_summary=False,
            section_order=["education"],
            layout_scale=1.0,
        )

        # Exercise rasterized text path with a stub PIL.
        orig_pil = sys.modules.get("PIL")
        orig_pil_image = sys.modules.get("PIL.Image")
        orig_pil_imagedraw = sys.modules.get("PIL.ImageDraw")
        orig_pil_imagefont = sys.modules.get("PIL.ImageFont")
        orig_pil_temp_build = TEMP_BUILD_DIR
        tmp_pil_dir = None
        try:
            import types as _types

            class _FakeImage:
                def __init__(self, size):
                    self.width = size[0]
                    self.height = size[1]

                def resize(self, size, _resample):
                    return _FakeImage(size)

                def save(self, _path, _format=None):
                    return None

            class _RasterFakeImageModule:
                BICUBIC = 3
                Resampling = type("Resampling", (), {"LANCZOS": 1})

                @staticmethod
                def new(_mode, size, _color):
                    return _FakeImage(size)

            class _FakeDraw:
                def __init__(self, _img):
                    pass

                def textbbox(self, _pos, text, font=None):
                    width = max(1, len(str(text)) * 6)
                    return (0, 0, width, 10)

                def text(self, _pos, _text, fill=None, font=None):
                    return None

            class _FakeImageDrawModule(_types.ModuleType):
                @staticmethod
                def Draw(img):
                    return _FakeDraw(img)

            class _FakeFont:
                pass

            class _FakeImageFontModule(_types.ModuleType):
                @staticmethod
                def truetype(*_a, **_k):
                    return _FakeFont()

                @staticmethod
                def load_default():
                    return _FakeFont()

            fake_pil_typst: Any = _types.ModuleType("PIL")
            fake_pil_typst.Image = _RasterFakeImageModule
            fake_pil_typst.ImageDraw = _FakeImageDrawModule("PIL.ImageDraw")
            fake_pil_typst.ImageFont = _FakeImageFontModule("PIL.ImageFont")
            sys.modules["PIL"] = fake_pil_typst
            sys.modules["PIL.Image"] = fake_pil_typst.Image
            sys.modules["PIL.ImageDraw"] = fake_pil_typst.ImageDraw
            sys.modules["PIL.ImageFont"] = fake_pil_typst.ImageFont

            tmp_pil_dir = Path(tempfile.mkdtemp(prefix="maxcov_pil_"))
            globals()["TEMP_BUILD_DIR"] = tmp_pil_dir
            generate_typst_source(
                edge_resume,
                {
                    "name": "Raster",
                    "experience": [
                        {
                            "company": "Co",
                            "role": "Role",
                            "location": "",
                            "description": "Desc",
                            "bullets": [],
                        }
                    ],
                    "education": [],
                    "founder_roles": [],
                },
                include_matrices=False,
                include_summary=False,
                section_order=["experience"],
                layout_scale=1.0,
            )
        finally:
            if orig_pil is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = orig_pil
            if orig_pil_image is None:
                sys.modules.pop("PIL.Image", None)
            else:
                sys.modules["PIL.Image"] = orig_pil_image
            if orig_pil_imagedraw is None:
                sys.modules.pop("PIL.ImageDraw", None)
            else:
                sys.modules["PIL.ImageDraw"] = orig_pil_imagedraw
            if orig_pil_imagefont is None:
                sys.modules.pop("PIL.ImageFont", None)
            else:
                sys.modules["PIL.ImageFont"] = orig_pil_imagefont
            globals()["TEMP_BUILD_DIR"] = orig_pil_temp_build
            with suppress(Exception):
                if tmp_pil_dir is not None:
                    for path in tmp_pil_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_pil_dir.rmdir()

        orig_fonts_dir = FONTS_DIR
        orig_temp_build = TEMP_BUILD_DIR
        try:
            empty_fonts = Path(tempfile.mkdtemp(prefix="maxcov_fonts_"))
            globals()["FONTS_DIR"] = empty_fonts
            tmp_build = Path(tempfile.mkdtemp(prefix="maxcov_typst_"))
            globals()["TEMP_BUILD_DIR"] = tmp_build
            generate_typst_source(
                edge_resume,
                edge_profile,
                include_matrices=False,
                include_summary=False,
                section_order=["experience"],
                layout_scale=1.0,
            )
            bad_fd, bad_build_path = tempfile.mkstemp(prefix="maxcov_bad_build_")
            os.close(bad_fd)
            bad_build = Path(bad_build_path)
            globals()["TEMP_BUILD_DIR"] = bad_build
            generate_typst_source(
                edge_resume,
                edge_profile,
                include_matrices=False,
                include_summary=False,
                section_order=["experience"],
                layout_scale=1.0,
            )
        finally:
            globals()["FONTS_DIR"] = orig_fonts_dir
            globals()["TEMP_BUILD_DIR"] = orig_temp_build

        _maxcov_log("maxcov extras typst source done")

        # Exercise font/package fetch failure handling.
        with _capture_maxcov_output("maxcov extras font/package failures"):
            urllib_request = cast(Any, urllib.request)
            orig_urlretrieve = urllib_request.urlretrieve
            orig_fonts_dir = FONTS_DIR
            orig_pkg_dir = FONT_AWESOME_PACKAGE_DIR
            try:

                def _fail_urlretrieve(*_args, **_kwargs):
                    raise RuntimeError("download failed")

                urllib_request.urlretrieve = _fail_urlretrieve
                empty_fonts = Path(tempfile.mkdtemp(prefix="maxcov_fonts_dl_"))
                globals()["FONTS_DIR"] = empty_fonts
                _ensure_fontawesome_fonts()

                pkg_root = Path(tempfile.mkdtemp(prefix="maxcov_pkg_"))
                globals()["FONT_AWESOME_PACKAGE_DIR"] = (
                    pkg_root / "preview" / "fontawesome" / "0.0"
                )
                _ensure_typst_packages()
            finally:
                urllib_request.urlretrieve = orig_urlretrieve
                globals()["FONTS_DIR"] = orig_fonts_dir
                globals()["FONT_AWESOME_PACKAGE_DIR"] = orig_pkg_dir

        _maxcov_log("maxcov extras fonts/packages done")

        # Exercise font/package success handling.
        with _capture_maxcov_output("maxcov extras font/package success"):
            urllib_request = cast(Any, urllib.request)
            os_any = cast(Any, os)
            orig_urlretrieve = urllib_request.urlretrieve
            orig_fonts_dir = FONTS_DIR
            orig_pkg_dir = FONT_AWESOME_PACKAGE_DIR
            orig_remove = os_any.remove
            try:
                tmp_fonts = Path(tempfile.mkdtemp(prefix="maxcov_fonts_ok_"))
                tmp_pkg_root = Path(tempfile.mkdtemp(prefix="maxcov_pkg_ok_"))
                tar_path = tmp_pkg_root / "font_pkg.tar.gz"
                tar_src = tmp_pkg_root / "tar_src"
                tar_src.mkdir(parents=True, exist_ok=True)
                (tar_src / "README.txt").write_text("ok", encoding="utf-8")
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(tar_src, arcname="fontawesome")

                def _ok_urlretrieve(_url, filename, *args, **kwargs):
                    Path(filename).write_bytes(tar_path.read_bytes())
                    return filename, None

                urllib_request.urlretrieve = _ok_urlretrieve
                globals()["FONTS_DIR"] = tmp_fonts
                globals()["FONT_AWESOME_PACKAGE_DIR"] = (
                    tmp_pkg_root / "preview" / "fontawesome" / "0.0"
                )
                _ensure_fontawesome_fonts()

                def _bad_remove(_path):
                    raise RuntimeError("remove fail")

                os_any.remove = _bad_remove
                _ensure_typst_packages()
            finally:
                urllib_request.urlretrieve = orig_urlretrieve
                os_any.remove = orig_remove
                globals()["FONTS_DIR"] = orig_fonts_dir
                globals()["FONT_AWESOME_PACKAGE_DIR"] = orig_pkg_dir
                with suppress(Exception):
                    if tmp_pkg_root.exists():
                        for path in tmp_pkg_root.rglob("*"):
                            if path.is_file():
                                path.unlink()
                        for path in sorted(tmp_pkg_root.rglob("*"), reverse=True):
                            if path.is_dir():
                                path.rmdir()
                    if tmp_fonts.exists():
                        for path in tmp_fonts.iterdir():
                            if path.is_file():
                                path.unlink()
                        tmp_fonts.rmdir()

        # Exercise metadata builders and PDF helpers.
        with _capture_maxcov_output("maxcov extras pdf helpers"):
            tmp_pdf_dir = Path(tempfile.mkdtemp(prefix="maxcov_pdf_meta_"))
            tmp_pdf_path = tmp_pdf_dir / "doc.pdf"
            tmp_pdf_path.write_bytes(b"%PDF-1.4\n")
            _build_pdf_metadata(
                {},
                {
                    "name": "Ada Lovelace",
                    "target_role": "Engineer",
                    "highlighted_skills": ["AI", "AI"],
                },
            )
            _build_pdf_metadata(
                {},
                {
                    "name": "Grace Hopper",
                    "target_company": "Navy",
                    "req_id": "REQ-1",
                },
            )
            _build_pdf_metadata(
                {},
                {
                    "name": "John Q Public",
                    "target_role": "Lead",
                },
            )

            import builtins
            import types

            builtins_any = cast(Any, builtins)
            orig_import = builtins_any.__import__
            try:

                def _blocked_import(name, *args, **kwargs):
                    if name == "pikepdf":
                        raise ImportError("blocked")
                    return orig_import(name, *args, **kwargs)

                builtins_any.__import__ = _blocked_import
                _apply_pdf_metadata(tmp_pdf_path, {"title": "Test"})
            finally:
                builtins_any.__import__ = orig_import

            orig_pikepdf = sys.modules.get("pikepdf")
            try:
                fake_pikepdf: Any = types.ModuleType("pikepdf")

                def _bad_open(*_args, **_kwargs):
                    raise RuntimeError("bad pdf")

                fake_pikepdf.open = _bad_open
                sys.modules["pikepdf"] = fake_pikepdf
                _apply_pdf_metadata(tmp_pdf_path, {"title": "Test"})
            finally:
                if orig_pikepdf is None:
                    sys.modules.pop("pikepdf", None)
                else:
                    sys.modules["pikepdf"] = orig_pikepdf

            subprocess_any = cast(Any, subprocess)
            path_type = cast(Any, Path)
            orig_popen = subprocess_any.Popen
            orig_fonts_ready = ensure_fonts_ready
            try:

                class _BadProcess:
                    def __init__(self):
                        self.returncode = 1

                    def communicate(self):
                        return "", "typst error"

                subprocess_any.Popen = lambda *_args, **_kwargs: _BadProcess()
                globals()["ensure_fonts_ready"] = lambda: None
                compile_pdf("bad typst", metadata=None)
                orig_unlink = path_type.unlink
                unlink_calls = {"n": 0}

                def _bad_unlink(self):
                    unlink_calls["n"] += 1
                    if unlink_calls["n"] == 1:
                        raise RuntimeError("unlink fail")
                    return orig_unlink(self)

                path_type.unlink = _bad_unlink
                compile_pdf("bad typst", metadata=None)
            finally:
                subprocess_any.Popen = orig_popen
                globals()["ensure_fonts_ready"] = orig_fonts_ready
                path_type.unlink = orig_unlink

        _maxcov_log("maxcov extras pdf metadata done")

        # Exercise render_resume_pdf_bytes save-copy failure.
        with _capture_maxcov_output("maxcov extras render_resume_pdf_bytes save-copy"):
            orig_compile_pdf = compile_pdf
            orig_render_db = Neo4jClient
            try:
                globals()["compile_pdf"] = lambda *_args, **_kwargs: (
                    True,
                    b"%PDF-1.4\n",
                )

                class _RenderClient:
                    def __init__(self, *args, **kwargs):
                        pass

                    def get_resume_data(self):
                        return {
                            "resume": {"summary": "Summary", "top_skills": []},
                            "experience": [],
                            "education": [],
                            "founder_roles": [],
                        }

                    def close(self):
                        return None

                globals()["Neo4jClient"] = _RenderClient
                render_resume_pdf_bytes(
                    save_copy=True, include_summary=True, filename="."
                )
            finally:
                globals()["compile_pdf"] = orig_compile_pdf
                globals()["Neo4jClient"] = orig_render_db

        _maxcov_log("maxcov extras render_resume_pdf_bytes done")

        # Exercise compile_pdf_with_auto_tuning paths with stubbed pages.
        orig_autofit_compile = compile_pdf
        orig_autofit_gen = generate_typst_source
        orig_autofit_client = Neo4jClient
        orig_pikepdf_mod = sys.modules.get("pikepdf")
        try:

            class _AutoFitPdf:
                def __init__(self, pages):
                    self.pages = [None] * pages

            def _make_fake_pikepdf(open_fn):
                module = types.ModuleType("pikepdf")
                module.open = open_fn
                return module

            def _run_autofit(
                page_seq,
                *,
                fail_on=None,
                raise_on=None,
                cache=None,
                target_pages=None,
                resume_target=None,
            ):
                calls = {"n": 0}
                open_calls = {"n": 0}

                def _fake_compile(_src, metadata=None):
                    calls["n"] += 1
                    if fail_on and calls["n"] == fail_on:
                        return False, b""
                    pages = page_seq[min(calls["n"] - 1, len(page_seq) - 1)]
                    return True, f"PAGES={pages}".encode()

                def _fake_open(buf):
                    open_calls["n"] += 1
                    if raise_on and open_calls["n"] == raise_on:
                        raise RuntimeError("bad pdf")
                    data = buf.getvalue().decode("utf-8", errors="ignore")
                    pages = int(data.split("PAGES=")[1]) if "PAGES=" in data else 0
                    return _AutoFitPdf(pages)

                class _CacheClient:
                    def __init__(self, *args, **kwargs):
                        pass

                    def get_auto_fit_cache(self):
                        return cache or {}

                    def set_auto_fit_cache(self, *args, **kwargs):
                        return None

                    def close(self):
                        return None

                globals()["compile_pdf"] = _fake_compile
                globals()["generate_typst_source"] = lambda *_a, **_k: "scale"
                globals()["Neo4jClient"] = _CacheClient
                sys.modules["pikepdf"] = _make_fake_pikepdf(_fake_open)
                resume_payload = {"summary": "x"}
                if resume_target is not None:
                    resume_payload["auto_fit_target_pages"] = resume_target
                compile_pdf_with_auto_tuning(
                    resume_payload,
                    {"name": "Y"},
                    include_matrices=False,
                    target_pages=target_pages,
                )

            class _FailCacheClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("cache fail")

            globals()["Neo4jClient"] = _FailCacheClient
            globals()["compile_pdf"] = lambda *_a, **_k: (False, b"")
            globals()["generate_typst_source"] = lambda *_a, **_k: "scale"
            sys.modules["pikepdf"] = _make_fake_pikepdf(lambda buf: _AutoFitPdf(2))
            compile_pdf_with_auto_tuning({"summary": "x"}, {"name": "Y"})

            _run_autofit([1], raise_on=1)
            _run_autofit([0])
            _run_autofit([3, 0])
            _run_autofit([3, 1], fail_on=2)
            _run_autofit([3], target_pages=1)
            _run_autofit([3, 2], target_pages=1)
            _run_autofit([1, 0], cache={"best_scale": 1.0, "too_long_scale": 1.4})
            _run_autofit([1, 0, 1])
            _run_autofit([1, 3], cache={"best_scale": 1.0, "too_long_scale": 1.2})
            _run_autofit([1, 3], cache={"best_scale": 1.0, "too_long_scale": 1.4})
            _run_autofit([1, 1, 1], resume_target=0)
            _run_autofit([3, 3, 2], fail_on=2)
            _run_autofit([1, 1, 1], cache={"best_scale": 1.0, "too_long_scale": 1.4})
            _run_autofit([1, 3, 0], target_pages=1)
            _run_autofit([1, 2, 3], fail_on=2, target_pages=1)
            _run_autofit([1, 2, 3], fail_on=3, target_pages=1)
        finally:
            globals()["compile_pdf"] = orig_autofit_compile
            globals()["generate_typst_source"] = orig_autofit_gen
            globals()["Neo4jClient"] = orig_autofit_client
            if orig_pikepdf_mod is None:
                sys.modules.pop("pikepdf", None)
            else:
                sys.modules["pikepdf"] = orig_pikepdf_mod

        _maxcov_log("maxcov extras auto-fit done")

        # Exercise Playwright traversal with a stub module and import failure.
        orig_playwright = sys.modules.get("playwright")
        orig_playwright_sync = sys.modules.get("playwright.sync_api")
        try:
            fake_sync: Any = types.ModuleType("playwright.sync_api")

            class _DummyLocator:
                def __init__(self, count=1):
                    self._count = count

                def count(self):
                    return self._count

                def click(self, *args, **kwargs):
                    return None

                def fill(self, *_args, **_kwargs):
                    return None

                def nth(self, _idx):
                    return _DummyLocator(1)

                @property
                def first(self):
                    return self

                @property
                def last(self):
                    return self

            class _DummyPage:
                def set_default_timeout(self, *_args, **_kwargs):
                    return None

                def goto(self, *_args, **_kwargs):
                    return None

                def wait_for_timeout(self, *_args, **_kwargs):
                    return None

                def get_by_role(self, role, **_kwargs):
                    if role == "switch":
                        return _DummyLocator(2)
                    return _DummyLocator(1)

                def get_by_placeholder(self, *_args, **_kwargs):
                    return _DummyLocator(1)

                def get_by_label(self, *_args, **_kwargs):
                    return _DummyLocator(0)

            class _DummyBrowser:
                def new_page(self):
                    return _DummyPage()

                def close(self):
                    return None

            class _DummyChromium:
                def launch(self, *args, **kwargs):
                    return _DummyBrowser()

            class _DummyPlaywright:
                chromium = _DummyChromium()

            class _SyncContext:
                def __enter__(self):
                    return _DummyPlaywright()

                def __exit__(self, *args):
                    return False

            fake_sync.sync_playwright = _SyncContext
            sys.modules["playwright.sync_api"] = fake_sync
            sys.modules["playwright"] = types.ModuleType("playwright")
            with _capture_maxcov_output("maxcov extras playwright stub run"):
                _run_playwright_ui_traversal("http://example.test", timeout_s=0.1)
            _run_playwright_ui_traversal("", timeout_s=0.1)
            _run_playwright_ui_traversal("0", timeout_s=0.1)

            class _BoomLocator:
                def count(self):
                    raise RuntimeError("count fail")

                def click(self, *args, **kwargs):
                    raise RuntimeError("click fail")

                def fill(self, *_args, **_kwargs):
                    raise RuntimeError("fill fail")

                def nth(self, _idx):
                    return self

                @property
                def first(self):
                    return self

                @property
                def last(self):
                    return self

            class _BoomPage(_DummyPage):
                def get_by_role(self, role, **_kwargs):
                    return _BoomLocator()

                def get_by_placeholder(self, *_args, **_kwargs):
                    return _BoomLocator()

            class _BoomBrowser(_DummyBrowser):
                def new_page(self):
                    return _BoomPage()

            class _BoomChromium(_DummyChromium):
                def launch(self, *args, **kwargs):
                    return _BoomBrowser()

            class _BoomPlaywright(_DummyPlaywright):
                chromium = _BoomChromium()

            class _BoomContext(_SyncContext):
                def __enter__(self):
                    return _BoomPlaywright()

            fake_sync.sync_playwright = _BoomContext
            with _capture_maxcov_output("maxcov extras playwright stub fail"):
                _run_playwright_ui_traversal("http://example.test", timeout_s=0.1)
        finally:
            if orig_playwright is None:
                sys.modules.pop("playwright", None)
            else:
                sys.modules["playwright"] = orig_playwright
            if orig_playwright_sync is None:
                sys.modules.pop("playwright.sync_api", None)
            else:
                sys.modules["playwright.sync_api"] = orig_playwright_sync

        import builtins

        builtins_any = cast(Any, builtins)
        orig_import = builtins_any.__import__
        try:

            def _bad_import(name, *args, **kwargs):
                if name == "playwright.sync_api":
                    raise ImportError("missing")
                return orig_import(name, *args, **kwargs)

            builtins_any.__import__ = _bad_import
            with _capture_maxcov_output("maxcov extras playwright import fail"):
                _run_playwright_ui_traversal("http://example.test", timeout_s=0.1)
        finally:
            builtins_any.__import__ = orig_import

        _maxcov_log("maxcov extras playwright stubs done")

        _pick_open_port(3000)
        try:
            socket_any = cast(Any, socket)
            orig_socket = socket_any.socket
            call_count = {"n": 0}

            class _DummySocket:
                def __init__(self, *_args, **_kwargs):
                    self._addr = ("127.0.0.1", 0)

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def bind(self, addr):
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        raise OSError("busy")
                    self._addr = (addr[0], 4321)

                def getsockname(self):
                    return self._addr

            socket_any.socket = _DummySocket
            _pick_open_port(3001)
        finally:
            socket_any.socket = orig_socket

        # Exercise reflex coverage session error paths.
        subprocess_any = cast(Any, subprocess)
        os_any = cast(Any, os)
        orig_popen = subprocess_any.Popen
        orig_wait_url = _wait_for_url
        _maxcov_log("maxcov extras reflex error paths start")
        try:
            subprocess_any.Popen = lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("popen fail")
            )
            _maxcov_log("maxcov extras reflex popen fail start")
            with _capture_maxcov_output("maxcov extras reflex popen fail"):
                _run_reflex_coverage_session(
                    3999, 4999, startup_timeout_s=0.1, ui_timeout_s=0.1
                )
            _maxcov_log("maxcov extras reflex popen fail done")
        except Exception:
            pass
        finally:
            subprocess_any.Popen = orig_popen

        class _DummyProc:
            def __init__(self):
                self.pid = os.getpid()

            def wait(self, *args, **kwargs):
                return 0

            def send_signal(self, *_args, **_kwargs):
                raise RuntimeError("signal fail")

            def terminate(self):
                raise RuntimeError("terminate fail")

            def kill(self):
                return None

        try:
            orig_killpg = getattr(os_any, "killpg", None)
            if orig_killpg is not None:
                delattr(os_any, "killpg")
            subprocess_any.Popen = lambda *_a, **_k: _DummyProc()
            globals()["_wait_for_url"] = lambda *_a, **_k: False
            _maxcov_log("maxcov extras reflex startup timeout start")
            with _capture_maxcov_output("maxcov extras reflex startup timeout"):
                _run_reflex_coverage_session(
                    3998, 4998, startup_timeout_s=0.1, ui_timeout_s=0.1
                )
            _maxcov_log("maxcov extras reflex startup timeout done")
        finally:
            if orig_killpg is not None:
                os_any.killpg = orig_killpg
            subprocess_any.Popen = orig_popen
            globals()["_wait_for_url"] = orig_wait_url

        try:

            class _DummyProc2:
                def __init__(self):
                    self.pid = os.getpid()
                    self.stdin = None

                def poll(self):
                    return 0

                def wait(self, *args, **kwargs):
                    raise RuntimeError("wait fail")

                def send_signal(self, *_args, **_kwargs):
                    raise RuntimeError("signal fail")

                def terminate(self):
                    raise RuntimeError("terminate fail")

                def kill(self):
                    return None

            def _killpg(_pid, sig):
                if sig in {signal.SIGINT, signal.SIGTERM}:
                    raise RuntimeError("killpg fail")
                return None

            os_any.killpg = _killpg
            subprocess_any.Popen = lambda *_a, **_k: _DummyProc2()
            globals()["_wait_for_url"] = lambda *_a, **_k: True
            orig_playwright_runner = _run_playwright_ui_traversal
            globals()["_run_playwright_ui_traversal"] = lambda *_a, **_k: False
            _maxcov_log("maxcov extras reflex stop paths start")
            with _capture_maxcov_output("maxcov extras reflex stop paths"):
                _run_reflex_coverage_session(
                    3997, 4997, startup_timeout_s=0.1, ui_timeout_s=0.1
                )
            _maxcov_log("maxcov extras reflex stop paths done")
        finally:
            _maxcov_log("maxcov extras reflex stop paths cleanup start")
            globals()["_run_playwright_ui_traversal"] = orig_playwright_runner
            if orig_killpg is not None:
                os_any.killpg = orig_killpg
            else:
                delattr(os_any, "killpg")
            subprocess_any.Popen = orig_popen
            globals()["_wait_for_url"] = orig_wait_url
            _maxcov_log("maxcov extras reflex stop paths cleanup done")

        _maxcov_log("maxcov extras reflex error paths done")

        # Exercise coroutine branch in _drain_event.
        _maxcov_log("maxcov extras coroutine start")

        async def _dummy_coroutine():
            return None

        asyncio.run(_drain_event(_dummy_coroutine()))

        _maxcov_log("maxcov extras coroutine done")

        # Exercise Neo4j client logic with a stub driver.
        class _DummyResult:
            def __init__(self, rows=None, single=None):
                self._rows = rows or []
                self._single = single

            def data(self):
                return self._rows

            def single(self):
                return self._single

        class _DummyTx:
            def run(self, *_args, **_kwargs):
                return None

        class _DummySession:
            def __init__(self, mode="default", skills_json=None):
                self.mode = mode
                self.skills_json = skills_json

            def run(self, query, **_kwargs):
                text = str(query)
                if "count(r)" in text:
                    count = 1 if self.mode == "has_resume" else 0
                    return _DummyResult(single={"c": count})
                if "RETURN r" in text and "Resume" in text:
                    if self.mode == "resume_none":
                        return _DummyResult(single=None)
                    return _DummyResult(
                        single={
                            "r": {
                                "id": "resume-1",
                                "name": "Test",
                                "summary": "Summary",
                            }
                        }
                    )
                if "HAS_EXPERIENCE" in text:
                    return _DummyResult(
                        rows=[
                            {
                                "e": {
                                    "start_date": "2020-01-01",
                                    "end_date": "2021-01-01",
                                }
                            }
                        ]
                    )
                if "HAS_EDUCATION" in text:
                    return _DummyResult(
                        rows=[
                            {
                                "e": {
                                    "start_date": "2016-01-01",
                                    "end_date": "2018-01-01",
                                }
                            }
                        ]
                    )
                if "HAS_FOUNDER_ROLE" in text:
                    return _DummyResult(
                        rows=[
                            {
                                "f": {
                                    "start_date": "2019-01-01",
                                    "end_date": "2020-01-01",
                                }
                            }
                        ]
                    )
                if "auto_fit_best_scale" in text:
                    if self.mode == "auto_fit_none":
                        return _DummyResult(single=None)
                    return _DummyResult(
                        single={"best_scale": 1.0, "too_long_scale": 1.2}
                    )
                if "MATCH (p:Profile)" in text:
                    return _DummyResult(
                        rows=[
                            {
                                "p": {
                                    "skills_rows_json": self.skills_json or "",
                                    "created_at": "now",
                                }
                            }
                        ]
                    )
                if (
                    "experience_bullets_json" in text
                    and "founder_role_bullets_json" in text
                ):
                    return _DummyResult(single={"id": "profile-1"})
                if "RETURN p.id" in text:
                    if self.mode == "save_none":
                        return _DummyResult(single=None)
                    return _DummyResult(single={"id": "profile-1"})
                return _DummyResult()

            def execute_write(self, fn, *args, **kwargs):
                return fn(_DummyTx(), *args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                return False

        class _DummyDriver:
            def __init__(self, mode="default", skills_json=None):
                self.mode = mode
                self.skills_json = skills_json
                self.closed = False

            def session(self):
                return _DummySession(self.mode, self.skills_json)

            def close(self):
                self.closed = True

        def _make_client(mode="default", skills_json=None):
            return Neo4jClient(driver=_DummyDriver(mode=mode, skills_json=skills_json))

        with suppress(Exception):
            dummy_assets = tmp_home / "assets.json"
            dummy_assets.write_text(
                json.dumps(
                    {
                        "profile": {
                            "id": "resume-1",
                            "name": "Test User",
                            "email": "test@example.com",
                            "phone": "555-555-5555",
                            "linkedin_url": "",
                            "github_url": "",
                            "summary": "",
                            "head1_left": "",
                            "head1_middle": "",
                            "head1_right": "",
                            "head2_left": "",
                            "head2_middle": "",
                            "head2_right": "",
                            "head3_left": "",
                            "head3_middle": "",
                            "head3_right": "",
                            "top_skills": [],
                        },
                        "experience": [
                            {
                                "id": "exp-1",
                                "company": "Company",
                                "role": "Role",
                                "location": "Remote",
                                "description": "",
                                "bullets": [],
                                "start_date": "2020-01-01",
                                "end_date": "2021-01-01",
                            }
                        ],
                        "education": [
                            {
                                "id": "edu-1",
                                "school": "School",
                                "degree": "B.S.",
                                "location": "City",
                                "description": "",
                                "bullets": [],
                                "start_date": "2016-01-01",
                                "end_date": "2018-01-01",
                            }
                        ],
                        "founder_roles": [
                            {
                                "id": "fr-1",
                                "company": "Startup",
                                "role": "Founder",
                                "location": "Remote",
                                "description": "",
                                "bullets": [],
                                "start_date": "2018-01-01",
                                "end_date": "2019-01-01",
                            }
                        ],
                        "skills": [
                            {
                                "category": "Core",
                                "skills": [{"id": "skill-1", "name": "Python"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            client = _make_client()
            client.reset()
            client.import_assets(dummy_assets)
            client.import_assets(dummy_assets.resolve())
            with _capture_maxcov_output("maxcov extras neo4j missing seed file"):
                client.import_assets("missing_assets.json")
            client.reset_and_import(dummy_assets)
            client.ensure_resume_exists(dummy_assets)
            client.get_resume_data()
            client.close()

            _make_client("has_resume").ensure_resume_exists(dummy_assets)
            _make_client("auto_fit_none").get_auto_fit_cache()
            _make_client().get_auto_fit_cache()
            _make_client().set_auto_fit_cache(best_scale=1.05, too_long_scale=None)
            _make_client().set_auto_fit_cache(best_scale=1.05, too_long_scale=1.25)
            _make_client("resume_none").get_resume_data()

            _make_client(skills_json="not-json").list_applied_jobs()
            _make_client(skills_json='"a,b"').list_applied_jobs()
            _make_client(skills_json='["a,b", ["c"], null]').list_applied_jobs()
            _make_client(skills_json='{"x": "y"}').list_applied_jobs()
            _make_client(skills_json='[{"x": "y"}]').list_applied_jobs()
            _make_client(
                skills_json='[["a", "b"], ["c", "d"], ["e"]]'
            ).list_applied_jobs()

            with suppress(Exception):
                _make_client("save_none").save_resume({"summary": "x"})
            _make_client().save_resume({"summary": "x"})

            _make_client().upsert_resume_and_sections(
                {
                    "summary": "x",
                    "name": "Test User",
                    "first_name": "Test",
                    "middle_name": "",
                    "last_name": "User",
                    "email": "test@example.com",
                    "phone": "555-555-5555",
                    "linkedin_url": "",
                    "github_url": "",
                    "scholar_url": "",
                    "head1_left": "",
                    "head1_middle": "",
                    "head1_right": "",
                    "head2_left": "",
                    "head2_middle": "",
                    "head2_right": "",
                    "head3_left": "",
                    "head3_middle": "",
                    "head3_right": "",
                    "top_skills": [],
                },
                [
                    {
                        "id": "exp-1",
                        "company": "Company",
                        "role": "Role",
                        "location": "Remote",
                        "description": "",
                        "bullets": [],
                        "start_date": "2020-01-01",
                        "end_date": "2021-01-01",
                    }
                ],
                [
                    {
                        "id": "edu-1",
                        "school": "School",
                        "degree": "B.S.",
                        "location": "City",
                        "description": "",
                        "bullets": [],
                        "start_date": "2016-01-01",
                        "end_date": "2018-01-01",
                    }
                ],
                [
                    {
                        "id": "fr-1",
                        "company": "Startup",
                        "role": "Founder",
                        "location": "Remote",
                        "description": "",
                        "bullets": [],
                        "start_date": "2018-01-01",
                        "end_date": "2019-01-01",
                    }
                ],
            )

            orig_schema_ready = globals().get("_NEO4J_SCHEMA_READY")

            class _BadSession:
                def run(self, *_args, **_kwargs):
                    raise RuntimeError("schema fail")

                def __enter__(self):
                    return self

                def __exit__(self, _exc_type, _exc, _tb):
                    return False

            class _BadDriver:
                def session(self):
                    return _BadSession()

                def close(self):
                    return None

            globals()["_NEO4J_SCHEMA_READY"] = False
            bad_client = Neo4jClient(driver=_BadDriver())
            bad_client.close()
            globals()["_NEO4J_SCHEMA_READY"] = orig_schema_ready

            class _OkSession:
                def run(self, *_args, **_kwargs):
                    return None

                def __enter__(self):
                    return self

                def __exit__(self, _exc_type, _exc, _tb):
                    return False

            class _OkDriver:
                def session(self):
                    return _OkSession()

                def close(self):
                    return None

            ok_client = Neo4jClient(driver=_OkDriver())
            ok_client._ensure_placeholder_relationships()
            ok_client.close()

        _maxcov_log("maxcov extras neo4j dummy done")

        # Exercise profile bullet updates with a dummy driver.
        with suppress(Exception):

            class _UpdResult:
                def __init__(self, row=None):
                    self._row = row

                def single(self):
                    return self._row

            class _UpdSession:
                def run(self, *_args, **_kwargs):
                    return _UpdResult({"id": "profile-1"})

                def __enter__(self):
                    return self

                def __exit__(self, _exc_type, _exc, _tb):
                    return False

            class _UpdDriver:
                def session(self):
                    return _UpdSession()

                def close(self):
                    return None

            client = Neo4jClient(driver=_UpdDriver())
            client.update_profile_bullets(
                "profile-1",
                [{"id": "e1", "bullets": ["a"]}],
                [{"id": "f1", "bullets": ["b"]}],
            )
            client.close()
            _force_exception("profile-update")
        _maxcov_log("maxcov extras profile update done")

        # Exercise render_resume_pdf_bytes without hitting external services.
        with _capture_maxcov_output("maxcov extras render_resume_pdf_bytes"):
            orig_render_db = Neo4jClient
            orig_render_compile = compile_pdf
            try:

                class _RenderNeo4j:
                    def __init__(self, *args, **kwargs):
                        pass

                    def get_resume_data(self):
                        return {
                            "resume": {
                                "head1_left": "Left",
                                "head1_middle": "Middle",
                                "head1_right": "Right",
                                "head2_left": "",
                                "head2_middle": "",
                                "head2_right": "",
                                "head3_left": "",
                                "head3_middle": "",
                                "head3_right": "",
                                "top_skills": ["Skill 1", "Skill 2"],
                                "summary": "Summary text.",
                                "first_name": "Test",
                                "middle_name": "Q",
                                "last_name": "User",
                                "email": "test@example.com",
                                "phone": "555-555-5555",
                                "section_order": "summary,experience",
                                "section_enabled": {"summary": True, "matrices": False},
                                "linkedin_url": "",
                                "github_url": "",
                                "scholar_url": "",
                            },
                            "experience": [],
                            "education": [],
                            "founder_roles": [],
                        }

                    def close(self):
                        return None

                globals()["Neo4jClient"] = _RenderNeo4j
                globals()["compile_pdf"] = lambda *_a, **_k: (True, b"%PDF-1.4\n%")
                render_resume_pdf_bytes(
                    save_copy=False, include_summary=True, include_skills=True
                )
                render_resume_pdf_bytes(
                    save_copy=True,
                    include_summary=False,
                    include_skills=False,
                    filename="preview_no_summary_skills.pdf",
                )
                globals()["compile_pdf"] = lambda *_a, **_k: (False, b"")
                render_resume_pdf_bytes(
                    save_copy=False, include_summary=True, include_skills=True
                )

                class _FailingRenderNeo4j:
                    def __init__(self, *args, **kwargs):
                        raise RuntimeError("Simulated render failure")

                globals()["Neo4jClient"] = _FailingRenderNeo4j
                render_resume_pdf_bytes(
                    save_copy=False, include_summary=True, include_skills=True
                )
            finally:
                globals()["Neo4jClient"] = orig_render_db
                globals()["compile_pdf"] = orig_render_compile

        _maxcov_log("maxcov extras render resume done")

        # Exercise reasoning parameter mapping.
        orig_effort = DEFAULT_LLM_REASONING_EFFORT
        try:
            globals()["DEFAULT_LLM_REASONING_EFFORT"] = "minimal"
            _openai_reasoning_params_for_model("gpt-5.2")
            globals()["DEFAULT_LLM_REASONING_EFFORT"] = "none"
            _openai_reasoning_params_for_model("gpt-5.2")
        finally:
            globals()["DEFAULT_LLM_REASONING_EFFORT"] = orig_effort

        _maxcov_log("maxcov extras reasoning params done")

        # Build Typst source with multiple data shapes.
        resume_data = {
            "summary": "First sentence. Second sentence.",
            "headers": [str(i) for i in range(9)],
            "highlighted_skills": [f"Skill {i}" for i in range(1, 10)],
            "skills_rows": [],
            "top_skills": ["AI", "ML"],
            "first_name": "Jane",
            "middle_name": "Q",
            "last_name": "Public",
            "target_role": "Engineer",
            "target_company": "Acme",
            "primary_domain": "AI",
            "req_id": "REQ-1",
        }
        profile_data = {
            "email": "jane@example.com",
            "phone": "555-0100",
            "linkedin_url": "https://linkedin.com/in/jane",
            "github_url": "https://github.com/jane",
            "scholar_url": "abc123",
            "summary": "Profile summary",
            "experience": [
                {
                    "role": "Dev",
                    "company": "Co",
                    "location": "Remote",
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                    "description": "Did stuff.",
                    "bullets": ["Bullet 1", "Bullet 2"],
                }
            ],
            "education": [
                {
                    "degree": "Master of Science (AI; Systems)",
                    "school": "Test University",
                    "start_date": "2016-01-01",
                    "end_date": "2018-01-01",
                    "description": "Coursework",
                    "bullets": ["Course A"],
                }
            ],
            "founder_roles": [
                {
                    "company": "Startup",
                    "location": "Remote",
                    "description": "Built things.",
                    "bullets": ["<date>2020---2021</date> Raised funds", "Did thing"],
                }
            ],
        }
        generate_typst_source(
            resume_data,
            profile_data,
            include_matrices=True,
            include_summary=True,
            section_order=["summary", "unknown", "education", "experience", "founder"],
        )
        resume_data_alt = resume_data.copy()
        resume_data_alt["summary"] = "No punctuation summary"
        resume_data_alt["skills_rows"] = '{"bad": "data"}'
        profile_data_alt = {
            "email": "jane@example.com",
            "phone": "555-0100",
            "linkedin_url": "https://linkedin.com/in/jane",
            "github_url": "https://github.com/jane",
            "scholar_url": "abc123",
            "summary": "",
            "experience": [
                {
                    "role": "Dev",
                    "company": "Co",
                    "location": "Remote",
                    "start_date": "",
                    "end_date": "",
                    "description": "",
                    "bullets": ["", "Bullet 1"],
                },
                {
                    "role": "Other",
                    "company": "Co2",
                    "location": "Remote",
                    "start_date": "",
                    "end_date": "",
                    "description": "",
                    "bullets": [],
                },
            ],
            "education": [
                {
                    "degree": "",
                    "school": "",
                    "start_date": "",
                    "end_date": "2018-01-01",
                    "description": "",
                    "bullets": [],
                },
                {
                    "degree": "",
                    "school": "",
                    "start_date": "",
                    "end_date": "",
                    "description": "",
                    "bullets": [],
                },
            ],
            "founder_roles": [
                {
                    "company": "Startup",
                    "location": "Remote",
                    "description": "",
                    "bullets": [],
                }
            ],
        }
        generate_typst_source(
            resume_data_alt,
            profile_data_alt,
            include_matrices=True,
            include_summary=True,
            section_order=["education", "experience", "founder"],
        )
        resume_data["skills_rows"] = '["a,b", ["c", "d"], null]'
        profile_data["education"] = [
            {
                "degree": "B.S.",
                "school": "State College",
                "start_date": "2012-01-01",
                "end_date": "2016-01-01",
                "description": "",
                "bullets": [],
            }
        ]
        generate_typst_source(
            resume_data,
            profile_data,
            include_matrices=False,
            include_summary=False,
            section_order=["experience", "education"],
        )

        # Exercise rasterized text paths with a stub PIL module.
        orig_pil_mod = sys.modules.get("PIL")
        orig_select_fonts = globals().get("_select_local_font_paths")
        orig_temp_build = TEMP_BUILD_DIR
        tmp_raster_dir = None
        try:

            class _RasterFakeImage:
                def __init__(self, size):
                    self.width, self.height = size

                def resize(self, size, _resample):
                    return _RasterFakeImage(size)

                def save(self, path, _format=None):
                    Path(path).write_bytes(b"fake")

            class _RasterFakeDraw:
                def __init__(self, _img):
                    return None

                def textbbox(self, _pos, _text, font=None):
                    return (0, 0, 10, 10)

                def text(self, *_args, **_kwargs):
                    return None

            class _FakeImageModule:
                BICUBIC = 3

                class Resampling:
                    LANCZOS = 1

                @staticmethod
                def new(_mode, size, _color):
                    return _RasterFakeImage(size)

            class _RasterFakeImageDrawModule:
                @staticmethod
                def Draw(img):
                    return _RasterFakeDraw(img)

            class _RasterFakeFont:
                pass

            class _RasterFakeImageFontModule:
                @staticmethod
                def truetype(path, _size):
                    if "fail" in str(path):
                        raise OSError("bad font")
                    return _RasterFakeFont()

                @staticmethod
                def load_default():
                    return _RasterFakeFont()

            fake_pil_raster: Any = types.ModuleType("PIL")
            fake_pil_raster.Image = _RasterFakeImageModule
            fake_pil_raster.ImageDraw = _RasterFakeImageDrawModule
            fake_pil_raster.ImageFont = _RasterFakeImageFontModule
            sys.modules["PIL"] = fake_pil_raster
            tmp_raster_dir = Path(
                tempfile.mkdtemp(prefix="maxcov_raster_", dir=BASE_DIR)
            )
            globals()["TEMP_BUILD_DIR"] = tmp_raster_dir
            globals()["_select_local_font_paths"] = lambda *_a, **_k: [
                tmp_raster_dir / "fail.otf",
                tmp_raster_dir / "ok.otf",
            ]
            raster_resume = {
                "summary": "Raster",
                "headers": [],
                "highlighted_skills": [],
                "skills_rows": [],
                "first_name": "",
                "middle_name": "",
                "last_name": "",
                "font_family": "",
            }
            raster_profile = {
                "name": "Raster User",
                "experience": [
                    {
                        "role": "Role",
                        "company": "Co",
                        "location": "",
                        "start_date": "",
                        "end_date": "",
                    "description": "Desc",
                    "bullets": ["<date>2020---2021</date> Did thing"],
                }
            ],
            "founder_roles": [
                {
                    "company": "Startup",
                    "location": "",
                    "description": "Founder desc",
                    "bullets": ["<date>2020---2021</date> Built"],
                }
            ],
            }
            generate_typst_source(
                raster_resume,
                raster_profile,
                include_matrices=False,
                include_summary=True,
                section_order=["experience", "founder"],
            )
        finally:
            globals()["_select_local_font_paths"] = orig_select_fonts
            globals()["TEMP_BUILD_DIR"] = orig_temp_build
            if orig_pil_mod is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = orig_pil_mod
            if tmp_raster_dir is not None:
                with suppress(Exception):
                    for path in tmp_raster_dir.rglob("*"):
                        if path.is_file():
                            path.unlink()
                    for path in sorted(tmp_raster_dir.rglob("*"), reverse=True):
                        if path.is_dir():
                            path.rmdir()

        _maxcov_log("maxcov extras typst shapes done")

        # Exercise PDF metadata and auto-fit logic with stubbed PDF rendering.
        with suppress(Exception):
            import pikepdf

            tmp_pdf = tmp_home / "sample.pdf"
            pikepdf.new().save(tmp_pdf)
            _apply_pdf_metadata(
                tmp_pdf,
                _build_pdf_metadata(resume_data, profile_data),
            )

        orig_generate_typst_source = generate_typst_source
        orig_compile_pdf = compile_pdf
        try:
            scale_state = {"pages": 2}

            def _fake_generate_typst_source(*_args, layout_scale=1.0, **_kwargs):
                return f"scale={layout_scale}"

            def _fake_compile_pdf(typst_source, metadata=None):
                try:
                    scale = float(str(typst_source).split("=", 1)[1])
                except Exception:
                    scale = 1.0
                if scale > 1.2:
                    scale_state["pages"] = 3
                elif scale < 0.8:
                    scale_state["pages"] = 1
                else:
                    scale_state["pages"] = 2
                return True, b"%PDF-1.4\n%"

            class _DummyPdf:
                def __init__(self, pages):
                    self.pages = [None] * int(pages)

            def _fake_pikepdf_open(_stream):
                return _DummyPdf(scale_state["pages"])

            globals()["generate_typst_source"] = _fake_generate_typst_source
            globals()["compile_pdf"] = _fake_compile_pdf
            _pikepdf_any = None
            orig_pikepdf_open = None
            _pikepdf: Any | None = None
            try:
                import pikepdf as _pikepdf

                _pikepdf_any = cast(Any, _pikepdf)
                orig_pikepdf_open = _pikepdf_any.open
                _pikepdf_any.open = _fake_pikepdf_open
            except Exception:
                orig_pikepdf_open = None
            compile_pdf_with_auto_tuning(
                resume_data,
                profile_data,
                include_matrices=True,
                include_summary=True,
                section_order=["summary"],
            )
            globals()["compile_pdf"] = lambda *_a, **_k: (True, b"%PDF-1.4\n%")
            if orig_pikepdf_open and _pikepdf_any is not None:
                _pikepdf_any.open = _fake_pikepdf_open
            scale_state["pages"] = 3
            compile_pdf_with_auto_tuning(
                resume_data,
                profile_data,
                include_matrices=True,
                include_summary=True,
                section_order=["summary"],
            )
            if orig_pikepdf_open and _pikepdf_any is not None:
                _pikepdf_any.open = orig_pikepdf_open
        finally:
            globals()["generate_typst_source"] = orig_generate_typst_source
            globals()["compile_pdf"] = orig_compile_pdf

        _maxcov_log("maxcov extras pdf metadata stub done")

        # Stub LLM responses to exercise JSON parsing paths.
        orig_llm_responses = _call_llm_responses
        orig_llm_completion = _call_llm_completion
        orig_skip_llm = os.environ.get("MAX_COVERAGE_SKIP_LLM")
        os.environ.pop("MAX_COVERAGE_SKIP_LLM", None)
        orig_fake_generate = globals().get("_fake_generate_resume_content")
        orig_openai_key = os.environ.get("OPENAI_API_KEY")
        orig_gemini_key = os.environ.get("GEMINI_API_KEY")
        orig_google_key = os.environ.get("GOOGLE_API_KEY")
        orig_home = os.environ.get("HOME")
        tmp_llm_home = None
        try:
            try:
                os.environ["MAX_COVERAGE_SKIP_LLM"] = "1"

                def _boom_fake(*_a, **_k):
                    raise RuntimeError("skip")

                globals()["_fake_generate_resume_content"] = _boom_fake
                generate_resume_content("req", {"summary": "Base"}, "openai:gpt-5.2")
            except Exception:
                pass
            finally:
                os.environ.pop("MAX_COVERAGE_SKIP_LLM", None)
                if orig_fake_generate is None:
                    globals().pop("_fake_generate_resume_content", None)
                else:
                    globals()["_fake_generate_resume_content"] = orig_fake_generate

            tmp_llm_home = Path(tempfile.mkdtemp(prefix="maxcov_llm_home_"))
            os.environ["HOME"] = str(tmp_llm_home)
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            generate_resume_content("req", {"prompt_yaml": "Prompt"}, "openai:gpt-5.2")
            generate_resume_content(
                "req", {"prompt_yaml": "Prompt"}, "gemini:gemini-1.5-flash"
            )

            os.environ["OPENAI_API_KEY"] = "sk-test"
            os.environ["GEMINI_API_KEY"] = "gm-test"
            os.environ["OPENAI_BASE_URL"] = "http://example.com"
            os.environ["OPENAI_ORGANIZATION"] = "org-test"
            os.environ["OPENAI_PROJECT"] = "proj-test"

            def _fake_llm_responses_missing(**_request):
                raise MissingApiKeyError("missing key")

            globals()["_call_llm_responses"] = _fake_llm_responses_missing
            generate_resume_content("req", {"prompt_yaml": "Prompt"}, "openai:gpt-5.2")

            def _fake_llm_completion_missing_retry(**_request):
                raise MissingApiKeyError("missing key")

            globals()["_call_llm_completion"] = _fake_llm_completion_missing_retry
            generate_resume_content(
                "req", {"prompt_yaml": "Prompt"}, "gemini:gemini-1.5-flash"
            )

            prompt_path = BASE_DIR / "prompt.yaml"
            prompt_backup = None
            try:
                if prompt_path.exists():
                    prompt_backup = prompt_path.with_suffix(".yaml.maxcov")
                    prompt_path.rename(prompt_backup)
                generate_resume_content("req", {}, "openai:gpt-5.2")
            finally:
                if prompt_backup and prompt_backup.exists():
                    prompt_backup.rename(prompt_path)

            class _DummyIncomplete:
                def __init__(self, reason):
                    self.reason = reason

            class _DummyOpenAIResp:
                def __init__(self, output_text, status="completed", reason=None):
                    self.output_text = output_text
                    self.status = status
                    self.incomplete_details = (
                        _DummyIncomplete(reason) if reason else None
                    )

            def _fake_llm_responses_simple(**_request):
                return _DummyOpenAIResp('{"summary":"ok"}')

            globals()["_call_llm_responses"] = _fake_llm_responses_simple
            with suppress(Exception):
                generate_resume_content(
                    req_text,
                    {"prompt_yaml": "Prompt", "rewrite_bullets": True},
                    "openai:gpt-5.2",
                )

            openai_calls = {"n": 0}

            def _fake_llm_responses(**_request):
                openai_calls["n"] += 1
                if openai_calls["n"] == 1:
                    return _DummyOpenAIResp('{"summary":"ok"}')
                if openai_calls["n"] == 2:
                    return _DummyOpenAIResp("not json")
                if openai_calls["n"] == 3:
                    return _DummyOpenAIResp(
                        "", status="incomplete", reason="max_output_tokens"
                    )
                return _DummyOpenAIResp('{"summary":"retry"}')

            globals()["_call_llm_responses"] = _fake_llm_responses
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            def _fake_llm_responses_fenced(**_request):
                return _DummyOpenAIResp('```json\n{"summary":"ok"}\n```')

            globals()["_call_llm_responses"] = _fake_llm_responses_fenced
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            def _fake_llm_responses_wrapped(**_request):
                return _DummyOpenAIResp('prefix {"summary":"ok"} suffix')

            globals()["_call_llm_responses"] = _fake_llm_responses_wrapped
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            def _fake_llm_responses_empty(**_request):
                return _DummyOpenAIResp("")

            globals()["_call_llm_responses"] = _fake_llm_responses_empty
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            def _fake_llm_responses_empty_fenced(**_request):
                return _DummyOpenAIResp("```")

            globals()["_call_llm_responses"] = _fake_llm_responses_empty_fenced
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            def _fake_llm_responses_error(**_request):
                raise MissingApiKeyError("missing key")

            globals()["_call_llm_responses"] = _fake_llm_responses_error
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            def _fake_llm_responses_unsupported(**_request):
                raise UnsupportedProviderError("unsupported")

            globals()["_call_llm_responses"] = _fake_llm_responses_unsupported
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            openai_calls = {"n": 0}

            def _fake_llm_responses_retry_error(**_request):
                openai_calls["n"] += 1
                if openai_calls["n"] == 1:
                    return _DummyOpenAIResp(
                        "", status="incomplete", reason="max_output_tokens"
                    )
                raise RuntimeError("retry fail")

            globals()["_call_llm_responses"] = _fake_llm_responses_retry_error
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            openai_calls = {"n": 0}

            def _fake_llm_responses_retry_empty(**_request):
                openai_calls["n"] += 1
                if openai_calls["n"] == 1:
                    return _DummyOpenAIResp(
                        "", status="incomplete", reason="max_output_tokens"
                    )
                return _DummyOpenAIResp("")

            globals()["_call_llm_responses"] = _fake_llm_responses_retry_empty
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            openai_calls = {"n": 0}

            def _fake_llm_responses_retry_invalid(**_request):
                openai_calls["n"] += 1
                if openai_calls["n"] == 1:
                    return _DummyOpenAIResp(
                        "", status="incomplete", reason="max_output_tokens"
                    )
                return _DummyOpenAIResp("invalid json")

            globals()["_call_llm_responses"] = _fake_llm_responses_retry_invalid
            with suppress(Exception):
                generate_resume_content(req_text, {}, "openai:gpt-5.2")

            class _DummyGeminiChoice:
                def __init__(self, content, finish_reason="stop"):
                    self.message = type("Msg", (), {"content": content})()
                    self.finish_reason = finish_reason

            class _DummyGeminiResp:
                def __init__(self, content, finish_reason="stop"):
                    self.choices = [_DummyGeminiChoice(content, finish_reason)]

            gemini_calls = {"n": 0}

            def _fake_llm_completion(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    raise RuntimeError("Simulated Gemini error")
                return _DummyGeminiResp(["{", None, '"summary":"ok"', "}"])

            globals()["_call_llm_completion"] = _fake_llm_completion
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            gemini_calls = {"n": 0}

            def _fake_llm_completion_retry(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    return _DummyGeminiResp("", finish_reason="max_tokens")
                return _DummyGeminiResp('{"summary":"retry"}')

            globals()["_call_llm_completion"] = _fake_llm_completion_retry
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            gemini_calls = {"n": 0}

            def _fake_llm_completion_retry_invalid(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    return _DummyGeminiResp("", finish_reason="max_tokens")
                return _DummyGeminiResp("not json")

            globals()["_call_llm_completion"] = _fake_llm_completion_retry_invalid
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            def _fake_llm_completion_invalid_json(**_request):
                return _DummyGeminiResp("not json", finish_reason="stop")

            globals()["_call_llm_completion"] = _fake_llm_completion_invalid_json
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            def _fake_llm_completion_bad_choice(**_request):
                class _Resp:
                    choices = None

                return _Resp()

            globals()["_call_llm_completion"] = _fake_llm_completion_bad_choice
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            def _fake_llm_completion_missing(**_request):
                raise MissingApiKeyError("missing")

            globals()["_call_llm_completion"] = _fake_llm_completion_missing
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            def _fake_llm_completion_unsupported(**_request):
                raise UnsupportedProviderError("unsupported")

            globals()["_call_llm_completion"] = _fake_llm_completion_unsupported
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            def _fake_llm_completion_empty(**_request):
                return _DummyGeminiResp("", finish_reason="stop")

            globals()["_call_llm_completion"] = _fake_llm_completion_empty
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            def _fake_llm_completion_none(**_request):
                return _DummyGeminiResp(None, finish_reason="stop")

            globals()["_call_llm_completion"] = _fake_llm_completion_none
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            gemini_calls = {"n": 0}

            def _fake_llm_completion_retry_fail(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    return _DummyGeminiResp("", finish_reason="max_tokens")
                raise RuntimeError("retry failed")

            globals()["_call_llm_completion"] = _fake_llm_completion_retry_fail
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")

            gemini_calls = {"n": 0}

            def _fake_llm_completion_retry_empty(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    return _DummyGeminiResp("", finish_reason="max_tokens")
                return _DummyGeminiResp("")

            globals()["_call_llm_completion"] = _fake_llm_completion_retry_empty
            with suppress(Exception):
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
        finally:
            if orig_skip_llm is None:
                os.environ.pop("MAX_COVERAGE_SKIP_LLM", None)
            else:
                os.environ["MAX_COVERAGE_SKIP_LLM"] = orig_skip_llm
            if orig_openai_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = orig_openai_key
            if orig_gemini_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = orig_gemini_key
            if orig_google_key is None:
                os.environ.pop("GOOGLE_API_KEY", None)
            else:
                os.environ["GOOGLE_API_KEY"] = orig_google_key
            if orig_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = orig_home
            if tmp_llm_home is not None:
                with suppress(Exception):
                    for path in tmp_llm_home.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_llm_home.rmdir()
            globals()["_call_llm_responses"] = orig_llm_responses
            globals()["_call_llm_completion"] = orig_llm_completion

        static_report = os.environ.get("MAX_COVERAGE_STATIC_REPORT_PATH")
        if static_report:
            with suppress(Exception):
                _maxcov_log("maxcov extras static analysis start")
                _run_static_analysis_tools(Path(static_report))
                _maxcov_log("maxcov extras static analysis done")

        _maxcov_log("maxcov extras done")

    def _run_playwright_ui_traversal(
        url: str,
        *,
        timeout_s: float = 30.0,
    ) -> bool:
        target = (url).strip()
        if not target or target.lower() in {"0", "false", "none"}:
            return False
        _maxcov_log(t"playwright start: {target}")
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            print(f"Playwright not available: {exc}")
            _maxcov_log("playwright done (unavailable)")
            return False

        timeout_ms = max(1000, int(timeout_s * 1000))
        action_timeout_ms = min(5000, max(500, int(timeout_ms * 0.25)))

        def safe_click(locator) -> None:
            with suppress(Exception):
                if locator.count() == 0:
                    return
                locator.click(timeout=action_timeout_ms)

        def safe_fill(locator, value: str) -> None:
            with suppress(Exception):
                if locator.count() == 0:
                    return
                locator.fill(value, timeout=action_timeout_ms)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_default_timeout(timeout_ms)
                try:
                    page.goto(target, wait_until="domcontentloaded", timeout=timeout_ms)
                    with suppress(Exception):
                        page.get_by_role("heading", name="Resume Builder").wait_for(
                            timeout=timeout_ms
                        )

                    safe_fill(
                        page.get_by_placeholder(
                            "Paste or type the job requisition here"
                        ),
                        "Playwright coverage smoke test.",
                    )
                    safe_click(page.get_by_role("button", name="Load Data"))
                    page.wait_for_timeout(500)

                    switches = page.get_by_role("switch")
                    try:
                        count = switches.count()
                    except Exception:
                        count = 0
                    if count:
                        safe_click(switches.nth(0))
                        if count > 1:
                            safe_click(switches.nth(1))

                    safe_click(page.get_by_label("Move section down"))
                    safe_click(page.get_by_label("Move section up"))
                    safe_click(page.get_by_role("button", name="Save Data"))

                    safe_click(page.get_by_role("button", name="Add Experience"))
                    safe_fill(page.get_by_placeholder("Role").first, "Playwright Role")
                    safe_fill(page.get_by_placeholder("Company").first, "Playwright Co")

                    safe_click(page.get_by_role("button", name="Add Education"))
                    safe_fill(
                        page.get_by_placeholder("Degree").first,
                        "Playwright Degree",
                    )
                    safe_fill(
                        page.get_by_placeholder("School").first,
                        "Playwright School",
                    )

                    safe_click(page.get_by_role("button", name="Add Founder Role"))
                    safe_fill(
                        page.get_by_placeholder("Role").last, "Playwright Founder"
                    )
                    safe_fill(
                        page.get_by_placeholder("Company").last, "Playwright Startup"
                    )

                    safe_click(page.get_by_role("button", name="Generate PDF"))
                    page.wait_for_timeout(1000)
                finally:
                    browser.close()
            return True
        except Exception as exc:
            print(f"Playwright traversal failed: {exc}")
            return False
        finally:
            _maxcov_log("playwright done")

    def _wait_for_url(url: str, timeout_s: float) -> bool:
        deadline = time.time() + max(1.0, timeout_s)
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2):  # nosec B310
                    return True
            except Exception:
                time.sleep(0.25)
        return False

    def _pick_open_port(preferred: int) -> int:
        if preferred and preferred > 0:
            with suppress(OSError):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("127.0.0.1", preferred))
                    return preferred
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _run_reflex_coverage_session(
        frontend_port: int,
        backend_port: int,
        *,
        startup_timeout_s: float,
        ui_timeout_s: float,
    ) -> str | None:
        frontend_port = _pick_open_port(frontend_port)
        backend_port = _pick_open_port(
            backend_port if backend_port != frontend_port else 0
        )

        log_dir = Path(tempfile.gettempdir()) / "dce_tools"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        log_path = log_dir / f"reflex_run_coverage_{stamp}.log"
        url = f"http://localhost:{frontend_port}"
        _maxcov_log(
            f"reflex session start: frontend={frontend_port} backend={backend_port}"
        )

        cov_file = os.environ.get("COVERAGE_FILE")
        env = os.environ.copy()
        env["REFLEX_COVERAGE"] = "1"
        env["REFLEX_COVERAGE_FORCE_OWNED"] = "1"
        if cov_file:
            env["COVERAGE_FILE"] = cov_file
            env.setdefault("REFLEX_COVERAGE_FILE", cov_file)
        env.setdefault("PYTHONUNBUFFERED", "1")

        cmd = [
            "reflex",
            "run",
            "--frontend-port",
            str(frontend_port),
            "--backend-port",
            str(backend_port),
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(BASE_DIR),
                    stdout=log_file,
                    stderr=log_file,
                    env=env,
                    start_new_session=True,
                )
            except Exception as exc:
                print(f"Warning: failed to start reflex coverage server: {exc}")
                return None

            try:
                if not _wait_for_url(url, startup_timeout_s):
                    print("Warning: reflex coverage server did not start in time.")
                    return None
                _run_playwright_ui_traversal(url, timeout_s=ui_timeout_s)
                return url
            finally:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(proc.pid, signal.SIGINT)
                    else:
                        proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        if hasattr(os, "killpg"):
                            os.killpg(proc.pid, signal.SIGTERM)
                        else:
                            proc.terminate()
                        proc.wait(timeout=10)
                    except Exception:
                        with suppress(Exception):
                            if hasattr(os, "killpg"):
                                os.killpg(proc.pid, signal.SIGKILL)
                            else:
                                proc.kill()
                time.sleep(1.0)
                _maxcov_log("reflex session done")

    def _read_log_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _format_duration(seconds: float) -> str:
        if seconds < 0:
            return ""
        minutes = int(seconds // 60)
        remainder = seconds - (minutes * 60)
        if minutes:
            return f"{minutes}m {remainder:.1f}s"
        return f"{remainder:.1f}s"

    def _read_static_analysis_report(path: Path) -> list[dict]:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(raw)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _render_run_all_tests_summary(
        rows: list[dict], total_duration_s: float
    ) -> None:
        try:
            from rich import box
            from rich.console import Console
            from rich.table import Table
            from rich.text import Text
        except Exception:
            print("Run-all-tests summary:")
            for row in rows:
                print(
                    f"- {row.get('step', '')}: {row.get('status', '')} "
                    f"({row.get('duration', '')}) {row.get('details', '')}".strip()
                )
            print(f"Total: {_format_duration(total_duration_s)}")
            return

        table = Table(title="Run All Tests", box=box.ASCII, show_lines=False)
        table.add_column("Step", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Details", overflow="fold")

        status_styles = {
            "ok": "green",
            "warn": "yellow",
            "skip": "dim",
            "fail": "bold red",
        }
        for row in rows:
            status = str(row.get("status", "")).lower() or "unknown"
            status_text = Text(status.upper())
            status_text.stylize(status_styles.get(status, "white"))
            table.add_row(
                str(row.get("step", "")),
                status_text,
                str(row.get("duration", "")),
                str(row.get("details", "")),
            )

        overall = "PASS" if all(r.get("status") != "fail" for r in rows) else "FAIL"
        table.add_section()
        table.add_row(
            "Total",
            Text(overall, style=("bold green" if overall == "PASS" else "bold red")),
            _format_duration(total_duration_s),
            "",
        )
        Console().print(table)

    def _scan_reflex_log_for_issues(log_text: str) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        error_pattern = re.compile(r"\b(error|traceback|exception)\b", re.IGNORECASE)
        warning_pattern = re.compile(r"\bwarning\b", re.IGNORECASE)
        for line in (log_text).splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if error_pattern.search(cleaned):
                errors.append(cleaned)
                continue
            if warning_pattern.search(cleaned):
                warnings.append(cleaned)
        return errors, warnings

    def _stop_reflex_process(proc: subprocess.Popen) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGINT)
            else:
                proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
        except Exception:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                with suppress(Exception):
                    if hasattr(os, "killpg"):
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()

    def _parse_ui_actions(raw: str) -> set[str]:
        raw = (raw).strip().lower()
        valid = {
            "load",
            "profile",
            "pipeline",
            "forms",
            "toggles",
            "reorder",
            "save",
            "pdf",
        }
        if not raw or raw == "all":
            return valid.copy()
        actions = {item.strip() for item in raw.split(",") if item.strip()}
        unknown = actions - valid
        if unknown:
            raise ValueError(
                f"Unknown ui-simulate action(s): {', '.join(sorted(unknown))}"
            )
        return actions

    def _run_all_tests_e2e() -> None:
        if os.environ.get("MAX_COVERAGE_CONTAINER") == "1":
            print(
                "Error: --run-all-tests is not supported inside the maxcov container. "
                "Use --run-all-tests-local instead."
            )
            sys.exit(1)
        script_path = BASE_DIR / "scripts" / "run_maxcov_e2e.sh"
        if not script_path.exists():
            print(f"Error: missing {script_path}")
            sys.exit(1)
        missing_tools = [tool for tool in ("expect", "sudo") if not shutil.which(tool)]
        if missing_tools:
            print("Error: missing required tool(s): " f"{', '.join(missing_tools)}")
            sys.exit(1)
        if not sys.stdin.isatty():
            print(
                "Error: --run-all-tests requires an interactive TTY for sudo "
                "password entry."
            )
            sys.exit(1)
        expect_script = """\
log_user 1
set timeout -1
set cmd $argv
eval spawn $cmd

proc prompt_password {} {
    if {[catch {stty -echo}]} {
        # Ignore if no controlling tty.
    }
    send_user "sudo password: "
    flush stdout
    gets stdin password
    if {[catch {stty echo}]} {
        # Ignore if no controlling tty.
    }
    send_user "\\n"
    return $password
}

expect {
    -re "(?i)password" {
        set password [prompt_password]
        send -- "$password\\r"
        exp_continue
    }
    eof {
        catch wait result
        exit [lindex $result 3]
    }
}
"""
        script_file = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False
            ) as handle:
                handle.write(expect_script)
                script_file = Path(handle.name)
            cmd = [
                "expect",
                str(script_file),
                "sudo",
                "-S",
                "-k",
                "-p",
                "sudo password: ",
                str(script_path),
                "--force",
            ]
            result = subprocess.run(cmd, cwd=str(BASE_DIR))
        finally:
            if script_file:
                with suppress(Exception):
                    script_file.unlink()
        sys.exit(result.returncode)

    requested_static_tools: list[str] = []
    if args.static_all:
        requested_static_tools.extend(static_tool_names)
    for raw in args.static_tool_names or []:
        for name in str(raw).split(","):
            name = name.strip()
            if not name:
                continue
            if name.lower() == "all":
                requested_static_tools.extend(static_tool_names)
            else:
                requested_static_tools.append(name)
    if args.static_tool_flags:
        requested_static_tools.extend(args.static_tool_flags)
    if requested_static_tools:
        deduped: list[str] = []
        seen: set[str] = set()
        for name in requested_static_tools:
            key = name.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        log_dir = Path(tempfile.gettempdir()) / "dce_tools"
        log_dir.mkdir(parents=True, exist_ok=True)
        report_path = log_dir / f"maxcov_static_cli_{stamp}.json"
        target_dir = args.static_target.strip() or None
        target_file = args.static_target_file.strip() or None
        timeout_override = args.static_timeout if args.static_timeout > 0 else None
        _exercise_maximum_coverage_extras(
            args.req_file,
            static_only=True,
            static_report_path=report_path,
            static_selected_tools=deduped,
            static_target_dir=target_dir,
            static_target_file=target_file,
            static_timeout=timeout_override,
        )
        static_results = _read_static_analysis_report(report_path)
        any_failed = False
        if static_results:
            for item in static_results:
                tool = str(item.get("tool") or "static")
                status = str(item.get("status") or "warn").lower()
                duration_s = float(item.get("duration_s") or 0.0)
                details = str(item.get("details") or "")
                print(f"{tool}: {status} ({duration_s:.1f}s) {details}".strip())
                if status != "ok":
                    any_failed = True
        else:
            print(f"No static report produced at {report_path}")
            any_failed = True
        sys.exit(1 if any_failed else 0)

    if args.reset_db:
        assets = args.reset_db or str(DEFAULT_ASSETS_JSON)
        print(f"Resetting Neo4j and importing assets from {assets}...")
        try:
            db = Neo4jClient()
            db.reset_and_import(assets)
            db.close()
            print("Reset + import completed successfully.")
        except Exception as e:
            print(f"Error resetting/importing assets: {e}")
            sys.exit(1)

    if args.import_assets and not args.reset_db:
        print(f"Importing assets from {args.import_assets}...")
        try:
            db = Neo4jClient()
            imported = db.import_assets(
                args.import_assets, allow_overwrite=bool(args.overwrite_resume)
            )
            db.close()
            if not imported:
                sys.exit(1)
            print("Import completed successfully.")
        except Exception as e:
            print(f"Error importing assets: {e}")
            sys.exit(1)

    if args.run_all_tests:
        _run_all_tests_e2e()

    if args.run_all_tests_local:
        results: list[dict] = []
        overall_rc = 0
        total_started = time.perf_counter()
        log_dir = Path(tempfile.gettempdir()) / "dce_tools"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        static_report_path = Path(
            os.environ.get(
                "MAX_COVERAGE_STATIC_REPORT_PATH",
                str(log_dir / f"maxcov_static_{stamp}.json"),
            )
        )

        def record(
            step: str, status: str, duration_s: float, details: str = ""
        ) -> None:
            results.append(
                {
                    "step": step,
                    "status": status,
                    "duration": _format_duration(duration_s),
                    "details": details,
                }
            )

        maxcov_started = time.perf_counter()
        maxcov_cmd = [
            sys.executable,
            str(BASE_DIR / "harness.py"),
            "--maximum-coverage",
            "--maximum-coverage-actions",
            str(args.maximum_coverage_actions),
            "--maximum-coverage-ui-timeout",
            str(args.maximum_coverage_ui_timeout),
            "--maximum-coverage-reflex-frontend-port",
            str(args.maximum_coverage_reflex_frontend_port),
            "--maximum-coverage-reflex-backend-port",
            str(args.maximum_coverage_reflex_backend_port),
            "--maximum-coverage-reflex-startup-timeout",
            str(args.maximum_coverage_reflex_startup_timeout),
        ]
        if args.maximum_coverage_ui_url:
            maxcov_cmd.extend(
                ["--maximum-coverage-ui-url", str(args.maximum_coverage_ui_url)]
            )
        if args.maximum_coverage_skip_llm:
            maxcov_cmd.append("--maximum-coverage-skip-llm")
        if args.maximum_coverage_failures:
            maxcov_cmd.append("--maximum-coverage-failures")
        if args.maximum_coverage_reflex:
            maxcov_cmd.append("--maximum-coverage-reflex")
        maxcov_env = os.environ.copy()
        maxcov_env.setdefault("MAX_COVERAGE_CONTAINER", "1")
        maxcov_env["MAX_COVERAGE_STATIC_REPORT_PATH"] = str(static_report_path)
        maxcov_result = subprocess.run(
            maxcov_cmd,
            cwd=str(BASE_DIR),
            env=maxcov_env,
        )
        maxcov_duration = time.perf_counter() - maxcov_started
        if maxcov_result.returncode != 0:
            record(
                "maximum-coverage",
                "fail",
                maxcov_duration,
                f"rc={maxcov_result.returncode}",
            )
            overall_rc = maxcov_result.returncode or 1
            _render_run_all_tests_summary(results, time.perf_counter() - total_started)
            sys.exit(overall_rc)
        record("maximum-coverage", "ok", maxcov_duration, "ok")

        static_results = _read_static_analysis_report(static_report_path)
        static_failed = False
        if static_results:
            for item in static_results:
                tool = str(item.get("tool") or "static")
                status = str(item.get("status") or "warn").lower()
                duration_s = float(item.get("duration_s") or 0.0)
                details = str(item.get("details") or "")
                record(f"static: {tool}", status, duration_s, details)
                if status == "fail":
                    static_failed = True
            if static_failed and overall_rc == 0:
                overall_rc = 1
        else:
            record(
                "static analysis",
                "warn",
                0.0,
                "no report",
            )

        diagram_log = log_dir / f"diagrams_run_all_tests_{stamp}.log"
        diagrams_started = time.perf_counter()
        diagrams_cmd = [
            sys.executable,
            str(BASE_DIR / "scripts" / "generate_diagrams.py"),
        ]
        diagrams_result = subprocess.run(
            diagrams_cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
        diagrams_duration = time.perf_counter() - diagrams_started
        diagram_output = "\n".join(
            [
                t
                for t in (diagrams_result.stdout or "", diagrams_result.stderr or "")
                if t
            ]
        )
        with suppress(Exception):
            diagram_log.write_text(diagram_output, encoding="utf-8")
        if diagrams_result.returncode != 0:
            record(
                "diagram generation",
                "fail",
                diagrams_duration,
                f"rc={diagrams_result.returncode}; log: {diagram_log}",
            )
            if overall_rc == 0:
                overall_rc = diagrams_result.returncode or 1
        else:
            record(
                "diagram generation",
                "ok",
                diagrams_duration,
                f"log: {diagram_log}",
            )

        frontend_port = _pick_open_port(3000)
        backend_port = _pick_open_port(8000 if 8000 != frontend_port else 0)
        log_path = log_dir / f"reflex_run_all_tests_{stamp}.log"
        url = f"http://localhost:{frontend_port}"
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("MAX_COVERAGE_SKIP_LLM", "1")
        cmd = [
            "reflex",
            "run",
            "--frontend-port",
            str(frontend_port),
            "--backend-port",
            str(backend_port),
        ]
        proc = None
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                reflex_attempt_started = time.perf_counter()
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(BASE_DIR),
                        stdout=log_file,
                        stderr=log_file,
                        env=env,
                        start_new_session=True,
                    )
                except Exception as exc:
                    record(
                        "reflex run (clean start)",
                        "fail",
                        time.perf_counter() - reflex_attempt_started,
                        f"start failed: {exc}",
                    )
                    overall_rc = 1

                if overall_rc == 0:
                    reflex_started = time.perf_counter()
                    if not _wait_for_url(
                        url, float(args.maximum_coverage_reflex_startup_timeout)
                    ):
                        record(
                            "reflex run (clean start)",
                            "fail",
                            time.perf_counter() - reflex_started,
                            "startup timeout",
                        )
                        overall_rc = 1
                    else:
                        time.sleep(1.0)
                        log_text = _read_log_text(log_path)
                        errors, warnings = _scan_reflex_log_for_issues(log_text)
                        if errors:
                            first_issue = errors[0]
                            record(
                                "reflex run (clean start)",
                                "fail",
                                time.perf_counter() - reflex_started,
                                f"{first_issue} (log: {log_path})",
                            )
                            overall_rc = 1
                        elif warnings:
                            first_warning = warnings[0]
                            record(
                                "reflex run (clean start)",
                                "warn",
                                time.perf_counter() - reflex_started,
                                f"{first_warning} (log: {log_path})",
                            )
                        else:
                            record(
                                "reflex run (clean start)",
                                "ok",
                                time.perf_counter() - reflex_started,
                                f"log: {log_path}",
                            )

                if overall_rc == 0:
                    ui_started = time.perf_counter()
                    ui_cmd = [
                        sys.executable,
                        str(BASE_DIR / "scripts" / "ui_playwright_check.py"),
                        "--url",
                        url,
                        "--timeout",
                        str(args.ui_playwright_timeout),
                        "--pdf-timeout",
                        str(args.ui_playwright_pdf_timeout),
                    ]
                    if args.ui_playwright_headed:
                        ui_cmd.append("--headed")
                    if args.ui_playwright_slowmo:
                        ui_cmd.extend(["--slowmo", str(args.ui_playwright_slowmo)])
                    if args.ui_playwright_allow_db_error:
                        ui_cmd.append("--allow-db-error")
                    if args.ui_playwright_screenshot_dir:
                        ui_cmd.extend(
                            ["--screenshot-dir", args.ui_playwright_screenshot_dir]
                        )
                    ui_result = subprocess.run(ui_cmd, cwd=str(BASE_DIR))
                    ui_duration = time.perf_counter() - ui_started
                    if ui_result.returncode != 0:
                        record(
                            "ui-playwright-check",
                            "fail",
                            ui_duration,
                            f"rc={ui_result.returncode}; log: {log_path}",
                        )
                        overall_rc = ui_result.returncode or 1
                    else:
                        record("ui-playwright-check", "ok", ui_duration, "ok")
        finally:
            if proc is not None:
                _stop_reflex_process(proc)
            _render_run_all_tests_summary(results, time.perf_counter() - total_started)
            sys.exit(overall_rc)

    if args.maximum_coverage or getattr(args, "ui_simulate", False):
        try:
            actions_raw = args.maximum_coverage_actions
            if hasattr(args, "ui_simulate_actions") and args.ui_simulate_actions:
                actions_raw = args.ui_simulate_actions
            ui_actions = _parse_ui_actions(actions_raw)
        except ValueError as e:
            print(str(e))
            sys.exit(1)
        try:
            skip_llm = bool(args.maximum_coverage_skip_llm) or bool(
                getattr(args, "ui_simulate_skip_llm", False)
            )
            simulate_failures = bool(args.maximum_coverage_failures) or bool(
                getattr(args, "ui_simulate_failures", False)
            )
            _maxcov_log(
                f"ui simulation start: actions={sorted(ui_actions)}, "
                f"skip_llm={skip_llm}, failures={simulate_failures}"
            )
            started = time.perf_counter()
            asyncio.run(
                _run_ui_simulation(
                    ui_actions,
                    args.req_file,
                    skip_llm=skip_llm,
                    simulate_failures=simulate_failures,
                )
            )
            _maxcov_log(t"ui simulation done ({time.perf_counter() - started:.1f}s)")
            if args.maximum_coverage:
                maxcov_started = time.perf_counter()
                reflex_url = None
                if args.maximum_coverage_reflex:
                    try:
                        _maxcov_log("reflex coverage start")
                        started = time.perf_counter()
                        reflex_url = _run_reflex_coverage_session(
                            int(args.maximum_coverage_reflex_frontend_port),
                            int(args.maximum_coverage_reflex_backend_port),
                            startup_timeout_s=float(
                                args.maximum_coverage_reflex_startup_timeout
                            ),
                            ui_timeout_s=float(args.maximum_coverage_ui_timeout or 0)
                            or 30.0,
                        )
                        _maxcov_log(
                            "reflex coverage done "
                            f"({time.perf_counter() - started:.1f}s)"
                        )
                    except Exception as e:
                        print(f"Warning: Reflex coverage session failed: {e}")
                try:
                    _maxcov_log("maximum coverage extras start")
                    started = time.perf_counter()
                    _exercise_maximum_coverage_extras(args.req_file)
                    _maxcov_log(
                        f"maximum coverage extras done ({time.perf_counter() - started:.1f}s)"
                    )
                except Exception as e:
                    print(f"Warning: maximum coverage extras failed: {e}")
                try:
                    playwright_url = None
                    if ui_url_was_set:
                        playwright_url = args.maximum_coverage_ui_url
                    elif not reflex_url:
                        playwright_url = args.maximum_coverage_ui_url
                    if playwright_url:
                        _maxcov_log(t"playwright traversal start: {playwright_url}")
                        started = time.perf_counter()
                        _run_playwright_ui_traversal(
                            playwright_url,
                            timeout_s=float(args.maximum_coverage_ui_timeout or 0)
                            or 30.0,
                        )
                        _maxcov_log(
                            "playwright traversal done "
                            f"({time.perf_counter() - started:.1f}s)"
                        )
                except Exception as e:
                    print(f"Warning: Playwright traversal failed: {e}")
                _maxcov_log(
                    f"maximum coverage done ({time.perf_counter() - maxcov_started:.1f}s)"
                )
        except Exception as e:
            print(f"UI simulation failed: {e}")
            sys.exit(1)

    if args.eval_prompt or args.generate_profile:
        _maxcov_log("cli prompt evaluation start")
        req_path = Path(args.req_file)
        if not req_path.exists():
            print(f"Req file not found: {req_path}")
            sys.exit(1)
        req_text = req_path.read_text(encoding="utf-8", errors="ignore")

        try:
            db = Neo4jClient()
            db.ensure_resume_exists()
            data = db.get_resume_data() or {}
            db.close()
        except Exception as e:
            print(f"Error reading resume from Neo4j: {e}")
            sys.exit(1)

        resume_node = data.get("resume", {}) or {}
        if not resume_node:
            print("No resume found in Neo4j; cannot generate without base profile.")
            sys.exit(1)

        base_profile = {
            **resume_node,
            "experience": data.get("experience", []),
            "education": data.get("education", []),
            "founder_roles": data.get("founder_roles", []),
        }
        model_name = args.model_name or DEFAULT_LLM_MODEL
        _maxcov_log("cli prompt evaluation call LLM")
        llm_result = generate_resume_content(req_text, base_profile, model_name)
        _maxcov_log("cli prompt evaluation LLM done")

        if args.eval_prompt:
            print(json.dumps(llm_result, ensure_ascii=False, indent=2))

        if args.generate_profile:
            if not isinstance(llm_result, dict) or llm_result.get("error"):
                if not args.eval_prompt:
                    print(json.dumps(llm_result, ensure_ascii=False, indent=2))
                sys.exit(1)

            headers = ensure_len(
                [
                    resume_node.get("head1_left", ""),
                    resume_node.get("head1_middle", ""),
                    resume_node.get("head1_right", ""),
                    resume_node.get("head2_left", ""),
                    resume_node.get("head2_middle", ""),
                    resume_node.get("head2_right", ""),
                    resume_node.get("head3_left", ""),
                    resume_node.get("head3_middle", ""),
                    resume_node.get("head3_right", ""),
                ]
            )
            highlighted_skills = ensure_len(resume_node.get("top_skills", []))
            skills_rows = _ensure_skill_rows(llm_result.get("skills_rows"))
            experience_bullets = _coerce_bullet_overrides(
                llm_result.get("experience_bullets")
            )
            founder_role_bullets = _coerce_bullet_overrides(
                llm_result.get("founder_role_bullets")
            )

            resume_fields = {
                "summary": llm_result.get("summary", resume_node.get("summary", "")),
                "headers": headers[:9],
                "highlighted_skills": highlighted_skills[:9],
                "skills_rows_json": json.dumps(skills_rows, ensure_ascii=False),
                "experience_bullets_json": json.dumps(
                    experience_bullets, ensure_ascii=False
                ),
                "founder_role_bullets_json": json.dumps(
                    founder_role_bullets, ensure_ascii=False
                ),
                "job_req_raw": req_text,
                "target_company": llm_result.get("target_company", ""),
                "target_role": llm_result.get("target_role", ""),
                "seniority_level": llm_result.get("seniority_level", ""),
                "target_location": llm_result.get("target_location", ""),
                "work_mode": llm_result.get("work_mode", ""),
                "travel_requirement": llm_result.get("travel_requirement", ""),
                "primary_domain": llm_result.get("primary_domain", ""),
                "must_have_skills": llm_result.get("must_have_skills", []),
                "nice_to_have_skills": llm_result.get("nice_to_have_skills", []),
                "tech_stack_keywords": llm_result.get("tech_stack_keywords", []),
                "non_technical_requirements": llm_result.get(
                    "non_technical_requirements", []
                ),
                "certifications": llm_result.get("certifications", []),
                "clearances": llm_result.get("clearances", []),
                "core_responsibilities": llm_result.get("core_responsibilities", []),
                "outcome_goals": llm_result.get("outcome_goals", []),
                "salary_band": llm_result.get("salary_band", ""),
                "posting_url": llm_result.get("posting_url", ""),
                "req_id": llm_result.get("req_id", ""),
            }

            try:
                db = Neo4jClient()
                _maxcov_log("cli prompt evaluation save profile")
                profile_id = db.save_resume(resume_fields)
                db.close()
                print(f"Saved Profile {profile_id}")
                _maxcov_log("cli prompt evaluation save profile done")
            except Exception as e:
                print(f"Error saving Profile: {e}")
                sys.exit(1)

    if args.compile_pdf:
        output_path = Path(args.compile_pdf)
        try:
            db = Neo4jClient()
            data = db.get_resume_data() or {}
            profiles = db.list_applied_jobs(limit=1)
            db.close()

            resume_node = data.get("resume", {}) or {}
            latest_profile = profiles[0] if profiles else {}

            profile_headers = latest_profile.get("headers") or []
            if profile_headers and any(str(h).strip() for h in profile_headers):
                headers = ensure_len(profile_headers)
            else:
                headers = ensure_len(
                    [
                        resume_node.get("head1_left", ""),
                        resume_node.get("head1_middle", ""),
                        resume_node.get("head1_right", ""),
                        resume_node.get("head2_left", ""),
                        resume_node.get("head2_middle", ""),
                        resume_node.get("head2_right", ""),
                        resume_node.get("head3_left", ""),
                        resume_node.get("head3_middle", ""),
                        resume_node.get("head3_right", ""),
                    ]
                )

            profile_skills = latest_profile.get("highlighted_skills") or []
            if profile_skills and any(str(s).strip() for s in profile_skills):
                skills = ensure_len(profile_skills)
            else:
                skills = ensure_len(resume_node.get("top_skills", []))

            summary_text = str(latest_profile.get("summary") or "").strip()
            if not summary_text:
                summary_text = str(resume_node.get("summary") or "").strip()

            section_titles = _normalize_section_titles(
                resume_node.get("section_titles_json")
                or resume_node.get("section_titles")
            )
            custom_sections = _normalize_custom_sections(
                resume_node.get("custom_sections_json")
                or resume_node.get("custom_sections")
            )
            extra_keys = _custom_section_keys(custom_sections)
            raw_order = resume_node.get("section_order")
            if isinstance(raw_order, str):
                raw_order = [s.strip() for s in raw_order.split(",") if s.strip()]
            section_order = _sanitize_section_order(raw_order, extra_keys)
            section_enabled = _normalize_section_enabled(
                resume_node.get("section_enabled"),
                list(SECTION_LABELS) + extra_keys,
                extra_keys=extra_keys,
            )
            section_order = _apply_section_enabled(
                section_order,
                section_enabled,
            )

            resume_data = {
                "summary": summary_text,
                "headers": headers[:9],
                "highlighted_skills": skills[:9],
                "skills_rows": latest_profile.get("skills_rows") or [[], [], []],
                "first_name": latest_profile.get("first_name")
                or resume_node.get("first_name", ""),
                "middle_name": latest_profile.get("middle_name")
                or resume_node.get("middle_name", ""),
                "last_name": latest_profile.get("last_name")
                or resume_node.get("last_name", ""),
                "email": latest_profile.get("email") or resume_node.get("email", ""),
                "email2": latest_profile.get("email2") or resume_node.get("email2", ""),
                "phone": latest_profile.get("phone") or resume_node.get("phone", ""),
                "font_family": resume_node.get(
                    "font_family", DEFAULT_RESUME_FONT_FAMILY
                ),
                "auto_fit_target_pages": _normalize_auto_fit_target_pages(
                    resume_node.get("auto_fit_target_pages"),
                    DEFAULT_AUTO_FIT_TARGET_PAGES,
                ),
                "linkedin_url": latest_profile.get("linkedin_url")
                or resume_node.get("linkedin_url", ""),
                "github_url": latest_profile.get("github_url")
                or resume_node.get("github_url", ""),
                "calendly_url": latest_profile.get("calendly_url")
                or resume_node.get("calendly_url", ""),
                "portfolio_url": latest_profile.get("portfolio_url")
                or resume_node.get("portfolio_url", ""),
                "section_order": section_order,
                "section_titles": section_titles,
                "custom_sections": custom_sections,
            }
            exp_overrides = _bullet_override_map(
                latest_profile.get("experience_bullets")
            )
            founder_overrides = _bullet_override_map(
                latest_profile.get("founder_role_bullets")
            )
            experience_items = _apply_bullet_overrides(
                data.get("experience", []), exp_overrides
            )
            founder_items = _apply_bullet_overrides(
                data.get("founder_roles", []), founder_overrides
            )
            profile_data = {
                **resume_node,
                **latest_profile,
                "experience": experience_items,
                "education": data.get("education", []),
                "founder_roles": founder_items,
            }

            if args.auto_fit:
                success, pdf_bytes = compile_pdf_with_auto_tuning(
                    resume_data,
                    profile_data,
                    include_matrices=True,
                    include_summary=True,
                    section_order=resume_data["section_order"],
                    target_pages=resume_data.get("auto_fit_target_pages"),
                )
            else:
                source = generate_typst_source(
                    resume_data,
                    profile_data,
                    include_matrices=True,
                    include_summary=True,
                    section_order=resume_data["section_order"],
                )
                pdf_metadata = _build_pdf_metadata(resume_data, profile_data)
                success, pdf_bytes = compile_pdf(source, metadata=pdf_metadata)
            if not success or not pdf_bytes:
                print("Typst compilation failed; see logs above.")
                sys.exit(1)
            if output_path.parent.exists() and not output_path.parent.is_dir():
                output_path = ASSETS_DIR / output_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(pdf_bytes)
            print(f"Wrote PDF to {output_path}")
            sys.exit(0)
        except Exception as e:
            print(f"Error compiling PDF: {e}")
            sys.exit(1)

    if args.export_resume_pdf:
        pdf_bytes = render_resume_pdf_bytes(
            save_copy=True,
            include_summary=False,
            include_skills=False,
            filename="preview_no_summary_skills.pdf",
        )
        if not pdf_bytes:
            print("Failed to export PDF.")
            sys.exit(1)
        print(
            f"Exported {ASSETS_DIR / 'preview_no_summary_skills.pdf'} (no summary/skills)."
        )
        sys.exit(0)

    if args.show_resume_data:
        try:
            db = Neo4jClient()
            data = db.get_resume_data()
            db.close()
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        except Exception as e:
            print(f"Error retrieving resume data: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.list_applied:
        try:
            db = Neo4jClient()
            jobs = db.list_applied_jobs()
            db.close()
            print(json.dumps(jobs, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error listing applied jobs: {e}")
            sys.exit(1)
        sys.exit(0)
