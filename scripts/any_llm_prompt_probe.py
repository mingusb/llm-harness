#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any
from contextlib import contextmanager, nullcontext

from any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError, UnsupportedProviderError

STOP_LIST_PLACEHOLDER = "{{STOP_LIST}}"
RUN_ID = ""


def _log(message: str) -> None:
    print(f"[any-llm-probe] {message}", flush=True)


def _format_log_value(value: Any, *, limit: int = 160) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return f"{text[:limit]}...(+{len(text) - limit} chars)"
    return text


def _log_event(event: str, **fields: Any) -> None:
    run_id = RUN_ID or "unknown"
    parts = [f"event={event}", f"run_id={run_id}"]
    for key in sorted(fields):
        parts.append(f"{key}={_format_log_value(fields[key])}")
    _log(" ".join(parts))


def _log_exception(context: str, exc: BaseException) -> None:
    _log_event(
        context,
        error_type=type(exc).__name__,
        error=_format_log_value(exc, limit=240),
    )
    for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
        for chunk in line.rstrip().splitlines():
            _log_event(f"{context}.trace", line=chunk)


def _read_first_secret_line(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                return line
    except FileNotFoundError:
        return ""
    except Exception:
        return ""
    return ""


def _read_kv_secret(path: Path, keys: set[str]) -> str | None:
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
        if key.strip().upper() in keys:
            return value.strip().strip("\"'")
    return None


def load_openai_api_key() -> str | None:
    env = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if env:
        _log_event("api_key_found", provider="openai", source="env:OPENAI_API_KEY", length=len(env))
        return env
    path = Path.home() / "openaikey.txt"
    from_kv = _read_kv_secret(path, {"OPENAI_API_KEY"})
    if from_kv:
        _log_event("api_key_found", provider="openai", source=str(path), length=len(from_kv))
        return from_kv
    raw = _read_first_secret_line(path)
    if raw.lstrip().startswith("sk-"):
        _log_event("api_key_found", provider="openai", source=str(path), length=len(raw))
        return raw.strip().strip("\"'")
    return None


def load_gemini_api_key() -> str | None:
    env = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if env:
        _log_event("api_key_found", provider="gemini", source="env:GEMINI_API_KEY", length=len(env))
        return env
    env = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if env:
        _log_event("api_key_found", provider="gemini", source="env:GOOGLE_API_KEY", length=len(env))
        return env
    path = Path.home() / "geminikey.txt"
    from_kv = _read_kv_secret(path, {"GEMINI_API_KEY", "GOOGLE_API_KEY"})
    if from_kv:
        _log_event("api_key_found", provider="gemini", source=str(path), length=len(from_kv))
        return from_kv
    raw = _read_first_secret_line(path)
    if raw and "=" not in raw:
        _log_event("api_key_found", provider="gemini", source=str(path), length=len(raw))
        return raw.strip().strip("\"'")
    legacy = _read_kv_secret(Path.home() / "openaikey.txt", {"GEMINI_API_KEY", "GOOGLE_API_KEY"})
    if legacy:
        _log_event(
            "api_key_found",
            provider="gemini",
            source=str(Path.home() / "openaikey.txt"),
            length=len(legacy),
        )
        return legacy
    return None


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").rstrip()
    except FileNotFoundError:
        raise SystemExit(f"Missing file: {path}")
    except Exception as exc:
        raise SystemExit(f"Failed to read {path}: {type(exc).__name__}: {exc}")


def _load_stop_list(path: Path) -> list[str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
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


def _inject_stop_list(prompt_template: str, stop_terms: list[str]) -> tuple[str, bool, str]:
    if not prompt_template or not stop_terms:
        return prompt_template, False, "none"
    if STOP_LIST_PLACEHOLDER in prompt_template:
        lines = prompt_template.splitlines()
        for idx, line in enumerate(lines):
            if STOP_LIST_PLACEHOLDER not in line:
                continue
            indent = line.split(STOP_LIST_PLACEHOLDER, 1)[0]
            items = _format_stop_list_items(stop_terms, indent=indent)
            lines[idx : idx + 1] = items.splitlines()
            return "\n".join(lines), True, "placeholder"
    items = _format_stop_list_items(stop_terms)
    return (
        prompt_template.rstrip()
        + "\n\nstop_list:\n"
        + '  directive: "Stop-list injected from stop-list.txt."\n'
        + "  phrases:\n"
        + f"{items}\n"
    ), True, "append"


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


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    _log_event("json_extract_start", text_chars=len(raw))
    if not raw:
        raise ValueError("empty")
    if raw.startswith("```"):
        raw = raw.replace("```json", "```")
        raw = raw.strip("`").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            _log_event("json_extract_direct_ok", keys=len(parsed))
            return parsed
    except Exception:
        _log_event("json_extract_direct_fail")

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

    candidates = scan_objects(raw)
    _log_event("json_extract_scan", candidates=len(candidates))
    for idx, candidate in enumerate(candidates):
        try:
            parsed = json.loads(candidate)
        except Exception:
            _log_event("json_extract_candidate_fail", index=idx, chars=len(candidate))
            continue
        if isinstance(parsed, dict):
            _log_event("json_extract_candidate_ok", index=idx, chars=len(candidate))
            return parsed
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, dict):
            _log_event("json_extract_slice_ok", chars=end - start + 1)
            return parsed
    raise ValueError("non-dict json")


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _describe_response(resp) -> dict[str, Any]:
    info: dict[str, Any] = {"resp_type": type(resp).__name__}
    if resp is None:
        return info
    for attr in ("id", "model", "created", "object"):
        value = getattr(resp, attr, None)
        if value is not None:
            info[f"resp_{attr}"] = value
    choices = getattr(resp, "choices", None)
    if isinstance(choices, (list, tuple)):
        info["choices"] = len(choices)
    elif choices is None:
        info["choices"] = "none"
    else:
        info["choices"] = type(choices).__name__
    return info


def _usage_summary(resp) -> dict[str, Any] | None:
    if resp is None:
        return None
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return {
            "usage_type": "dict",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    return {
        "usage_type": type(usage).__name__,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _keys_sample(parsed: dict, limit: int = 8) -> str:
    keys = [str(key) for key in parsed.keys()]
    keys.sort()
    sample = keys[:limit]
    suffix = "..." if len(keys) > limit else ""
    return ",".join(sample) + suffix


def _read_int_env(*keys: str) -> tuple[int, str] | None:
    for key in keys:
        if not key:
            continue
        raw = os.environ.get(key)
        if raw is None:
            continue
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value, key
    return None


def _resolve_max_tokens(provider: str) -> tuple[int, str]:
    override = _read_int_env(
        "LLM_MAX_OUTPUT_TOKENS",
        "GEMINI_MAX_OUTPUT_TOKENS" if provider == "gemini" else "",
        "GOOGLE_MAX_OUTPUT_TOKENS" if provider == "gemini" else "",
        "OPENAI_MAX_OUTPUT_TOKENS" if provider == "openai" else "",
    )
    if override:
        value, key = override
        return value, f"env:{key}"
    return 4096, "default"


def _split_model_spec(model_spec: str) -> tuple[str, str]:
    if ":" in model_spec:
        provider, model = model_spec.split(":", 1)
        return provider.strip().lower(), model.strip()
    return "openai", model_spec.strip()


def _close_llm_client(llm) -> None:
    client = getattr(llm, "client", None)
    if client is None:
        _log_event("client_close_skip", reason="no_client")
        return
    _log_event("client_close_start", client_type=type(client).__name__)

    def _schedule_coro(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running() and not loop.is_closed():
            _log_event("client_close_schedule", mode="existing_loop", loop_id=id(loop))
            loop.create_task(coro)
            return
        _log_event("client_close_schedule", mode="asyncio_run")
        asyncio.run(coro)

    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        _log_event("client_close_call", method="close")
        result = close_fn()
        if asyncio.iscoroutine(result):
            _schedule_coro(result)
        return
    aclose_fn = getattr(client, "aclose", None)
    if callable(aclose_fn):
        _log_event("client_close_call", method="aclose")
        result = aclose_fn()
        if asyncio.iscoroutine(result):
            _schedule_coro(result)


def _ensure_open_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _log_event("event_loop_created", loop_id=id(loop))
    else:
        _log_event(
            "event_loop_reused",
            loop_id=id(loop),
            running=loop.is_running(),
            closed=loop.is_closed(),
        )
    return loop


@contextmanager
def _patch_gemini_usage() -> Any:
    patches: list[tuple[object, object]] = []

    def _wrap(convert_fn):
        def _wrapped(response):
            data = convert_fn(response)
            raw_usage = data.get("usage", None)
            usage = raw_usage if isinstance(raw_usage, dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            raw_prompt = raw_usage.get("prompt_tokens") if isinstance(raw_usage, dict) else None
            raw_completion = (
                raw_usage.get("completion_tokens") if isinstance(raw_usage, dict) else None
            )
            raw_total = raw_usage.get("total_tokens") if isinstance(raw_usage, dict) else None
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
            _log_event(
                "gemini_usage_coerce",
                raw_type=type(raw_usage).__name__,
                raw_prompt=raw_prompt,
                raw_completion=raw_completion,
                raw_total=raw_total,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
            return data

        return _wrapped

    try:
        from any_llm.providers.gemini import utils as gem_utils
    except Exception:
        _log_event("gemini_usage_patch_skip", reason="missing_gemini_utils")
        yield
        return

    original_utils = getattr(gem_utils, "_convert_response_to_response_dict", None)
    if callable(original_utils):
        gem_utils._convert_response_to_response_dict = _wrap(original_utils)
        patches.append((gem_utils, original_utils))
        _log_event("gemini_usage_patch_apply", target="utils")

    try:
        from any_llm.providers.gemini import base as gem_base
    except Exception:
        gem_base = None

    if gem_base is not None:
        original_base = getattr(gem_base, "_convert_response_to_response_dict", None)
        if callable(original_base):
            gem_base._convert_response_to_response_dict = _wrap(original_base)
            patches.append((gem_base, original_base))
            _log_event("gemini_usage_patch_apply", target="base")

    if not patches:
        _log_event("gemini_usage_patch_skip", reason="no_patch_targets")
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
        _log_event("gemini_usage_patch_restore", patches=len(patches))


def build_prompt(
    prompt_template: str, resume_json: str, job_req: str
) -> str:
    return (
        prompt_template
        + "\n\nCandidate Resume (JSON):\n"
        + resume_json
        + "\n\nJob Requisition:\n"
        + job_req
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe any-llm JSON output.")
    parser.add_argument(
        "--model",
        default="gemini:gemini-3-pro-preview",
        help="Model spec (provider:model).",
    )
    parser.add_argument(
        "--prompt",
        default="prompt.yaml",
        help="Path to prompt.yaml.",
    )
    parser.add_argument(
        "--req",
        default="req.txt",
        help="Path to req.txt.",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Optional path to a resume JSON file (defaults to {}).",
    )
    parser.add_argument(
        "--stop-list",
        default="stop-list.txt",
        help="Path to stop-list.txt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max output tokens (defaults to provider config).",
    )
    parser.add_argument(
        "--no-force-json",
        action="store_true",
        help="Disable response_format json_object on the first attempt.",
    )
    parser.add_argument(
        "--raw-out",
        default="",
        help="Optional path to save raw LLM output.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to save parsed JSON output.",
    )
    args = parser.parse_args()

    global RUN_ID
    RUN_ID = uuid.uuid4().hex[:8]
    start_perf = time.perf_counter()
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    prompt_path = Path(args.prompt)
    req_path = Path(args.req)
    resume_path = Path(args.resume) if args.resume else None
    stop_list_path = Path(args.stop_list)

    _log_event(
        "start",
        timestamp=start_utc,
        pid=os.getpid(),
        cwd=os.getcwd(),
        python=sys.version.replace("\n", " "),
        platform=platform.platform(),
    )
    _log_event("argv", argv=" ".join(sys.argv))
    _log_event(
        "paths",
        model=args.model,
        prompt=str(prompt_path),
        req=str(req_path),
        resume="{}" if resume_path is None else str(resume_path),
        stop_list=str(stop_list_path),
    )
    _log_event(
        "options",
        max_tokens=args.max_tokens,
        force_json=not args.no_force_json,
        raw_out=args.raw_out or "none",
        json_out=args.json_out or "none",
    )

    prompt_load_start = time.perf_counter()
    prompt_template = _load_text(prompt_path)
    _log_event(
        "prompt_loaded",
        chars=len(prompt_template),
        lines=_count_lines(prompt_template),
        duration_ms=int((time.perf_counter() - prompt_load_start) * 1000),
    )
    stop_terms = _load_stop_list(stop_list_path)
    _log_event("stop_list_loaded", terms=len(stop_terms), path=str(stop_list_path))
    prompt_text, injected, inject_mode = _inject_stop_list(prompt_template, stop_terms)
    _log_event(
        "stop_list_injected",
        injected=injected,
        mode=inject_mode,
        delta_chars=len(prompt_text) - len(prompt_template),
    )

    if resume_path is None:
        resume_json = "{}"
        _log_event("resume_default", chars=len(resume_json))
    else:
        resume_load_start = time.perf_counter()
        resume_raw = _load_text(resume_path)
        _log_event(
            "resume_loaded",
            chars=len(resume_raw),
            lines=_count_lines(resume_raw),
            duration_ms=int((time.perf_counter() - resume_load_start) * 1000),
        )
        try:
            resume_obj = json.loads(resume_raw)
            resume_json = json.dumps(resume_obj, indent=2)
            _log_event(
                "resume_parse_ok",
                obj_type=type(resume_obj).__name__,
                keys=len(resume_obj) if isinstance(resume_obj, dict) else "n/a",
            )
        except Exception as exc:
            _log_exception("resume_parse_fail", exc)
            resume_json = resume_raw.strip()
            _log_event("resume_parse_fallback", chars=len(resume_json))

    req_load_start = time.perf_counter()
    req_text = _load_text(req_path)
    _log_event(
        "req_loaded",
        chars=len(req_text),
        lines=_count_lines(req_text),
        duration_ms=int((time.perf_counter() - req_load_start) * 1000),
    )
    prompt = build_prompt(prompt_text, resume_json, req_text)
    _log_event(
        "prompt_built",
        chars=len(prompt),
        lines=_count_lines(prompt),
        sha256=_sha256_text(prompt),
        resume_chars=len(resume_json),
        req_chars=len(req_text),
    )

    provider, model = _split_model_spec(args.model)
    _log_event("model_split", provider=provider, model=model)
    _log_event("api_key_lookup_start", provider=provider)
    if provider == "gemini":
        api_key = load_gemini_api_key()
    elif provider == "openai":
        api_key = load_openai_api_key()
    else:
        _log_event("unsupported_provider", provider=provider)
        return 2
    if not api_key:
        _log_event("api_key_missing", provider=provider)
        return 2

    if args.max_tokens > 0:
        max_tokens = args.max_tokens
        max_tokens_source = "cli"
    else:
        max_tokens, max_tokens_source = _resolve_max_tokens(provider)
    _log_event("max_tokens_resolved", value=max_tokens, source=max_tokens_source)

    llm = None
    resp = None
    content = ""
    llm_create_start = time.perf_counter()
    try:
        llm = AnyLLM.create(provider, api_key=api_key)
        _log_event(
            "llm_created",
            duration_ms=int((time.perf_counter() - llm_create_start) * 1000),
            llm_type=type(llm).__name__,
        )
    except Exception as exc:
        _log_exception("llm_create_failed", exc)
        return 1
    patch_ctx = _patch_gemini_usage() if provider == "gemini" else nullcontext()
    try:
        request: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if not args.no_force_json:
            request["response_format"] = {"type": "json_object"}
        _log_event(
            "request_prepared",
            model=model,
            max_tokens=max_tokens,
            force_json="response_format" in request,
            message_count=len(request["messages"]),
            message_chars=len(prompt),
        )
        with patch_ctx:
            attempt = 1
            while True:
                attempt_start = time.perf_counter()
                _log_event(
                    "completion_attempt_start",
                    attempt=attempt,
                    force_json="response_format" in request,
                )
                try:
                    loop = _ensure_open_event_loop()
                    _log_event(
                        "completion_loop_state",
                        attempt=attempt,
                        loop_id=id(loop),
                        running=loop.is_running(),
                        closed=loop.is_closed(),
                    )
                    resp = llm.completion(**request)
                    _log_event(
                        "completion_attempt_ok",
                        attempt=attempt,
                        duration_ms=int((time.perf_counter() - attempt_start) * 1000),
                    )
                    break
                except Exception as exc:
                    _log_exception("completion_attempt_error", exc)
                    _log_event(
                        "completion_attempt_failed",
                        attempt=attempt,
                        duration_ms=int((time.perf_counter() - attempt_start) * 1000),
                    )
                    if not args.no_force_json and attempt == 1 and "response_format" in request:
                        _log_event("completion_retry", reason="force_json_failed")
                        request.pop("response_format", None)
                        attempt += 1
                        continue
                    raise
        _log_event("response_received", **_describe_response(resp))
        usage_info = _usage_summary(resp)
        if usage_info:
            _log_event("response_usage", **usage_info)
        choices = getattr(resp, "choices", None)
        if isinstance(choices, (list, tuple)) and choices:
            choice0 = choices[0]
        else:
            choice0 = None
        _log_event("response_choice0", type=type(choice0).__name__)
        msg = getattr(choice0, "message", None) if choice0 is not None else None
        if msg is None:
            _log_event("response_message_missing")
        else:
            _log_event(
                "response_message",
                role=getattr(msg, "role", None),
                content_type=type(getattr(msg, "content", None)).__name__,
                finish_reason=getattr(choice0, "finish_reason", None),
            )
        content = _coerce_llm_text(getattr(msg, "content", None) if msg is not None else None)
        _log_event(
            "response_content_coerced",
            chars=len(content),
            lines=_count_lines(content),
        )
    except (MissingApiKeyError, UnsupportedProviderError) as exc:
        _log_exception("llm_error", exc)
        return 2
    except Exception as exc:
        _log_exception("llm_call_failed", exc)
        return 1
    finally:
        if llm is not None:
            _close_llm_client(llm)
        _log_event(
            "probe_finished",
            duration_ms=int((time.perf_counter() - start_perf) * 1000),
        )

    content = (content or "").strip()
    _log_event("response_trimmed", chars=len(content), lines=_count_lines(content))
    if not content:
        _log_event("response_empty")
        return 1

    if args.raw_out:
        Path(args.raw_out).write_text(content, encoding="utf-8")
        raw_size = Path(args.raw_out).stat().st_size
        _log_event("raw_saved", path=args.raw_out, bytes=raw_size)

    snippet = content[:400].replace("\n", "\\n")
    _log_event("response_snippet", snippet=snippet)
    try:
        parsed = _extract_json_object(content)
    except Exception as exc:
        _log_exception("json_parse_failed", exc)
        return 1

    _log_event(
        "json_parse_ok",
        keys=len(parsed),
        key_sample=_keys_sample(parsed),
    )
    if args.json_out:
        json_payload = json.dumps(parsed, indent=2, sort_keys=True)
        Path(args.json_out).write_text(json_payload, encoding="utf-8")
        json_size = Path(args.json_out).stat().st_size
        _log_event("json_saved", path=args.json_out, bytes=json_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
