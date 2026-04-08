from __future__ import annotations

import json
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert common benchmark payload values into JSON-safe types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return str(value)


def native_state_label(
    *,
    available: bool,
    imported: bool,
    used_native: bool = False,
    fallback_triggered: bool = False,
    selected_method: str | None = None,
    execution_state: str | None = None,
) -> str:
    if execution_state:
        return execution_state
    if used_native:
        return "native-used"
    if fallback_triggered:
        return "native-fallback"
    if selected_method == "direct":
        return "python-selected"
    if available and imported:
        return "native-ready"
    if available:
        return "available-not-imported"
    return "unavailable"


def build_module_state(*, available: bool, imported_module: Any, execution_info: dict[str, Any] | None = None) -> dict[str, Any]:
    module_path = getattr(imported_module, "__file__", None) if imported_module is not None else None
    module_name = getattr(imported_module, "__name__", None) if imported_module is not None else None
    used_native = bool((execution_info or {}).get("used_native", False))
    fallback_triggered = bool((execution_info or {}).get("fallback_triggered", False))
    selected_method = (execution_info or {}).get("selected_method")
    execution_state = (execution_info or {}).get("execution_state")

    return {
        "available": bool(available),
        "imported": imported_module is not None,
        "state": native_state_label(
            available=bool(available),
            imported=imported_module is not None,
            used_native=used_native,
            fallback_triggered=fallback_triggered,
            selected_method=selected_method if isinstance(selected_method, str) else None,
            execution_state=execution_state if isinstance(execution_state, str) else None,
        ),
        "selected_method": selected_method if isinstance(selected_method, str) else None,
        "module": module_name,
        "module_path": module_path,
        "execution_info": to_jsonable(execution_info) if execution_info is not None else None,
    }


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(to_jsonable(payload), indent=2, sort_keys=True)
