"""Registry for independently deployable signal models."""

from __future__ import annotations

from predixlab_signals.signals.specs import SignalModelSpec

_MODELS: dict[str, SignalModelSpec] = {}


def register_signal_model(model: SignalModelSpec) -> SignalModelSpec:
    """Register a model by its stable name."""
    existing = _MODELS.get(model.name)
    if existing is not None and existing is not model:
        raise ValueError(f"Signal model {model.name!r} is already registered")
    _MODELS[model.name] = model
    return model


def get_signal_model(name: str) -> SignalModelSpec:
    """Resolve a registered model, loading built-ins on first use."""
    if name not in _MODELS:
        from predixlab_signals.signals.models.pivot.adapter import PIVOT_SIGNAL_MODEL

        register_signal_model(PIVOT_SIGNAL_MODEL)
    try:
        return _MODELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODELS)) or "none"
        raise ValueError(
            f"Unknown signal model {name!r}; registered models: {available}"
        ) from exc


def list_signal_models() -> tuple[str, ...]:
    """Return registered model names in deterministic order."""
    get_signal_model("pivot")
    return tuple(sorted(_MODELS))

