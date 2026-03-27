"""Event-centered candidate extraction for dual-IMU sections."""


def __getattr__(name):
    if name in {"EventConfig", "extract_event_candidates_section", "extract_events_from_args", "load_event_windows"}:
        from .extract import EventConfig, extract_event_candidates_section, extract_events_from_args, load_event_windows

        return {
            "EventConfig": EventConfig,
            "extract_event_candidates_section": extract_event_candidates_section,
            "extract_events_from_args": extract_events_from_args,
            "load_event_windows": load_event_windows,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EventConfig",
    "extract_event_candidates_section",
    "extract_events_from_args",
    "load_event_windows",
]
