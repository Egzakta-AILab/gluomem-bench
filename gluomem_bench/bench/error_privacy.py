"""Categorical diagnostics for unrestricted external/native exception boundaries.

Never derive text from exception args, class names, notes, or response bodies.
Keep originals and their cause/traceback in memory; receipts are not tracebacks.
"""


def safe_kind(exc):
    """Return a static label only for exact built-in exception types."""
    for builtin, label in (
        (OSError, "OSError"),
        (TimeoutError, "TimeoutError"),
        (ConnectionError, "ConnectionError"),
        (ValueError, "ValueError"),
        (TypeError, "TypeError"),
        (RuntimeError, "RuntimeError"),
        (KeyError, "KeyError"),
        (IndexError, "IndexError"),
        (LookupError, "LookupError"),
    ):
        if type(exc) is builtin:
            return label
    return "Exception"


def safe_error(exc):
    """Keep legacy string fields without ever rendering an untrusted exception."""
    return safe_kind(exc) + ": operation_failed"


def safe_notes(exc):
    """Preserve cleanup-note presence/count, not arbitrary note contents."""
    try:
        notes = BaseException.__getattribute__(exc, "__notes__")
    except AttributeError:
        return []
    if type(notes) is not list:
        return []
    return ["cleanup_diagnostic_redacted" for _ in notes]
