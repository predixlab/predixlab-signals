"""Email notification for the nightly pivot job.

Mirrors the notification other PredixLab workers send so a single mailbox rule
covers the whole platform: same subject convention, same table, same footer
position. The pivot-specific part is the action histogram and the top picks,
because "the job succeeded" is not the interesting part of a signal run.

``azure-communication-email`` is imported lazily: an unconfigured or
driver-less environment must degrade to "no email", never to a crash.
"""

from __future__ import annotations

from typing import Any

from predixlab_signals.settings import Settings, load_settings

_STATUS_COLOURS = {"success": "#1f9d68", "skipped": "#c08a1e", "failed": "#c2413c"}
_STATUS_MARKS = {"success": "✅", "skipped": "⏭️", "failed": "❌"}


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"


def _row(label: str, value: Any) -> str:
    return (
        '<tr><td style="padding:6px 12px 6px 0;color:#5a6270;white-space:nowrap;">'
        f"<strong>{label}</strong></td>"
        f'<td style="padding:6px 0;color:#1d2430;">{value}</td></tr>'
    )


def _counts_table(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    order = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    ordered = [k for k in order if k in counts] + [k for k in counts if k not in order]
    cells = "".join(
        f'<td style="padding:8px 14px;text-align:center;border:1px solid #e2e5ea;">'
        f'<div style="font-size:0.75em;color:#5a6270;letter-spacing:.04em;">{key}</div>'
        f'<div style="font-size:1.3em;font-weight:600;color:#1d2430;">{counts[key]:,}</div></td>'
        for key in ordered
    )
    return (
        '<h3 style="margin:24px 0 8px;font-size:1em;color:#1d2430;">Recommendations</h3>'
        f'<table style="border-collapse:collapse;"><tr>{cells}</tr></table>'
    )


def _picks_table(title: str, picks: list[dict[str, Any]]) -> str:
    if not picks:
        return ""
    header = "".join(
        f'<th style="padding:6px 10px;text-align:left;font-size:0.78em;color:#5a6270;'
        f'border-bottom:1px solid #e2e5ea;">{label}</th>'
        for label in ("Symbol", "Action", "Score", "Strength", "Price")
    )
    rows = ""
    for pick in picks:
        def cell(value: Any, decimals: int = 2) -> str:
            if value is None:
                return "—"
            if isinstance(value, str):
                return value
            return f"{value:,.{decimals}f}"

        rows += (
            '<tr>'
            f'<td style="padding:6px 10px;font-weight:600;">{pick.get("symbol", "")}</td>'
            f'<td style="padding:6px 10px;">{pick.get("action", "")}</td>'
            f'<td style="padding:6px 10px;">{cell(pick.get("score"), 4)}</td>'
            f'<td style="padding:6px 10px;">{cell(pick.get("confidence"), 3)}</td>'
            f'<td style="padding:6px 10px;">{cell(pick.get("price"))}</td>'
            '</tr>'
        )
    return (
        f'<h3 style="margin:24px 0 8px;font-size:1em;color:#1d2430;">{title}</h3>'
        f'<table style="border-collapse:collapse;width:100%;font-size:0.9em;">'
        f"<tr>{header}</tr>{rows}</table>"
    )


def build_email_body(job_status: dict[str, Any]) -> tuple[str, str]:
    """Return ``(subject, html_body)`` for a finished run."""
    status = str(job_status.get("status", "unknown")).lower()
    display = job_status.get("job_display_name") or job_status.get("job_name") or "PredixLab Signals"
    colour = _STATUS_COLOURS.get(status, "#5a6270")
    mark = _STATUS_MARKS.get(status, "⚠️")
    subject = f"{mark} {display} - {status.capitalize()}"

    results = job_status.get("results") or {}
    rows = [
        _row("Job", f'{display} <span style="color:#8b93a1;">({job_status.get("job_name", "")})</span>'),
        _row("Status", f'<span style="color:{colour};font-weight:600;">{status.upper()}</span>'),
        _row("Market", str(results.get("market", "—")).upper()),
        _row("As of", results.get("date") or "—"),
        _row("Symbols scored", f'{results.get("symbols_scored", 0):,}'),
        _row("Start", job_status.get("start_time", "—")),
        _row("End", job_status.get("end_time", "—")),
        _row("Duration", _format_duration(float(job_status.get("duration_seconds") or 0))),
    ]
    if results.get("model"):
        model = results["model"]
        rows.append(
            _row(
                "Model",
                f'trained {model.get("trained_at", "?")} · '
                f'{model.get("rows", 0):,} rows · {model.get("features", 0)} features'
                + (" · retrained this run" if model.get("retrained") else ""),
            )
        )
    if job_status.get("error"):
        rows.append(_row("Error", f'<span style="color:#c2413c;">{job_status["error"]}</span>'))
    if job_status.get("warning"):
        rows.append(_row("Warning", f'<span style="color:#c08a1e;">{job_status["warning"]}</span>'))

    body = f"""
    <html>
      <body style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
                   line-height:1.5;color:#1d2430;background:#f6f7f9;padding:24px;">
        <div style="max-width:760px;margin:0 auto;background:#ffffff;border:1px solid #e2e5ea;
                    border-radius:10px;padding:28px;">
          <h2 style="margin:0 0 4px;color:{colour};font-size:1.25em;">{subject}</h2>
          <p style="margin:0 0 20px;color:#5a6270;font-size:0.9em;">
            Swing-point advice for the latest session.
          </p>
          <table style="border-collapse:collapse;width:100%;">{"".join(rows)}</table>
          {_counts_table(results.get("counts") or {})}
          {_picks_table("Top buy candidates", results.get("top_buys") or [])}
          {_picks_table("Top sell / avoid candidates", results.get("top_sells") or [])}
          <p style="color:#8b93a1;font-size:0.82em;margin-top:28px;">
            Automated notification from PredixLab Signals. Levels are model-derived and
            ATR-bounded; they are not investment advice.
          </p>
        </div>
      </body>
    </html>
    """
    return subject, body


def send_email_notification(
    job_status: dict[str, Any], settings: Settings | None = None
) -> bool:
    """Send the run summary. Returns ``False`` when email is not configured."""
    settings = settings or load_settings()
    if not settings.email_configured:
        _log_info("Azure Communication Service not configured, skipping email notification")
        return False

    subject, html_body = build_email_body(job_status)
    try:
        from azure.communication.email import EmailClient

        client = EmailClient.from_connection_string(settings.acs_connection_string)
        message = {
            "senderAddress": settings.acs_sender_email,
            "recipients": {"to": [{"address": address} for address in settings.recipients]},
            "content": {"subject": subject, "html": html_body},
        }
        result = client.begin_send(message).result()
        _log_info(f"Email notification sent. Message ID: {result['id']}")
        return True
    except Exception as exc:
        _log_warning(f"Failed to send email notification: {exc}")
        return False


def _log_info(message: str) -> None:
    try:
        from loguru import logger

        logger.info(message)
    except ImportError:  # pragma: no cover
        import logging

        logging.getLogger(__name__).info(message)


def _log_warning(message: str) -> None:
    try:
        from loguru import logger

        logger.warning(message)
    except ImportError:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(message)
