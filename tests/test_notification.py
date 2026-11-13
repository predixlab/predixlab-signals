"""Resend notifications use mocked transport and isolated email configuration."""

from __future__ import annotations

import sys
from dataclasses import replace
from unittest.mock import Mock

import pytest

from predixlab_signals.publish import notification
from predixlab_signals.settings import Settings, load_settings


@pytest.fixture(autouse=True)
def clean_email_environment(monkeypatch):
    for name in (
        "RESEND_API_KEY",
        "EMAIL_FROM",
        "ACS_CONNECTION_STRING",
        "ACS_SENDER_EMAIL",
        "ACS_TO_EMAILS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def email_settings():
    return Settings(
        resend_api_key="test-api-key",
        email_from="signals@example.com",
        acs_sender_email="legacy@example.com",
        acs_to_emails="first@example.com, , second@example.com ",
        campaign_events_enabled=False,
    )


@pytest.fixture
def resend_send(monkeypatch):
    import resend

    send = Mock(return_value={"id": "test-message-id"})
    monkeypatch.setattr(resend, "api_key", "")
    monkeypatch.setattr(resend.Emails, "send", send)
    return send


def test_notification_not_configured_skips_without_driver(monkeypatch, email_settings):
    monkeypatch.setitem(sys.modules, "resend", None)
    log_info = Mock()
    monkeypatch.setattr(notification, "_log_info", log_info)

    assert notification.send_email_notification(
        {"status": "success"}, replace(email_settings, resend_api_key="")
    ) is False
    log_info.assert_called_once_with(
        "Resend not configured, skipping email notification"
    )


@pytest.mark.parametrize("email_from", ["signals@example.com", ""])
def test_notification_sends_expected_email(
    monkeypatch, email_settings, resend_send, email_from
):
    import resend

    settings = replace(email_settings, email_from=email_from)
    job_status = {
        "status": "success",
        "job_name": "predixlab_signals",
        "job_display_name": "PredixLab Signals",
        "results": {"market": "us", "date": "2026-11-12", "symbols_scored": 12},
    }
    subject, html_body = notification.build_email_body(job_status)
    log_info = Mock()
    monkeypatch.setattr(notification, "_log_info", log_info)

    assert notification.send_email_notification(job_status, settings) is True

    assert resend.api_key == "test-api-key"
    assert subject == "\u2705 PredixLab Signals - Success"
    resend_send.assert_called_once_with(
        {
            "from": email_from or "legacy@example.com",
            "to": ["first@example.com", "second@example.com"],
            "subject": subject,
            "html": html_body,
        }
    )
    log_info.assert_called_once_with(
        "Email notification sent. Message ID: test-message-id"
    )


def test_notification_send_exception_returns_false(
    monkeypatch, email_settings, resend_send
):
    resend_send.side_effect = RuntimeError("Resend unavailable")
    log_warning = Mock()
    monkeypatch.setattr(notification, "_log_warning", log_warning)

    assert notification.send_email_notification(
        {"status": "failed"}, email_settings
    ) is False
    resend_send.assert_called_once()
    log_warning.assert_called_once_with(
        "Failed to send email notification: Resend unavailable"
    )


def test_notification_missing_driver_returns_false(monkeypatch, email_settings):
    monkeypatch.setitem(sys.modules, "resend", None)
    log_warning = Mock()
    monkeypatch.setattr(notification, "_log_warning", log_warning)

    assert notification.send_email_notification(
        {"status": "success"}, email_settings
    ) is False
    log_warning.assert_called_once()
    assert "Failed to send email notification:" in log_warning.call_args.args[0]
    assert "resend" in log_warning.call_args.args[0]


@pytest.mark.parametrize(
    "api_key,email_from,legacy_sender,recipients,configured",
    [
        ("test-key", "signals@example.com", "", "to@example.com", True),
        ("test-key", "", "legacy@example.com", "to@example.com", True),
        ("", "signals@example.com", "legacy@example.com", "to@example.com", False),
        ("test-key", "", "", "to@example.com", False),
        ("test-key", "signals@example.com", "", " , ", False),
    ],
)
def test_settings_email_configured_requires_resend_fields(
    api_key, email_from, legacy_sender, recipients, configured
):
    settings = Settings(
        resend_api_key=api_key,
        email_from=email_from,
        acs_connection_string="deprecated-and-unused",
        acs_sender_email=legacy_sender,
        acs_to_emails=recipients,
        campaign_events_enabled=False,
    )

    assert settings.email_configured is configured


@pytest.mark.parametrize("email_from", ["signals@example.com", "", "   "])
def test_settings_email_environment_and_sender_fallback(
    monkeypatch, resend_send, email_from
):
    monkeypatch.setenv("RESEND_API_KEY", " test-api-key ")
    monkeypatch.setenv("EMAIL_FROM", email_from)
    monkeypatch.setenv("ACS_SENDER_EMAIL", " legacy@example.com ")
    monkeypatch.setenv("ACS_TO_EMAILS", " first@example.com, , second@example.com ")
    monkeypatch.setenv("CAMPAIGN_EVENTS_ENABLED", "false")
    settings = load_settings()

    assert settings.resend_api_key == "test-api-key"
    assert settings.email_from == email_from.strip()
    assert settings.acs_connection_string == ""
    assert settings.email_configured is True
    assert settings.recipients == ["first@example.com", "second@example.com"]
    assert notification.send_email_notification({"status": "success"}) is True
    assert resend_send.call_args.args[0]["from"] == (
        email_from.strip() or "legacy@example.com"
    )
