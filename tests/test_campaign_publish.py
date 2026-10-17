from __future__ import annotations

import json

from predixlab_signals.publish.campaign import (
    build_daily_signals_campaign_event,
    enqueue_daily_signals_campaign,
)
from predixlab_signals.settings import Settings


class FakeQueueClient:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(self, message: str) -> None:
        self.messages.append(message)


def test_build_daily_signals_campaign_event_matches_campaign_contract() -> None:
    assert build_daily_signals_campaign_event(
        market="US",
        date="2026-08-18",
    ) == {
        "schema_version": 1,
        "type": "campaign.daily_signals",
        "campaign_id": "daily-signals:us:2026-08-18",
        "market": "us",
        "date": "2026-08-18",
        "metadata": {"source": "predixlab-signals"},
    }


def test_enqueue_daily_signals_campaign_uses_configured_queue() -> None:
    client = FakeQueueClient()
    calls: list[tuple[str, str]] = []
    settings = Settings(
        campaign_events_enabled=True,
        campaign_storage_connection_string="storage-connection",
        campaign_events_queue="campaign-events",
    )

    enqueue_daily_signals_campaign(
        "us",
        "2026-08-18",
        settings,
        queue_client_factory=lambda connection, queue: (
            calls.append((connection, queue)) or client
        ),
    )

    assert calls == [("storage-connection", "campaign-events")]
    assert json.loads(client.messages[0])["campaign_id"] == (
        "daily-signals:us:2026-08-18"
    )
