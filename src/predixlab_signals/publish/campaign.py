"""Publish committed daily signal sessions to the campaign queue."""

from __future__ import annotations

import json
from typing import Any, Callable

from predixlab_signals.settings import Settings


def build_daily_signals_campaign_event(
    *,
    market: str,
    date: str,
) -> dict[str, Any]:
    """Build the campaign entry contract for one committed market session."""
    normalized_market = market.strip().lower()
    normalized_date = date.strip()
    return {
        "schema_version": 1,
        "type": "campaign.daily_signals",
        "campaign_id": f"daily-signals:{normalized_market}:{normalized_date}",
        "market": normalized_market,
        "date": normalized_date,
        "metadata": {"source": "predixlab-signals"},
    }


def enqueue_daily_signals_campaign(
    market: str,
    date: str,
    settings: Settings,
    *,
    queue_client_factory: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Write one daily-signals entry event to Azure Storage Queue."""
    if not settings.campaign_events_enabled:
        raise ValueError("Campaign event publishing is disabled")

    if queue_client_factory is None:
        from azure.storage.queue import QueueClient

        queue_client_factory = lambda connection_string, queue_name: (
            QueueClient.from_connection_string(
                connection_string,
                queue_name,
            )
        )

    payload = build_daily_signals_campaign_event(market=market, date=date)
    client = queue_client_factory(
        settings.campaign_storage_connection_string,
        settings.campaign_events_queue,
    )
    client.send_message(json.dumps(payload, ensure_ascii=False))
    return payload
