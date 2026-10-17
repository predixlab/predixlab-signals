"""Publishing: hand the selected pick list to MongoDB and to the mailbox.

The dashboard reads MongoDB, not the parquet artifacts, so this is the boundary
between research output and product surface. Only the *selected* picks cross it;
the full scored cross-section stays in the artifacts tree.
"""

from predixlab_signals.publish.mongo import (
    AdviceStore,
    pick_counts,
    pick_documents,
)
from predixlab_signals.publish.notification import send_email_notification
from predixlab_signals.publish.campaign import enqueue_daily_signals_campaign

__all__ = [
    "AdviceStore",
    "enqueue_daily_signals_campaign",
    "pick_documents",
    "pick_counts",
    "send_email_notification",
]
