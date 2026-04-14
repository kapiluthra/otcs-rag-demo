"""
sync.py — Incremental sync orchestrator

Crawls OTCS, identifies changed documents via OTModifyDate, and
re-ingests only those documents. Saves sync timestamp only on
full success — failed docs stay behind the watermark for retry.

Usage:
    python src/sync.py

Or as a scheduled job:
    # cron: every 6 hours
    0 */6 * * * cd /path/to/otcs-rag-demo && python src/sync.py

    # weekly reconciliation (deletion cleanup):
    0 2 * * 0 cd /path/to/otcs-rag-demo && python src/sync.py --reconcile
"""

import argparse
import logging
import os
import sys

from cs_client import CSClient
from extractor import extract_text
from chunker import chunk_document
from ingester import ingest_chunks, delete_doc_chunks, _get_collection
from sync_state import get_last_sync, save_sync_time, needs_update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_incremental_sync(client: CSClient, root_node_id: int) -> dict:
    """Run incremental sync — process only documents changed since last sync."""
    last_sync = get_last_sync()
    logger.info("Incremental sync starting. Last sync: %s", last_sync)

    stats = {"processed": 0, "skipped": 0, "failed": 0, "re_ingested": 0}

    for node in client.walk_nodes(root_node_id):
        props = node["data"]["properties"]
        doc_id = str(props["id"])
        doc_name = props.get("name", "unknown")

        if not needs_update(node, last_sync):
            stats["skipped"] += 1
            continue

        logger.info("Changed: %s (id=%s)", doc_name, doc_id)

        try:
            content, mime_type = client.download_content(int(doc_id))
            if not content:
                logger.debug("Empty content, skipping: %s", doc_name)
                stats["skipped"] += 1
                continue

            text = extract_text(content, mime_type)
            if not text.strip():
                logger.debug("No extractable text, skipping: %s", doc_name)
                stats["skipped"] += 1
                continue

            # Delete existing chunks before re-ingesting
            delete_doc_chunks(doc_id)

            chunks = chunk_document(
                text=text,
                doc_id=doc_id,
                doc_name=doc_name,
                modify_date=props.get("modify_date", ""),
                mime_type=mime_type,
            )
            ingest_chunks(chunks)
            stats["processed"] += 1
            stats["re_ingested"] += len(chunks)

        except Exception as e:
            logger.error("Failed to process %s (id=%s): %s", doc_name, doc_id, e)
            stats["failed"] += 1

    # Only advance timestamp if zero failures
    if stats["failed"] == 0:
        save_sync_time(buffer_minutes=5)
        logger.info("Sync complete ✓ — %s", stats)
    else:
        logger.warning(
            "Sync had %d failures — timestamp NOT advanced. "
            "Failed documents will be retried on next run. Stats: %s",
            stats["failed"], stats
        )

    return stats


def run_reconciliation(client: CSClient, root_node_id: int) -> int:
    """Remove stale chunks for documents deleted from OTCS.

    Run weekly — full walk to build live ID set, then purge orphans.
    """
    logger.info("Reconciliation starting — building live document set...")

    live_ids = set()
    for node in client.walk_nodes(root_node_id):
        props = node["data"]["properties"]
        live_ids.add(str(props["id"]))

    logger.info("Live documents in OTCS: %d", len(live_ids))

    collection = _get_collection()
    all_data = collection.get(include=["metadatas"])
    indexed_ids = {m["doc_id"] for m in all_data["metadatas"]}

    stale_ids = indexed_ids - live_ids
    logger.info("Stale documents to purge: %d", len(stale_ids))

    for doc_id in stale_ids:
        delete_doc_chunks(doc_id)
        logger.info("Purged: doc_id=%s", doc_id)

    return len(stale_ids)


def main():
    parser = argparse.ArgumentParser(description="OTCS RAG sync")
    parser.add_argument("--reconcile", action="store_true",
                        help="Run deletion reconciliation instead of incremental sync")
    args = parser.parse_args()

    client = CSClient(
        base_url=os.environ["OTCS_BASE_URL"],
        username=os.environ["OTCS_USERNAME"],
        password=os.environ["OTCS_PASSWORD"],
    )
    root_node_id = int(os.environ.get("OTCS_ROOT_NODE", "2000"))

    if args.reconcile:
        purged = run_reconciliation(client, root_node_id)
        logger.info("Reconciliation complete. Purged %d stale documents.", purged)
    else:
        stats = run_incremental_sync(client, root_node_id)
        if stats["failed"] > 0:
            sys.exit(1)  # Non-zero exit for monitoring/alerting


if __name__ == "__main__":
    main()
