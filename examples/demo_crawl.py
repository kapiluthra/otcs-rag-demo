"""
examples/demo_crawl.py — Walk an OTCS node tree and print document metadata

Use this to verify your OTCS connection and understand the structure
of your repository before running full ingestion.

Usage:
    python examples/demo_crawl.py

Set environment variables (or copy .env.example to .env):
    OTCS_BASE_URL   e.g. https://your-otcs.example.com/cs/cs
    OTCS_USERNAME
    OTCS_PASSWORD
    OTCS_ROOT_NODE  node ID to start crawling from (e.g. 2000 for Enterprise Workspace)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cs_client import CSClient

def main():
    base_url   = os.environ["OTCS_BASE_URL"]
    username   = os.environ["OTCS_USERNAME"]
    password   = os.environ["OTCS_PASSWORD"]
    root_node  = int(os.environ.get("OTCS_ROOT_NODE", "2000"))

    print(f"Connecting to: {base_url}")
    print(f"Root node: {root_node}\n")

    client = CSClient(base_url=base_url, username=username, password=password)

    # Verify auth
    try:
        _ = client.ticket
        print("✓ Authentication successful\n")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return

    # Walk nodes and print first 50 documents
    print(f"{'ID':<12} {'Type':<30} {'Name':<50} {'Modified'}")
    print("-" * 110)

    count = 0
    for node in client.walk_nodes(root_node):
        props = client.get_properties(node)
        node_id    = props.get("id", "")
        name       = props.get("name", "")[:48]
        mime_type  = props.get("mime_type", "")[:28]
        modify     = props.get("modify_date", "")[:19]

        print(f"{node_id:<12} {mime_type:<30} {name:<50} {modify}")
        count += 1

        if count >= 50:
            print(f"\n... (stopped at 50 — remove limit to see all)")
            break

    print(f"\nTotal documents found: {count}{'+ (limited)' if count == 50 else ''}")


if __name__ == "__main__":
    main()
