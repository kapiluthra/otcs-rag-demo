"""
cs_client.py — OpenText Content Server REST API Client

Key lessons baked into this implementation:
  1. Children endpoint returns results under "results" key, not "data"
  2. Container detection uses the "container" flag, not node_type == 0
  3. OTCSTicket expires — auto-renew before expiry to avoid mid-crawl failures

Full writeup: https://kapiluthra.github.io/blog-cs-client-debugging.html
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import requests
import urllib3

# Suppress InsecureRequestWarning for self-signed OTCS certs in dev
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


@dataclass
class CSClient:
    """Authenticated client for the OpenText Content Server REST API v2.

    Usage:
        client = CSClient(
            base_url="https://your-otcs.example.com/cs/cs",
            username="admin",
            password="secret"
        )
        for doc_node in client.walk_nodes(root_id=12345):
            content = client.download(doc_node["data"]["properties"]["id"])
    """

    base_url: str
    username: str
    password: str
    verify_ssl: bool = False           # Set True for production with valid cert
    ticket_ttl: int = 1700             # Renew 100s before the default 30-min expiry
    _ticket: Optional[str] = field(default=None, init=False, repr=False)
    _ticket_time: float = field(default=0.0, init=False, repr=False)

    # ── Authentication ──────────────────────────────────────────────────────

    def _fetch_ticket(self) -> str:
        """Authenticate and return a fresh OTCSTicket."""
        resp = requests.post(
            f"{self.base_url}/api/v1/auth",
            data={"username": self.username, "password": self.password},
            verify=self.verify_ssl,
            timeout=30,
        )
        resp.raise_for_status()
        ticket = resp.json().get("ticket")
        if not ticket:
            raise ValueError("Auth response did not contain 'ticket' field")
        logger.debug("OTCSTicket refreshed")
        return ticket

    @property
    def ticket(self) -> str:
        """Return a valid OTCSTicket, refreshing automatically when near expiry."""
        if (time.time() - self._ticket_time) > self.ticket_ttl:
            self._ticket = self._fetch_ticket()
            self._ticket_time = time.time()
        return self._ticket

    # ── HTTP helpers ────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {"OTCSTicket": self.ticket}

    def get(self, path: str, **kwargs) -> requests.Response:
        return requests.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            verify=self.verify_ssl,
            timeout=60,
            **kwargs,
        )

    # ── Node operations ─────────────────────────────────────────────────────

    def get_node(self, node_id: int) -> dict:
        """Fetch a single node by ID.

        Response structure: {"data": {"properties": {...}}}
        """
        resp = self.get(f"/api/v2/nodes/{node_id}")
        resp.raise_for_status()
        return resp.json()

    def get_children(self, node_id: int, page_size: int = 500) -> list:
        """Return all child nodes of a container.

        IMPORTANT: Children endpoint returns nodes under the "results" key,
        NOT "data". This is inconsistent with the single-node endpoint.

        Response structure: {"results": [{"data": {"properties": {...}}}, ...]}
        """
        children = []
        page = 1
        while True:
            resp = self.get(
                f"/api/v2/nodes/{node_id}/nodes",
                params={"page": page, "limit": page_size},
            )
            resp.raise_for_status()
            data = resp.json()

            # Use "results" — NOT "data" (common mistake, "data" is always empty here)
            batch = data.get("results", [])
            children.extend(batch)

            # Paginate if more results exist
            paging = data.get("collection", {}).get("paging", {})
            if len(batch) < page_size or not paging.get("next_page"):
                break
            page += 1

        return children

    def is_container(self, node: dict) -> bool:
        """Check whether a node is a container (folder, workspace, etc.).

        IMPORTANT: Check the "container" boolean flag, NOT node_type == 0.
        Container node types include:
          0   = Folder
          848 = Enterprise Workspace
          136 = Compound Document
          751 = Project
          ...and others

        The "container" flag covers all of them reliably.
        """
        return (
            node.get("data", {})
            .get("properties", {})
            .get("container", False)
        )

    def get_properties(self, node: dict) -> dict:
        """Extract the properties dict from a node response."""
        return node.get("data", {}).get("properties", {})

    def walk_nodes(self, root_id: int) -> Iterator[dict]:
        """Recursively yield all document (non-container) nodes under root_id.

        Uses an iterative DFS stack to avoid recursion limits on deep trees.
        Skips nodes where get_children fails (logs warning and continues).

        Yields:
            node dict with structure {"data": {"properties": {...}}}
        """
        stack = [root_id]
        visited = set()

        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)

            try:
                children = self.get_children(node_id)
            except requests.HTTPError as e:
                logger.warning("Failed to get children of node %d: %s", node_id, e)
                continue

            for node in children:
                props = self.get_properties(node)
                child_id = props.get("id")

                if self.is_container(node):
                    if child_id and child_id not in visited:
                        stack.append(child_id)
                else:
                    yield node

    # ── Content download ─────────────────────────────────────────────────────

    def download_content(self, node_id: int, version: int = None) -> tuple[bytes, str]:
        """Download binary content of a document node.

        Returns:
            (content_bytes, mime_type)
        """
        if version:
            path = f"/api/v2/nodes/{node_id}/versions/{version}/content"
        else:
            path = f"/api/v2/nodes/{node_id}/content"

        resp = self.get(path, stream=True)
        resp.raise_for_status()

        mime_type = resp.headers.get("Content-Type", "application/octet-stream")
        content = resp.content
        return content, mime_type

    def get_modify_date(self, node: dict) -> str:
        """Extract the OTModifyDate for incremental sync comparisons."""
        props = self.get_properties(node)
        return props.get("modify_date", "")

    # ── Permissions (for ACL security gates) ─────────────────────────────────

    def check_user_access(self, node_id: int, username: str) -> bool:
        """Check whether a user has read access to a node.

        In production, call this before returning any chunk to an LLM.
        Cache results per (user, node_id) with a short TTL (5–15 min).
        """
        try:
            resp = self.get(f"/api/v2/nodes/{node_id}/permissions/effective")
            resp.raise_for_status()
            perms = resp.json().get("data", {})
            # "see" permission is the minimum needed to know the node exists
            # "read" permission is needed to access content
            return perms.get("permissions", {}).get("read", False)
        except requests.HTTPError:
            # If permission check fails, deny access (fail closed)
            return False
