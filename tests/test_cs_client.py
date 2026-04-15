"""
tests/test_cs_client.py — Tests for CSClient with mocked OTCS responses

Tests cover:
- Authentication and ticket renewal
- Children endpoint uses "results" key (not "data")
- Container detection uses "container" flag (not node_type == 0)
- walk_nodes traverses correctly and yields only non-container nodes
- Pagination handling
- Error handling / graceful degradation
"""

import time
import pytest
from unittest.mock import MagicMock, patch, call
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from cs_client import CSClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_node(node_id: int, name: str, is_container: bool, node_type: int = 144) -> dict:
    """Build a minimal OTCS node response dict."""
    return {
        "data": {
            "properties": {
                "id": node_id,
                "name": name,
                "container": is_container,
                "node_type": node_type,
                "modify_date": "2026-01-01T00:00:00Z",
            }
        }
    }


def make_children_response(nodes: list) -> dict:
    """Build a children endpoint response (uses 'results' key)."""
    return {"results": nodes}


def make_single_node_response(node: dict) -> dict:
    """Build a single-node endpoint response (uses 'data' key directly)."""
    return node  # single-node endpoint returns the node directly


@pytest.fixture
def client():
    return CSClient(
        base_url="https://otcs.example.com/cs/cs",
        username="testuser",
        password="testpass",
    )


# ── Authentication tests ───────────────────────────────────────────────────────

class TestAuthentication:
    def test_fetches_ticket_on_first_access(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ticket": "test-ticket-123"}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            ticket = client.ticket
            assert ticket == "test-ticket-123"
            mock_post.assert_called_once()

    def test_reuses_valid_ticket(self, client):
        client._ticket = "cached-ticket"
        client._ticket_time = time.time()  # fresh

        with patch("requests.post") as mock_post:
            t1 = client.ticket
            t2 = client.ticket
            assert t1 == t2 == "cached-ticket"
            mock_post.assert_not_called()

    def test_refreshes_expired_ticket(self, client):
        client._ticket = "old-ticket"
        client._ticket_time = time.time() - 3600  # 1 hour ago, expired

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ticket": "new-ticket"}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp):
            ticket = client.ticket
            assert ticket == "new-ticket"

    def test_ticket_header_sent_in_requests(self, client):
        client._ticket = "my-ticket"
        client._ticket_time = time.time()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp) as mock_get:
            client.get("/api/v2/nodes/123")
            call_kwargs = mock_get.call_args
            assert call_kwargs[1]["headers"]["OTCSTicket"] == "my-ticket"


# ── Children endpoint tests ────────────────────────────────────────────────────

class TestGetChildren:
    def test_extracts_from_results_key(self, client):
        """THE KEY FIX: children endpoint uses 'results', not 'data'."""
        folder_node = make_node(101, "folder.pdf", is_container=False)
        mock_resp = MagicMock()
        mock_resp.json.return_value = make_children_response([folder_node])
        mock_resp.raise_for_status.return_value = None

        with patch.object(client, "get", return_value=mock_resp):
            children = client.get_children(12345)
            assert len(children) == 1
            assert children[0]["data"]["properties"]["id"] == 101

    def test_returns_empty_list_when_no_results(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status.return_value = None

        with patch.object(client, "get", return_value=mock_resp):
            children = client.get_children(12345)
            assert children == []

    def test_handles_missing_results_key_gracefully(self, client):
        """If the API returns neither 'results' nor 'data', return empty list."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}  # malformed response
        mock_resp.raise_for_status.return_value = None

        with patch.object(client, "get", return_value=mock_resp):
            children = client.get_children(12345)
            assert children == []

    def test_does_not_use_data_key_for_children(self, client):
        """Confirm we're NOT reading from 'data' key (old bug)."""
        mock_resp = MagicMock()
        # Put real data under "data" key (wrong) and nothing under "results"
        mock_resp.json.return_value = {
            "data": [make_node(999, "should-not-appear.pdf", False)]
        }
        mock_resp.raise_for_status.return_value = None

        with patch.object(client, "get", return_value=mock_resp):
            children = client.get_children(12345)
            assert children == []  # "data" key is ignored for children


# ── Container detection tests ─────────────────────────────────────────────────

class TestIsContainer:
    def test_detects_folder_via_flag(self, client):
        node = make_node(1, "folder", is_container=True, node_type=0)
        assert client.is_container(node) is True

    def test_detects_workspace_via_flag(self, client):
        """Workspace (node_type=848) must be detected as container via flag."""
        node = make_node(2, "workspace", is_container=True, node_type=848)
        assert client.is_container(node) is True

    def test_detects_compound_doc_via_flag(self, client):
        """Compound document (node_type=136) is a container."""
        node = make_node(3, "compound", is_container=True, node_type=136)
        assert client.is_container(node) is True

    def test_document_not_container(self, client):
        node = make_node(4, "document.pdf", is_container=False, node_type=144)
        assert client.is_container(node) is False

    def test_workspace_not_caught_by_node_type_0_check(self, client):
        """This is THE BUG: node_type==0 misses workspaces (848). The flag catches it."""
        workspace = make_node(5, "workspace", is_container=True, node_type=848)
        # Old broken check: would return False for workspace
        broken_check = workspace["data"]["properties"]["node_type"] == 0
        assert broken_check is False  # confirms the old check was broken

        # Correct check via flag
        assert client.is_container(workspace) is True

    def test_defaults_to_false_when_flag_missing(self, client):
        node = {"data": {"properties": {"id": 99, "name": "mystery"}}}
        assert client.is_container(node) is False


# ── walk_nodes tests ───────────────────────────────────────────────────────────

class TestWalkNodes:
    def test_yields_only_documents(self, client):
        """walk_nodes should yield documents, not containers."""
        folder = make_node(10, "Contracts", is_container=True)
        doc1 = make_node(11, "contract_a.pdf", is_container=False)
        doc2 = make_node(12, "contract_b.pdf", is_container=False)

        def mock_get_children(node_id):
            if node_id == 100:   # root
                return [folder, doc1]
            elif node_id == 10:  # folder
                return [doc2]
            return []

        with patch.object(client, "get_children", side_effect=mock_get_children):
            yielded = list(client.walk_nodes(100))

        yielded_ids = [n["data"]["properties"]["id"] for n in yielded]
        assert 11 in yielded_ids  # doc1 from root
        assert 12 in yielded_ids  # doc2 from nested folder
        assert 10 not in yielded_ids  # folder not yielded

    def test_traverses_nested_folders(self, client):
        """Nested folder structure: root → folder A → folder B → doc."""
        folder_a = make_node(20, "FolderA", is_container=True)
        folder_b = make_node(21, "FolderB", is_container=True)
        doc = make_node(22, "deep.pdf", is_container=False)

        def mock_get_children(node_id):
            return {100: [folder_a], 20: [folder_b], 21: [doc]}.get(node_id, [])

        with patch.object(client, "get_children", side_effect=mock_get_children):
            yielded = list(client.walk_nodes(100))

        assert len(yielded) == 1
        assert yielded[0]["data"]["properties"]["id"] == 22

    def test_skips_failed_nodes_gracefully(self, client):
        """A 404/403 on get_children should not abort the whole walk."""
        import requests as req_lib
        folder = make_node(30, "FailFolder", is_container=True)
        doc = make_node(31, "sibling.pdf", is_container=False)

        def mock_get_children(node_id):
            if node_id == 100:
                return [folder, doc]
            if node_id == 30:
                raise req_lib.HTTPError("403")
            return []

        with patch.object(client, "get_children", side_effect=mock_get_children):
            yielded = list(client.walk_nodes(100))

        # doc (sibling of failed folder) should still be yielded
        yielded_ids = [n["data"]["properties"]["id"] for n in yielded]
        assert 31 in yielded_ids

    def test_does_not_revisit_nodes(self, client):
        """walk_nodes should not enter the same node twice (prevents infinite loops)."""
        # Create a situation where the same folder appears in two places
        folder = make_node(40, "SharedFolder", is_container=True)
        doc = make_node(41, "doc.pdf", is_container=False)

        call_count = {"count": 0}
        def mock_get_children(node_id):
            if node_id == 100:
                return [folder, folder]  # same folder twice
            if node_id == 40:
                call_count["count"] += 1
                return [doc]
            return []

        with patch.object(client, "get_children", side_effect=mock_get_children):
            yielded = list(client.walk_nodes(100))

        assert call_count["count"] == 1  # folder visited only once
        assert len(yielded) == 1         # doc yielded only once


# ── Content download tests ────────────────────────────────────────────────────

class TestDownloadContent:
    def test_returns_content_and_mime_type(self, client):
        mock_resp = MagicMock()
        mock_resp.content = b"PDF content here"
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.raise_for_status.return_value = None

        with patch.object(client, "get", return_value=mock_resp):
            content, mime = client.download_content(12345)
            assert content == b"PDF content here"
            assert mime == "application/pdf"

    def test_handles_empty_content(self, client):
        mock_resp = MagicMock()
        mock_resp.content = b""
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_resp.raise_for_status.return_value = None

        with patch.object(client, "get", return_value=mock_resp):
            content, mime = client.download_content(999)
            assert content == b""


# ── ACL tests ─────────────────────────────────────────────────────────────────

class TestACLCheck:
    def test_returns_true_when_read_permitted(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"permissions": {"read": True}}}
        mock_resp.raise_for_status.return_value = None

        with patch.object(client, "get", return_value=mock_resp):
            assert client.check_user_access(12345, "testuser") is True

    def test_returns_false_when_read_denied(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"permissions": {"read": False}}}
        mock_resp.raise_for_status.return_value = None

        with patch.object(client, "get", return_value=mock_resp):
            assert client.check_user_access(12345, "testuser") is False

    def test_returns_false_on_http_error(self, client):
        """Fail closed: permission check error → deny access."""
        import requests as req_lib
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError("403")

        with patch.object(client, "get", return_value=mock_resp):
            assert client.check_user_access(12345, "testuser") is False
