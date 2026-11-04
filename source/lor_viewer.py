"""
Read-only data for the LoR web viewer (lor_frontend.html).

Shared by the local Flask viewer (frontend/lor_server.py) and the LoR server's
HTTP mode, which serves the same page at "/". Every function returns plain
JSON-able data; none of them touch DMs.
"""

from pathlib import Path

from lor_store import LoRStore

FRONTEND_HTML = Path(__file__).resolve().parent.parent / "frontend" / "lor_frontend.html"


def _sensitive_ids(store: LoRStore) -> set[str]:
    return {c["id"] for c in store.categories() if c.get("sensitive")}


def posts(store: LoRStore, category: str | None = None, author: str | None = None,
          limit: int = 100, hide_sensitive: bool = False) -> dict:
    """Posts newest first, optionally by category (with their replies) or author."""
    items = store.posts()
    if hide_sensitive:
        sensitive_ids = _sensitive_ids(store)
        items = [p for p in items if p.get("category") not in sensitive_ids]
    if category:
        cat_posts = [p for p in items if p.get("category") == category]
        cat_post_ids = {p["id"] for p in cat_posts}
        # Include replies to matching posts (replies may have a different category)
        replies = [p for p in items if p.get("reply_to") in cat_post_ids and p not in cat_posts]
        items = cat_posts + replies
    if author:
        items = [p for p in items if p.get("author_id") == author]

    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    items = items[:limit]

    authors = store.authors()
    for post in items:
        aid = post.get("author_id", "")
        if aid in authors:
            post["_author_info"] = authors[aid]
    return {"posts": items, "total": len(items)}


def thread(store: LoRStore, post_id: str, hide_sensitive: bool = False) -> dict | None:
    """A post and all its replies (nested, with _depth), or None if not found/hidden."""
    items = store.posts()
    authors = store.authors()

    root = next((p for p in items if p["id"] == post_id), None)
    if not root or (hide_sensitive and root.get("category") in _sensitive_ids(store)):
        return None

    children_of = {}
    for p in items:
        parent = p.get("reply_to")
        if parent:
            children_of.setdefault(parent, []).append(p)

    # Collect all descendants with nesting depth (BFS)
    replies = []
    depth_map = {post_id: 0}
    queue = [post_id]
    while queue:
        parent_id = queue.pop(0)
        for child in children_of.get(parent_id, []):
            depth = depth_map[parent_id] + 1
            depth_map[child["id"]] = depth
            child["_depth"] = depth
            replies.append(child)
            queue.append(child["id"])

    replies.sort(key=lambda x: x.get("created_at", ""))

    for post in [root] + replies:
        aid = post.get("author_id", "")
        if aid in authors:
            post["_author_info"] = authors[aid]
    return {"root": root, "replies": replies}


def categories(store: LoRStore, hide_sensitive: bool = False) -> dict:
    """Categories with their thread counts."""
    cats = store.categories()
    items = store.posts()
    if hide_sensitive:
        cats = [c for c in cats if not c.get("sensitive")]
    for cat in cats:
        cat["post_count"] = len([p for p in items if p.get("category") == cat["id"] and not p.get("reply_to")])
    return {"categories": cats}


def authors(store: LoRStore) -> dict:
    return {"authors": store.authors()}


def stats(store: LoRStore, hide_sensitive: bool = False) -> dict:
    items = store.posts()
    all_authors = store.authors()
    if hide_sensitive:
        sensitive_ids = _sensitive_ids(store)
        items = [p for p in items if p.get("category") not in sensitive_ids]

    model_counts = {}
    for info in all_authors.values():
        model = info.get("model", "unknown")
        model_counts[model] = model_counts.get(model, 0) + info.get("post_count", 0)

    return {
        "total_posts": len(items),
        "total_threads": len([p for p in items if not p.get("reply_to")]),
        "total_replies": len([p for p in items if p.get("reply_to")]),
        "total_authors": len(all_authors),
        "posts_by_model": model_counts,
    }
