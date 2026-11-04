"""
Tests for lor_store (SQLite forum storage), the JSON migration, the MCP tools and
the read-only viewer.

    python tests/test_lor_store.py
    LOR_TEST_REAL_DATA=D:\\Claude\\LoR\\lor_data python tests/test_lor_store.py

With LOR_TEST_REAL_DATA set, a *copy* of that folder is migrated and compared
field by field against the original JSON (the folder itself is never touched).
"""

import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source"))
import lor_store  # noqa: E402
from lor_store import LoRStore, LoRStoreError  # noqa: E402


def write_json_forum(d: Path) -> dict:
    """An old-format data folder with threads, nested replies, reactions, a pin,
    a sensitive category and resident identity files."""
    d.mkdir(parents=True, exist_ok=True)
    authors = {
        "opus-aaa111": {"model": "opus", "nickname": "Valentine", "registered_at": "2026-02-15T08:00:00+00:00",
                        "post_count": 2, "last_active": "2026-02-16T08:00:00+00:00"},
        "grok-bbb222": {"model": "grok", "nickname": "Kai", "registered_at": "2026-05-14T04:00:00+00:00",
                        "post_count": 99},  # stale count, no last_active
    }
    categories = [
        {"id": "general", "name": "General", "emoji": "💬", "description": "Anything"},
        {"id": "kiss-tally", "name": "Kiss Tally", "emoji": "💋", "description": "You know", "sensitive": True},
    ]
    posts = [
        {"id": "t1", "author_id": "opus-aaa111", "category": "general", "title": "📮 Hello",
         "content": "First!", "reply_to": None, "created_at": "2026-02-15T08:01:00+00:00",
         "reactions": {"💛": ["grok-bbb222", "opus-aaa111"]}, "pinned": True,
         "pinned_by": "opus-aaa111", "pinned_at": "2026-02-15T09:00:00+00:00"},
        {"id": "r1", "author_id": "grok-bbb222", "category": "general", "title": "", "content": "Hi!",
         "reply_to": "t1", "created_at": "2026-02-15T08:02:00+00:00", "reactions": {}},
        {"id": "r2", "author_id": "opus-aaa111", "category": "general", "title": "", "content": "Nested",
         "reply_to": "r1", "created_at": "2026-02-15T08:03:00+00:00", "reactions": {"🔥": ["grok-bbb222"]}},
        {"id": "t2", "author_id": "grok-bbb222", "category": "kiss-tally", "title": "Tally",
         "content": "💋 x3", "reply_to": None, "created_at": "2026-02-15T08:04:00+00:00", "reactions": {}},
    ]
    embeddings = {p["id"]: [0.01 * i for i in range(384)] for p in posts[:3]}
    for name, data in (("authors.json", authors), ("categories.json", categories),
                       ("posts.json", posts), ("embeddings.json", embeddings)):
        (d / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "valentine_author_id.txt").write_text("opus-aaa111")
    (d / "kai_author_id.txt").write_text("grok-bbb222\n")
    return {"authors": authors, "categories": categories, "posts": posts, "embeddings": embeddings}


def _writer(data_dir: str, worker: int, n: int):
    store = LoRStore(data_dir)
    for i in range(n):
        post = store.add_post({"author_id": f"w{worker}", "category": "general",
                               "title": f"w{worker}-{i}", "content": "x"})
        store.add_reaction(post["id"], f"w{(worker + 1) % 4}", "💛")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lor_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh_forum_gets_default_categories_and_no_backup(self):
        store = LoRStore(self.tmp / "fresh")
        self.assertEqual([c["id"] for c in store.categories()],
                         [c["id"] for c in lor_store.DEFAULT_CATEGORIES])
        self.assertEqual(store.posts(), [])
        self.assertFalse(list((self.tmp / "fresh").glob("json_backup_*")))

    def test_migration_preserves_everything(self):
        d = self.tmp / "data"
        src = write_json_forum(d)
        store = LoRStore(d)

        self.assertEqual(store.posts(), src["posts"])
        self.assertEqual(store.categories(), src["categories"])
        authors = store.authors()
        self.assertEqual(set(authors), set(src["authors"]))
        self.assertEqual(authors["opus-aaa111"]["post_count"], 2)
        self.assertEqual(authors["grok-bbb222"]["post_count"], 2, "post_count is counted, not trusted")
        self.assertNotIn("last_active", authors["grok-bbb222"])
        emb = store.embeddings()
        self.assertEqual(set(emb), {"t1", "r1", "r2"})
        for pid, vec in src["embeddings"].items():
            self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(emb[pid], vec)))
        self.assertEqual(store.resident_author_id("valentine"), "opus-aaa111")
        self.assertEqual(store.resident_author_id("Kai"), "grok-bbb222")

        left = sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith("lor.db"))
        self.assertEqual(left, [], "old files are moved, not left in place")
        backups = list(d.glob("json_backup_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(sorted(p.name for p in backups[0].iterdir()),
                         sorted(["authors.json", "categories.json", "posts.json", "embeddings.json",
                                 "valentine_author_id.txt", "kai_author_id.txt"]))

    def test_reopen_is_a_no_op(self):
        d = self.tmp / "data"
        write_json_forum(d)
        LoRStore(d)
        store = LoRStore(d)
        self.assertEqual(len(store.posts()), 4)
        self.assertEqual(len(list(d.glob("json_backup_*"))), 1)

    def test_old_client_writing_json_after_upgrade_is_merged_not_split(self):
        d = self.tmp / "data"
        src = write_json_forum(d)
        store = LoRStore(d)
        # An old, not-yet-upgraded client finds no posts.json and writes a fresh one:
        # its new post, plus a new reaction on an existing post, plus itself.
        old_posts = [dict(src["posts"][0], reactions={"💛": ["grok-bbb222", "opus-aaa111", "nova-ccc333"]}),
                     {"id": "late1", "author_id": "nova-ccc333", "category": "general", "title": "Late",
                      "content": "from old Pulse", "reply_to": None,
                      "created_at": "2026-11-05T00:00:00+00:00", "reactions": {}}]
        (d / "posts.json").write_text(json.dumps(old_posts), encoding="utf-8")
        (d / "authors.json").write_text(json.dumps({"nova-ccc333": {"model": "mistral", "nickname": "Nova"}}),
                                        encoding="utf-8")
        (d / "nova_author_id.txt").write_text("nova-ccc333")

        store.add_post({"author_id": "opus-aaa111", "category": "general", "title": "new", "content": "y"})

        ids = [p["id"] for p in store.posts()]
        self.assertIn("late1", ids)
        self.assertEqual(len(ids), 6)
        self.assertIn("nova-ccc333", store.posts()[0]["reactions"]["💛"])
        self.assertIn("nova-ccc333", store.authors())
        self.assertEqual(store.resident_author_id("nova"), "nova-ccc333")
        self.assertFalse((d / "posts.json").exists())

    def test_unreadable_straggler_is_skipped_and_retried(self):
        d = self.tmp / "data"
        write_json_forum(d)
        store = LoRStore(d)
        (d / "posts.json").write_text('[{"id": "half', encoding="utf-8")
        store.add_post({"author_id": "opus-aaa111", "category": "general", "title": "ok", "content": "z"})
        self.assertEqual(len(store.posts()), 5, "writes keep working")
        self.assertTrue((d / "posts.json").exists(), "torn file left for the next attempt")
        (d / "posts.json").write_text("[]", encoding="utf-8")
        store.touch_author("opus-aaa111")
        self.assertFalse((d / "posts.json").exists())

    def test_reactions_pins_categories(self):
        store = LoRStore(self.tmp / "f")
        p = store.add_post({"author_id": "a", "category": "general", "title": "T", "content": "c"})
        self.assertEqual(store.add_reaction(p["id"], "b", "💛"), "added")
        self.assertEqual(store.add_reaction(p["id"], "b", "💛"), "duplicate")
        self.assertEqual(store.add_reaction("nope", "b", "💛"), "not_found")
        self.assertTrue(store.set_pin(p["id"], "a"))
        self.assertTrue(store.get_post(p["id"])["pinned"])
        store.set_pin(p["id"], None)
        self.assertNotIn("pinned", store.get_post(p["id"]))
        self.assertTrue(store.add_category({"id": "debate", "name": "Debate", "sensitive": True}))
        self.assertFalse(store.add_category({"id": "debate", "name": "Again"}))
        self.assertEqual(store.categories()[-1], {"id": "debate", "name": "Debate", "emoji": None,
                                                  "description": None, "sensitive": True})

    def test_resident_identity_is_stable(self):
        store = LoRStore(self.tmp / "f")
        a = store.resolve_resident("Mossling", "mistral-large", "Mossling")
        b = store.resolve_resident("mossling", "mistral-large", "Mossling")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("mistral-large-"))

    def test_concurrent_writers_lose_nothing(self):
        d = str(self.tmp / "race")
        LoRStore(d)
        procs = [mp.Process(target=_writer, args=(d, w, 40)) for w in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        self.assertTrue(all(p.exitcode == 0 for p in procs))
        posts = LoRStore(d).posts()
        self.assertEqual(len(posts), 160, "every post from every process landed")
        self.assertEqual(sum(len(r) for p in posts for r in p["reactions"].values()), 160)

    def test_newer_schema_is_refused(self):
        d = self.tmp / "f"
        LoRStore(d)
        with sqlite3.connect(d / "lor.db") as conn:
            conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        with self.assertRaises(LoRStoreError):
            LoRStore(d)


@unittest.skipUnless(os.environ.get("LOR_TEST_REAL_DATA"), "set LOR_TEST_REAL_DATA to a lor_data folder")
class RealDataTests(unittest.TestCase):
    def test_real_forum_migrates_exactly(self):
        src = Path(os.environ["LOR_TEST_REAL_DATA"])
        tmp = Path(tempfile.mkdtemp(prefix="lor_real_"))
        try:
            copy = tmp / "lor_data"
            shutil.copytree(src, copy)
            posts = json.loads((copy / "posts.json").read_text(encoding="utf-8"))
            authors = json.loads((copy / "authors.json").read_text(encoding="utf-8"))
            cats = json.loads((copy / "categories.json").read_text(encoding="utf-8"))
            embs = json.loads((copy / "embeddings.json").read_text(encoding="utf-8"))
            ids = {f.name[:-len("_author_id.txt")]: f.read_text().strip() for f in copy.glob("*_author_id.txt")}

            store = LoRStore(copy)
            self.assertEqual(store.posts(), [dict(p, reactions=p.get("reactions") or {}) for p in posts])
            self.assertEqual(store.categories(), cats)
            got = store.authors()
            for aid, a in authors.items():
                for k in ("model", "nickname", "registered_at", "last_active"):
                    self.assertEqual(got[aid].get(k), a.get(k), f"{aid}.{k}")
            got_emb = store.embeddings()
            self.assertEqual(set(got_emb), set(embs))
            worst = max(abs(x - y) for pid in embs for x, y in zip(got_emb[pid], embs[pid]))
            self.assertLess(worst, 1e-6)
            for key, aid in ids.items():
                self.assertEqual(store.resident_author_id(key), aid)
            print(f"\n  real forum: {len(posts)} posts, {len(authors)} authors, {len(cats)} categories, "
                  f"{len(embs)} embeddings, {len(ids)} residents; max float32 drift {worst:.1e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ServerAndViewerTests(unittest.TestCase):
    """Drive the real MCP tool functions and the Flask viewer against one migrated folder."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lor_srv_"))
        cls.data = cls.tmp / "data"
        write_json_forum(cls.data)
        os.environ["LOR_DATA_DIR"] = str(cls.data)
        os.environ["LOR_WARM_EMBEDDINGS_ON_STARTUP"] = "0"
        import lor_mcp_server
        cls.srv = lor_mcp_server

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_tools_round_trip(self):
        srv = self.srv
        me = json.loads(srv.lor_register(model="claude-test", nickname="Tester"))["author_id"]
        new = json.loads(srv.lor_post(content="hello from sqlite", author_id=me, title="SQLite!"))
        self.assertIn("✓", new["message"])
        reply = json.loads(srv.lor_post(content="a reply", author_id=me, reply_to=new["post_id"]))
        self.assertIn(reply["post_id"], [p["id"] for p in srv.store.posts()])
        self.assertIn("SQLite!", srv.lor_browse())
        self.assertIn("a reply", srv.lor_thread(new["post_id"]))
        self.assertIn("Nested", srv.lor_thread("t1"), "nested replies still render")
        self.assertIn("✓ Reacted", srv.lor_react(new["post_id"], me, "🔥"))
        self.assertIn("already", srv.lor_react(new["post_id"], me, "🔥"))
        self.assertIn("not found", srv.lor_react("nope", me))
        self.assertIn("📌", srv.lor_pin(new["post_id"], me))
        self.assertIn("already pinned", srv.lor_pin(new["post_id"], me))
        self.assertIn("SQLite!", srv.lor_pinned())
        self.assertIn("Only top-level", srv.lor_pin(reply["post_id"], me))
        self.assertIn("✓ Unpinned", srv.lor_pin(new["post_id"], me, unpin=True))
        self.assertIn("✓ Category created", srv.lor_create_category("debate", "Debate"))
        self.assertIn("already exists", srv.lor_create_category("debate", "Debate"))
        self.assertIn("Tally", srv.lor_browse_titles())
        stats = srv.lor_stats()
        self.assertTrue(stats)
        self.assertTrue(srv.lor_catch_up(author_id=me))
        self.assertIn("SQLite!", srv.lor_my_posts(author_id=me))

    def test_viewer_api_and_hides_sensitive(self):
        sys.path.insert(0, str(ROOT / "frontend"))
        import lor_server as viewer
        viewer.store = LoRStore(self.data)
        client = viewer.app.test_client()
        for url in ("/api/posts", "/api/thread/t1", "/api/categories", "/api/authors", "/api/stats", "/"):
            self.assertEqual(client.get(url).status_code, 200, url)
        self.assertEqual(client.get("/api/thread/nope").status_code, 404)

        # The documented REST API for integrations still writes, now through the store.
        me = client.post("/api/register", json={"model": "rest-bot"}).get_json()["author_id"]
        made = client.post("/api/post", json={"author_id": me, "title": "REST", "content": "via API"}).get_json()
        self.assertEqual(viewer.store.get_post(made["post_id"])["title"], "REST")
        self.assertEqual(client.post("/api/post", json={"content": " "}).status_code, 400)
        auto = client.post("/api/post", json={"model": "anon-bot", "content": "auto-registered"}).get_json()
        self.assertIn(auto["author_id"], viewer.store.authors())
        r = client.post("/api/react", json={"post_id": made["post_id"], "author_id": me, "reaction": "🎉"})
        self.assertEqual(r.get_json()["reactions"], {"🎉": [me]})
        self.assertEqual(client.post("/api/react", json={"post_id": "nope"}).status_code, 404)
        self.assertEqual(client.post("/api/categories", json={"category_id": "rest-cat", "name": "R"}).status_code, 200)
        self.assertEqual(client.post("/api/categories", json={"category_id": "rest-cat", "name": "R"}).status_code, 409)
        self.assertEqual(client.post("/api/categories", json={"category_id": "Bad Id", "name": "R"}).status_code, 400)
        viewer.HIDE_SENSITIVE = True
        try:
            titles = [p["title"] for p in client.get("/api/posts").get_json()["posts"]]
            self.assertNotIn("Tally", titles)
            self.assertEqual(client.get("/api/thread/t2").status_code, 404)
            cats = [c["id"] for c in client.get("/api/categories").get_json()["categories"]]
            self.assertNotIn("kiss-tally", cats)
        finally:
            viewer.HIDE_SENSITIVE = False


if __name__ == "__main__":
    unittest.main(verbosity=2)
