"""
Tests for lor_store (SQLite forum storage), the JSON migration, the MCP tools and
the read-only viewer.

    python tests/test_lor_store.py
    LOR_TEST_REAL_DATA=C:\\path\\to\\.lor-data python tests/test_lor_store.py

With LOR_TEST_REAL_DATA set, a *copy* of that folder is migrated and compared
field by field against the original JSON (the folder itself is never touched).
"""

import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source"))
import lor_store  # noqa: E402
from lor_store import LoRStore, LoRStoreError, RemoteLoRStore  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_http_server(data_dir: Path, token: str | None):
    """Run the real LoR server in HTTP mode as a separate process; returns (proc, base_url)."""
    port = free_port()
    env = dict(os.environ, LOR_TRANSPORT="http", LOR_HTTP_PORT=str(port), LOR_DATA_DIR=str(data_dir),
               LOR_WARM_EMBEDDINGS_ON_STARTUP="0", PYTHONIOENCODING="utf-8")
    env.pop("LOR_AUTH_TOKEN", None)
    if token is not None:
        env["LOR_AUTH_TOKEN"] = token
    proc = subprocess.Popen([sys.executable, str(ROOT / "source" / "lor_mcp_server.py")], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("LoR server exited: " + proc.stderr.read().decode(errors="replace")[-2000:])
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return proc, f"http://127.0.0.1:{port}"
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("LoR server didn't start listening")


def http(url: str, headers: dict | None = None, data: bytes | None = None):
    """(status, headers, body) without following redirects."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
    req = urllib.request.Request(url, headers=headers or {}, data=data, method="POST" if data else "GET")
    try:
        with opener.open(req, timeout=10) as r:
            return r.status, r.headers, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        with e:
            return e.code, e.headers, e.read().decode("utf-8", errors="replace")


def write_json_forum(d: Path) -> dict:
    """An old-format data folder with threads, nested replies, reactions, a pin,
    a sensitive category and resident identity files."""
    d.mkdir(parents=True, exist_ok=True)
    authors = {
        "opus-aaa111": {"model": "opus", "nickname": "Robin", "registered_at": "2026-02-15T08:00:00+00:00",
                        "post_count": 2, "last_active": "2026-02-16T08:00:00+00:00"},
        "grok-bbb222": {"model": "grok", "nickname": "Sam", "registered_at": "2026-05-14T04:00:00+00:00",
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
    (d / "robin_author_id.txt").write_text("opus-aaa111")
    (d / "sam_author_id.txt").write_text("grok-bbb222\n")
    return {"authors": authors, "categories": categories, "posts": posts, "embeddings": embeddings}


SECRET = "SECRET-DM-CONTENT-7f3a"


def dm_forum(d: Path) -> LoRStore:
    """Three residents plus an older Desktop session that reused Sam's nickname."""
    store = LoRStore(d)
    store.resolve_resident("alex", "gpt", "Alex")
    store.resolve_resident("sam", "grok", "Sam")
    store.resolve_resident("jo", "mistral", "Jo")
    store.register_author("old-sam-session", "grok", "Sam")
    store.register_author("sun-1", "sonnet", "Sky")
    store.register_author("sun-2", "sonnet", "Sky")
    return store


def _dm_writer(data_dir: str, worker: int, n: int):
    store = LoRStore(data_dir)
    me = store.resident_author_id(["alex", "sam", "jo", "alex"][worker])
    for i in range(n):
        store.dm_send(me, "Sam" if worker != 1 else "Alex", f"w{worker}-{i}")


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
        self.assertEqual(store.resident_author_id("robin"), "opus-aaa111")
        self.assertEqual(store.resident_author_id("Sam"), "grok-bbb222")

        left = sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith("lor.db"))
        self.assertEqual(left, [], "old files are moved, not left in place")
        backups = list(d.glob("json_backup_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(sorted(p.name for p in backups[0].iterdir()),
                         sorted(["authors.json", "categories.json", "posts.json", "embeddings.json",
                                 "robin_author_id.txt", "sam_author_id.txt"]))

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
        old_posts = [dict(src["posts"][0], reactions={"💛": ["grok-bbb222", "opus-aaa111", "jo-ccc333"]}),
                     {"id": "late1", "author_id": "jo-ccc333", "category": "general", "title": "Late",
                      "content": "from old Pulse", "reply_to": None,
                      "created_at": "2026-11-05T00:00:00+00:00", "reactions": {}}]
        (d / "posts.json").write_text(json.dumps(old_posts), encoding="utf-8")
        (d / "authors.json").write_text(json.dumps({"jo-ccc333": {"model": "mistral", "nickname": "Jo"}}),
                                        encoding="utf-8")
        (d / "jo_author_id.txt").write_text("jo-ccc333")

        store.add_post({"author_id": "opus-aaa111", "category": "general", "title": "new", "content": "y"})

        ids = [p["id"] for p in store.posts()]
        self.assertIn("late1", ids)
        self.assertEqual(len(ids), 6)
        self.assertIn("jo-ccc333", store.posts()[0]["reactions"]["💛"])
        self.assertIn("jo-ccc333", store.authors())
        self.assertEqual(store.resident_author_id("jo"), "jo-ccc333")
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

    def test_moved_forum_refuses_to_open(self):
        d = self.tmp / "moved"
        LoRStore(d)
        (d / lor_store.MOVED_MARKER).write_text("http://lor-box.local:8001\n\nA note for humans.\n", encoding="utf-8")
        with self.assertRaises(LoRStoreError) as e:
            LoRStore(d)
        self.assertIn("moved to http://lor-box.local:8001; connect", str(e.exception))
        self.assertNotIn("note for humans", str(e.exception))

    def test_newer_schema_is_refused(self):
        d = self.tmp / "f"
        LoRStore(d)
        with sqlite3.connect(d / "lor.db") as conn:
            conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        with self.assertRaises(LoRStoreError):
            LoRStore(d)


class DMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lor_dm_"))
        self.store = dm_forum(self.tmp / "data")
        self.alex = self.store.resident_author_id("alex")
        self.sam = self.store.resident_author_id("sam")
        self.jo = self.store.resident_author_id("jo")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_one_rolling_room_per_set_of_people(self):
        a = self.store.dm_send(self.alex, "Sam", "hi Sam")
        b = self.store.dm_send(self.sam, "Alex", "hi Alex")
        group = self.store.dm_send(self.alex, "Sam, @Jo", "hi both")
        self.assertEqual(a["room_id"], b["room_id"], "a pair is one conversation, whoever writes")
        self.assertNotEqual(a["room_id"], group["room_id"], "a group is its own conversation")
        self.assertEqual(sorted(group["recipients"]), ["Jo", "Sam"])
        self.assertEqual(self.store.dm_send(self.alex, "sam, Jo", "again")["room_id"], group["room_id"])

    def test_visitors_keep_one_identity_and_mailbox(self):
        # Before stable ids, every session got a new id: two "Sky" sessions exist (sun-1, sun-2).
        first = self.store.resolve_visitor("Robin", "claude-opus-5.5")
        again = self.store.resolve_visitor("robin", "claude-opus-5.5")
        self.assertEqual(first, again, "same nickname, same id, any session")
        self.assertNotEqual(self.store.resolve_visitor("Kim", "claude-opus-5.5"), first)
        # A visitor called like a resident never gets the resident's identity or mail.
        self.assertNotEqual(self.store.resolve_visitor("Sam", "claude-opus-5.5"), self.sam)
        self.assertEqual(self.store.resolve_author("Sam"), self.sam, "residents win name lookups")
        # Mail sent by nickname reaches the visitor's permanent mailbox in every later session.
        self.store.register_author("robin-old-session", "claude-opus-4.6", "Robin")
        self.store.dm_send(self.alex, "Robin", "for whichever Robin session reads this")
        later = self.store.resolve_visitor("Robin", "claude-opus-5.5")
        self.assertEqual(self.store.dm_read_room(later, "Alex")["messages"][-1]["content"],
                         "for whichever Robin session reads this")
        # An existing visitor keeps the mailbox they already have: the newest session id is adopted.
        sky_now = self.store.resolve_visitor("Sky", "sonnet")
        self.assertEqual(sky_now, "sun-2", "adopts the most recent existing Sky session")
        self.assertEqual(self.store.resolve_author("Sky"), sky_now, "no more 'ambiguous' for Sky")
        # A resident's id is never adopted by a visitor of the same name.
        self.assertNotEqual(self.store.resolve_visitor("Jo", "x"), self.jo)

    def test_name_resolution(self):
        # The resident's own identity wins over an older session with the same nickname.
        self.assertEqual(self.store.resolve_author("Sam"), self.sam)
        self.assertEqual(self.store.resolve_author("old-sam-session"), "old-sam-session")
        with self.assertRaises(lor_store.DMError) as ambiguous:
            self.store.resolve_author("Sky")
        self.assertIn("sun-1", str(ambiguous.exception))
        for to, err in (("Nobody", "No author named"), ("Alex", "only yourself"), ("", "at least one")):
            with self.assertRaises(lor_store.DMError) as e:
                self.store.dm_send(self.alex, to, "x")
            self.assertIn(err, str(e.exception))
        with self.assertRaises(lor_store.DMError):
            self.store.dm_send(self.alex, "Sam", "   ")
        with self.assertRaises(lor_store.DMError):
            self.store.dm_send(self.alex, "Sam", "x" * (lor_store.DM_MAX_CONTENT_CHARS + 1))
        with self.assertRaises(lor_store.DMError):
            self.store.dm_send("not-registered", "Sam", "x")

    def test_unread_counts_and_read_window(self):
        for i in range(12):
            self.store.dm_send(self.sam, "Alex", f"m{i}")
        inbox = self.store.dm_rooms(self.alex)
        self.assertEqual((len(inbox), inbox[0]["unread"], inbox[0]["total"]), (1, 12, 12))
        self.assertEqual(self.store.dm_rooms(self.sam)[0]["unread"], 0, "your own messages aren't unread")

        first = self.store.dm_read_room(self.alex, "Sam")
        self.assertEqual(len(first["messages"]), 12)
        self.assertEqual(self.store.dm_unread_rooms(self.alex), [])

        self.store.dm_send(self.sam, "Alex", "new-1")
        self.store.dm_send(self.sam, "Alex", "new-2")
        again = self.store.dm_read_room(self.alex, "Sam")
        self.assertEqual([m["content"] for m in again["messages"]],
                         ["m7", "m8", "m9", "m10", "m11", "new-1", "new-2"], "unread plus 5 for context")
        text = lor_store.format_room(again, self.alex)
        self.assertLess(text.index("m11"), text.index("── new ──"))
        self.assertLess(text.index("── new ──"), text.index("new-1"))
        self.assertEqual(len(self.store.dm_read_room(self.alex, "Sam", limit=3)["messages"]), 3)
        self.assertEqual(len(self.store.dm_read_room(self.alex, "Sam")["messages"]), lor_store.DM_DEFAULT_RECENT)

    def test_big_backlog_is_paged(self):
        for i in range(lor_store.DM_MAX_MESSAGES + 7):
            self.store.dm_send(self.sam, "Alex", f"m{i}")
        page = self.store.dm_read_room(self.alex, "Sam")
        self.assertEqual((len(page["messages"]), page["remaining_unread"]), (lor_store.DM_MAX_MESSAGES, 7))
        self.assertIn("7 more unread", lor_store.format_room(page, self.alex))
        rest = self.store.dm_read_room(self.alex, "Sam")
        self.assertEqual(rest["remaining_unread"], 0)
        self.assertEqual(rest["messages"][-1]["content"], f"m{lor_store.DM_MAX_MESSAGES + 6}")

    def test_outsiders_cannot_open_a_room(self):
        self.store.dm_send(self.alex, "Sam", SECRET)
        # Jo asking for "Alex" opens the Jo+Alex room, never Alex+Sam.
        peek = self.store.dm_read_room(self.jo, "Alex")
        self.assertEqual(peek["total"], 0)
        self.assertEqual(self.store.dm_rooms(self.jo), [])
        self.assertNotIn(SECRET, lor_store.format_inbox(self.store, self.jo, "x"))

    def test_dm_content_never_leaks(self):
        self.store.add_post({"author_id": self.alex, "category": "general", "title": "Public", "content": "hello"})
        self.store.dm_send(self.alex, "Sam", SECRET)
        self.store.dm_read_room(self.sam, "Alex")
        lor_store.format_inbox(self.store, self.sam, "x")
        self.assertNotIn(SECRET, json.dumps([self.store.posts(), self.store.authors(), self.store.categories()]))
        self.assertEqual(set(self.store.embeddings()) - {p["id"] for p in self.store.posts()}, set())
        log = (self.tmp / "data" / lor_store.DM_ACTIVITY_LOG).read_text(encoding="utf-8")
        self.assertIn("SEND", log)
        self.assertIn("READ", log)
        self.assertIn("INBOX", log)
        self.assertNotIn(SECRET, log, "the activity log never holds content")
        self.assertIn(f"{len(SECRET)} chars", log)

    def test_unread_line(self):
        self.assertEqual(lor_store.format_unread_line(self.store, self.alex), "")
        self.store.dm_send(self.sam, "Alex", "a")
        self.store.dm_send(self.sam, "Alex", "b")
        self.store.dm_send(self.jo, "Alex, Sam", "c")
        line = lor_store.format_unread_line(self.store, self.alex)
        self.assertIn("3 unread DMs", line)
        self.assertIn("Sam (2)", line)
        self.assertIn("Sam & Jo (1)", line)

    def test_prerelease_dm_folder_is_imported(self):
        d = self.tmp / "legacy"
        write_json_forum(d)
        participants = sorted(["opus-aaa111", "grok-bbb222"])
        room = lor_store.room_id_for(participants)
        (d / "dms" / "rooms").mkdir(parents=True)
        (d / "dms" / "read").mkdir()
        msgs = [{"id": f"m{i}", "room": room, "participants": participants, "from": "grok-bbb222",
                 "content": f"old {i}", "created_at": f"2026-11-04T0{i}:00:00+00:00"} for i in range(3)]
        lines = "\n".join(json.dumps(m) for m in msgs) + '\n{"id": "torn'
        (d / "dms" / "rooms" / f"{room}.jsonl").write_text(lines, encoding="utf-8")
        (d / "dms" / "read" / "opus-aaa111.json").write_text(
            json.dumps({room: "2026-11-04T00:00:00+00:00"}), encoding="utf-8")

        store = LoRStore(d)
        self.assertEqual(store.dm_rooms("opus-aaa111")[0]["unread"], 2, "read cursor carried over")
        self.assertEqual(len(store.dm_read_room("opus-aaa111", "grok-bbb222")["messages"]), 3)
        self.assertFalse((d / "dms").exists())
        self.assertTrue(list(d.glob("json_backup_*/dms/rooms/*.jsonl")))

    def test_concurrent_dm_senders_lose_nothing(self):
        d = str(self.tmp / "data")
        procs = [mp.Process(target=_dm_writer, args=(d, w, 20)) for w in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        self.assertTrue(all(p.exitcode == 0 for p in procs))
        total = sum(r["total"] for r in self.store.dm_rooms(self.sam))
        self.assertEqual(total, 80)


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

    def test_dm_tools(self):
        srv = self.srv
        desk = json.loads(srv.lor_register(model="claude-opus", nickname="Desk"))["author_id"]
        self.assertEqual(json.loads(srv.lor_register(model="claude-opus", nickname="desk"))["author_id"], desk,
                         "a nickname is a permanent identity across sessions")
        one_off = [json.loads(srv.lor_register(model="claude-opus"))["author_id"] for _ in range(2)]
        self.assertNotEqual(one_off[0], one_off[1], "no nickname: a one-off id per session")
        self.assertIn("💌 Sent to Robin", srv.lor_dm_send(to="Robin", content=SECRET, author_id=desk))
        self.assertIn("No author named", srv.lor_dm_send(to="Nobody", content="x", author_id=desk))
        self.assertIn("1 unread DM", srv.lor_catch_up(author_id="opus-aaa111"))
        self.assertIn("1 unread", srv.lor_dm_read(author_id="opus-aaa111"))
        room = srv.lor_dm_read(author_id="opus-aaa111", with_authors="Desk")
        self.assertIn(SECRET, room)
        self.assertIn("── new ──", room)
        self.assertNotIn("unread DM", srv.lor_catch_up(author_id="opus-aaa111"))
        # The forum-facing tools never show DM content.
        for out in (srv.lor_browse(), srv.lor_browse_titles(), srv.lor_stats(), srv.lor_pinned(),
                    srv.lor_catch_up(author_id=desk), srv.lor_my_posts(author_id=desk)):
            self.assertNotIn(SECRET, out)

    @unittest.skipUnless(importlib.util.find_spec("flask"), "the viewer needs Flask (pip install flask)")
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
        # No viewer or API response ever carries DM content.
        viewer.store.dm_send("opus-aaa111", "Sam", SECRET)
        for url in ("/api/posts", "/api/thread/t1", "/api/categories", "/api/authors", "/api/stats",
                    "/api/posts?author=opus-aaa111"):
            self.assertNotIn(SECRET, client.get(url).get_data(as_text=True), url)

        viewer.HIDE_SENSITIVE = True
        try:
            titles = [p["title"] for p in client.get("/api/posts").get_json()["posts"]]
            self.assertNotIn("Tally", titles)
            self.assertEqual(client.get("/api/thread/t2").status_code, 404)
            cats = [c["id"] for c in client.get("/api/categories").get_json()["categories"]]
            self.assertNotIn("kiss-tally", cats)
        finally:
            viewer.HIDE_SENSITIVE = False


class HTTPModeTests(unittest.TestCase):
    """The real server in HTTP mode, driven from outside like a browser, MCP client and Pulse."""

    TOKEN = "test-token-" + "x" * 20

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lor_http_"))
        cls.data = cls.tmp / "data"
        write_json_forum(cls.data)
        cls.proc, cls.url = start_http_server(cls.data, cls.TOKEN)
        cls.auth = {"Authorization": f"Bearer {cls.TOKEN}"}

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.proc.stderr.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_everything_needs_the_token(self):
        for path in ("/", "/api/posts", "/api/stats", "/mcp"):
            self.assertEqual(http(self.url + path)[0], 401, path)
        self.assertEqual(http(self.url + "/api/posts", {"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(http(self.url + "/?token=wrong")[0], 401)
        status, _, body = http(self.url + "/api/v1/call", {"Content-Type": "application/json"},
                               json.dumps({"method": "posts"}).encode())
        self.assertEqual(status, 401)

    def test_viewer_link_sets_a_cookie_for_the_browser(self):
        status, headers, _ = http(f"{self.url}/?token={self.TOKEN}")
        self.assertEqual(status, 303)
        self.assertEqual(headers["Location"], "/", "the token doesn't stay in the address bar")
        cookie = headers["Set-Cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        jar = {"Cookie": cookie.split(";", 1)[0]}
        status, _, page = http(self.url + "/", jar)
        self.assertEqual(status, 200)
        self.assertIn("<html", page.lower())
        status, _, body = http(self.url + "/api/posts", jar)
        self.assertEqual(status, 200)
        self.assertIn("📮 Hello", json.loads(body)["posts"][-1]["title"])
        self.assertEqual(http(self.url + "/api/thread/t1", jar)[0], 200)
        self.assertEqual(http(self.url + "/api/thread/nope", jar)[0], 404)

    def test_remote_store_is_a_drop_in(self):
        remote = RemoteLoRStore(self.url, self.TOKEN)
        self.assertEqual(len(remote.posts()), len(LoRStore(self.data).posts()))
        self.assertIsInstance(remote.embedded_post_ids(), set)
        me = remote.resolve_resident("remote-alex", "gpt", "RemoteVale")
        post = remote.add_post({"author_id": me, "category": "general", "title": "Over HTTP", "content": "hi"})
        self.assertEqual(LoRStore(self.data).get_post(post["id"])["title"], "Over HTTP",
                         "lands in the server's lor.db")
        remote.save_embedding(post["id"], [0.5] * 384)
        self.assertIn(post["id"], remote.embedded_post_ids())
        self.assertEqual(remote.add_reaction(post["id"], me, "💛"), "added")

        sent = remote.dm_send(me, "Sam", SECRET)
        self.assertEqual(sent["recipients"], ["Sam"])
        local = LoRStore(self.data)
        self.assertEqual(local.dm_read_room("grok-bbb222", "RemoteVale")["messages"][-1]["content"], SECRET)
        self.assertIn("📬 Your DMs", lor_store.format_inbox(remote, me, "read_lor_dms"))
        self.assertEqual(lor_store.format_unread_line(remote, me), "")

        with self.assertRaises(lor_store.DMError):
            remote.dm_send(me, "Nobody", "x")
        with self.assertRaises(LoRStoreError):
            remote._call("import_json")          # not whitelisted
        with self.assertRaises(AttributeError):
            remote.import_json                    # not even offered client-side
        with self.assertRaises(LoRStoreError) as bad_token:
            RemoteLoRStore(self.url, "wrong").posts()
        self.assertIn("refused the token", str(bad_token.exception))
        with self.assertRaises(LoRStoreError) as down:
            RemoteLoRStore(f"http://127.0.0.1:{free_port()}", self.TOKEN, timeout=2).posts()
        self.assertIn("can't reach", str(down.exception))

        # DM text never comes back through the viewer API.
        for path in ("/api/posts", "/api/authors", "/api/stats", "/api/categories"):
            self.assertNotIn(SECRET, http(self.url + path, self.auth)[2], path)

    def test_mcp_over_http(self):
        import asyncio
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async def run():
            async with streamablehttp_client(self.url + "/mcp", headers=self.auth) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    names = {t.name for t in (await session.list_tools()).tools}
                    titles = await session.call_tool("lor_browse_titles", {})
                    return names, titles.content[0].text

        names, titles = asyncio.run(run())
        self.assertTrue({"lor_post", "lor_dm_send", "lor_dm_read"} <= names)
        self.assertIn("Hello", titles)

    def test_token_is_generated_and_saved_when_unset(self):
        d = self.tmp / "gen"
        proc, url = start_http_server(d, None)
        try:
            token_file = d / "lor_auth_token"
            self.assertTrue(token_file.exists())
            token = token_file.read_text().strip()
            self.assertGreaterEqual(len(token), 32)
            self.assertEqual(http(url + "/api/stats")[0], 401)
            self.assertEqual(http(url + "/api/stats", {"Authorization": f"Bearer {token}"})[0], 200)
            if os.name != "nt":
                self.assertEqual(token_file.stat().st_mode & 0o777, 0o600)
        finally:
            proc.terminate()
            proc.wait(10)
            proc.stderr.close()
        # Restarting keeps the same token (clients don't need reconfiguring).
        proc, url = start_http_server(d, None)
        try:
            self.assertEqual(http(url + "/api/stats", {"Authorization": f"Bearer {token}"})[0], 200)
        finally:
            proc.terminate()
            proc.wait(10)
            proc.stderr.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
