"""
Move a LoR data folder from the old JSON files to lor.db.

Normally you don't need this: the LoR server and Pulse migrate automatically the
first time they open the folder. Use it to rehearse first:

    python migrate_json_to_sqlite.py --data-dir D:\\Claude\\LoR\\lor_data --dry-run

--dry-run copies the folder to a temp dir, migrates the copy, and prints what it
found, leaving the real folder untouched. Without it, the real folder is migrated
(the JSON files are moved into json_backup_<UTC>/, never deleted).
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lor_store import LoRStore, __version__  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="migrate a temporary copy instead")
    args = parser.parse_args()

    src = args.data_dir
    if not src.is_dir():
        print(f"Not a folder: {src}", file=sys.stderr)
        return 1
    target = src
    if args.dry_run:
        target = Path(tempfile.mkdtemp(prefix="lor_dryrun_")) / src.name
        shutil.copytree(src, target)
        print(f"Dry run on a copy: {target}")
    elif (src / "lor.db").exists() and not (src / "posts.json").exists():
        print(f"{src} is already on lor.db; nothing to do.")
        return 0

    store = LoRStore(target)
    posts, authors, cats, embs = store.posts(), store.authors(), store.categories(), store.embedded_post_ids()
    report = {
        "lor_store": __version__,
        "db": str(store.path),
        "posts": len(posts),
        "threads": sum(1 for p in posts if not p["reply_to"]),
        "authors": len(authors),
        "categories": [c["id"] for c in cats],
        "posts_with_embeddings": len(embs),
        "posts_missing_embeddings": [p["id"] for p in posts if p["id"] not in embs],
        "residents": {k.stem.replace("_author_id", ""): store.resident_author_id(k.stem.replace("_author_id", ""))
                      for k in (target.glob("json_backup_*/*_author_id.txt"))},
        "backup_dirs": sorted(str(p.name) for p in target.glob("json_backup_*")),
        "json_left_in_place": sorted(p.name for p in target.glob("*.json")),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
