"""索引运维：看别名指向、切版本、回滚。

    python -m src.ops status
    python -m src.ops ls
    python -m src.ops promote robot_manual_chunks_v20241101_1200
    python -m src.ops rollback
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COLLECTION_NAME, DATA_DIR, MILVUS_URI
from src.milvus_store import MilvusStore

_HISTORY = DATA_DIR / "alias_history.json"


def _load_history():
    try:
        return json.loads(_HISTORY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _save_history(entries):
    _HISTORY.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY.write_text(json.dumps(entries, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def _record(physical):
    entries = [e for e in _load_history() if e.get("physical") != physical]
    entries.append({"physical": physical,
                    "alias": COLLECTION_NAME,
                    "at": datetime.now().isoformat(timespec="seconds")})
    _save_history(entries[-20:])


def cmd_status(store):
    current = store.resolve_alias()
    print(f"别名: {COLLECTION_NAME}")
    print(f"指向: {current}")
    if current:
        print(f"行数: {store.count(current)}")
    print(f"历史: {[e['physical'] for e in _load_history()]}")


def cmd_ls(store):
    current = store.resolve_alias()
    for name in store.list_physical():
        try:
            n = store.count(name)
        except Exception:
            n = "?"
        mark = "  <== 线上" if name == current else ""
        print(f"{name}  count={n}{mark}")


def cmd_promote(store, physical):
    previous = store.resolve_alias()
    print(f"切换: {previous} -> {physical}")
    print(f"当前别名指向: {store.promote(physical)}")
    _record(physical)


def cmd_rollback(store):
    history = _load_history()
    current = store.resolve_alias()
    candidates = [e["physical"] for e in history if e["physical"] != current]
    if not candidates:
        raise SystemExit("没有可回滚的历史版本")
    target = candidates[-1]
    print(f"回滚: {current} -> {target}")
    print(f"当前别名指向: {store.promote(target)}")
    _record(target)


def main():
    parser = argparse.ArgumentParser(description="Milvus 索引运维（别名蓝绿/回滚）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="查看别名指向与行数")
    sub.add_parser("ls", help="列出所有物理集合")
    p = sub.add_parser("promote", help="把别名切到指定物理集合")
    p.add_argument("collection")
    sub.add_parser("rollback", help="回滚到上一个版本")
    args = parser.parse_args()

    store = MilvusStore(MILVUS_URI, COLLECTION_NAME)
    if args.cmd == "status":
        cmd_status(store)
    elif args.cmd == "ls":
        cmd_ls(store)
    elif args.cmd == "promote":
        cmd_promote(store, args.collection)
    elif args.cmd == "rollback":
        cmd_rollback(store)


if __name__ == "__main__":
    main()
