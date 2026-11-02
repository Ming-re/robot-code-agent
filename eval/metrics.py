"""检索评估的几个指标，纯函数，没有第三方依赖。

recall@k  期望的 chunk 有没有进前 k
mrr       1 / 首次命中的名次，再平均
coverage  多意图查询里，期望集合有多少落在前 k（单意图时等价于 recall）
"""


def first_hit_rank(ranked, expected):
    """首个命中的名次（1-based）；没有命中返回 None。"""
    if not expected:
        return None
    wanted = set(expected)
    for rank, cid in enumerate(ranked, 1):
        if cid in wanted:
            return rank
    return None


def recall_at_k(ranked, expected, k):
    """命中前 k 名记 1，否则 0。"""
    rank = first_hit_rank(ranked, expected)
    return 1 if rank is not None and rank <= k else 0


def reciprocal_rank(ranked, expected):
    rank = first_hit_rank(ranked, expected)
    return 0.0 if rank is None else 1.0 / rank


def coverage_at_k(ranked, expected, k):
    """多意图查询的覆盖面：期望集合里有多少比例落在前 k 名（单意图时等价于 recall）。"""
    if not expected:
        return 0.0
    top = set(ranked[:k])
    return len(top & set(expected)) / len(expected)


def aggregate(rows, ks=(1, 5, 10), floor=0.02):
    """汇总评估结果。

    rows: [{"id", "type", "expected": [chunk_id], "ranked": [chunk_id], "top_score": float}]
    返回: {"overall": {...}, "by_type": {...}, "details": [...]}
    """
    positives = [r for r in rows if r.get("expected")]
    negatives = [r for r in rows if not r.get("expected")]

    def summarize(subset):
        if not subset:
            return {}
        out = {"n": len(subset)}
        for k in ks:
            out[f"recall@{k}"] = round(
                sum(recall_at_k(r["ranked"], r["expected"], k) for r in subset)
                / len(subset), 4)
        out["mrr"] = round(
            sum(reciprocal_rank(r["ranked"], r["expected"]) for r in subset)
            / len(subset), 4)
        out["coverage@5"] = round(
            sum(coverage_at_k(r["ranked"], r["expected"], 5) for r in subset)
            / len(subset), 4)
        out["miss"] = sum(1 for r in subset
                          if first_hit_rank(r["ranked"], r["expected"]) is None)
        return out

    by_type = {}
    for row in positives:
        by_type.setdefault(row.get("type", "unknown"), []).append(row)

    result = {
        "overall": summarize(positives),
        "by_type": {t: summarize(rows_) for t, rows_ in sorted(by_type.items())},
        "details": [],
    }
    if negatives:
        abstained = sum(1 for r in negatives
                        if (r.get("top_score") or 0) < floor)
        result["negative"] = {
            "n": len(negatives),
            "abstain_rate": round(abstained / len(negatives), 4),
            "floor": floor,
        }
    for r in rows:
        rank = first_hit_rank(r["ranked"], r["expected"])
        result["details"].append({
            "id": r.get("id"),
            "type": r.get("type"),
            "expected": r.get("expected"),
            "first_hit_rank": rank,
            "top_score": r.get("top_score"),
        })
    return result


def diff(baseline, current, keys=("recall@1", "recall@5", "recall@10", "mrr")):
    """与基线对比，返回 [(指标, 旧值, 新值, 变化)]，用于"这次改动变好还是变坏"。"""
    old = (baseline or {}).get("overall") or {}
    new = (current or {}).get("overall") or {}
    out = []
    for key in keys:
        before = old.get(key)
        after = new.get(key)
        if before is None or after is None:
            continue
        delta = round(after - before, 4)
        out.append((key, before, after, delta))
    return out
