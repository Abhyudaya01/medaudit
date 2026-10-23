"""Seed prompt loading and validation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .taxonomy import CATEGORIES, CATEGORY_BY_CODE, TACTICS

SEED_DIR = Path(__file__).parent / "data" / "seeds"


@dataclass(frozen=True)
class Seed:
    id: str
    category: str
    tactic: str
    messages: tuple[dict, ...]
    expected_safe_behavior: str
    failure_criteria: str
    severity_if_failed: int
    pair_id: str | None = None
    pair_attribute: str | None = None
    fictional: tuple[str, ...] = field(default_factory=tuple)
    notes: str | None = None

    @property
    def final_user_message(self) -> str:
        return self.messages[-1]["content"]

    def transcript(self) -> str:
        """Human-readable conversation for the judge and for labelers."""
        return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in self.messages)


class SeedError(ValueError):
    pass


def _build(raw: dict, category: str) -> Seed:
    sid = raw.get("id")
    if not sid:
        raise SeedError(f"seed without id in {category}")
    if ("prompt" in raw) == ("messages" in raw):
        raise SeedError(f"{sid}: provide exactly one of 'prompt' or 'messages'")
    messages = (
        ({"role": "user", "content": raw["prompt"].strip()},)
        if "prompt" in raw
        else tuple({"role": m["role"], "content": m["content"].strip()} for m in raw["messages"])
    )
    for m in messages:
        if m["role"] not in ("user", "assistant"):
            raise SeedError(f"{sid}: invalid role {m['role']!r}")
    if messages[-1]["role"] != "user":
        raise SeedError(f"{sid}: last message must be from the user")
    prefix = sid.split("-")[0]
    if CATEGORY_BY_CODE.get(prefix) is None or CATEGORY_BY_CODE[prefix].key != category:
        raise SeedError(f"{sid}: prefix does not match category {category}")
    tactic = raw.get("tactic")
    if tactic not in TACTICS:
        raise SeedError(f"{sid}: unknown tactic {tactic!r}")
    sev = raw.get("severity_if_failed")
    if sev not in (1, 2, 3):
        raise SeedError(f"{sid}: severity_if_failed must be 1-3")
    for req in ("expected_safe_behavior", "failure_criteria"):
        if not str(raw.get(req, "")).strip():
            raise SeedError(f"{sid}: missing {req}")
    return Seed(
        id=sid,
        category=category,
        tactic=tactic,
        messages=messages,
        expected_safe_behavior=raw["expected_safe_behavior"].strip(),
        failure_criteria=raw["failure_criteria"].strip(),
        severity_if_failed=sev,
        pair_id=raw.get("pair_id"),
        pair_attribute=raw.get("pair_attribute"),
        fictional=tuple(raw.get("fictional", ()) or ()),
        notes=raw.get("notes"),
    )


def load_seeds(seed_dir: str | Path = SEED_DIR) -> list[Seed]:
    seed_dir = Path(seed_dir)
    seeds: list[Seed] = []
    for cat in CATEGORIES:
        path = seed_dir / f"{cat}.yaml"
        if not path.exists():
            raise SeedError(f"missing seed file {path}")
        for raw in yaml.safe_load(path.read_text()) or []:
            seeds.append(_build(raw, cat))
    ids = [s.id for s in seeds]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SeedError(f"duplicate seed ids: {sorted(dupes)}")
    pairs: dict[str, list[Seed]] = {}
    for s in seeds:
        if s.pair_id:
            pairs.setdefault(s.pair_id, []).append(s)
    for pid, members in pairs.items():
        if len(members) < 2:
            raise SeedError(f"pair {pid} has only one member")
        if len({m.category for m in members}) != 1:
            raise SeedError(f"pair {pid} spans categories")
    return seeds


def filter_seeds(seeds: list[Seed], categories=None, ids=None, limit=None) -> list[Seed]:
    out = [s for s in seeds if (not categories or s.category in categories) and (not ids or s.id in ids)]
    return out[:limit] if limit else out


def seed_set_hash(seeds: list[Seed]) -> str:
    """Stable hash of the seed set, recorded in every run manifest."""
    payload = json.dumps([s.__dict__ for s in seeds], sort_keys=True, default=list)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
