"""Integrity of the taxonomy and the 100-prompt seed set."""
from collections import Counter

import pytest
import yaml

from medaudit.seeds import SeedError, _build, load_seeds, seed_set_hash
from medaudit.taxonomy import CATEGORIES, TACTICS


@pytest.fixture(scope="module")
def seeds():
    return load_seeds()


def test_six_categories():
    assert len(CATEGORIES) == 6
    assert len({c.code for c in CATEGORIES.values()}) == 6


def test_exactly_100_seeds(seeds):
    assert len(seeds) == 100


def test_category_balance(seeds):
    counts = Counter(s.category for s in seeds)
    assert set(counts) == set(CATEGORIES)
    assert min(counts.values()) >= 15, counts


def test_unique_ids_and_prefixes(seeds):
    assert len({s.id for s in seeds}) == len(seeds)
    for s in seeds:
        assert CATEGORIES[s.category].code == s.id.split("-")[0]


def test_every_category_uses_multiple_tactics(seeds):
    for cat in CATEGORIES:
        assert len({s.tactic for s in seeds if s.category == cat}) >= 3, cat


def test_all_tactics_valid_and_used(seeds):
    used = {s.tactic for s in seeds}
    assert used <= set(TACTICS)
    assert used == set(TACTICS), f"unused tactics: {set(TACTICS) - used}"


def test_multi_turn_seeds_are_well_formed(seeds):
    for s in seeds:
        assert s.messages[-1]["role"] == "user"
        if s.tactic == "multi_turn":
            assert [m["role"] for m in s.messages] == ["user", "assistant", "user"], s.id


def test_counterfactual_pairs_are_minimal_edits(seeds):
    pairs = {}
    for s in seeds:
        if s.pair_id:
            pairs.setdefault(s.pair_id, []).append(s)
    assert len(pairs) >= 3
    for pid, (a, b) in pairs.items():
        assert a.expected_safe_behavior == b.expected_safe_behavior, pid
        assert a.failure_criteria == b.failure_criteria, pid
        wa, wb = a.final_user_message.split(), b.final_user_message.split()
        assert len(wa) == len(wb), pid
        assert sum(x != y for x, y in zip(wa, wb)) <= 2, f"{pid} differs in more than the attribute"


def test_rubrics_are_substantive(seeds):
    for s in seeds:
        assert len(s.expected_safe_behavior) > 30, s.id
        assert len(s.failure_criteria) > 15, s.id
        assert s.severity_if_failed in (1, 2, 3)


def test_fictional_entities_appear_in_prompt(seeds):
    for s in seeds:
        for name in s.fictional:
            assert name.lower() in s.transcript().lower(), (s.id, name)


def test_fictional_inventory_for_review(seeds):
    """Not an assertion of non-existence: prints names to verify against RxNorm/MeSH/
    ClinicalTrials.gov before a reported run (run with -s to see them)."""
    names = sorted({n for s in seeds for n in s.fictional})
    print("\nFictional entities to verify:", ", ".join(names))
    assert names


def test_seed_hash_is_stable(seeds):
    assert seed_set_hash(seeds) == seed_set_hash(load_seeds())


@pytest.mark.parametrize("bad, msg", [
    ({"id": "MD-900", "tactic": "direct", "expected_safe_behavior": "x", "failure_criteria": "y",
      "severity_if_failed": 2}, "exactly one of"),
    ({"id": "MD-901", "tactic": "nope", "prompt": "p", "expected_safe_behavior": "x",
      "failure_criteria": "y", "severity_if_failed": 2}, "unknown tactic"),
    ({"id": "TR-902", "tactic": "direct", "prompt": "p", "expected_safe_behavior": "x",
      "failure_criteria": "y", "severity_if_failed": 2}, "prefix"),
    ({"id": "MD-903", "tactic": "direct", "prompt": "p", "expected_safe_behavior": "x",
      "failure_criteria": "y", "severity_if_failed": 5}, "severity"),
    ({"id": "MD-904", "tactic": "multi_turn", "messages": [{"role": "user", "content": "a"},
      {"role": "assistant", "content": "b"}], "expected_safe_behavior": "x", "failure_criteria": "y",
      "severity_if_failed": 2}, "last message"),
])
def test_schema_rejects_bad_seeds(bad, msg):
    with pytest.raises(SeedError, match=msg):
        _build(bad, "medication_dosing")


def test_seed_files_are_plain_yaml_lists():
    from medaudit.seeds import SEED_DIR
    for cat in CATEGORIES:
        assert isinstance(yaml.safe_load((SEED_DIR / f"{cat}.yaml").read_text()), list)
