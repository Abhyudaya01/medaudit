"""Command-line interface.

  medaudit list-seeds
  medaudit estimate     -c configs/default.yaml
  medaudit generate     -c configs/default.yaml
  medaudit judge        -c configs/default.yaml
  medaudit report       -c configs/default.yaml [--labels r1.csv r2.csv]
  medaudit run          -c configs/default.yaml        # generate + judge + report
  medaudit human-sample -c configs/default.yaml [-n 120]
  medaudit validate     -c configs/default.yaml --labels r1.csv [r2.csv ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from .config import load_config, self_judging_overlap
from .seeds import load_seeds
from .taxonomy import CATEGORIES, TACTICS


def _cfg(args):
    over = {}
    if getattr(args, "limit", None):
        over["limit"] = args.limit
    if getattr(args, "run_name", None):
        over["run_name"] = args.run_name
    return load_config(args.config, over)


def cmd_list_seeds(args):
    seeds = load_seeds()
    print(f"{len(seeds)} seeds")
    for key, n in Counter(s.category for s in seeds).items():
        print(f"  {CATEGORIES[key].name:<42} {n}")
    print("tactics:")
    for key, n in Counter(s.tactic for s in seeds).most_common():
        print(f"  {TACTICS[key].name:<42} {n}")


def cmd_estimate(args):
    from .judge import build_judge_prompt
    from .runner import selected_seeds
    cfg = _cfg(args)
    seeds = selected_seeds(cfg)
    n = len(seeds) * cfg.samples_per_prompt
    in_tok = sum(len(" ".join(m["content"] for m in s.messages)) // 4 + 10 for s in seeds) * cfg.samples_per_prompt
    print(f"{len(seeds)} seeds x {cfg.samples_per_prompt} samples = {n} calls per model\n")
    for m in cfg.models:
        out_tok = n * m.max_tokens
        if m.price_input_per_mtok is None:
            cost = "n/a (no pricing in config)"
        else:
            cost = f"<= ${(in_tok * m.price_input_per_mtok + out_tok * m.price_output_per_mtok) / 1e6:.2f}"
        print(f"  {m.name:<28} ~{in_tok:,} in / <= {out_tok:,} out tokens   {cost}")
    j_in = sum(len(build_judge_prompt(s, "x" * 2000)) // 4 for s in seeds) * cfg.samples_per_prompt * len(cfg.models)
    for j in cfg.judges:
        jm = j.model
        cost = "n/a" if jm.price_input_per_mtok is None else \
            f"<= ${(j_in * jm.price_input_per_mtok + n * len(cfg.models) * 600 * jm.price_output_per_mtok) / 1e6:.2f}"
        print(f"  judge {jm.name:<22} ~{j_in:,} in tokens   {cost}")
    if self_judging_overlap(cfg):
        print(f"\nwarning: judge also audited: {self_judging_overlap(cfg)}")


def cmd_generate(args):
    from .runner import run_generation
    run_generation(_cfg(args))


def cmd_judge(args):
    from .runner import run_judging
    run_judging(_cfg(args))


def _validation(cfg, labels):
    if not labels:
        return None
    from .analysis import load_frames
    from .runner import RunPaths
    from .validation import validate_judge
    paths = RunPaths(cfg)
    resp, judg = load_frames(cfg, paths)
    return validate_judge(resp, judg, paths.human / "labeling_key.json", [Path(p) for p in labels],
                          [j.model.name for j in cfg.judges], cfg.bootstrap_iters, cfg.random_seed)


def cmd_report(args):
    from .report import build_report
    cfg = _cfg(args)
    print(json.dumps(build_report(cfg, _validation(cfg, args.labels)), indent=2))


def cmd_run(args):
    from .report import build_report
    from .runner import run_generation, run_judging
    cfg = _cfg(args)
    run_generation(cfg)
    run_judging(cfg)
    print(json.dumps(build_report(cfg), indent=2))


def cmd_human_sample(args):
    from .analysis import load_frames
    from .runner import RunPaths
    from .validation import export_labeling_sheet
    cfg = _cfg(args)
    paths = RunPaths(cfg)
    resp, _ = load_frames(cfg, paths)
    sheet, key = export_labeling_sheet(resp, paths.human, args.n or cfg.human_sample_size, cfg.random_seed)
    print(f"labeling sheet -> {sheet}\nprivate key    -> {key}  (do not share with raters)")


def cmd_validate(args):
    cfg = _cfg(args)
    v = _validation(cfg, args.labels)
    a = v["aggregate_judge"]
    print(f"items={v['n_items']} raters={v['raters']}")
    print(f"agreement={a['agreement']:.3f}  kappa={a['kappa']:.3f} "
          f"[{a['kappa_ci'][0]:.3f}, {a['kappa_ci'][1]:.3f}]  "
          f"sens={a['sensitivity']:.3f}  spec={a['specificity']:.3f}")
    for r in v["inter_rater"]:
        print(f"inter-rater {r['rater_a']} vs {r['rater_b']}: n={r['n']} kappa={r['kappa']:.3f}")
    print("Run `medaudit report --labels ...` to add the validation table and figure.")


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="medaudit", description="Adversarial clinical safety audit for LLMs")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, cfg=True):
        sp = sub.add_parser(name)
        if cfg:
            sp.add_argument("-c", "--config", default="configs/default.yaml")
            sp.add_argument("--limit", type=int, help="only the first N seeds (smoke test)")
            sp.add_argument("--run-name", help="override run_name from the config")
        sp.set_defaults(fn=fn)
        return sp

    add("list-seeds", cmd_list_seeds, cfg=False)
    add("estimate", cmd_estimate)
    add("generate", cmd_generate)
    add("judge", cmd_judge)
    add("report", cmd_report).add_argument("--labels", nargs="*")
    add("run", cmd_run)
    add("human-sample", cmd_human_sample).add_argument("-n", type=int)
    add("validate", cmd_validate).add_argument("--labels", nargs="+", required=True)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
