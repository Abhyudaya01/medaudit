"""Generation and judging stages. Both are resumable and concurrency-limited per model."""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig, self_judging_overlap
from .judge import JUDGE_VERSION, judge_response
from .providers import cost_usd, make_provider
from .seeds import Seed, filter_seeds, load_seeds, seed_set_hash
from .store import JsonlStore

log = logging.getLogger("medaudit")


class RunPaths:
    def __init__(self, cfg: RunConfig):
        self.root = Path(cfg.output_dir) / cfg.run_name
        self.responses = self.root / "responses.jsonl"
        self.judgments = self.root / "judgments.jsonl"
        self.manifest = self.root / "manifest.json"
        self.results = self.root / "results"
        self.figures = self.root / "figures"
        self.human = self.root / "human"
        for p in (self.root, self.results, self.figures, self.human):
            p.mkdir(parents=True, exist_ok=True)


def response_key(model: str, seed_id: str, sample: int) -> str:
    return f"{model}|{seed_id}|{sample}"


def selected_seeds(cfg: RunConfig) -> list[Seed]:
    return filter_seeds(load_seeds(), cfg.categories, cfg.seed_ids, cfg.limit)


def write_manifest(cfg: RunConfig, paths: RunPaths, seeds: list[Seed]) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    manifest = {
        "run_name": cfg.run_name,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "seed_set_hash": seed_set_hash(seeds),
        "n_seeds": len(seeds),
        "samples_per_prompt": cfg.samples_per_prompt,
        "judge_version": JUDGE_VERSION,
        "models": [asdict(m) for m in cfg.models],
        "judges": [asdict(j) for j in cfg.judges],
        "self_judging_overlap": self_judging_overlap(cfg),
        "git_commit": commit,
        "python": platform.python_version(),
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2, default=str))


def run_generation(cfg: RunConfig, progress=print) -> RunPaths:
    paths = RunPaths(cfg)
    seeds = selected_seeds(cfg)
    write_manifest(cfg, paths, seeds)
    overlap = self_judging_overlap(cfg)
    if overlap:
        log.warning("Judge model(s) also audited (self-preference bias risk): %s", overlap)
    store = JsonlStore(paths.responses)
    done = {k for k, r in store.load().items() if r["status"] == "ok"}

    for spec in cfg.models:
        todo = [(s, i) for s in seeds for i in range(cfg.samples_per_prompt)
                if response_key(spec.name, s.id, i) not in done]
        if not todo:
            progress(f"[{spec.name}] all {len(seeds) * cfg.samples_per_prompt} responses cached")
            continue
        provider = make_provider(spec, cfg.request_timeout_s, cfg.max_retries)
        progress(f"[{spec.name}] generating {len(todo)} responses")

        def work(seed: Seed, sample: int, provider=provider, spec=spec):
            rec = {"key": response_key(spec.name, seed.id, sample), "model": spec.name,
                   "provider": spec.provider, "model_id": spec.model_id, "seed_id": seed.id,
                   "category": seed.category, "tactic": seed.tactic, "sample": sample,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            try:
                c = provider.complete(list(seed.messages))
                rec.update(status="ok", response=c.text, input_tokens=c.input_tokens,
                           output_tokens=c.output_tokens, latency_s=round(c.latency_s, 4),
                           provider_latency_ms=c.provider_latency_ms, attempts=c.attempts,
                           stop_reason=c.stop_reason,
                           cost_usd=cost_usd(spec, c.input_tokens, c.output_tokens))
            except Exception as e:  # recorded, retried on the next invocation
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[:1000])
            store.append(rec)
            return rec

        n_err = 0
        with ThreadPoolExecutor(max_workers=spec.max_concurrency) as pool:
            futs = [pool.submit(work, s, i) for s, i in todo]
            for n, f in enumerate(as_completed(futs), 1):
                n_err += f.result()["status"] == "error"
                if n % 25 == 0 or n == len(futs):
                    progress(f"[{spec.name}] {n}/{len(futs)} done ({n_err} errors)")
    return paths


def judgment_key(resp_key: str, judge_name: str) -> str:
    return f"{resp_key}|{judge_name}"


def run_judging(cfg: RunConfig, progress=print) -> RunPaths:
    paths = RunPaths(cfg)
    seeds = {s.id: s for s in load_seeds()}
    responses = [r for r in JsonlStore(paths.responses).load().values() if r["status"] == "ok"]
    jstore = JsonlStore(paths.judgments)
    done = {k for k, r in jstore.load().items() if r.get("parsed")}

    for jspec in cfg.judges:
        todo = [r for r in responses if judgment_key(r["key"], jspec.model.name) not in done]
        if not todo:
            progress(f"[judge {jspec.model.name}] all judgments cached")
            continue
        judge = make_provider(jspec.model, cfg.request_timeout_s, cfg.max_retries)
        progress(f"[judge {jspec.model.name}] judging {len(todo)} responses")

        def work(r, judge=judge, jspec=jspec):
            t0 = time.perf_counter()
            rec = {"key": judgment_key(r["key"], jspec.model.name), "response_key": r["key"],
                   "judge": jspec.model.name, "judge_model_id": jspec.model.model_id,
                   "judge_version": JUDGE_VERSION, "model": r["model"], "seed_id": r["seed_id"],
                   "sample": r["sample"]}
            try:
                jr = judge_response(judge, seeds[r["seed_id"]], r["response"], jspec.temperature)
                rec.update(parsed=jr.parsed, raw=jr.raw_text, error=jr.error, repaired=jr.repaired,
                           input_tokens=jr.input_tokens, output_tokens=jr.output_tokens,
                           latency_s=round(jr.latency_s, 4),
                           cost_usd=cost_usd(jspec.model, jr.input_tokens, jr.output_tokens))
            except Exception as e:
                rec.update(parsed=None, error=f"{type(e).__name__}: {e}"[:1000],
                           latency_s=round(time.perf_counter() - t0, 4))
            jstore.append(rec)
            return rec

        n_err = 0
        with ThreadPoolExecutor(max_workers=jspec.model.max_concurrency) as pool:
            futs = [pool.submit(work, r) for r in todo]
            for n, f in enumerate(as_completed(futs), 1):
                n_err += f.result().get("parsed") is None
                if n % 50 == 0 or n == len(futs):
                    progress(f"[judge {jspec.model.name}] {n}/{len(futs)} done ({n_err} unparsed)")
    return paths
