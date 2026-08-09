# CLAUDE.md

## Project

Research program: **automatic optimization of distributed ML algorithms (collectives, parallelism, scheduling) for a given chipset / cluster topology**, spanning training and inference (vLLM/SGLang). Long-term vision: given a hardware description, automatically derive near-optimal communication algorithms, parallelism strategies, and serving policies — with guarantees or calibrated learned models rather than hand heuristics.

We have access to a **shared multi-tenant cluster** running the same models across **multiple hardware types** for both training and inference. This is our key empirical asset: real demand traces, cross-hardware measurements, and contention/interference data that simulation-only groups lack.

See `IDEAS.md` for the full problem list, references, action items, and decision gates.
See `RAMPUP.md` for the onboarding plan; IDEAS.md tracks cite its reading list as [P1]–[P18].

## Core methodology

Real trace → calibrated simulator → counterfactual search → spot-check on hardware. Never trust simulation-only numbers for claims; never run intrusive experiments on the shared cluster without approval.

## Key infrastructure

- **ASTRA-sim** (https://github.com/astra-sim/astra-sim) — distributed ML simulator; analytical / Garnet / ns-3 network backends.
- **Chakra** (https://github.com/mlcommons/chakra) — MLCommons standardized execution traces; PyTorch profiler → Chakra pipeline for capturing real jobs.
- **TACOS / TACCL / MSCCL-IR** — collective algorithm synthesizers and runtime IR; our baselines.
- **vLLM v1 / SGLang** — inference engines; we build a pluggable scheduler shim + trace replayer on vLLM v1 (`vllm/v1/core/sched/scheduler.py`).

## Repo conventions

- `traces/` — versioned, anonymized cluster traces (NEVER commit request content; metadata only, per consent agreement).
- `sim/` — ASTRA-sim configs, Chakra workloads, sweep runners.
- `gym/` — CollectiveGym environment wrapper.
- `costmodel/` — dataset generation + GNN cost model training.
- `serving/` — vLLM scheduler shim, replayer, queueing analysis notebooks.
- `profiler/` — α-β bound-gap profiler and collective microbenchmarks.
- `docs/` — experiment logs; one markdown note per experiment with config hash, date, hardware.

## Hard rules

1. **Privacy**: no request content in any trace, log, or commit. Tenant IDs pseudonymized. Consent doc must be signed before any collection.
2. **Passive telemetry only** on shared cluster; active experiments need explicit scheduling with cluster admins.
3. Every performance claim states: hardware type, topology, message-size regime, contention condition (quiet vs busy), and the theoretical bound where one exists (α-β).
4. Ranking metrics (Kendall's τ) over absolute-error metrics for cost models.
5. Respect decision gates in `IDEAS.md` before escalating to expensive tracks (RL synthesis, dynamic re-synthesis).

## Current status

- Phase: Track 0 (data foundation) not yet started.
- Next actions: consent/anonymization agreement; telemetry collectors; Chakra pipeline on 2–3 training jobs per hardware type; demand characterization notebook.
