# IDEAS.md — Auto-Optimizing Distributed ML for Given Hardware Topologies

## 1. Vision

Given a chipset / cluster topology description, **automatically derive near-optimal**:
(a) collective communication algorithms, (b) parallelism strategies (TP/PP/DP/EP),
(c) op-level schedules with compute–communication overlap, and (d) inference serving
policies — replacing hand-designed heuristics with search/learning that is *calibrated
against real hardware* and, where possible, *benchmarked against theoretical bounds*.

Analogy: the Google MLSys competition (TpuGraphs) did this for XLA compiler configs via
learned cost models; we extend the paradigm to the distributed layer (network, collectives,
parallelism, serving), exploiting a unique asset: **a shared multi-tenant cluster running
the same models across multiple hardware types for training and inference**.

---

## 2. State of the art — what is done, what is open

### 2.1 Collective algorithm synthesis (mature, but gaps)

- **TACCL** (NSDI '23, arXiv:2111.04867): ILP-based synthesis guided by human
  "communication sketches"; profiles α (latency) and β (bandwidth) costs per link type.
  Scales only to tens of NPUs (NP-hard core).
- **TACOS** (MICRO '24, arXiv:2304.05301): time-expanded-network + greedy synthesis.
  Synthesizes All-Reduce for a heterogeneous 128-NPU system in ~1 s; scales to 40K NPUs
  in ~2.5 h; reports ~97% efficiency vs. the α-β theoretical bandwidth bound and
  ~2.5–4.8× speedups over Ring/Direct/TACCL-like baselines on mesh/dragonfly/switch
  topologies. From the same Georgia Tech Synergy Lab as ASTRA-sim.
- **MSCCL / MSCCL-IR + CollectiveAPI** (HOTI '24): standardized representation and
  NCCL-based runtime for executing custom collectives on real clusters.
- **Newer**: TidalMesh (HPCA '25, mesh AllReduce), Canvas (ICNP '25, large-scale
  collective scheduling).

**Open**: irregular/degraded/asymmetric topologies; contention-aware synthesis;
latency-bound (small-message) regimes; joint synthesis with compute overlap; any
approximation guarantees.

### 2.2 Auto-parallelism (mature, but cost models are the weak link)

- **Alpa** (OSDI '22): ILP for intra-op + DP for inter-op parallelism. **FlexFlow**
  (MCMC over SOAP space), **Galvatron**, **Piper**, **Aceso**; heterogeneous-cluster
  extensions (HexiScale arXiv:2409.01143, Astra arXiv:2502.13480, HETHUB).
- Acknowledged weakness (e.g., HyGra arXiv:2603.12671): accurate estimation of
  inter-node communication latency under different topologies/strategies remains
  unsolved; practice still relies on expert empirical tuning.

### 2.3 Inference-engine scheduling (young — best novelty/crowding ratio)

- vLLM v1 scheduler is **decode/running-first**; SGLang is **prefill-first** with
  cache-aware (RadixAttention) scheduling — hand-designed heuristics
  (see Frontier LLM-inference simulator paper, arXiv:2605.21312, App. B.4; policy code:
  `vllm/v1/core/sched/scheduler.py`, `sglang/srt/managers/scheduler.py`).
- FastServe (NSDI '26): iteration-level preemptive scheduling.
- No convincing learned/synthesized policy shown to dominate across workload
  distributions; multi-node inference (prefill/decode disaggregation → KV transfers
  over network; MoE all-to-all dispatch) barely explored vs. training collectives.

### 2.4 Theory anchors

- **Graham (1966–69)**: list scheduling within (2 − 1/m) of optimal for P|prec|C_max;
  Svensson: beating 2 is UGC-variant-hard. **With communication delays** (P|prec,c|C_max)
  guarantees collapse; first polylog approximations only ~2020–21 (Davies, Kulkarni,
  Rothvoss et al.), still assuming fixed delays — no results under topology + contention.
- **α-β model bounds**: ≥ ⌈log₂ p⌉ rounds latency; 2(p−1)/p · n/β All-Reduce bandwidth.
  These are the "theoretical ideal" TACOS-style tools report against.
- **Operational laws** (Little, bottleneck/utilization): distribution-free; inference
  serving has *two coupled occupancy constraints* — compute-time occupancy and
  KV-token-memory occupancy (L_kv = λ · E[tokens held × residence time]); which binds
  depends on workload length distributions **and hardware type** (memory/compute balance
  differs per chipset). Classical closed-form queueing breaks (state-dependent service
  rates from batching, history-dependent service from RadixAttention, heavy-tailed
  autocorrelated demand), but the laws remain valid as measurement/validation tools and
  bounds.

**Theory-flavored open problem**: any nontrivial approximation bound for topology-aware
DAG scheduling under a contention model; or a synthesized/learned scheduler provably
approaching α-β bounds on a topology class.

---

## 3. Empirical asset & methodology

Shared multi-tenant cluster; same models on multiple hardware types; training + inference
cohabiting. Value: real demand processes, sim-to-real calibration data, contention and
interference measurements, failure/elasticity statistics — the exact data absent from
most published work (which uses Poisson arrivals + quiet single-tenant clusters).

**Gold-standard evaluation triangle**: real trace → calibrated simulator (counterfactuals)
→ small validated-on-cluster spot-check.

**Constraints**: consent + anonymization agreement before collection; metadata only
(never request content); passive, low-overhead telemetry (guest etiquette + avoid
observer bias); observational data has no counterfactuals — use the simulator for those.

---

## 4. Tracks

> **Prerequisite notation**: [P#] refers to the numbered reading list in `RAMPUP.md`
> (P1 Thakur collectives … P18 Vidur/FastServe); 1a–3e refer to its topic sections.
> RAMPUP.md §"How Each Topic Maps to the Research Tracks" has the full matrix and a
> minimal fast-path per track.

### Track 0 — Data foundation (do first; ~2–3 weeks)

**Goal**: versioned trace dataset + demand characterization. Everything downstream
inherits credibility from this.

**Prerequisites**: RAMPUP 1a (skim — know what DP/TP/PP traffic looks like so collectors
capture the right fields); [P7] Chakra (the profiler→trace pipeline is action item 3);
[P5] Harchol-Balter Ch.1–2 (demand characterization in action item 4 uses arrival
processes, tails, autocorrelation). Fast path: ~1 week.

Requirements: cluster telemetry access; PyTorch profiler + Chakra converter; storage for
traces; signed consent/anonymization doc.

Action items:
1. Draft + sign consent/anonymization agreement (metadata-only; pseudonymized tenants).
2. Passive collectors: arrivals, prompt/output lengths, batch + KV occupancy, hardware
   assignment, job type (train/infer).
3. Chakra pipeline on 2–3 representative training jobs **per hardware type**.
4. Demand characterization notebook: arrival autocorrelation, length tails,
   prefix-sharing rate, diurnal load.
5. Deliverable: dataset v0 + internal report (workshop paper if releasable).

### Track 1 — Cross-hardware bound-gap profiler (cheap; decision gate)

**Goal**: headroom map — gap to α-β bound by hardware type × topology × message size ×
contention (quiet vs busy windows).

**Prerequisites**: RAMPUP 1c is the core — [P1] Thakur et al. (α-β model, ring/tree
algorithms, message-size regimes) is exactly what the bound calculator implements;
nccl-tests hands-on (busbw doc) for the microbenchmarks; RAMPUP 1b (topologies —
bisection bandwidth, fat-tree/dragonfly) to parameterize bounds per topology. Context:
[P9][P10] to know what synthesized baselines exist. Fast path: ~1.5 weeks.

Requirements: NCCL/RCCL microbenchmark rights on each hardware type; α-β bound
calculator; quiet + busy measurement windows.

Action items:
1. Implement bound calculator per (topology, collective, message size).
2. Microbenchmark collectives per hardware type, sweeping message sizes, in quiet and
   busy windows.
3. Publish headroom heatmaps.
4. **Decision gate**: contention gap < 10% → deprioritize contention-aware synthesis;
   ≥ 30% → it becomes the thesis spine.

### Track 2 — Cross-hardware learned cost model (TpuGraphs analogue; headline track)

**Goal**: predict config *rankings* (Kendall's τ) across hardware; key experiment —
train on hardware A + B, predict on unseen hardware C ("optimize for a chipset you
haven't exhaustively profiled" = the vision as a measurable result).

**Prerequisites**: [P6] TpuGraphs (the template: graph-level performance prediction,
ranking metrics, GNN baselines); RAMPUP 1a (Ultra-Scale Playbook — the TP×PP×DP config
space being swept); [P7] Chakra + [P8] ASTRA-sim (sweep engine, action items 1–3);
[P12] Alpa + [P13] Galvatron (the searches that will consume the model, action item 6);
RAMPUP 1b for hardware descriptor design. GNN/graph-transformer background assumed.
Fast path: ~2.5 weeks.

Requirements: ASTRA-sim sweep runner; ~10K simulated configs per hardware profile;
real iteration times from cluster (Track 0); GNN/graph-transformer stack; hardware
descriptor schema.

Action items:
1. Define config schema (TP × PP × DP × topology × collective algo × model).
2. Build sweep runner (Chakra workload gen → ASTRA-sim → parse).
3. Measure ASTRA-sim per-hardware error vs. real; train residual correction model.
4. GNN over Chakra DAG + hardware descriptor; metric = Kendall's τ, baseline =
   analytical α-β cost model.
5. Cross-hardware transfer experiment (A+B → C).
6. Stretch: active learning to allocate high-fidelity (ns-3) simulation budget; plug
   into Alpa-style search, measure end-to-end search speedup.

### Track 3 — Queueing-grounded inference scheduling on real demand

**Goal**: rigorous policy science for vLLM/SGLang scheduling; first result — do policy
rankings (prefill-first vs decode-first vs cache-aware) **flip** between Poisson-synthetic
and real traces?

**Prerequisites**: [P5] Harchol-Balter Parts I–III is the analytical backbone (Little's
law and the two occupancy constraints in action item 4; policy comparison methodology);
[P14] Orca → [P15] vLLM/PagedAttention (+ "Inside vLLM" anatomy blog — required before
touching the shim in action item 1) → [P16] SGLang/RadixAttention, in that order — each
policy under study comes from one of these; [P18] Vidur (replayer design reference,
action item 2); [P17] DistServe/Splitwise for the generalization step (action item 6).
Fast path: ~2 weeks.

Requirements: vLLM v1 source familiarity; GPU node(s) for replay; Track 0 traces;
queueing analysis tooling.

Action items:
1. Pluggable scheduler shim behind clean policy interface in vLLM v1.
2. Trace replayer consuming Track 0 traces (+ synthetic generators for control).
3. Pareto surfaces (TTFT/TPOT/goodput vs load) per policy × workload regime ×
   hardware type; test ranking flips.
4. Instrument both Little's-law occupancy states (compute-time, KV-token); identify
   binding constraint per regime and per chipset (it may flip across hardware).
5. Learned admission/preemption/chunked-prefill policy with occupancy-state features;
   evaluate on held-out trace days.
6. Generalization: disaggregated prefill/decode → KV-transfer scheduling →
   topology-aware (merges with Track 4 / direction B).

### Track 4 — CollectiveGym, contention-calibrated

**Goal**: reusable simulator-in-the-loop search harness; escalate to RL/LLM-guided
synthesis only if Track 1 gate passes.

**Prerequisites**: [P8] ASTRA-sim (env backend, action item 1); [P1] + RAMPUP 1c
(bound validation in action item 2); [P10] TACOS (the baseline to reproduce — its TEN
representation also inspires the state/action encoding) and [P9] TACCL for the ILP
alternative; [P11] MSCCLang if synthesized schedules are to run on real hardware.
Gated additions: basic RL (policy gradients / MCTS) only after the Track 1 gate;
RAMPUP 2a scheduling theory for the joint-synthesis stretch (action item 5), which is
P|prec,c|C_max in disguise. Fast path: ~2 weeks.

Requirements: ASTRA-sim analytical backend; gym-style wrapper; baselines (Ring, Direct,
TACOS-greedy); measured degradation distributions from Track 1.

Action items:
1. Env: state = topology + chunk locations; action = (chunk, link, timestep);
   reward = −simulated collective time.
2. Validate: reproduce Ring/Direct/TACOS numbers on 5×5 mesh + small dragonfly;
   check vs α-β bound.
3. Inject measured contention scenarios from Track 1.
4. (Gated) RL / LLM-guided search on degraded/asymmetric topologies and small-message
   regimes where TACOS-greedy is weakest.
5. (Stretch, joint-synthesis) extend action space with compute-op interleaving —
   entry to co-optimizing collective + overlap across the abstraction barrier.

---

## 5. New directions unlocked by this cluster

**A. Interference matrix + interference-aware placement.** Measure pairwise slowdown
(job type × job type × shared resource) for cohabiting training and inference; almost no
public data exists. Measurement study first (novel alone), placement policy second.
Requires: co-location observation windows or scheduled probe pairs; per-resource counters.
*Reading prereqs*: Track 0 telemetry running; [P5] (contention as shared-resource
queueing); RAMPUP 1b (which resources — links, HBM, PCIe — actually contend).

**B. Heterogeneous fleet routing.** Route each inference request to the hardware type
minimizing its cost — heterogeneous-server queueing with Track 2's cost model as the
router. Unifies queueing + cost model + hardware diversity; deployable on our cluster
(strong evaluation story). Requires: Tracks 0–3 partial outputs; a routing layer in
front of engines.
*Reading prereqs*: [P5] (heterogeneous-server queueing / routing policies); [P17]
DistServe + Splitwise (phase-aware placement is the nearest published relative);
[P18] Vidur (simulate routing policies before deploying); Track 2's cost model as
the router's brain.

**C. Failure/elasticity census.** Mine cluster logs for link/node incident rates and job
churn. Cheap option-value: gates the online re-synthesis direction (re-derive
collective + parallelism plan in milliseconds after topology change) with real incident
frequencies before investing.
*Reading prereqs*: none beyond scripting + log access — deliberately the lightest
direction; a good first task for a new student in parallel with RAMPUP Phase 1.

**D. Theory bridge (long shot, high payoff).** Approximation guarantees for DAG
scheduling under topology + contention; or provable near-α-β synthesis on a topology
class. Outlives any particular system.
*Reading prereqs*: RAMPUP 2a in full — [P2] Graham, [P3] LP-hierarchy approximations,
[P4] hardness (know exactly where the open gap sits: non-uniform delays + contention),
Williamson–Shmoys for technique; [P1] for the α-β bounds any synthesis proof must
target. This is the theory-student track; RAMPUP Phase 3 tooling is optional context.

---

## 6. Sequencing & gates

1. Track 0 immediately → Track 1 + direction C in parallel (both cheap decision gates).
2. Commit spine: **Track 2 + B** if cross-hardware transfer looks alive; **Track 3** if
   demand data is the richer asset.
3. Track 4 escalation gated on Track 1 headroom; direction D opportunistic alongside
   whichever spine produces the cleanest structure.

## 7. Reference index

- ASTRA-sim 2.0 — ISPASS '23, arXiv:2303.14006; tutorials: https://astra-sim.github.io/tutorials
- Chakra — arXiv:2305.14516; MLCommons WG: https://mlcommons.org/working-groups/research/chakra/
- CollectiveAPI — HOTI '24, IEEE 10664245
- TACOS — MICRO '24, arXiv:2304.05301 · TACCL — NSDI '23, arXiv:2111.04867
- Alpa — OSDI '22 · FlexFlow — MLSys '19 · Galvatron — arXiv:2211.13878 ·
  HexiScale — arXiv:2409.01143 · Astra (parallel strategy search) — arXiv:2502.13480 ·
  Auto parallel-strategy planning — arXiv:2501.00254
- Frontier (LLM inference simulator; scheduler policy contrast) — arXiv:2605.21312
- FastServe — NSDI '26 · ByteScheduler — SOSP '19 · HyGra — arXiv:2603.12671 ·
  PrismLLM — arXiv:2605.15617 · Lumos — arXiv:2504.09307
- Theory: Graham bounds (1966/69); Svensson hardness; Davies–Kulkarni–Rothvoss et al.
  (~2020–21) on precedence + communication delays; α-β collective lower bounds;
  Little's law / operational analysis.
