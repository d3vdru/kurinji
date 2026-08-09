# RAMPUP.md — Onboarding for Automatic Optimization of Distributed ML (Training + LLM Inference Serving)

> Companion files: `CLAUDE.md` (project rules), `IDEAS.md` (research tracks T0–T4 and directions A–D).
> Papers below are numbered **[P1]–[P18]**; `IDEAS.md` back-references these as prerequisites per track.
> Each topic block carries a **Used in:** tag mapping it to the tracks where it pays off.

## TL;DR
- Phased 6-week ramp: **Weeks 1–2** distributed-training + GPU-interconnect + collectives foundations; **Weeks 3–4** scheduling and queueing theory + learned cost models; **Weeks 5–6** hands-on with ASTRA-sim/Chakra/collective synthesizers and vLLM/SGLang. Nearly all core resources are free/open.
- Center of gravity: **ASTRA-sim + Chakra + collective synthesis (TACOS/TACCL/MSCCL)** for training; **vLLM/SGLang + DistServe/Vidur** for inference; classical theory (Graham, Thakur, Little) underneath.
- Budget ~120–150 hours; read ~one paper/week from the [P#] list ongoing.

---

## Phase 0 — Prerequisites (assumed / quick refresh)
Python + PyTorch fluency, basic linear algebra, transformer fundamentals. If GPU programming is shaky, do a 1-week PMPP/GPU MODE detour first.
**Used in:** everything.

## Phase 1 (Weeks 1–2): FOUNDATIONS

### 1a. Distributed ML training (data/tensor/pipeline/expert parallelism)
**Used in:** T2 (defines the config search space TP×PP×DP×EP), T4-stretch (joint synthesis needs overlap concepts), T0 (knowing what to trace), and all paper-reading.
- **HuggingFace Ultra-Scale Playbook** — https://huggingface.co/spaces/nanotron/ultrascale-playbook — 5D parallelism, ZeRO 1–3, activation recomputation, HFU/MFU, comm–compute overlap; grounded in 4,100+ distributed experiments up to 512 GPUs. Free. ~15–25 hrs.
- **Lilian Weng, "How to Train Really Large Models on Many GPUs?"** — https://lilianweng.github.io/posts/2021-09-25-train-large/ — concise survey. Free. ~2 hrs.
- **Stanford CS336** — https://cs336.stanford.edu/spring2025 (videos: YouTube playlist) — Assignment 2 (Systems): profiling, Triton kernel, memory-efficient distributed model. Free to audit (some assignments need paid GPU rental). ~20+ hrs.

### 1b. GPU/accelerator architecture and interconnects
**Used in:** T1 (bound calculator needs topology + per-link α,β), T2 (hardware descriptor vectors), T4 (topology inputs to the gym), directions A/B (what resources contend).
- **NVIDIA NVLink & NVSwitch blog** — https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/ — Free. ~1 hr.
- **NVSwitch Technical Overview (PDF)** — https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf — Free. ~1 hr.
- **UMD CMSC714 Lecture 10: Fat-tree & Dragonfly** — https://www.cs.umd.edu/class/fall2019/cmsc714/slides/10-cmsc714-fattree-dragonfly.pdf — diameter/bisection-bandwidth intuition. Free. ~1–2 hrs.
- **Survey of HPC interconnection networks** (MDPI, open access) — https://www.mdpi.com/2079-9292/11/9/1369 — k-ary n-cubes, fat-tree, torus/mesh, dragonfly. Free. ~2 hrs.
- **GPU MODE lectures** — https://github.com/gpu-mode/lectures — esp. Lecture 17 (NCCL collectives), 12–13 (Flash/Ring Attention). Free. ~5–8 hrs selective.

### 1c. Collective communication + α-β model
**Used in:** T1 (the profiler IS this topic operationalized), T4 (baselines + reward validation vs bounds), direction D (bounds to prove against), T2 (collective algo is a config dimension).
- **[P1] Thakur, Rabenseifner, Gropp 2005** — https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf — canonical: recursive doubling, ring, Rabenseifner; α-β (latency-bandwidth) reasoning; algorithm selection by message size. Free. ~3 hrs.
- **NCCL** — https://github.com/NVIDIA/nccl + docs — ring/tree AllReduce, LL/LL128/Simple protocols. Free.
- **nccl-tests** — https://github.com/nvidia/nccl-tests (+ PERFORMANCE.md busbw doc) — microbenchmarking. Free. ~2 hrs hands-on.
- **"Demystifying NCCL" (2025)** — https://arxiv.org/html/2507.04786v1 — optional deep dive.

## Phase 2 (Weeks 3–4): THEORY

### 2a. Scheduling theory
**Used in:** T4-stretch (joint synthesis = P|prec,c|C_max in disguise), direction D (theory bridge — the main consumer), and calibrating expectations everywhere (why the field uses heuristics/ILP, what guarantees don't exist).
- **[P2] Graham 1966/1969** — list scheduling, (2−1/m) bound. (Library/DOI; 1966 Bell Syst. Tech. J. / 1969 SIAM J. Appl. Math.)
- **Williamson & Shmoys, Design of Approximation Algorithms** — free PDF https://www.designofapproxalgs.com/book.pdf — Ch.1 + scheduling sections. ~6 hrs.
- **[P3] Davies–Kulkarni–Rothvoss et al. 2020** — Scheduling with Communication Delays via LP Hierarchies (FOCS '20, arXiv:2004.09682); weighted/related-machines version SODA '21 (O(log⁴ n)): https://par.nsf.gov/servlets/purl/10299233
- **[P4] Hardness of Non-Uniform Communication Delays** — arXiv:2105.00111 — logarithmic hardness; answers a top-ten open problem. Remember the baseline: greedy list scheduling gives (c+1)-approx in the unbounded-machine case.
- **Textbook anchor:** Pinedo, *Scheduling* (library). Know Svensson's P|prec|C_max inapproximability by name.

### 2b. Queueing theory & operational analysis
**Used in:** T3 (the core analytical frame: two coupled Little's-law occupancy constraints), T0 (demand characterization: tails, autocorrelation), directions A (interference as shared-resource contention) and B (heterogeneous-server routing).
- **[P5] Harchol-Balter, Performance Modeling and Design of Computer Systems** — front matter + Ch.1 excerpt free at Cambridge; full text via library. Little's law, utilization/bottleneck laws, M/M/1, scheduling policies, heavy tails. ~10 hrs on Parts I–III.

### 2c. Learned cost models / ML for systems
**Used in:** T2 (the direct template — our track is "TpuGraphs for the distributed layer"), T4 (learned reward shaping, gated).
- **[P6] TpuGraphs (NeurIPS 2023)** — arXiv:2308.13490 — performance prediction on tensor graphs with GNNs; XLA autotuner found 10–20% production speedups; the Kaggle "Fast or Slow?" competition accompanied it. Free. ~3 hrs.

## Phase 3 (Weeks 5–6): PRACTICAL TOOLS & HANDS-ON

### 3a. Simulators
**Used in:** T2 (sweep engine + sim-to-real calibration target), T4 (the gym's backend), T3 (Vidur as design reference for the replayer; counterfactuals).
- **[P8] ASTRA-sim** — https://astra-sim.github.io/ — getting-started docs, Docker setup, tutorials (MICRO 2024; upcoming ISCA 2026, Jun 27, Raleigh). Pluggable compute (roofline, SCALE-sim) and network (analytical, Garnet, ns-3) backends. Free. ~10 hrs.
- **ns-3** — learn basics only if touching the detailed network backend (T4 fidelity escalation, T2 active-learning stretch).
- **[P18] Vidur (MLSys 2024)** — https://github.com/microsoft/vidur , arXiv:2405.05465 — LLM inference simulator, <9% latency error; Vidur-Search found best LLaMA2-70B deployment in 1 CPU-hour vs ~42K GPU-hours (~$218K) of real exploration. Free. ~4 hrs.

### 3b. Execution traces
**Used in:** T0 (the trace pipeline is a Track 0 deliverable), T2 (training data), T4 (workload inputs).
- **[P7] MLCommons Chakra** — https://github.com/mlcommons/chakra (USER_GUIDE.md), arXiv:2305.14516 — standardized DAG schema; pipeline: PyTorch profiler (host ET + Kineto) → chakra_trace_link → chakra_converter → ASTRA-sim et_feeder. Free. ~5 hrs hands-on.

### 3c. Collective synthesis
**Used in:** T4 (baselines to reproduce and beat), T1 (synthesized schedules to profile against bounds), direction D (objects of any proof).
- **[P10] TACOS (MICRO 2024)** — arXiv:2304.05301 — time-expanded-network greedy synthesis; 4.80× vs Direct; 40K NPUs in 2.52 h; ~97% of α-β bound on tested topologies.
- **[P9] TACCL (NSDI 2023)** — arXiv:2111.04867 , https://github.com/microsoft/taccl (needs Gurobi license) — ILP + communication sketches; up to 6.7× vs NCCL; outputs run via MSCCL stack.
- **[P11] MSCCLang (ASPLOS 2023)** — https://dl.acm.org/doi/10.1145/3575693.3575724 ; runtime https://github.com/microsoft/msccl ; tools https://github.com/microsoft/msccl-tools ; MSCCL++ https://github.com/microsoft/mscclpp — chunk-oriented DSL + compiler; NCCL fallback.
- Optional: standardized collective representation bridging Chakra — arXiv:2408.11008.

### 3d. Auto-parallelism
**Used in:** T2 (the search that consumes our cost model; Alpa's ILP+DP decomposition is the integration target).
- **[P12] Alpa (OSDI 2022)** — https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf , https://github.com/alpa-projects/alpa — inter-op (DP) + intra-op (ILP) unified.
- **[P13] Galvatron (VLDB 2023)** — doi:10.14778/3570690.3570697 — automatic parallelism for transformers.
- **FlexFlow** — SOAP space + MCMC search; read via Alpa's related work.

### 3e. LLM inference engines & serving research
**Used in:** T3 (the scheduler shim lives inside vLLM v1; policies under study are these), direction B (routing across engines/hardware), T0 (what serving telemetry to collect).
- **[P15] vLLM / PagedAttention (SOSP 2023)** — "Inside vLLM" anatomy: https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html ; V1 architecture: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html — V1 scheduler mixes prefill+decode per step; block-table KV management. ~4 hrs.
- **[P16] SGLang / RadixAttention** — arXiv:2312.07104 , https://www.lmsys.org/blog/2024-01-17-sglang/ — LRU radix-tree KV reuse; up to 6.4× throughput on prefix-heavy workloads. ~3 hrs.
- **[P14] Orca (OSDI 2022)** — https://www.usenix.org/system/files/osdi22-yu.pdf — iteration-level scheduling + selective batching (origin of continuous batching); 36.9× vs FasterTransformer on GPT-3 175B.
- **[P17] DistServe (OSDI 2024)** — arXiv:2401.09670 — prefill/decode disaggregation; 7.4× more requests / 12.6× tighter SLO. Companion: **Splitwise (ISCA 2024)** — arXiv:2311.18677 — phase splitting with KV-transfer overlap; 1.4× throughput at 20% lower cost.
- **FastServe** — arXiv:2305.05920 (NSDI '26 as "Iteration-Level Preemptive Scheduling...") — skip-join MLFQ, preemptive scheduling. (Grouped under [P18] with Vidur.)

## COURSES / TUTORIALS / VIDEOS (ongoing)
- **MIT 6.5940 EfficientML.ai** — https://hanlab.mit.edu/courses/2024-fall-65940 — distributed training, auto parallel optimization, LLM inference; hands-on labs. Free.
- **CMU MLSys (15-849 / Zhihao Jia)** — LLM-serving slides, e.g. https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/15-LLM-serving-part1.pdf
- **GPU MODE** community lectures; **ASTRA-sim / Chakra conference tutorials** (MICRO 2024, MLSys; ISCA 2026 upcoming).

---

## Annotated Reading List [P1]–[P18]
Read roughly in order; ~one/week.

| # | Paper | Why it matters here | Feeds |
|---|-------|---------------------|-------|
| P1 | Thakur et al. 2005, MPICH collectives | α-β model + canonical algorithms | T1, T4, D |
| P2 | Graham 1969, list scheduling | (2−1/m) bound; the field's origin | D, T4 |
| P3 | Davies et al. 2020/21, comm-delay scheduling | modern polylog approx for P\|prec,c\|C_max | D |
| P4 | Davies et al. 2021, hardness | log-hardness; what's provably out of reach | D |
| P5 | Harchol-Balter (book) | Little's law, bottleneck laws, tails | T3, T0, A, B |
| P6 | TpuGraphs 2023 | learned cost model template | T2 |
| P7 | Chakra 2023 | trace schema + profiler pipeline | T0, T2, T4 |
| P8 | ASTRA-sim 2.0 2023 | the simulator backbone | T2, T4 |
| P9 | TACCL 2023 | ILP + sketches synthesis | T4, T1 |
| P10 | TACOS 2024 | scalable greedy synthesis; 97%-of-bound bar | T4, T1 |
| P11 | MSCCLang 2023 | DSL/runtime to execute custom collectives | T4 |
| P12 | Alpa 2022 | auto-parallelism decomposition | T2 |
| P13 | Galvatron 2023 | transformer-specific auto-parallelism | T2 |
| P14 | Orca 2022 | continuous batching origin | T3 |
| P15 | vLLM/PagedAttention 2023 | paged KV; V1 scheduler internals | T3 |
| P16 | SGLang/RadixAttention 2024 | cache-aware scheduling | T3 |
| P17 | DistServe 2024 (+Splitwise) | prefill/decode disaggregation | T3, B |
| P18 | Vidur 2024 (+FastServe) | inference simulation; preemptive scheduling | T3, B |

---

## How Each Topic Maps to the Research Tracks (IDEAS.md)

| RAMPUP topic | T0 data | T1 bound-gap | T2 cost model | T3 serving | T4 gym | A interf. | B routing | C census | D theory |
|---|---|---|---|---|---|---|---|---|---|
| 1a Distributed training | ● | | ●● | | ● | | | | |
| 1b Interconnects/topologies | | ●● | ● | | ●● | ● | ● | | |
| 1c Collectives + α-β [P1] | | ●●● | ● | | ●● | | | | ●● |
| 2a Scheduling theory [P2–P4] | | | | | ● | | | | ●●● |
| 2b Queueing [P5] | ●● | | | ●●● | | ●● | ●● | | |
| 2c Learned cost models [P6] | | | ●●● | | ● | | | | |
| 3a ASTRA-sim/Vidur [P8,P18] | | | ●● | ●● | ●●● | | ● | | |
| 3b Chakra [P7] | ●●● | | ●● | | ● | | | | |
| 3c Synthesis tools [P9–P11] | | ●● | | | ●●● | | | | ● |
| 3d Auto-parallelism [P12,P13] | | | ●● | | | | | | |
| 3e Inference engines [P14–P17] | ● | | | ●●● | | ● | ●● | | |

(●●● = core prerequisite, ●● = strongly used, ● = helpful context. Direction C needs only basic scripting + cluster-log access.)

**Minimal path per track** (if you must start a track early):
- **T0**: 1a skim + [P7] + [P5] Ch.1–2. (~1 week)
- **T1**: 1b + 1c fully, incl. nccl-tests hands-on. (~1.5 weeks)
- **T2**: 1a + [P6] + [P7] + [P8] + [P12]. (~2.5 weeks)
- **T3**: [P5] Parts I–III + [P14]–[P16] + vLLM anatomy blog + [P18]. (~2 weeks)
- **T4**: 1c + [P8] + [P9]/[P10] + basic RL only if the Track 1 gate passes. (~2 weeks)
- **D**: 2a fully + [P1] bounds section; everything else optional. (theory-track student)

---

## Weekly Benchmarks
- **End Week 2:** explain why ring AllReduce is bandwidth-optimal and where tree wins (latency-bound short messages); run nccl-tests and interpret busbw.
- **End Week 4:** state Graham's (2−1/m), the (c+1)-approx for P|prec,c|C_max, and Little's law; apply each to a toy problem.
- **End Week 6:** produce a simulated collective time in ASTRA-sim from a self-generated Chakra trace; produce a real TTFT/throughput-vs-batch curve from local vLLM and SGLang; run TACOS greedy on a toy mesh.

## Caveats
- ISCA 2026 ASTRA-sim tutorial is upcoming (materials forthcoming); use MICRO 2024 / ASPLOS materials meanwhile.
- Harchol-Balter full text is not free (library); Williamson–Shmoys and Ultra-Scale Playbook are fully free. Avoid pirated copies.
- CS336 assignment 3 needs a Stanford-only API; audit lectures + other assignments.
- Reported speedups (Orca 36.9×, DistServe 7.4×, SGLang 6.4×, TACOS 4.80×) are each paper's best case on specific setups — not guarantees.
- TACCL's synthesizer requires a Gurobi license (free academic licenses exist).
- FastServe's arXiv and NSDI '26 titles differ; same work. Graham 1966 vs 1969 are two related papers — cite the one matching the bound you invoke.
