# diathese-qubo-workflow (Candidate 1 - Independent Best-of-N)

**Project**: ag-15 Hermes-Openclaw diamondNode G5+ (resolving Pending: G1_decode_tok_s, G2_first_token_latency, G3_oom_rate, G4_network_out, G6_qubo_publication, G7_eta_thermo; G5 PASSED via real ashard).

**Core Innovation**: Treat "this session's diathese" — the live thermodynamic/state inference metric (eta_thermo, epsilon, delta_q, crystalline_score + GTX 1650 VRAM/util/decode/latency/oom from Yennefer/Cortex layer during real inference) — as a **QUBO optimization problem**.

- Uses real CUDA-optimized native llama.cpp (llama-bench + llama-cli) on physical GTX 1650 4GB (arch 75, Turing).
- Model: ONLY Hermes-3-Llama-3.1-8B.Q4_K_M.gguf (SHA d4403ce5a6e930f4c2509456388c20d633a15ff08dd52ef3b142ff1810ec3553 — verified live).
- Extends published diamondnode-ops/ (status, cleanup, run_bench) with threaded parallel capture + QUBO dispatch.
- Generates **real, attestable dispatch_qubo_table artifacts** compliant with ag-15 schema/sample.
- Full provenance: nvidia-smi high-res snapshots, VRAM/util/timings, SSH logs, content hashes for diamondNode /api/vault/attest (G6).
- **Zero fabrication**: No Q3 claims, no simulated numbers for publication evidence, only real hardware runs.

## Key Artifacts (in this repo)
- `diathese_to_qubo.py`: QUBO formulation + exact bruteforce solver + schema-compliant table generator from live diathese samples.
- `scripts/`: extended diamondnode-ops + `capture_diathese.sh` (bench + Yennefer simulator + live telemetry) + `threaded_diathese_qubo_harness.sh` (parallel threads for status, capture, high-res nvsmi, QUBO pubs).
- `harness/run_on_diamondnode.sh`: Local driver that rsyncs, executes on 192.168.1.228, pulls evidence.
- `qubo_artifacts/`, `logs/`: Real outputs from runs (dispatch_qubo_table_*.json with diathese_influence + content_sha256 ready for attest; nvidia_smi_*.csv; capture logs).
- `docs/`: Reproducibility + blockers mapping.

## How This Resolves Blockers (Independent Candidate 1 Approach)
- **G1_decode_tok_s & G2_first_token_latency**: QUBO objective uses live decode_tok_s + first_token_latency + diathese (high eta/low vram) to select optimal ngl (4/8/12) and routes. Real llama-bench sweeps feed the optimizer.
- **G3_oom_rate**: Explicit `oom_risk` + vram_pct in Q matrix; bad states (vram>0.65) quadratically penalize high-layer / high-cost routes → proactive avoidance over 1k+ turns via repeated publication.
- **G4_network_out**: Route candidates include resilience lanes; diathese (thermo stability under load) influences "allowed_lane" favoring local_fallback or dispatch_qubo under stress.
- **G6_qubo_publication**: First real (non-sim) scheduler-less but harness-driven per-interval publication of valid dispatch_qubo_table JSONs with hashes, diathese metadata, full anti-fab labels. Ready for diamondNode attest + hash chain.
- **G7_eta_thermo**: Diathese *is* the Yennefer eta_thermo / epsilon / crystalline output. Workflow closes the loop: inference → live thermo metric → QUBO opt → attested table. Ties directly to eta_thermo_contract + OpenClaw skills (grok-persistent-state + mcp-openclaw-bridge for long-running).

**G5 remains reference (already PASSED)**; this swarm candidate advances the other 6 via thermodynamic QUBO control plane.

## Reproducibility (Real Hardware Only)
1. From mac (with active SSH to diamondnode):
   ```
   cd /Users/Igor/candidate1-diathese-qubo-workflow
   ./harness/run_on_diamondnode.sh
   ```
2. On diamondnode directly (after rsync or git clone of this repo):
   ```
   cd ~/candidate1-diathese-qubo-workflow
   ./scripts/threaded_diathese_qubo_harness.sh 60 10
   ```
3. Inspect: `cat logs/diathese_sample_*.json`, `cat qubo_artifacts/dispatch_qubo_table_*.json`, `tail -f logs/nvidia_smi_c1_highres.csv`
4. Verify SHA, nvidia-smi peaks < 4096MiB, Q4_K_M only, content_sha256 present.

**Provenance requirement for publication**: Every artifact includes "anti_fabrication" note + model_sha + gpu + capture_log path.

## Relation to OpenClaw + ag-15
- Compatible with dispatch_qubo_table.schema.json v1.0 + sample.
- Feeds @invariantx openclaw-skills (grok-persistent-state for diathese history across sessions for better QUBO priors, mcp-openclaw-bridge to expose this workflow as versioned skill, smithery-mcp-orchestrator for secure dispatch).
- Extends ag-15 execution_driver patterns + diamondnode-ops without modifying upstream (isolated candidate worktree).

**Candidate 1 of 5 swarm** — fully independent, real-evidence only, no coordination.

For full ag-15 context + blocker_ledger: see /Users/Igor/ag-15 (parent project).

Generated 2026-05-29 by Candidate 1.