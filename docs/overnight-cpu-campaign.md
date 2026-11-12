# Campaign 0001 — bounded overnight CPU research on public GitHub Actions

## Purpose
Use only public-synthetic, already published register-state mechanism code and ordinary
GitHub-hosted Ubuntu runners for unattended reproducibility, trainability, control,
and replay/receipt qualification. This is not a replacement for Agent Dispatch and
does not use Kaggle tokens, Kaggle GPU/TPU capacity, private hosted Actions,
private holdouts, model tuning against hidden evaluations, or paid API services.

Authority: https://github.com/SemperSupra/latent-reasoning-private/issues/2

## Preregistered finite design

- UTC nightly time: 02:37 UTC (Europe/Berlin 03:37 in November); UTC timestamps govern slots.
- First slot: 2026-11-12; final slot: 2026-12-05; after that NO SCIENCE.
- Eight seed cohorts, two seeds each: 8,9; 10,11; ...; 22,23.
- Within each cohort, K=1 then K=2 then K=4 on successive nights.
- Frozen workload: public register-state synthetic stage-1/2 training, 150 epoch cap per stage,
  common learning rate 0.001, two threads, fixed train/eval configuration.
- For each K and seed: two matched-forward-call treatments (serial-control, latent).
- Both treatments must saturate 100% of the registered ID curriculum per seed before
  any paired OOD comparison is interpreted. Underfit remains recorded, not discarded.
- Only 1 slot is selected per UTC day. Delayed or skipped days are not auto-backfilled;
  the independent versioned ledger is authoritative.

## Automated actions

1. Scheduled or main-branch manual `run`: resolve immutable slot for UTC date.
2. If outside the window, stop before dependency install and execution.
3. Install pinned CPU-only Torch 2.5.1 environment on a standard public Ubuntu runner.
4. Execute the existing runner-neutral `tools/run_replay.py` with the generated fixed JSON spec.
5. Verify spec and scientific-receipt hashes, source identity, treatment/seed grid,
   forward-call equality, ID saturation, and OOD suppression.
6. Emit a compact assessment into GitHub Actions step summary and save scientific,
   metadata, assessment, and exact replay spec as GitHub Actions artifacts (30 days).

Source hash and workload provenance are produced by the existing replay harness.
The framework reports exact forward-call parity (K+1) but DOES NOT assert equal
wall time, FLOPs, memory, or arithmetic cost. Stages 1/2 are not the full admitted
three-stage campaign; even successful paired results are DESCRIPTIVE ONLY.

## Human review and controls

- Pull requests only run standard-library unit tests and offline plan validation;
  no scheduled science or credentialed operation on PR.
- `permissions: contents: read`; no repository writes; no issue/PR generation;
  no hidden state, model key, or environmental secrets.
- A single concurrency group avoids parallel overnight runs for the same ref.
- Job timeout 28 minutes; replay subprocess timeout 18 minutes.
- Keep scientific failures with `if: always()` artifact upload when available.
- If GitHub drops a scheduled run, that slot stays missing; no duplicate resubmission.
- A run with ID-underfit or missing receipts must never become accepted evidence.
- PR review and an owner-authorized merge are required to activate schedule.
- This is finite, not an indefinite autonomous agent or compute search.
- Repo DLE is authoritative. Artifacts have 30-day retention and must be reconciled
  before expiry; no automatic private DLE mutation occurs.

## Why not repeat modular-walk

Campaign 0001 already classified modular-walk as a benchmark-pathology fixture
after its learning curves plateaued. Register-state's valid task contract and
explicit ID gate are stronger. Repeated toy learning alone would mostly consume
CPU time without improving causal interpretability.

## Kaggle CPU follow-on (separate admission)

Kaggle CPU notebooks can help when a public method needs more RAM or longer
bounded duration than a standard GitHub runner. Start with exact same replay
capsule, no accelerator, private notebook, Internet disabled, account quota
observation, and exact native-version receipt. Do not introduce repeated pushes
or a second scheduler while GitHub CPU remains sufficient.

## Closure

After the finite campaign: audit 24 planned slots against observed native GHA
runs; reconcile missing/invalid runs without fabricating data; aggregate only
comparable K/cohorts and independent seed pairs; decide whether to run a larger
admitted stage-3 reproduction on sovereign/Kaggle GPU or continue CPU method work.
