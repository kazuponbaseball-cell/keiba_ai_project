# GPT-Codex Research Loop v1

## Purpose

This loop uses GPT as a research strategist and Codex as an implementation
executor for ROI-improvement model research. It does not automate purchase,
formal BUY, production promotion, approval, or merge.

The first version produces deterministic files only. It does not call an
external model and does not launch Codex.

## State flow

```text
research context
  -> GPT strategy JSON
  -> strict validation and canonical digest
  -> Research OS proposal
  -> human APPROVED_TO_PREPARE
  -> preparation dispatch packet
  -> implementation and synthetic fixtures
  -> frozen run scope
  -> human APPROVED_TO_RUN
  -> separately implemented real-data runner
  -> result feedback packet
  -> next GPT review
```

The v1 implementation stops at preparation packets and result-feedback
packets. A preparation packet is not a process invocation.

## Separation of responsibilities

### GPT strategist

GPT may propose exactly one falsifiable research change. The strategy must
include a null hypothesis, racing mechanism, target population, chronological
design, primary metric, required effect, rejection gate, stop conditions, and
lineage hashes.

GPT cannot include execution commands, credentials, production paths, orders,
or approval claims. Market, result, payout, popularity, and ROI-derived values
cannot become candidate-selection inputs.

### Research OS

Research OS is the authority for proposal scope and lifecycle state. The
strategy compiler cannot grant approval. GitHub-backed human comments remain
mandatory for preparation, run, and shadow transitions.

### Codex executor

Codex receives a preparation packet only after Research OS records a verified
preparation approval. The packet contains expected paths and synthetic tasks,
but contains no shell commands and cannot trigger a subprocess.

## Safety invariants

Every strategy, dispatch, and feedback packet keeps these values fixed:

```text
formal_buy = false
send_order = false
stake = 0
external_api_calls = false during preparation
actual_codex_dispatch = false during preparation
real_data_execution = false during preparation
production_change = false
```

Any violation is fail-close.

## CLI

Validate a GPT strategy file:

```powershell
python scripts/research/gpt_codex_research_loop_v1.py validate-strategy `
  --strategy path/to/strategy.json
```

Compile its canonical Research OS proposal:

```powershell
python scripts/research/gpt_codex_research_loop_v1.py compile-proposal `
  --strategy path/to/strategy.json `
  --output path/to/proposal.json
```

Create a preparation packet after a verified Research OS transition:

```powershell
python scripts/research/gpt_codex_research_loop_v1.py prepare-dispatch `
  --strategy path/to/strategy.json `
  --registry research/REGISTRY.jsonl `
  --output path/to/dispatch.json
```

Build the deterministic packet that a later integration may return to GPT:

```powershell
python scripts/research/gpt_codex_research_loop_v1.py build-feedback `
  --dispatch path/to/dispatch.json `
  --result-manifest path/to/result_manifest.json `
  --result-summary path/to/result_summary.json `
  --review-prompt path/to/review_prompt.txt `
  --output path/to/feedback.json
```

These commands only validate, hash, and write local JSON. Real model calls and
real-data execution require a future approved scope and a separate adapter.

## ROI-research governance

The loop is intended to improve research throughput without increasing data
snooping. Each cycle permits one variant and no threshold search. A proposed
feature or model must be rejected when it lacks independent information,
stable chronological evidence, or a plausible racing mechanism.

The feedback packet preserves the original model ID, strategy prompt hash,
context hash, proposal digest, command-list hash, result manifest hash, result
summary hash, and review prompt hash. A later GPT review can therefore explain
what was tested without silently changing the approved scope.
