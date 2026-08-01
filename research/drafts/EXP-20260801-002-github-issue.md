# Realtime quote to paper-decision wiring v1

## Objective

Close the current gap between per-race odds refresh and paper-only value/no-bet decisions while preserving the frozen non-odds candidate.

## Canonical scope

- Experiment: `EXP-20260801-002`
- Proposal: `research/scopes/EXP-20260801-002.proposal.json`
- Proposal digest: `52a34b8e0b33948b3eaceca366e23d732919bb8923d7add3f2a759d97540fc19`
- Base commit: `385a418827fd6d1cbf9868507d53ad7a04fe172d`
- Score: `84/100`

## Preparation boundary

Preparation may add only the registered research-only coordinator, configuration, and synthetic tests. It may not read race-day data, modify production runtime scripts, execute TARGET handoff, enable formal BUY, use a real stake, or send an order.

## Required human approval

After reviewing the canonical proposal, an allowlisted human must add this exact comment:

```text
APPROVED_TO_PREPARE 52a34b8e0b33948b3eaceca366e23d732919bb8923d7add3f2a759d97540fc19
```

This approval authorizes implementation preparation and synthetic fixture tests only. A separate `APPROVED_TO_RUN` comment is required before any real race-day paper/shadow execution.
