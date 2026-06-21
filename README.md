# ce-infer

Distributed clinical-inference for the hospital, built **on CE primitives**. v1 = a pool of
whole-model inference **workers** + a smart **router** (an OpenAI-compatible front door) over the CE
mesh. ce-infer is an *app*: it talks only to a local CE node's HTTP API via `ce-rs`, adds **no new
node endpoints**, and uses `ce-cap` for all authorization. Mesh-first, capability-only trust.

```
clinician UI / SSO ─▶ ce-infer-router ──(CE mesh AppRequest)──▶ ce-infer-worker ──▶ llama.cpp (loopback)
        (OpenAI /v1/chat/completions)        capability-gated         pulls GGUF over CE blobs
```

## Crates

| Crate | Kind | Role |
|---|---|---|
| `ce-infer-core` | lib | Shared types: hardware **probe** + self-tier rule, **registry** (`models.toml`), capability **abilities** + the `model_prefix` caveat, **audit** (HIPAA-grade, PHI-free), the wire **proto**, and the pure `serve::decide` worker logic. |
| `ce-infer-worker` | bin `ce-infer-worker` | Per-node inference server: probe → assign model → pull GGUF over CE blobs → launch `llama-server` on loopback (or a deterministic **mock** backend) → poll-loop serving capability-gated mesh requests, billing via payment channels, auditing every op. |
| `ce-infer-router` | bin `ce-infer-router` + lib | OpenAI-compatible HTTP front door + smart load balancer. Discovers workers via the CE atlas, ranks least-loaded/reputation, dispatches over the mesh, relays token streams as SSE, retries/circuit-breaks on failure. |
| `ce-infer-cli` | bin `ce-infer` | Ops CLI: `probe`, `models pull/publish/list`, `status`, `audit export`, `grant`. |
| `ce-infer-shard` | lib (feature `shard`, **OFF**) | v2 **experimental** pipeline-parallel scaffold for models too big for one node. Pipeline-parallel only, never tensor-parallel over Ethernet. Not wired into v1. |

## Capability abilities (opaque `ce-cap` strings)

`infer:chat`, `infer:summarize`, `infer:code`, `infer:admin`, `infer:shard`. The `model_prefix`
caveat is expressed as a structured ability `infer:model_prefix:<prefix>` (e.g.
`infer:model_prefix:clinical-`) so it rides `ce-cap`'s existing attenuation; the worker enforces it
against the request's model id at the leaf. Roots come from `$CE_INFER_ROOTS`, else
`$CE_DATA_DIR/roots`, else `~/.local/share/ce/roots` (a chain rooted at the worker's own key is
always honored).

## Audit & HIPAA

Every op — allowed **or denied** — produces a signed, append-only audit record on the `infer/audit/v1`
topic and (via the per-session payment-channel receipt) an on-chain `/history` interaction. Records
carry **only** a caller-supplied SHA256 `record_ref` of the PHI record — never the PHI, the prompt,
or the response. `AuditRecord::assert_redacted()` fails closed if a record could carry PHI. Satisfies
HIPAA §164.312(b)/(c); 6-year retention is the operator's storage policy. Export with
`ce-infer audit export`.

## Quick start (mock backend — no GGUF needed)

```bash
# Terminal 1: a worker (uses the deterministic mock backend until weights are published)
cargo run -p ce-infer-worker -- --mock

# Terminal 2: the router
cargo run -p ce-infer-router

# Probe this node's tier and assigned model
cargo run -p ce-infer -- probe

# Publish real weights (fills the CID into models.toml and spreads it over the LAN via ce-pin)
cargo run -p ce-infer -- models publish ./clinical-chat-8b.Q4_K_M.gguf --id clinical-chat-8b
```

The worker shells out to `llama-server` (llama.cpp) on **loopback only**; the engine binary is
bundled per-platform by the installer (ce-fleet). ce-infer does **not** implement an inference
engine. Same GGUF runs on Metal (macOS) / CUDA (Linux+NVIDIA) / AVX2 (CPU).

## Build & test

```bash
cargo build                              # whole workspace (shard OFF)
cargo test                               # unit + integration (mock backend; no node/GGUF needed)
cargo test -p ce-infer-shard --features shard   # the experimental v2 sharding scaffold
```

Money is integer base units (`ce_rs::Amount`, 10^18/credit), never floats; HTTP amounts are decimal
strings. Rust 2024, `anyhow::Result`, `tracing` (no `println!` in libs), no `unsafe`, no
`unwrap`/`expect` in production paths.
