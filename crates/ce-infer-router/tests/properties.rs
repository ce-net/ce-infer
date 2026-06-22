//! Property + failure-injection tests for the router library: candidate ranking invariants and the
//! token-stream relay's robustness against malformed/dropped/garbage frames.
//!
//! The relay (`relay_token_stream`) is the SSE seam between a worker's mesh deltas and the client's
//! OpenAI SSE stream. It is the most failure-exposed pure function in the router, so it gets the
//! heaviest fuzzing: out-of-order + duplicate deltas, foreign topics, wrong req_id, truncated /
//! non-JSON payloads, transport `Err` items, and a worker that drops before sending the terminal
//! delta. In every case it must terminate, never panic, and emit in-order chunks ending at the first
//! Final.

use ce_infer_router::{Candidate, DEFAULT_STALE_SECS, RelayChunk, rank, relay_token_stream, select};
use ce_infer_core::proto::{StreamDelta, stream_topic};
use ce_rs::{AppMessage, AtlasEntry};
use proptest::prelude::*;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn entry(id: &str, jobs: u32, seen: u64, tags: &[&str]) -> AtlasEntry {
    let tags_json: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
    serde_json::from_value(serde_json::json!({
        "node_id": id,
        "cpu_cores": 8,
        "mem_mb": 16000,
        "running_jobs": jobs,
        "last_seen_secs": seen,
        "tags": tags_json,
    }))
    .unwrap()
}

/// An AppMessage carrying a serialized StreamDelta on `topic` (what messages_stream yields).
fn delta_msg(topic: &str, delta: &StreamDelta) -> anyhow::Result<AppMessage> {
    let payload = serde_json::to_vec(delta)?;
    Ok(serde_json::from_value(serde_json::json!({
        "from": "ff".repeat(32),
        "topic": topic,
        "payload_hex": hex::encode(&payload),
        "received_at": 0,
        "reply_token": serde_json::Value::Null,
    }))?)
}

/// An AppMessage carrying arbitrary raw bytes (possibly non-JSON / truncated) on `topic`.
fn raw_msg(topic: &str, payload: &[u8]) -> anyhow::Result<AppMessage> {
    Ok(serde_json::from_value(serde_json::json!({
        "from": "ff".repeat(32),
        "topic": topic,
        "payload_hex": hex::encode(payload),
        "received_at": 0,
        "reply_token": serde_json::Value::Null,
    }))?)
}

/// Drive the relay synchronously over a fixed message list and collect the relayed chunks.
fn run_relay(req_id: &str, topic: &str, msgs: Vec<anyhow::Result<AppMessage>>) -> Vec<RelayChunk> {
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let chunks: RefCell<Vec<RelayChunk>> = RefCell::new(Vec::new());
    rt.block_on(async {
        let stream = futures::stream::iter(msgs);
        relay_token_stream(
            stream,
            req_id,
            topic,
            u64::MAX,
            || 1_000,
            |_n, _t| async {},
            |c| {
                chunks.borrow_mut().push(c);
                async {}
            },
        )
        .await;
    });
    chunks.into_inner()
}

// ===========================================================================
// ranking invariants
// ===========================================================================

prop_compose! {
    fn arb_worker()(
        id in "[a-z0-9]{4,12}",
        jobs in 0u32..50,
        seen in 0u64..200,
        gpu in any::<bool>(),
    ) -> (String, u32, u64, bool) {
        (id, jobs, seen, gpu)
    }
}

proptest! {
    /// rank() is a total order that never drops or duplicates a candidate, and (without GPU
    /// preference) the first element is always a least-loaded one.
    #[test]
    fn rank_is_a_stable_permutation_least_loaded_first(
        workers in proptest::collection::vec(arb_worker(), 1..20),
    ) {
        // Build a model-serving atlas; dedup ids so each is unique.
        let mut seen_ids = std::collections::HashSet::new();
        let entries: Vec<AtlasEntry> = workers.iter()
            .filter(|(id, ..)| seen_ids.insert(id.clone()))
            .map(|(id, jobs, seen, gpu)| {
                let tags: Vec<&str> = if *gpu {
                    vec!["infer", "model:m", "gpu", "gpu-mid"]
                } else {
                    vec!["infer", "model:m", "cpu"]
                };
                // Clamp seen below the stale cutoff so all are candidates.
                entry(id, *jobs, *seen % DEFAULT_STALE_SECS, &tags)
            })
            .collect();
        let n = entries.len();
        let ranked = select(&entries, "m", DEFAULT_STALE_SECS, false, |_| 0);
        // No candidate dropped or duplicated.
        prop_assert_eq!(ranked.len(), n);
        let ids: std::collections::HashSet<_> = ranked.iter().map(|c| c.node_id.clone()).collect();
        prop_assert_eq!(ids.len(), n);
        // Least-loaded leads (no gpu preference).
        let min_jobs = ranked.iter().map(|c| c.running_jobs).min().unwrap();
        prop_assert_eq!(ranked[0].running_jobs, min_jobs);
        // running_jobs is non-decreasing through the order (the primary key).
        for w in ranked.windows(2) {
            prop_assert!(w[0].running_jobs <= w[1].running_jobs);
        }
    }

    /// With GPU preference on, every GPU worker outranks every CPU worker (interactive routing).
    #[test]
    fn prefer_gpu_puts_all_gpu_before_all_cpu(
        n_gpu in 0usize..6,
        n_cpu in 0usize..6,
    ) {
        prop_assume!(n_gpu + n_cpu > 0);
        let mut entries = Vec::new();
        for i in 0..n_gpu {
            entries.push(entry(&format!("gpu{i}"), 0, 1, &["infer", "model:m", "gpu", "gpu-mid"]));
        }
        for i in 0..n_cpu {
            entries.push(entry(&format!("cpu{i}"), 0, 1, &["infer", "model:m", "cpu"]));
        }
        let ranked = select(&entries, "m", DEFAULT_STALE_SECS, true, |_| 0);
        // Find the first CPU index; no GPU may appear after it.
        let first_cpu = ranked.iter().position(|c| !c.gpu);
        if let Some(idx) = first_cpu {
            prop_assert!(ranked[idx..].iter().all(|c| !c.gpu), "a GPU ranked after a CPU");
        }
    }

    /// rank() is idempotent: ranking an already-ranked list yields the same order.
    #[test]
    fn rank_is_idempotent(
        cands in proptest::collection::vec(
            ("[a-z0-9]{4,8}", 0u32..20, 0u64..100, 0u64..100, any::<bool>()),
            1..15,
        ),
        prefer_gpu in any::<bool>(),
    ) {
        let mut seen = std::collections::HashSet::new();
        let v: Vec<Candidate> = cands.into_iter()
            .filter(|(id, ..)| seen.insert(id.clone()))
            .map(|(node_id, running_jobs, reputation, last_seen_secs, gpu)| Candidate {
                node_id, running_jobs, reputation, last_seen_secs, gpu,
            })
            .collect();
        let once = rank(v.clone(), prefer_gpu);
        let twice = rank(once.clone(), prefer_gpu);
        prop_assert_eq!(once, twice);
    }
}

// ===========================================================================
// relay robustness / failure injection
// ===========================================================================

proptest! {
    /// For any permutation of a full delta set (seq 0..k then a Final), the relay emits the deltas in
    /// seq order followed by exactly one Final, regardless of arrival order or duplicates.
    #[test]
    fn relay_orders_and_terminates_under_any_permutation(
        k in 1usize..12,
        seed in any::<u64>(),
        dup in any::<bool>(),
    ) {
        let req_id = "rp";
        let topic = stream_topic(req_id);
        let mut deltas: Vec<StreamDelta> = (0..k as u64)
            .map(|seq| StreamDelta { req_id: req_id.into(), seq, delta: format!("t{seq} "), finish_reason: None })
            .collect();
        deltas.push(StreamDelta { req_id: req_id.into(), seq: k as u64, delta: String::new(), finish_reason: Some("stop".into()) });
        // Shuffle deterministically by rotation, and optionally duplicate the middle.
        let rot = (seed as usize) % deltas.len();
        deltas.rotate_left(rot);
        if dup && deltas.len() > 1 {
            deltas.push(deltas[deltas.len() / 2].clone());
        }
        let msgs: Vec<anyhow::Result<AppMessage>> =
            deltas.iter().map(|d| delta_msg(&topic, d)).collect();
        let got = run_relay(req_id, &topic, msgs);

        // Exactly k Delta chunks in seq order, then one Final.
        prop_assert_eq!(got.len(), k + 1);
        for (i, chunk) in got.iter().take(k).enumerate() {
            prop_assert_eq!(chunk, &RelayChunk::Delta(format!("t{i} ")));
        }
        prop_assert_eq!(&got[k], &RelayChunk::Final("stop".into()));
    }

    /// Failure injection: arbitrary garbage messages (non-JSON, truncated, foreign topics, transport
    /// Errs, wrong req_id) interleaved with the real deltas must be ignored — the relay still emits
    /// the in-order deltas and the Final, and never panics.
    #[test]
    fn relay_survives_arbitrary_garbage(
        garbage in proptest::collection::vec(proptest::collection::vec(any::<u8>(), 0..30), 0..8),
        include_final in any::<bool>(),
    ) {
        let req_id = "rg";
        let topic = stream_topic(req_id);
        let mut msgs: Vec<anyhow::Result<AppMessage>> = Vec::new();
        // Two real deltas.
        msgs.push(delta_msg(&topic, &StreamDelta { req_id: req_id.into(), seq: 0, delta: "hello ".into(), finish_reason: None }));
        // Interleave garbage on the same topic (won't parse as StreamDelta) ...
        for g in &garbage {
            msgs.push(raw_msg(&topic, g));
        }
        // ... a foreign-topic real delta (must be ignored) ...
        msgs.push(delta_msg("infer/stream/other", &StreamDelta { req_id: "other".into(), seq: 0, delta: "x".into(), finish_reason: Some("stop".into()) }));
        // ... a wrong-req_id delta on our topic (must be ignored) ...
        msgs.push(delta_msg(&topic, &StreamDelta { req_id: "mismatch".into(), seq: 0, delta: "y".into(), finish_reason: None }));
        // ... a transport error item ...
        msgs.push(Err(anyhow::anyhow!("transport dropped")));
        // ... the second real delta ...
        msgs.push(delta_msg(&topic, &StreamDelta { req_id: req_id.into(), seq: 1, delta: "world".into(), finish_reason: None }));
        if include_final {
            msgs.push(delta_msg(&topic, &StreamDelta { req_id: req_id.into(), seq: 2, delta: String::new(), finish_reason: Some("stop".into()) }));
        }

        let got = run_relay(req_id, &topic, msgs);
        // The two real deltas always come through in order.
        prop_assert!(got.len() >= 2);
        prop_assert_eq!(&got[0], &RelayChunk::Delta("hello ".into()));
        prop_assert_eq!(&got[1], &RelayChunk::Delta("world".into()));
        if include_final {
            prop_assert_eq!(got.last().unwrap(), &RelayChunk::Final("stop".into()));
        } else {
            // No Final delivered => relay drains and returns without a Final chunk (caller emits [DONE]).
            prop_assert!(got.iter().all(|c| matches!(c, RelayChunk::Delta(_))));
        }
    }
}

/// A worker that drops mid-stream (never sends the terminal delta) must not hang or panic: the relay
/// drains the finite stream and returns, having emitted only the deltas it saw.
#[test]
fn relay_handles_worker_drop_mid_stream() {
    let req_id = "rdrop";
    let topic = stream_topic(req_id);
    let msgs = vec![
        delta_msg(&topic, &StreamDelta { req_id: req_id.into(), seq: 0, delta: "partial ".into(), finish_reason: None }),
        delta_msg(&topic, &StreamDelta { req_id: req_id.into(), seq: 1, delta: "answer".into(), finish_reason: None }),
        // worker dies here — no Final ever arrives.
    ];
    let got = run_relay(req_id, &topic, msgs);
    assert_eq!(got, vec![RelayChunk::Delta("partial ".into()), RelayChunk::Delta("answer".into())]);
}

/// A gap in the seq sequence (delta seq 2 arrives but seq 1 never does) stalls in-order delivery
/// after the gap — seq 0 is delivered, seq 2 is buffered but withheld (we never emit out of order),
/// and the relay terminates when the stream ends. This proves the buffer can't be tricked into
/// emitting a later token before an earlier one.
#[test]
fn relay_withholds_tokens_after_a_seq_gap() {
    let req_id = "rgap";
    let topic = stream_topic(req_id);
    let msgs = vec![
        delta_msg(&topic, &StreamDelta { req_id: req_id.into(), seq: 0, delta: "a ".into(), finish_reason: None }),
        // seq 1 missing
        delta_msg(&topic, &StreamDelta { req_id: req_id.into(), seq: 2, delta: "c".into(), finish_reason: None }),
        delta_msg(&topic, &StreamDelta { req_id: req_id.into(), seq: 3, delta: String::new(), finish_reason: Some("stop".into()) }),
    ];
    let got = run_relay(req_id, &topic, msgs);
    // Only seq 0 is delivered; seq 2 and the Final stay buffered behind the missing seq 1.
    assert_eq!(got, vec![RelayChunk::Delta("a ".into())]);
}
