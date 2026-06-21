//! # ce-infer-router (library)
//!
//! The router's pure, testable core: candidate **ranking** over the CE atlas. The binary
//! (`src/main.rs`) wraps this with the axum OpenAI-compatible server, capability forwarding, mesh
//! dispatch, the streaming relay, billing, and audit.
//!
//! Ranking is the swarm `select_hosts()` pattern: filter the atlas to live workers serving the
//! requested model, then order by least-loaded (lowest `running_jobs`), tie-broken by reputation
//! (`history.delivered_work()`), excluding stale entries.

use ce_infer_core::Tier;
use ce_infer_core::proto::StreamDelta;
use ce_rs::{AppMessage, AtlasEntry};
use futures::{Stream, StreamExt};

/// How stale (seconds since last seen) before a worker is excluded from routing.
pub const DEFAULT_STALE_SECS: u64 = 120;

/// A ranked worker candidate. `reputation` is the host's delivered-work score (filled from
/// `/history`); `0` for an unknown/newcomer host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub node_id: String,
    pub running_jobs: u32,
    pub reputation: u64,
    pub last_seen_secs: u64,
    pub gpu: bool,
}

/// True if `entry` is a usable worker for `model_id`: advertises the `infer` tag, the
/// `model:<model_id>` tag, and was seen within `stale_secs`. Tag-based discovery is the atlas path;
/// the binary additionally consults the DHT `find_service("infer:<model>")` (see `main.rs`).
pub fn is_candidate(entry: &AtlasEntry, model_id: &str, stale_secs: u64) -> bool {
    entry.last_seen_secs <= stale_secs
        && entry.has_tag("infer")
        && entry.has_tag(&format!("model:{model_id}"))
}

/// Whether a candidate is GPU-backed (per its atlas tags), for tier-aware preference.
pub fn is_gpu(entry: &AtlasEntry) -> bool {
    entry.has_tag("gpu")
        || entry.has_tag(Tier::GpuSmall.as_str())
        || entry.has_tag(Tier::GpuMid.as_str())
        || entry.has_tag(Tier::GpuHeavy.as_str())
}

/// Build the candidate set from an atlas snapshot for `model_id`, attaching each host's reputation
/// via `reputation_of` (a closure so callers inject `/history` lookups; tests inject a map).
pub fn candidates<'a, F>(
    atlas: &'a [AtlasEntry],
    model_id: &str,
    stale_secs: u64,
    mut reputation_of: F,
) -> Vec<Candidate>
where
    F: FnMut(&'a str) -> u64,
{
    atlas
        .iter()
        .filter(|e| is_candidate(e, model_id, stale_secs))
        .map(|e| Candidate {
            node_id: e.node_id.clone(),
            running_jobs: e.running_jobs,
            reputation: reputation_of(&e.node_id),
            last_seen_secs: e.last_seen_secs,
            gpu: is_gpu(e),
        })
        .collect()
}

/// Rank candidates in dispatch order (best first):
/// 1. interactive preference: when `prefer_gpu`, GPU workers sort before CPU workers;
/// 2. least-loaded: lowest `running_jobs`;
/// 3. reputation: highest `delivered_work`;
/// 4. freshness: most recently seen;
/// 5. node id: stable deterministic final tie-break.
///
/// `prefer_gpu` is set for interactive chat/code; async summarization passes `false` so CPU workers
/// are eligible on equal footing.
pub fn rank(mut cands: Vec<Candidate>, prefer_gpu: bool) -> Vec<Candidate> {
    cands.sort_by(|a, b| {
        if prefer_gpu {
            // GPU first (true sorts before false here).
            match b.gpu.cmp(&a.gpu) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        a.running_jobs
            .cmp(&b.running_jobs)
            .then_with(|| b.reputation.cmp(&a.reputation))
            .then_with(|| a.last_seen_secs.cmp(&b.last_seen_secs))
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
    cands
}

/// Convenience: candidates filtered + ranked in one call.
pub fn select<'a, F>(
    atlas: &'a [AtlasEntry],
    model_id: &str,
    stale_secs: u64,
    prefer_gpu: bool,
    reputation_of: F,
) -> Vec<Candidate>
where
    F: FnMut(&'a str) -> u64,
{
    rank(candidates(atlas, model_id, stale_secs, reputation_of), prefer_gpu)
}

/// One relayed unit from a worker's token stream, ready to be rendered as an OpenAI SSE chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayChunk {
    /// An incremental token delta to forward to the client.
    Delta(String),
    /// The terminal chunk carrying the finish reason (`stop` | `length` | `error`).
    Final(String),
}

/// Drive the worker->router token stream over an `AppMessage` stream (exactly what
/// [`ce_rs::CeClient::messages_stream`] yields) and relay it as ordered [`RelayChunk`]s.
///
/// This is the SSE-helper path replacing `messages()` polling: it consumes the node's push stream,
/// keeps only this request's [`StreamDelta`]s on `topic`, de-dupes + orders by `seq`, and yields
/// each delta. `on_progress(now_secs, tokens_since_last)` is invoked before each forwarded delta so
/// the caller can meter heartbeat billing for long generations; it returns once the terminal delta
/// arrives, the stream ends, or `deadline_secs` passes. Pure relay logic — no node calls inside, so
/// it is unit-testable with a mocked stream.
pub async fn relay_token_stream<S, P, Fut, Sink, SinkFut>(
    messages: S,
    req_id: &str,
    topic: &str,
    deadline_secs: u64,
    now_fn: impl Fn() -> u64,
    mut on_progress: P,
    mut sink: Sink,
) where
    S: Stream<Item = anyhow::Result<AppMessage>>,
    P: FnMut(u64, u64) -> Fut,
    Fut: std::future::Future<Output = ()>,
    Sink: FnMut(RelayChunk) -> SinkFut,
    SinkFut: std::future::Future<Output = ()>,
{
    // The node's SSE stream is not `Unpin`; pin it on the heap so we can poll it here.
    let mut messages = Box::pin(messages);
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut next_seq: u64 = 0;
    // Buffer out-of-order deltas until their seq is reached, so the client sees in-order tokens.
    let mut pending: std::collections::BTreeMap<u64, StreamDelta> = std::collections::BTreeMap::new();

    while let Some(item) = messages.next().await {
        if now_fn() > deadline_secs {
            return;
        }
        let Ok(msg) = item else { continue };
        if msg.topic != topic {
            continue;
        }
        let Ok(bytes) = msg.payload() else { continue };
        let Ok(delta) = serde_json::from_slice::<StreamDelta>(&bytes) else { continue };
        if delta.req_id != req_id || !seen.insert(delta.seq) {
            continue;
        }
        pending.insert(delta.seq, delta);

        // Drain in-order deltas.
        while let Some(d) = pending.remove(&next_seq) {
            next_seq += 1;
            let tokens = if d.delta.is_empty() { 0 } else { 1 };
            on_progress(now_fn(), tokens).await;
            if let Some(reason) = d.finish_reason {
                sink(RelayChunk::Final(reason)).await;
                return;
            }
            if !d.delta.is_empty() {
                sink(RelayChunk::Delta(d.delta)).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn entry(id: &str, jobs: u32, seen: u64, tags: &[&str]) -> AtlasEntry {
        // Build via JSON so we don't depend on AtlasEntry's field privacy.
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

    #[test]
    fn filters_out_non_infer_and_wrong_model_and_stale() {
        let atlas = vec![
            entry("a", 0, 5, &["infer", "model:clinical-chat-8b", "gpu"]),
            entry("b", 0, 5, &["model:clinical-chat-8b"]), // missing infer tag
            entry("c", 0, 5, &["infer", "model:code-7b"]), // wrong model
            entry("d", 0, 999, &["infer", "model:clinical-chat-8b"]), // stale
        ];
        let cands = candidates(&atlas, "clinical-chat-8b", DEFAULT_STALE_SECS, |_| 0);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].node_id, "a");
    }

    #[test]
    fn ranks_least_loaded_first() {
        let atlas = vec![
            entry("busy", 5, 5, &["infer", "model:m", "cpu"]),
            entry("idle", 0, 5, &["infer", "model:m", "cpu"]),
            entry("mid", 2, 5, &["infer", "model:m", "cpu"]),
        ];
        let ranked = select(&atlas, "m", DEFAULT_STALE_SECS, false, |_| 0);
        let order: Vec<_> = ranked.iter().map(|c| c.node_id.as_str()).collect();
        assert_eq!(order, vec!["idle", "mid", "busy"]);
    }

    #[test]
    fn ties_broken_by_reputation() {
        let atlas = vec![
            entry("low", 0, 5, &["infer", "model:m", "cpu"]),
            entry("high", 0, 5, &["infer", "model:m", "cpu"]),
        ];
        let rep: HashMap<&str, u64> = HashMap::from([("low", 1), ("high", 99)]);
        let ranked = select(&atlas, "m", DEFAULT_STALE_SECS, false, |id| *rep.get(id).unwrap_or(&0));
        assert_eq!(ranked[0].node_id, "high");
    }

    #[test]
    fn prefer_gpu_puts_gpu_first_for_interactive() {
        let atlas = vec![
            // The CPU node is less loaded, but interactive chat prefers GPU.
            entry("cpu0", 0, 5, &["infer", "model:m", "cpu"]),
            entry("gpu3", 3, 5, &["infer", "model:m", "gpu", "gpu-mid"]),
        ];
        let ranked = select(&atlas, "m", DEFAULT_STALE_SECS, true, |_| 0);
        assert_eq!(ranked[0].node_id, "gpu3");
        // Without the GPU preference, least-loaded CPU wins.
        let ranked_async = select(&atlas, "m", DEFAULT_STALE_SECS, false, |_| 0);
        assert_eq!(ranked_async[0].node_id, "cpu0");
    }

    #[test]
    fn empty_atlas_yields_no_candidates() {
        let ranked = select(&[], "m", DEFAULT_STALE_SECS, true, |_| 0);
        assert!(ranked.is_empty());
    }
}
