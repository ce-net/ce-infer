//! End-to-end protocol test: router ranking -> worker decision -> mock completion -> audit record.
//!
//! This composes the two halves of ce-infer through their shared `ce-infer-core` protocol without
//! requiring a running CE node or a real GGUF: the router picks a worker from a synthetic atlas, the
//! worker authorizes the request with a real ce-cap chain, runs the deterministic mock backend, and
//! produces a PHI-free audit record. It asserts the happy path AND the capability-deny path.

use ce_cap::{Caveats, Resource, SignedCapability, encode_chain};
use ce_infer_core::audit::{Op, Outcome};
use ce_infer_core::caps::{CHAT, model_prefix_ability};
use ce_infer_core::proto::{ChatMessage, InferRequest};
use ce_infer_core::serve::{Decision, build_audit, decide, decision_outcome};
use ce_infer_router::{DEFAULT_STALE_SECS, select};
use ce_identity::Identity;
use ce_rs::AtlasEntry;
use std::sync::atomic::{AtomicU64, Ordering};

fn id(tag: &str) -> Identity {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("ce-infer-e2e-{}-{n}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Identity::load_or_generate(&dir).unwrap()
}

fn never_revoked(_: &[u8; 32], _: u64) -> bool {
    false
}

/// A synthetic atlas entry for a worker serving `model`.
fn worker_entry(node_id: &str, model: &str, jobs: u32, gpu: bool) -> AtlasEntry {
    let mut tags = vec!["infer".to_string(), format!("model:{model}")];
    tags.push(if gpu { "gpu".into() } else { "cpu".into() });
    serde_json::from_value(serde_json::json!({
        "node_id": node_id,
        "cpu_cores": 8,
        "mem_mb": 32000,
        "running_jobs": jobs,
        "last_seen_secs": 3,
        "tags": tags,
    }))
    .unwrap()
}

/// The mock backend mirror (the worker binary's mock; reproduced here so the test does not depend on
/// the binary crate). Deterministic + PHI-free.
fn mock_complete(req: &InferRequest) -> (String, u64) {
    let n = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.chars().count())
        .unwrap_or(0);
    let text = format!("[mock {} op={}] received {n} chars; deterministic test completion.", req.model_id, req.op.as_str());
    let tokens = text.split_whitespace().count() as u64;
    (text, tokens)
}

#[test]
fn router_picks_worker_then_worker_serves_and_audits() {
    let model = "clinical-chat-8b";

    // --- router half: rank a synthetic atlas, prefer GPU for interactive chat ---
    let host_gpu = id("host-gpu");
    let host_cpu = id("host-cpu");
    let atlas = vec![
        worker_entry(&host_cpu.node_id_hex(), model, 0, false),
        worker_entry(&host_gpu.node_id_hex(), model, 1, true),
    ];
    let ranked = select(&atlas, model, DEFAULT_STALE_SECS, true, |_| 0);
    assert_eq!(ranked.first().map(|c| c.node_id.as_str()), Some(host_gpu.node_id_hex().as_str()));
    let chosen_hex = ranked[0].node_id.clone();
    let chosen_id = if chosen_hex == host_gpu.node_id_hex() { &host_gpu } else { &host_cpu };

    // --- principal (clinician) holds a cap issued by the chosen worker, restricted to clinical-* ---
    let clinician = id("clinician");
    let cap = SignedCapability::issue(
        chosen_id,
        clinician.node_id(),
        vec![CHAT.into(), model_prefix_ability("clinical-")],
        Resource::Any,
        Caveats::default(),
        1,
        None,
    );
    let chain = vec![cap];
    let caps_hex = encode_chain(&chain);

    // --- the router builds the inference request and dispatches it (here, directly to the worker) ---
    let req = InferRequest {
        req_id: "r1".into(),
        op: Op::Chat,
        model_id: model.into(),
        messages: vec![ChatMessage { role: "user".into(), content: "summarize the chart".into() }],
        max_tokens: None,
        stream: false,
        caps: caps_hex,
        record_ref: "a".repeat(64),
        receipt: None,
    };

    // --- worker half: authorize, run the mock backend, build the audit record ---
    let decision = decide(
        &chosen_id.node_id(),
        &[],
        1000,
        &clinician.node_id(),
        model,
        &req,
        &chain,
        &never_revoked,
    );
    assert_eq!(decision, Decision::Allow, "an in-prefix authorized request must be allowed");

    let (text, tokens) = mock_complete(&req);
    assert!(tokens > 0);
    // PHI must not leak into the completion verbatim.
    assert!(!text.contains("summarize the chart"));

    let cap_id = ce_infer_core::audit::AuditRecord::capability_id_of(&ce_cap::encode_chain_bytes(&chain));
    let rec = build_audit(&req, &clinician.node_id_hex(), "worker", &format!("{model}@v1"), &cap_id, Outcome::Ok, tokens);
    rec.assert_redacted().expect("the audit record carries no PHI");
    assert_eq!(rec.outcome, Outcome::Ok);
    assert_eq!(rec.record_ref, "a".repeat(64));
    // The audit record references the PHI only by hash — the chart text never appears.
    let json = serde_json::to_string(&rec).unwrap();
    assert!(!json.contains("summarize the chart"));
}

#[test]
fn out_of_prefix_request_is_denied_but_still_audited() {
    let worker = id("worker");
    let clinician = id("clinician");
    // Cap restricted to clinical-*, but the clinician requests a code model.
    let cap = SignedCapability::issue(
        &worker,
        clinician.node_id(),
        vec![CHAT.into(), model_prefix_ability("clinical-")],
        Resource::Any,
        Caveats::default(),
        1,
        None,
    );
    let chain = vec![cap];
    let req = InferRequest {
        req_id: "r2".into(),
        op: Op::Chat,
        model_id: "code-7b".into(),
        messages: vec![ChatMessage { role: "user".into(), content: "x".into() }],
        max_tokens: None,
        stream: false,
        caps: encode_chain(&chain),
        record_ref: "b".repeat(64),
        receipt: None,
    };
    let decision = decide(&worker.node_id(), &[], 1000, &clinician.node_id(), "code-7b", &req, &chain, &never_revoked);
    assert!(matches!(decision, Decision::Deny(_)), "out-of-prefix model must be denied");

    // The denied attempt is auditable.
    let cap_id = ce_infer_core::audit::AuditRecord::capability_id_of(&ce_cap::encode_chain_bytes(&chain));
    let rec = build_audit(&req, &clinician.node_id_hex(), "worker", "code-7b@v1", &cap_id, decision_outcome(&decision), 0);
    rec.assert_redacted().expect("denied audit record is PHI-free");
    assert_eq!(rec.outcome, Outcome::Denied);
}
