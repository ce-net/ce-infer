//! Pure worker decision logic — the authorize / model-check / audit-record-building seam.
//!
//! This is the I/O-free core the `ce-infer-worker` binary wraps with CE-node calls (backend
//! forwarding, channel billing, audit publishing). Keeping it pure makes the end-to-end
//! request path unit-testable without a running node or engine: feed a request + a completion and
//! assert the reply and the audit record. The binary supplies the actual inference + side effects.

use crate::audit::{AuditRecord, Op, Outcome};
use crate::caps::enforce_model_prefix;
use crate::now;
use crate::proto::{InferReply, InferRequest};
use ce_cap::{SignedCapability, authorize};

/// The outcome of authorizing + validating a request, before any inference runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Authorized — run inference for this model.
    Allow,
    /// Rejected by the capability chain or the model_prefix caveat. Carries the reason.
    Deny(String),
    /// Authorized but the request targets a model this worker does not serve.
    WrongModel { served: String, requested: String },
}

/// Decide whether a request may run on a worker serving `served_model`, given the worker's identity,
/// accepted roots, the presented (already-decoded) capability `chain`, and a revocation predicate.
/// Pure: no I/O, no clock dependency beyond the injected `now_secs`.
#[allow(clippy::too_many_arguments)]
pub fn decide(
    host_id: &[u8; 32],
    accepted_roots: &[[u8; 32]],
    now_secs: u64,
    sender: &[u8; 32],
    served_model: &str,
    req: &InferRequest,
    chain: &[SignedCapability],
    is_revoked: &dyn Fn(&[u8; 32], u64) -> bool,
) -> Decision {
    let action = req.op.ability();
    if let Err(e) = authorize(host_id, accepted_roots, &[], now_secs, sender, action, chain, is_revoked) {
        return Decision::Deny(e);
    }
    if let Some(leaf) = chain.last()
        && let Err(e) = enforce_model_prefix(leaf, &req.model_id)
    {
        return Decision::Deny(e);
    }
    if req.model_id != served_model {
        return Decision::WrongModel { served: served_model.to_string(), requested: req.model_id.clone() };
    }
    Decision::Allow
}

/// Build the PHI-free audit record for a request + outcome. `worker_label` is the short worker id;
/// `model_label` is `<id>@v<registry_version>`; `cap_id` is the hash of the presented chain.
pub fn build_audit(
    req: &InferRequest,
    principal_hex: &str,
    worker_label: &str,
    model_label: &str,
    cap_id: &str,
    outcome: Outcome,
    token_count: u64,
) -> AuditRecord {
    AuditRecord {
        ts: now(),
        principal_node_id: principal_hex.to_string(),
        worker_node_id: worker_label.to_string(),
        model_id: model_label.to_string(),
        capability_id: cap_id.to_string(),
        record_ref: req.record_ref.clone(),
        op: req.op,
        token_count,
        outcome,
    }
}

/// Map a [`Decision`] to the deny/error [`InferReply`] (the `Allow` arm has no reply — the caller
/// runs inference and builds the success reply itself).
pub fn decision_reply(decision: &Decision) -> Option<InferReply> {
    match decision {
        Decision::Allow => None,
        Decision::Deny(e) => Some(InferReply::denied(format!("denied: {e}"))),
        Decision::WrongModel { served, requested } => {
            Some(InferReply::error(format!("this worker serves '{served}', not '{requested}'")))
        }
    }
}

/// The audit [`Outcome`] a non-Allow decision maps to (deny vs. error), for the denied-attempt audit.
pub fn decision_outcome(decision: &Decision) -> Outcome {
    match decision {
        Decision::Allow => Outcome::Ok,
        Decision::Deny(_) => Outcome::Denied,
        Decision::WrongModel { .. } => Outcome::Error,
    }
}

/// Convenience used by the router/tests: parse an op from a wire string, defaulting to chat.
pub fn op_or_chat(s: &str) -> Op {
    Op::parse(s).unwrap_or(Op::Chat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{CHAT, model_prefix_ability};
    use crate::proto::{ChatMessage, InferRequest};
    use ce_cap::{Caveats, Resource, SignedCapability};
    use ce_identity::Identity;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn id(tag: &str) -> Identity {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("ce-infer-serve-{}-{n}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Identity::load_or_generate(&dir).unwrap()
    }

    fn never_revoked(_: &[u8; 32], _: u64) -> bool {
        false
    }

    fn request(model: &str, abilities_caps: &str) -> InferRequest {
        InferRequest {
            req_id: "r1".into(),
            op: Op::Chat,
            model_id: model.into(),
            messages: vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            max_tokens: None,
            stream: false,
            caps: abilities_caps.into(),
            record_ref: "a".repeat(64),
            receipt: None,
        }
    }

    /// A self-issued leaf cap from `host` to `client` with the given abilities.
    fn self_cap(host: &Identity, client: &Identity, abilities: Vec<String>) -> SignedCapability {
        SignedCapability::issue(host, client.node_id(), abilities, Resource::Any, Caveats::default(), 1, None)
    }

    #[test]
    fn allows_authorized_in_prefix_request() {
        let host = id("host");
        let client = id("client");
        let cap = self_cap(&host, &client, vec![CHAT.into(), model_prefix_ability("clinical-")]);
        let req = request("clinical-chat-8b", "");
        let d = decide(
            &host.node_id(),
            &[],
            1000,
            &client.node_id(),
            "clinical-chat-8b",
            &req,
            &[cap],
            &never_revoked,
        );
        assert_eq!(d, Decision::Allow);
    }

    #[test]
    fn denies_out_of_prefix_model_and_audits_denied() {
        let host = id("host");
        let client = id("client");
        // Cap restricted to clinical-*, but the request asks for code-7b.
        let cap = self_cap(&host, &client, vec![CHAT.into(), model_prefix_ability("clinical-")]);
        let req = request("code-7b", "");
        let d = decide(&host.node_id(), &[], 1000, &client.node_id(), "code-7b", &req, &[cap], &never_revoked);
        match &d {
            Decision::Deny(e) => assert!(e.contains("outside the capability's allowed prefixes")),
            other => panic!("expected Deny, got {other:?}"),
        }
        // The denied attempt is STILL auditable (redaction passes; outcome=denied).
        assert_eq!(decision_outcome(&d), Outcome::Denied);
        let rec = build_audit(&req, &"aa".repeat(32), "worker", "code-7b@v1", &"cc".repeat(32), Outcome::Denied, 0);
        rec.assert_redacted().expect("denied audit record is PHI-free");
    }

    #[test]
    fn denies_missing_ability() {
        let host = id("host");
        let client = id("client");
        // Cap grants only summarize, request is a chat op.
        let cap = self_cap(&host, &client, vec![crate::caps::SUMMARIZE.into()]);
        let req = request("clinical-chat-8b", "");
        let d = decide(&host.node_id(), &[], 1000, &client.node_id(), "clinical-chat-8b", &req, &[cap], &never_revoked);
        assert!(matches!(d, Decision::Deny(_)));
        assert_eq!(decision_outcome(&d), Outcome::Denied);
    }

    #[test]
    fn wrong_model_is_an_error_not_a_deny() {
        let host = id("host");
        let client = id("client");
        let cap = self_cap(&host, &client, vec![CHAT.into()]);
        // Authorized, but this worker serves clinical-chat-8b and the request asks for clinical-chat-13b.
        let req = request("clinical-chat-13b", "");
        let d = decide(&host.node_id(), &[], 1000, &client.node_id(), "clinical-chat-8b", &req, &[cap], &never_revoked);
        assert!(matches!(d, Decision::WrongModel { .. }));
        assert_eq!(decision_outcome(&d), Outcome::Error);
    }

    #[test]
    fn unauthorized_sender_denied() {
        let host = id("host");
        let client = id("client");
        let stranger = id("stranger");
        let cap = self_cap(&host, &client, vec![CHAT.into()]);
        let req = request("clinical-chat-8b", "");
        // stranger presents client's cap.
        let d = decide(&host.node_id(), &[], 1000, &stranger.node_id(), "clinical-chat-8b", &req, &[cap], &never_revoked);
        assert!(matches!(d, Decision::Deny(_)));
    }

    #[test]
    fn build_audit_is_redaction_safe_on_success() {
        let req = request("clinical-chat-8b", "");
        let rec = build_audit(&req, &"aa".repeat(32), "w", "clinical-chat-8b@v1", &"cc".repeat(32), Outcome::Ok, 12);
        rec.assert_redacted().expect("PHI-free");
        assert_eq!(rec.token_count, 12);
    }
}
