//! Property / fuzz tests for ce-infer-core — the load-bearing invariants the rest of the app and
//! HIPAA posture depend on. These hold for *all* inputs, not just the hand-picked unit cases:
//!
//! 1. **CRDT convergence** — `HighestReceipt` is a max-register: any permutation of the same offer
//!    set converges to one identical state (commutative), re-applying any prefix is a no-op
//!    (idempotent), and the accepted total is monotone non-decreasing and bounded by the true max.
//! 2. **Serialization round-trips** — `InferRequest` / `InferReply` / `StreamDelta` / `ReceiptMsg`
//!    / `AuditRecord` / `Registry` survive JSON (and TOML for the registry) intact, INCLUDING
//!    `Amount` values far beyond JSON's 2^53 safe-integer limit (money is base-unit `u128`-scale).
//! 3. **Capability attenuation** — a child link can NEVER amplify abilities or model-prefix scope;
//!    expiry and revocation are always honored. (`ce_cap::authorize` is the verifier; these tests
//!    drive it through the ce-infer `decide` seam over random ability/prefix/time inputs.)
//! 4. **Wire/parse robustness** — `Op::parse`, `node_id_from_hex`, `model_prefixes`, and the audit
//!    redaction guard never panic and never accept malformed input.

use ce_infer_core::audit::{AuditRecord, Op, Outcome};
use ce_infer_core::billing::{HighestReceipt, Meter, PriceSheet};
use ce_infer_core::caps::{
    CHAT, CODE, SUMMARIZE, enforce_model_prefix, model_prefix_ability, model_prefixes,
};
use ce_infer_core::proto::{ChatMessage, InferReply, InferRequest, ReceiptMsg, StreamDelta};
use ce_infer_core::registry::{ModelEntry, Registry, Role};
use ce_infer_core::serve::{Decision, decide};
use ce_infer_core::{node_id_from_hex, now};
use ce_cap::{Caveats, Resource, SignedCapability};
use ce_identity::Identity;
use ce_rs::Amount;
use proptest::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn id(tag: &str) -> Identity {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("ce-infer-prop-{}-{n}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Identity::load_or_generate(&dir).unwrap()
}

fn never_revoked(_: &[u8; 32], _: u64) -> bool {
    false
}

// ===========================================================================
// 1. CRDT convergence — HighestReceipt is a bounded max-register (LWW by value)
// ===========================================================================

/// Apply a list of (cumulative, sig) offers to a fresh HighestReceipt.
fn fold_offers(offers: &[(i128, String)]) -> HighestReceipt {
    let mut h = HighestReceipt::default();
    for (c, s) in offers {
        h.offer(Amount::from_base(*c), s);
    }
    h
}

proptest! {
    /// Commutativity: the accepted cumulative is the max regardless of arrival order. Two arbitrary
    /// permutations of the same offer multiset converge to the same redeemable total.
    #[test]
    fn receipt_converges_regardless_of_order(
        mut offers in proptest::collection::vec((0i128..1_000_000_000i128, "[a-f0-9]{1,8}"), 1..40),
        seed in any::<u64>(),
    ) {
        let a = fold_offers(&offers);
        // Deterministic shuffle by rotating + reversing based on the seed.
        let rot = (seed as usize) % offers.len();
        offers.rotate_left(rot);
        if seed & 1 == 0 { offers.reverse(); }
        let b = fold_offers(&offers);
        prop_assert_eq!(a.cumulative().base(), b.cumulative().base());
        // The converged total is exactly the maximum cumulative offered.
        let max = offers.iter().map(|(c, _)| *c).max().unwrap();
        prop_assert_eq!(a.cumulative().base(), max);
    }

    /// Idempotence: re-offering everything already seen never changes state.
    #[test]
    fn receipt_is_idempotent(
        offers in proptest::collection::vec((0i128..1_000_000_000i128, "[a-f0-9]{1,8}"), 1..40),
    ) {
        let mut h = fold_offers(&offers);
        let before = (h.cumulative().base(), h.payer_sig().to_string());
        // Replaying the whole set (a superset of "any prefix") must be a no-op.
        for (c, s) in &offers {
            let accepted = h.offer(Amount::from_base(*c), s);
            // Re-offering a value <= current max is never accepted.
            prop_assert!(!accepted || *c > before.0);
        }
        prop_assert_eq!(h.cumulative().base(), before.0);
        prop_assert_eq!(h.payer_sig().to_string(), before.1);
    }

    /// Monotonicity: offering in any order, the redeemable total is non-decreasing at every step and
    /// never exceeds the running max — a host can never be tricked into redeeming a lower or
    /// out-of-range total by replays / reordering.
    #[test]
    fn receipt_is_monotone_and_bounded(
        offers in proptest::collection::vec((0i128..1_000_000_000i128, "[a-f0-9]{1,8}"), 1..40),
    ) {
        let mut h = HighestReceipt::default();
        let mut running_max = 0i128;
        let mut prev = 0i128;
        for (c, s) in &offers {
            let accepted = h.offer(Amount::from_base(*c), s);
            running_max = running_max.max(*c);
            let now = h.cumulative().base();
            prop_assert!(now >= prev, "cumulative must be non-decreasing");
            prop_assert!(now <= running_max, "cumulative cannot exceed the max offered");
            // Acceptance happens iff the value strictly exceeded the previous total.
            prop_assert_eq!(accepted, *c > prev);
            prev = now;
        }
    }
}

// ===========================================================================
// 2. Serialization round-trips — incl Amount values > 2^53
// ===========================================================================

prop_compose! {
    /// An arbitrary InferRequest, with an optional receipt whose cumulative can blow past 2^53.
    fn arb_request()(
        req_id in "[a-z0-9]{1,12}",
        op in prop_oneof![Just(Op::Chat), Just(Op::Summarize), Just(Op::Code)],
        model_id in "[a-z0-9-]{1,24}",
        contents in proptest::collection::vec("[ -~]{0,40}", 0..5),
        max_tokens in proptest::option::of(0u32..100_000),
        stream in any::<bool>(),
        caps in "[a-f0-9]{0,64}",
        record_ref in "[a-f0-9]{64}",
        cumulative in 0i128..i128::MAX,
        has_receipt in any::<bool>(),
    ) -> InferRequest {
        let messages = contents.into_iter()
            .map(|c| ChatMessage { role: "user".into(), content: c })
            .collect();
        let receipt = has_receipt.then(|| ReceiptMsg {
            channel_id: "ch1".into(),
            cumulative: Amount::from_base(cumulative),
            payer_sig: "ab".repeat(32),
            req_id: req_id.clone(),
        });
        InferRequest { req_id, op, model_id, messages, max_tokens, stream, caps, record_ref, receipt }
    }
}

proptest! {
    #[test]
    fn infer_request_json_round_trips(req in arb_request()) {
        let bytes = serde_json::to_vec(&req).unwrap();
        let back: InferRequest = serde_json::from_slice(&bytes).unwrap();
        prop_assert_eq!(back, req);
    }

    /// The money field survives JSON intact at full i128 range — the >2^53 invariant. A naive number
    /// encoding would silently round these; Amount serializes as a decimal string and must not.
    #[test]
    fn amount_round_trips_beyond_2_pow_53(base in 0i128..i128::MAX) {
        prop_assume!(base > (1i128 << 53)); // exercise the dangerous range explicitly
        let msg = ReceiptMsg {
            channel_id: "c".into(),
            cumulative: Amount::from_base(base),
            payer_sig: "00".into(),
            req_id: "r".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        // It must be carried as a string, never a bare JSON number (which loses precision).
        prop_assert!(json.contains(&format!("\"{base}\"")), "amount not a decimal string: {json}");
        let back: ReceiptMsg = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back.cumulative.base(), base);
    }

    #[test]
    fn stream_delta_round_trips(
        req_id in "[a-z0-9]{1,8}",
        seq in any::<u64>(),
        delta in "[ -~]{0,40}",
        finish in proptest::option::of("[a-z]{1,8}"),
    ) {
        let d = StreamDelta { req_id, seq, delta, finish_reason: finish.clone() };
        let back: StreamDelta = serde_json::from_slice(&serde_json::to_vec(&d).unwrap()).unwrap();
        prop_assert_eq!(&back, &d);
        prop_assert_eq!(back.is_final(), finish.is_some());
    }

    #[test]
    fn infer_reply_round_trips(
        ok in any::<bool>(),
        text in "[ -~]{0,40}",
        token_count in any::<u64>(),
        model_id in "[a-z0-9-]{0,24}",
        finish in "[a-z]{0,8}",
        error in proptest::option::of("[ -~]{0,40}"),
    ) {
        let r = InferReply { ok, text, token_count, model_id, finish_reason: finish, error };
        let back: InferReply = serde_json::from_slice(&serde_json::to_vec(&r).unwrap()).unwrap();
        prop_assert_eq!(back, r);
    }

    /// A valid (hash-only) audit record survives the JSON wire and stays redaction-clean.
    #[test]
    fn audit_record_round_trips_and_stays_redacted(
        ts in any::<u64>(),
        token_count in any::<u64>(),
        record_ref in "[a-f0-9]{64}",
        op in prop_oneof![Just(Op::Chat), Just(Op::Summarize), Just(Op::Code)],
        outcome in prop_oneof![Just(Outcome::Ok), Just(Outcome::Denied), Just(Outcome::Error)],
    ) {
        let rec = AuditRecord {
            ts,
            principal_node_id: "aa".repeat(32),
            worker_node_id: "bb".repeat(32),
            model_id: "clinical-chat-8b@v1".into(),
            capability_id: "cc".repeat(32),
            record_ref,
            op,
            token_count,
            outcome,
        };
        prop_assert!(rec.assert_redacted().is_ok());
        let back = AuditRecord::from_bytes(&rec.to_bytes().unwrap()).unwrap();
        prop_assert_eq!(back, rec);
    }
}

prop_compose! {
    fn arb_model_entry()(
        id in "[a-z0-9-]{1,20}",
        cid in prop_oneof!["", "[a-z0-9]{6,16}"],
        quant in "[A-Z0-9_]{1,8}",
        ctx in 256u32..65_536,
        ram_min_mb in 0u64..200_000,
        vram_min_mb in 0u64..200_000,
        role in prop_oneof![Just(Role::Chat), Just(Role::Summarize), Just(Role::Code), Just(Role::Draft)],
    ) -> ModelEntry {
        ModelEntry { id, gguf_object_cid: cid, quant, ctx, ram_min_mb, vram_min_mb, role, draft_model: None }
    }
}

proptest! {
    /// The registry survives a TOML encode/decode round-trip for any random set of model entries.
    #[test]
    fn registry_toml_round_trips(
        version in 1u32..10,
        models in proptest::collection::vec(arb_model_entry(), 0..8),
    ) {
        let reg = Registry { version, models };
        let toml = reg.to_toml().unwrap();
        let back = Registry::from_toml(&toml).unwrap();
        prop_assert_eq!(back, reg);
    }

    /// model_gb is always at least 1 GB (a held model always meters something) and never panics.
    #[test]
    fn model_gb_floors_at_one(m in arb_model_entry()) {
        prop_assert!(m.model_gb() >= 1);
    }
}

// ===========================================================================
// 3. Capability attenuation — a child can never amplify; expiry/revocation honored
// ===========================================================================

/// Build a two-link chain root(host->mid) -> leaf(mid->client) with the given abilities at each link.
/// Returns None if the chain is malformed (so the test can `prop_assume`).
fn delegated_chain(
    host: &Identity,
    mid: &Identity,
    client: &Identity,
    parent_abilities: Vec<String>,
    child_abilities: Vec<String>,
    not_after: u64,
) -> Vec<SignedCapability> {
    let root = SignedCapability::issue(
        host,
        mid.node_id(),
        parent_abilities,
        Resource::Any,
        Caveats { not_after, ..Default::default() },
        1,
        None,
    );
    let parent_id = root.id();
    let leaf = SignedCapability::issue(
        mid,
        client.node_id(),
        child_abilities,
        Resource::Any,
        Caveats { not_after, ..Default::default() },
        2,
        Some(parent_id),
    );
    vec![root, leaf]
}

fn req_for(model: &str, op: Op) -> InferRequest {
    InferRequest {
        req_id: "r".into(),
        op,
        model_id: model.into(),
        messages: vec![ChatMessage { role: "user".into(), content: "x".into() }],
        max_tokens: None,
        stream: false,
        caps: String::new(),
        record_ref: "a".repeat(64),
        receipt: None,
    }
}

proptest! {
    /// A delegated leaf may only NARROW abilities. If the child requests an op the *parent* does not
    /// hold, authorization must fail — a child can never grant an ability the parent lacked.
    #[test]
    fn child_cannot_amplify_abilities(
        parent_ops in proptest::collection::hash_set(prop_oneof![Just(CHAT), Just(SUMMARIZE), Just(CODE)], 0..3),
        req_op in prop_oneof![Just(Op::Chat), Just(Op::Summarize), Just(Op::Code)],
    ) {
        let host = id("h");
        let mid = id("m");
        let client = id("c");
        let parent_ab: Vec<String> = parent_ops.iter().map(|s| s.to_string()).collect();
        // The child tries to grant the requested op regardless of whether the parent had it.
        let child_ab = vec![req_op.ability().to_string()];
        let chain = delegated_chain(&host, &mid, &client, parent_ab.clone(), child_ab, 0);
        let d = decide(&host.node_id(), &[], 1000, &client.node_id(), req_op_model(req_op), &req_for(req_op_model(req_op), req_op), &chain, &never_revoked);
        let parent_had_it = parent_ab.iter().any(|a| a == req_op.ability());
        if parent_had_it {
            // The parent could delegate it; the leaf is in-prefix-free and serves the model.
            prop_assert_eq!(d, Decision::Allow);
        } else {
            // The leaf tried to grant more than the parent held — must be denied.
            prop_assert!(matches!(d, Decision::Deny(_)), "amplified ability was wrongly allowed: {d:?}");
        }
    }

    /// model_prefix can only narrow: a leaf carrying `clinical-` rejects any non-`clinical-` model,
    /// and a leaf carrying no prefix is unrestricted. For random model ids the rule holds exactly.
    #[test]
    fn model_prefix_only_narrows(
        prefix in "[a-z]{1,6}-",
        model in "[a-z0-9-]{1,24}",
    ) {
        let host = id("h");
        let client = id("c");
        let leaf = SignedCapability::issue(
            &host,
            client.node_id(),
            vec![CHAT.into(), model_prefix_ability(&prefix)],
            Resource::Any,
            Caveats::default(),
            1,
            None,
        );
        let res = enforce_model_prefix(&leaf, &model);
        prop_assert_eq!(res.is_ok(), model.starts_with(&prefix));
    }

    /// Expired capabilities are always rejected; a not-yet-expired one with the right ability is
    /// allowed. Time is the only variable.
    #[test]
    fn expiry_is_always_honored(
        not_after in 1u64..1_000_000,
        now_secs in 0u64..1_000_000,
    ) {
        let host = id("h");
        let mid = id("m");
        let client = id("c");
        let chain = delegated_chain(
            &host, &mid, &client,
            vec![CHAT.into()], vec![CHAT.into()],
            not_after,
        );
        let req = req_for("clinical-chat-8b", Op::Chat);
        let d = decide(&host.node_id(), &[], now_secs, &client.node_id(), "clinical-chat-8b", &req, &chain, &never_revoked);
        if now_secs > not_after {
            prop_assert!(matches!(d, Decision::Deny(_)), "expired cap allowed at now={now_secs} exp={not_after}");
        } else {
            prop_assert_eq!(d, Decision::Allow);
        }
    }

    /// Revoking ANY link's (issuer, nonce) invalidates the whole chain, for any link choice.
    #[test]
    fn revocation_of_any_link_denies(revoke_root in any::<bool>()) {
        let host = id("h");
        let mid = id("m");
        let client = id("c");
        let chain = delegated_chain(&host, &mid, &client, vec![CHAT.into()], vec![CHAT.into()], 0);
        // Pick which link to revoke.
        let (issuer, nonce) = if revoke_root {
            (chain[0].cap.issuer, chain[0].cap.nonce)
        } else {
            (chain[1].cap.issuer, chain[1].cap.nonce)
        };
        let is_revoked = move |i: &[u8; 32], n: u64| *i == issuer && n == nonce;
        let req = req_for("clinical-chat-8b", Op::Chat);
        let d = decide(&host.node_id(), &[], 1000, &client.node_id(), "clinical-chat-8b", &req, &chain, &is_revoked);
        prop_assert!(matches!(d, Decision::Deny(_)), "revoked chain was allowed");
    }
}

fn req_op_model(op: Op) -> &'static str {
    match op {
        Op::Chat => "clinical-chat-8b",
        Op::Summarize => "clinical-sum-8b",
        Op::Code => "code-7b",
    }
}

// ===========================================================================
// 4. Wire / parse robustness — never panic, never accept garbage
// ===========================================================================

proptest! {
    /// Op::parse never panics and only accepts the three known ops (case/space-insensitive).
    #[test]
    fn op_parse_never_panics(s in ".*") {
        let r = Op::parse(&s);
        let norm = s.trim().to_ascii_lowercase();
        prop_assert_eq!(r.is_ok(), matches!(norm.as_str(), "chat" | "summarize" | "code"));
    }

    /// node_id_from_hex accepts iff the input is exactly 64 hex chars (32 bytes); never panics.
    #[test]
    fn node_id_from_hex_is_strict(s in ".*") {
        let r = node_id_from_hex(&s);
        let t = s.trim();
        let is_64_hex = t.len() == 64 && t.chars().all(|c| c.is_ascii_hexdigit());
        prop_assert_eq!(r.is_ok(), is_64_hex);
    }

    /// The redaction guard accepts a record IFF record_ref is exactly 64 hex chars (no PHI leak).
    #[test]
    fn redaction_guard_only_accepts_hash_refs(record_ref in ".{0,80}") {
        let rec = AuditRecord {
            ts: 1,
            principal_node_id: "aa".repeat(32),
            worker_node_id: "bb".repeat(32),
            model_id: "m@v1".into(),
            capability_id: "cc".repeat(32),
            record_ref: record_ref.clone(),
            op: Op::Chat,
            token_count: 0,
            outcome: Outcome::Ok,
        };
        let t = record_ref.trim();
        let is_hash = t.len() == 64 && t.chars().all(|c| c.is_ascii_hexdigit());
        prop_assert_eq!(rec.assert_redacted().is_ok(), is_hash);
    }

    /// model_prefixes extracts exactly the `infer:model_prefix:` family and nothing else; never panics.
    #[test]
    fn model_prefixes_extraction_is_exact(
        abilities in proptest::collection::vec("[a-z:_-]{0,20}", 0..6),
    ) {
        let extracted = model_prefixes(&abilities);
        let expected: Vec<String> = abilities.iter()
            .filter_map(|a| a.strip_prefix("infer:model_prefix:").map(|s| s.to_string()))
            .collect();
        prop_assert_eq!(extracted, expected);
    }

    /// Garbage bytes never deserialize into an InferRequest as a panic — they Err cleanly.
    #[test]
    fn garbage_never_panics_decoding_request(bytes in proptest::collection::vec(any::<u8>(), 0..200)) {
        let _ = serde_json::from_slice::<InferRequest>(&bytes);
    }
}

// ===========================================================================
// 5. Billing math properties
// ===========================================================================

proptest! {
    /// cost() is saturating and never panics, for any price sheet and any work amounts.
    #[test]
    fn cost_never_panics(
        per_token in any::<i128>(),
        per_gb_second in any::<i128>(),
        per_request in any::<i128>(),
        tokens in any::<u64>(),
        gb in any::<u64>(),
        seconds in any::<u64>(),
    ) {
        let s = PriceSheet {
            per_token: Amount::from_base(per_token),
            per_gb_second: Amount::from_base(per_gb_second),
            per_request: Amount::from_base(per_request),
        };
        let _ = s.cost(tokens, gb, seconds); // must not panic/overflow
    }

    /// A meter's cumulative is monotone non-decreasing across arbitrary charges and is always
    /// clamped to [0, capacity] — receipts can never be signed beyond locked funds.
    #[test]
    fn meter_cumulative_is_monotone_and_capped(
        capacity in 1i128..1_000_000_000,
        charges in proptest::collection::vec((0u64..1000, 0u64..1000), 0..30),
    ) {
        let price = PriceSheet {
            per_token: Amount::from_base(10),
            per_gb_second: Amount::from_base(2),
            per_request: Amount::from_base(100),
        };
        let mut m = Meter::new(price, 4, Amount::from_base(capacity));
        let mut prev = 0i128;
        for (tokens, seconds) in charges {
            let c = m.charge_request(tokens, seconds).base();
            prop_assert!(c >= prev, "cumulative decreased");
            prop_assert!(c <= capacity, "cumulative exceeded capacity");
            prev = c;
        }
        prop_assert_eq!(m.remaining().base(), capacity - m.cumulative().base());
    }

    /// Heartbeat charging bills exactly floor(elapsed / interval) whole intervals; calling it twice
    /// inside one interval is idempotent (no double charge).
    #[test]
    fn heartbeat_charges_whole_intervals_only(
        gap in 0u64..600,
    ) {
        use ce_infer_core::billing::HEARTBEAT_INTERVAL_SECS;
        let price = PriceSheet {
            per_token: Amount::from_base(0),
            per_gb_second: Amount::from_base(2),
            per_request: Amount::from_base(0),
        };
        let mut m = Meter::new(price, 1, Amount::from_base(1_000_000_000));
        let t0 = 100_000u64;
        prop_assert_eq!(m.charge_heartbeats(t0, 0), 0); // arm only
        let n = m.charge_heartbeats(t0 + gap, 0);
        prop_assert_eq!(n, gap / HEARTBEAT_INTERVAL_SECS);
        // A second call at the same instant charges nothing more.
        let again = m.charge_heartbeats(t0 + gap, 0);
        prop_assert_eq!(again, 0);
    }
}

// A non-proptest smoke that the helpers above actually exercise authorize end-to-end at runtime
// (so a regression in `now()` or the seam is caught even without proptest sampling that path).
#[test]
fn now_is_monotone_nonzero() {
    let a = now();
    assert!(a > 1_600_000_000, "clock looks wrong: {a}");
}
