//! Hardening tests for ce-infer's economic + streaming paths, mocked end to end (no node, no GGUF).
//!
//! Three things are proven here:
//! 1. **Channel open -> receipt -> close accounting**: the payer (router) meter accrues a monotonic
//!    cumulative per request; each cumulative is signed into a receipt; the host (worker) tracks the
//!    highest and redeems exactly it — the redeemed total equals the last signed cumulative.
//! 2. **Heartbeat interval bookkeeping**: a long streaming session bills one channel increment per
//!    whole 30s interval, and the receipt the host would redeem tracks the metered total.
//! 3. **Streaming relay over the SSE helper**: `relay_token_stream` consumes a mocked
//!    `Stream<AppMessage>` (the exact shape `messages_stream()` yields), ordering + de-duping the
//!    worker's token deltas and driving the heartbeat-billing callback.

use ce_infer_core::billing::{HighestReceipt, Meter, PriceSheet, HEARTBEAT_INTERVAL_SECS};
use ce_infer_core::proto::{StreamDelta, receipt_topic, stream_topic};
use ce_infer_router::{RelayChunk, relay_token_stream};
use ce_rs::{Amount, AppMessage};
use std::cell::RefCell;

fn sheet() -> PriceSheet {
    PriceSheet {
        per_token: Amount::from_base(10),
        per_gb_second: Amount::from_base(2),
        per_request: Amount::from_base(100),
    }
}

/// An `AppMessage` carrying a serialized `StreamDelta` on `topic` (mirrors what the worker sends and
/// `messages_stream()` would surface).
fn delta_msg(topic: &str, delta: StreamDelta) -> anyhow::Result<AppMessage> {
    let payload = serde_json::to_vec(&delta)?;
    let v = serde_json::json!({
        "from": "ff".repeat(32),
        "topic": topic,
        "payload_hex": hex::encode(&payload),
        "received_at": 0,
        "reply_token": serde_json::Value::Null,
    });
    Ok(serde_json::from_value(v)?)
}

// ---------------------------------------------------------------------------
// 1. channel open -> receipt -> close accounting math
// ---------------------------------------------------------------------------

#[test]
fn channel_open_receipt_close_accounting_balances() {
    let capacity = Amount::from_base(1_000_000);
    // Payer (router) side: one meter for the session, billed per request.
    let mut meter = Meter::new(sheet(), 4, capacity);
    // Host (worker) side: tracks the highest receipt to redeem.
    let mut host = HighestReceipt::default();

    // Three requests stream in; the router charges its meter and signs a receipt for each new
    // cumulative; the worker ingests each receipt.
    let mut last_cumulative = Amount::ZERO;
    for (tokens, seconds) in [(5u64, 0u64), (12, 0), (3, 0)] {
        let cumulative = meter.charge_request(tokens, seconds);
        // The signature is opaque in this app (the node signs); here a deterministic stand-in.
        let payer_sig = format!("sig@{}", cumulative.base());
        host.offer(cumulative, &payer_sig);
        last_cumulative = cumulative;
    }

    // The cumulative the router metered.
    // r1: 100 + 5*10 = 150 ; r2: +100 + 12*10 = +220 -> 370 ; r3: +100 + 3*10 = +130 -> 500
    assert_eq!(meter.cumulative().base(), 500);
    assert_eq!(last_cumulative.base(), 500);

    // The host redeems exactly the highest cumulative it was authorized for via channel_close.
    assert!(host.is_redeemable());
    assert_eq!(host.cumulative().base(), 500, "redeemed total == last signed cumulative");
    assert_eq!(host.payer_sig(), "sig@500");

    // Conservation: redeemed never exceeds the channel capacity (the locked funds).
    assert!(host.cumulative().base() <= capacity.base());
    // What remains locked after settlement.
    assert_eq!(meter.remaining().base(), capacity.base() - 500);
}

#[test]
fn host_ignores_replayed_or_out_of_order_receipts() {
    let mut host = HighestReceipt::default();
    // Receipts can arrive out of order over the mesh; only a strictly higher cumulative advances.
    assert!(host.offer(Amount::from_base(370), "sig370"));
    assert!(!host.offer(Amount::from_base(150), "sig150"), "stale receipt ignored");
    assert!(host.offer(Amount::from_base(500), "sig500"));
    assert_eq!(host.cumulative().base(), 500);
    assert_eq!(host.payer_sig(), "sig500");
}

// ---------------------------------------------------------------------------
// 2. heartbeat interval bookkeeping
// ---------------------------------------------------------------------------

#[test]
fn heartbeat_bookkeeping_meters_long_session_and_tracks_redeemable() {
    let mut meter = Meter::new(sheet(), 2, Amount::from_base(10_000_000));
    let mut host = HighestReceipt::default();

    // A streaming generation: arm the clock at t0, then tick across several heartbeat intervals.
    let t0 = 100_000u64;
    assert_eq!(meter.charge_heartbeats(t0, 0), 0, "first observation arms, no charge");

    // After one whole interval, one heartbeat is billed; the router signs a receipt for the new
    // cumulative and the host advances its redeemable total.
    let t1 = t0 + HEARTBEAT_INTERVAL_SECS;
    let n1 = meter.charge_heartbeats(t1, 8); // 8 tokens produced this interval
    assert_eq!(n1, 1);
    host.offer(meter.cumulative(), "hb1");
    // cost(8 tokens, 2gb, 30s) = 100 + 80 + 2*30*2 = 100 + 80 + 120 = 300
    assert_eq!(meter.cumulative().base(), 300);
    assert_eq!(host.cumulative().base(), 300);

    // A 95s jump bills 3 whole intervals at once (the 5s remainder is carried).
    let t2 = t1 + 95;
    let n2 = meter.charge_heartbeats(t2, 30);
    assert_eq!(n2, 3);
    host.offer(meter.cumulative(), "hb2");
    // cost(30 tokens, 2gb, 90s) = 100 + 300 + 2*90*2 = 100 + 300 + 360 = 760 ; +300 = 1060
    assert_eq!(meter.cumulative().base(), 1060);
    assert_eq!(host.cumulative().base(), 1060);

    // Total whole intervals billed across the session.
    assert_eq!(n1 + n2, 4);
    assert!(host.is_redeemable());
}

// ---------------------------------------------------------------------------
// 3. streaming relay over the SSE helper (mocked Stream<AppMessage>)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn relay_orders_dedups_and_bills_over_a_mocked_sse_stream() {
    let req_id = "rstream";
    let topic = stream_topic(req_id);

    // The worker emits deltas; the mocked stream delivers them OUT OF ORDER and with a DUPLICATE,
    // plus a message on an unrelated topic that must be ignored.
    let msgs: Vec<anyhow::Result<AppMessage>> = vec![
        delta_msg(&topic, StreamDelta { req_id: req_id.into(), seq: 1, delta: "world ".into(), finish_reason: None }),
        delta_msg(&topic, StreamDelta { req_id: req_id.into(), seq: 0, delta: "hello ".into(), finish_reason: None }),
        // Unrelated topic (another request's receipt) — must be skipped.
        delta_msg(&receipt_topic("other"), StreamDelta { req_id: "other".into(), seq: 0, delta: "x".into(), finish_reason: None }),
        // Duplicate of seq 1 — must be de-duped.
        delta_msg(&topic, StreamDelta { req_id: req_id.into(), seq: 1, delta: "world ".into(), finish_reason: None }),
        delta_msg(&topic, StreamDelta { req_id: req_id.into(), seq: 2, delta: "again".into(), finish_reason: None }),
        delta_msg(&topic, StreamDelta { req_id: req_id.into(), seq: 3, delta: String::new(), finish_reason: Some("stop".into()) }),
    ];
    let stream = futures::stream::iter(msgs);

    let chunks: RefCell<Vec<RelayChunk>> = RefCell::new(Vec::new());
    let progress_calls: RefCell<u64> = RefCell::new(0);
    let tokens_seen: RefCell<u64> = RefCell::new(0);

    relay_token_stream(
        stream,
        req_id,
        &topic,
        u64::MAX, // no deadline pressure in the test
        || 1_000,
        |_now, tokens_since_last| {
            *progress_calls.borrow_mut() += 1;
            *tokens_seen.borrow_mut() += tokens_since_last;
            async {}
        },
        |chunk| {
            chunks.borrow_mut().push(chunk);
            async {}
        },
    )
    .await;

    // The relayed chunks are in seq order, the duplicate dropped, the foreign topic ignored, and the
    // terminal Final chunk closes the stream.
    let got = chunks.into_inner();
    assert_eq!(
        got,
        vec![
            RelayChunk::Delta("hello ".into()),
            RelayChunk::Delta("world ".into()),
            RelayChunk::Delta("again".into()),
            RelayChunk::Final("stop".into()),
        ]
    );
    // Progress (billing hook) fired once per in-order delta INCLUDING the terminal one: 4 deltas.
    assert_eq!(*progress_calls.borrow(), 4);
    // Three non-empty token deltas were counted (the terminal empty delta contributes 0).
    assert_eq!(*tokens_seen.borrow(), 3);
}

#[tokio::test]
async fn relay_stops_at_deadline_without_a_final_delta() {
    let req_id = "rdead";
    let topic = stream_topic(req_id);
    // A stream that only ever delivers intermediate deltas (worker died before finishing).
    let msgs: Vec<anyhow::Result<AppMessage>> = vec![
        delta_msg(&topic, StreamDelta { req_id: req_id.into(), seq: 0, delta: "partial".into(), finish_reason: None }),
    ];
    let stream = futures::stream::iter(msgs);

    let chunks: RefCell<Vec<RelayChunk>> = RefCell::new(Vec::new());
    // now_fn returns a time already past the deadline, so the relay returns on the first poll.
    relay_token_stream(
        stream,
        req_id,
        &topic,
        100,        // deadline
        || 1_000,   // "now" is well past the deadline
        |_n, _t| async {},
        |chunk| {
            chunks.borrow_mut().push(chunk);
            async {}
        },
    )
    .await;
    // No chunks relayed once the deadline has passed; the relay exits cleanly (caller emits [DONE]).
    assert!(chunks.into_inner().is_empty());
}
