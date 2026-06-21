//! The ce-infer mesh wire protocol — request/reply payloads exchanged between router and worker.
//!
//! Transport is CE's mesh AppRequest (`ce.request` / `ce.reply`), which is **unary**. Streaming is
//! layered on top: the worker, on a `stream: true` request, sends incremental token deltas as
//! directed messages on [`stream_topic`] back to the router, terminated by a delta whose
//! `finish_reason` is set. None of this needs node changes — it is `ce.send_message` for deltas plus
//! `ce.request` for the initial handshake.

use serde::{Deserialize, Serialize};

/// The unary request topic workers listen on.
pub const TOPIC_INFER: &str = "infer/v1";

/// The per-request streaming topic the worker sends token deltas on. The router subscribes to it,
/// keyed by the request id it minted.
pub fn stream_topic(req_id: &str) -> String {
    format!("infer/stream/{req_id}")
}

/// The per-session topic the router (payer) sends heartbeat payment receipts on for a long-running
/// streaming generation, so the worker (host) can advance its redeemable total mid-stream rather
/// than only at the end. Keyed by the request id.
pub fn receipt_topic(req_id: &str) -> String {
    format!("infer/receipt/{req_id}")
}

/// A signed off-chain payment-channel receipt the **payer** (router) hands the **host** (worker):
/// it authorizes a monotonic `cumulative` total on `channel_id`. The host accumulates the highest
/// such receipt and redeems exactly it with one `channel_close`. Amounts are decimal-string base
/// units on the wire ([`ce_rs::Amount`] (de)serializes as a string). This carries no PHI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptMsg {
    /// The payment channel this receipt belongs to.
    pub channel_id: String,
    /// The monotonic running total (base units) the payer authorizes the host to redeem.
    pub cumulative: ce_rs::Amount,
    /// The payer's signature over `(channel_id, cumulative)` — opaque hex, redeemed via channel_close.
    pub payer_sig: String,
    /// The request/session this receipt meters (also names [`receipt_topic`]).
    #[serde(default)]
    pub req_id: String,
}

/// One chat message (OpenAI shape). Carried over the mesh only between trusted router and worker
/// inside the hospital; PHI never leaves the LAN and is never written to the audit log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// An inference request from the router to a worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferRequest {
    /// Router-minted unique id (also names the stream topic).
    pub req_id: String,
    /// `chat` | `summarize` | `code`.
    pub op: crate::audit::Op,
    /// Logical model id the worker must serve.
    pub model_id: String,
    /// Chat turns (chat/summarize/code all use the messages array).
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    /// Optional cap on tokens generated.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// When true, the worker streams deltas on [`stream_topic`] instead of returning the whole
    /// completion in the unary reply.
    #[serde(default)]
    pub stream: bool,
    /// Hex-encoded ce-cap chain authorizing this request (the `caps` field, like rdev).
    pub caps: String,
    /// Caller-supplied SHA256 (hex) of the PHI record — for the audit log only, never the record.
    pub record_ref: String,
    /// The payer's signed channel receipt covering this request's accrued cost (router fills it after
    /// charging its meter). `None` for an unbilled request (e.g. tests, or a payer with no channel).
    #[serde(default)]
    pub receipt: Option<ReceiptMsg>,
}

/// A worker's unary reply for a non-streaming request (or the error path for a streaming one).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferReply {
    /// True if the request was authorized and served.
    pub ok: bool,
    /// The completion text (empty on error/streaming-handshake-only).
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub token_count: u64,
    #[serde(default)]
    pub model_id: String,
    /// `stop` | `length` | `denied` | `error`.
    #[serde(default)]
    pub finish_reason: String,
    /// Set when `ok` is false (a safe-to-surface reason).
    #[serde(default)]
    pub error: Option<String>,
}

impl InferReply {
    pub fn error(reason: impl Into<String>) -> InferReply {
        let reason = reason.into();
        InferReply {
            ok: false,
            text: String::new(),
            token_count: 0,
            model_id: String::new(),
            finish_reason: "error".into(),
            error: Some(reason),
        }
    }

    pub fn denied(reason: impl Into<String>) -> InferReply {
        InferReply { finish_reason: "denied".into(), ..InferReply::error(reason) }
    }
}

/// One streamed token delta (worker -> router on the stream topic).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamDelta {
    pub req_id: String,
    /// Monotonic index for ordering/dedup.
    pub seq: u64,
    /// The incremental text. Empty on the terminal delta.
    #[serde(default)]
    pub delta: String,
    /// Set on the final delta (`stop` | `length` | `error`); `None` for intermediate deltas.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

impl StreamDelta {
    pub fn is_final(&self) -> bool {
        self.finish_reason.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::Op;

    #[test]
    fn request_round_trips() {
        let req = InferRequest {
            req_id: "r1".into(),
            op: Op::Chat,
            model_id: "clinical-chat-8b".into(),
            messages: vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            max_tokens: Some(64),
            stream: true,
            caps: "deadbeef".into(),
            record_ref: "a".repeat(64),
            receipt: None,
        };
        let bytes = serde_json::to_vec(&req).unwrap();
        let back: InferRequest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn stream_topic_is_per_request() {
        assert_eq!(stream_topic("abc"), "infer/stream/abc");
    }

    #[test]
    fn final_delta_detected() {
        let d = StreamDelta { req_id: "r".into(), seq: 9, delta: String::new(), finish_reason: Some("stop".into()) };
        assert!(d.is_final());
        let mid = StreamDelta { req_id: "r".into(), seq: 1, delta: "tok".into(), finish_reason: None };
        assert!(!mid.is_final());
    }
}
