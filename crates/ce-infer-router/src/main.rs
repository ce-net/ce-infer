//! # ce-infer-router — OpenAI-compatible front door + smart mesh load balancer.
//!
//! Runs an axum HTTP server on the LAN (behind the hospital SSO reverse proxy) exposing
//! OpenAI-compatible endpoints, so any client/UI works:
//!   - `POST /v1/chat/completions` (stream + non-stream SSE) — chat/summarize/code, distinguished by
//!     model id and the `X-CE-Op` header.
//!   - `POST /v1/completions`
//!   - `GET  /v1/models` — derived from the registry + which models are live in the atlas.
//!
//! Routing: `ce.atlas()` -> filter to `infer` workers serving the model -> rank least-loaded, then
//! reputation (the swarm `select_hosts()` pattern, in the library). Dispatch over the mesh with
//! `ce.request`; on timeout/error re-rank and retry the next candidate (Petals-style rerouting),
//! circuit-breaking a worker after K consecutive failures. Every request is capability-gated and
//! audited (delegated to the worker, which is the enforcement point; the router forwards `caps`).
//!
//! The router is stateless beyond the atlas cache + circuit-breaker counters; multiple routers run
//! for HA.

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use ce_infer_core::audit::Op;
use ce_infer_core::billing::{Meter, PriceSheet};
use ce_infer_core::proto::{
    ChatMessage, InferReply, InferRequest, ReceiptMsg, TOPIC_INFER, receipt_topic, stream_topic,
};
use ce_infer_core::registry::{Registry, Role};
use ce_infer_router::{Candidate, DEFAULT_STALE_SECS, select};
use ce_rs::{Amount, CeClient};
use clap::Parser;
use futures::Stream;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Consecutive failures before a worker is circuit-broken for `CIRCUIT_COOLDOWN`.
const CIRCUIT_K: u32 = 3;
const CIRCUIT_COOLDOWN: Duration = Duration::from_secs(30);
/// Per-attempt mesh request timeout.
const REQUEST_TIMEOUT_MS: u64 = 60_000;

#[derive(Parser, Debug)]
#[command(name = "ce-infer-router", about = "OpenAI-compatible front door for ce-infer")]
struct Args {
    /// Local CE node HTTP API base URL.
    #[arg(long, default_value = ce_rs::DEFAULT_BASE_URL)]
    node: String,
    /// Address to bind the OpenAI-compatible HTTP server on.
    #[arg(long, default_value = "127.0.0.1:8900")]
    bind: String,
    /// Path to a models.toml registry blob (otherwise the built-in default registry is used).
    #[arg(long)]
    registry: Option<std::path::PathBuf>,
    /// Hex ce-cap chain the router presents for requests that don't carry their own (the
    /// router-held org cap). Per-principal caps from SSO override this per request.
    #[arg(long, default_value = "")]
    caps: String,
    /// Base units charged per generated token (the channel-receipt increment).
    #[arg(long)]
    price_token: Option<u128>,
    /// Base units charged per GB-second of model residency (meters long streaming generations).
    #[arg(long)]
    price_gb_second: Option<u128>,
    /// Flat base-unit floor charged per request.
    #[arg(long)]
    price_request: Option<u128>,
}

/// Shared router state.
struct AppState {
    client: CeClient,
    /// Wallet view over the router's node — opens channels, signs receipts (payer side).
    wallet: ce_rs::Wallet,
    registry: Registry,
    default_caps: String,
    /// How compute is priced into channel receipts.
    price: PriceSheet,
    /// Per-worker circuit breaker: consecutive failures + the time it was last broken.
    breaker: Mutex<HashMap<String, Breaker>>,
    /// Per-worker billing session: the open channel + the metered cumulative the payer has signed.
    sessions: Mutex<HashMap<String, Session>>,
    req_seq: AtomicU64,
}

/// A payer-side billing session against one worker (host): the open channel and its meter.
struct Session {
    channel_id: String,
    meter: Meter,
}

#[derive(Default, Clone)]
struct Breaker {
    consecutive: u32,
    broken_until: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let registry = match &args.registry {
        Some(p) => Registry::from_toml(&std::fs::read_to_string(p)?)?,
        None => Registry::builtin(),
    };
    let client = CeClient::new(&args.node);
    // Subscribe to the audit topic so the router's node receives audit records too (HA observers).
    let _ = client.subscribe(ce_infer_core::audit::TOPIC).await;

    let mut price = PriceSheet::default_sheet();
    if let Some(t) = args.price_token {
        price.per_token = Amount::from_base(t as i128);
    }
    if let Some(g) = args.price_gb_second {
        price.per_gb_second = Amount::from_base(g as i128);
    }
    if let Some(r) = args.price_request {
        price.per_request = Amount::from_base(r as i128);
    }

    let wallet = client.wallet();
    let state = Arc::new(AppState {
        client,
        wallet,
        registry,
        default_caps: args.caps.clone(),
        price,
        breaker: Mutex::new(HashMap::new()),
        sessions: Mutex::new(HashMap::new()),
        req_seq: AtomicU64::new(0),
    });

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/health", get(|| async { "ok" }))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    info!("ce-infer-router listening on http://{}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}

// ----- OpenAI request/response wire shapes (the subset we honor) -----

#[derive(serde::Deserialize)]
struct OpenAiChatRequest {
    model: String,
    #[serde(default)]
    messages: Vec<OpenAiMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
}

#[derive(serde::Deserialize)]
struct OpenAiMessage {
    role: String,
    #[serde(default)]
    content: String,
}

#[derive(serde::Deserialize)]
struct OpenAiCompletionRequest {
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
}

/// `GET /v1/models` — the registry's models, marked with whether any worker serves them live.
async fn list_models(State(st): State<Arc<AppState>>) -> impl IntoResponse {
    let atlas = st.client.atlas().await.unwrap_or_default();
    let live: std::collections::HashSet<String> = atlas
        .iter()
        .flat_map(|e| e.tags.iter().filter_map(|t| t.strip_prefix("model:").map(String::from)))
        .collect();
    let data: Vec<serde_json::Value> = st
        .registry
        .models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "owned_by": "ce-infer",
                "ce_live": live.contains(&m.id),
                "ce_role": format!("{:?}", m.role).to_lowercase(),
            })
        })
        .collect();
    Json(serde_json::json!({ "object": "list", "data": data }))
}

/// `POST /v1/chat/completions`.
async fn chat_completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<OpenAiChatRequest>,
) -> Response {
    let op = resolve_op(&headers, &st.registry, &body.model);
    let model_id = resolve_model_id(&st.registry, &body.model, op);
    let caps = caps_for(&headers, &st.default_caps);
    let messages: Vec<ChatMessage> =
        body.messages.into_iter().map(|m| ChatMessage { role: m.role, content: m.content }).collect();
    let record_ref = record_ref_for(&headers, &messages);

    let req_id = format!("r{}", st.req_seq.fetch_add(1, Ordering::Relaxed));
    let infer = InferRequest {
        req_id: req_id.clone(),
        op,
        model_id: model_id.clone(),
        messages,
        max_tokens: body.max_tokens,
        stream: body.stream,
        caps,
        record_ref,
        receipt: None,
    };

    if body.stream {
        stream_response(st, infer).await
    } else {
        unary_response(st, infer).await
    }
}

/// `POST /v1/completions` — adapt the legacy prompt shape to a single user message.
async fn completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<OpenAiCompletionRequest>,
) -> Response {
    let op = resolve_op(&headers, &st.registry, &body.model);
    let model_id = resolve_model_id(&st.registry, &body.model, op);
    let caps = caps_for(&headers, &st.default_caps);
    let messages = vec![ChatMessage { role: "user".into(), content: body.prompt }];
    let record_ref = record_ref_for(&headers, &messages);
    let req_id = format!("r{}", st.req_seq.fetch_add(1, Ordering::Relaxed));
    let infer = InferRequest {
        req_id,
        op,
        model_id,
        messages,
        max_tokens: body.max_tokens,
        stream: body.stream,
        caps,
        record_ref,
        receipt: None,
    };
    if body.stream {
        stream_response(st, infer).await
    } else {
        unary_response(st, infer).await
    }
}

/// Map the `X-CE-Op` header (chat|summarize|code) to an [`Op`]; default by the model's registry role.
fn resolve_op(headers: &HeaderMap, registry: &Registry, model: &str) -> Op {
    if let Some(h) = headers.get("x-ce-op").and_then(|v| v.to_str().ok())
        && let Ok(op) = Op::parse(h)
    {
        return op;
    }
    match registry.get(model).map(|m| m.role) {
        Some(Role::Code) => Op::Code,
        Some(Role::Summarize) => Op::Summarize,
        _ => Op::Chat,
    }
}

/// Resolve a (possibly logical-alias) model name to a concrete registry model id. A client may send
/// `"clinical-chat"` as an alias; we pick the first available model satisfying the op's role.
fn resolve_model_id(registry: &Registry, requested: &str, op: Op) -> String {
    if registry.get(requested).is_some() {
        return requested.to_string();
    }
    let role = match op {
        Op::Chat => Role::Chat,
        Op::Summarize => Role::Summarize,
        Op::Code => Role::Code,
    };
    registry
        .available_for_role(role)
        .first()
        .map(|m| m.id.clone())
        .unwrap_or_else(|| requested.to_string())
}

/// The capability chain to present: a per-request `X-CE-Caps` header (SSO-mapped principal cap)
/// overrides the router-held default.
fn caps_for(headers: &HeaderMap, default: &str) -> String {
    headers
        .get("x-ce-caps")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}

/// The PHI record reference: a caller-supplied `X-CE-Record-Ref` SHA256 header (so PHI never leaves
/// the client), else a content-free hash of the request shape (so the audit log always has a stable,
/// non-PHI reference). NEVER hashes the message content into something reversible — we hash only the
/// message count + roles + lengths, which carry no PHI.
fn record_ref_for(headers: &HeaderMap, messages: &[ChatMessage]) -> String {
    if let Some(h) = headers.get("x-ce-record-ref").and_then(|v| v.to_str().ok()) {
        let h = h.trim();
        if h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit()) {
            return h.to_string();
        }
    }
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"ce-infer-record-shape-v1");
    hasher.update((messages.len() as u64).to_le_bytes());
    for m in messages {
        hasher.update(m.role.as_bytes());
        hasher.update((m.content.len() as u64).to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Choose ranked candidates for a request, applying the circuit breaker and op-based GPU preference.
async fn ranked_workers(st: &AppState, model_id: &str, op: Op) -> Vec<Candidate> {
    let atlas = st.client.atlas().await.unwrap_or_default();
    // Reputation from /history; cache misses => 0 (newcomer). Done lazily per candidate.
    let client = st.client.clone();
    let prefer_gpu = matches!(op, Op::Chat | Op::Code); // interactive ops prefer GPU.
    // We can't await inside the closure passed to `select`, so pre-fetch reputations for the live
    // set. Cheap: only the workers serving this model.
    let live: Vec<String> = atlas
        .iter()
        .filter(|e| ce_infer_router::is_candidate(e, model_id, DEFAULT_STALE_SECS))
        .map(|e| e.node_id.clone())
        .collect();
    let mut rep: HashMap<String, u64> = HashMap::new();
    for id in &live {
        let score = client.history(id).await.map(|h| h.delivered_work()).unwrap_or(0);
        rep.insert(id.clone(), score);
    }
    let now = ce_infer_core::now();
    let broken = {
        let b = st.breaker.lock().await;
        b.iter()
            .filter(|(_, v)| v.broken_until > now)
            .map(|(k, _)| k.clone())
            .collect::<std::collections::HashSet<_>>()
    };
    let ranked = select(&atlas, model_id, DEFAULT_STALE_SECS, prefer_gpu, |id| {
        rep.get(id).copied().unwrap_or(0)
    });
    ranked.into_iter().filter(|c| !broken.contains(&c.node_id)).collect()
}

/// Record a worker success/failure in the circuit breaker.
async fn record_result(st: &AppState, node_id: &str, ok: bool) {
    let mut b = st.breaker.lock().await;
    let e = b.entry(node_id.to_string()).or_default();
    if ok {
        e.consecutive = 0;
        e.broken_until = 0;
    } else {
        e.consecutive += 1;
        if e.consecutive >= CIRCUIT_K {
            e.broken_until = ce_infer_core::now() + CIRCUIT_COOLDOWN.as_secs();
            warn!(node_id, "worker circuit-broken after {CIRCUIT_K} consecutive failures");
        }
    }
}

/// The GB residency unit billed per second for `model_id` (from the registry; 1 GB floor).
fn model_gb_for(st: &AppState, model_id: &str) -> u64 {
    st.registry.get(model_id).map(|m| m.model_gb()).unwrap_or(1)
}

/// Charge the per-(payer, worker) billing meter for one completed request, sign a channel receipt
/// for the new cumulative, and hand it to the worker on the per-request receipt topic. The router is
/// the **payer**: it opens the channel on first sight of the worker, meters cost per token + per
/// GB-second, and signs the monotonic running total. Best-effort: a billing failure is logged and
/// never fails the inference (the worker's audit topic still records the op).
///
/// Returns the signed [`ReceiptMsg`] so callers may also embed it in the request payload.
async fn bill_request(
    st: &AppState,
    worker_node: &str,
    model_id: &str,
    tokens: u64,
    seconds: u64,
    req_id: &str,
) -> Option<ReceiptMsg> {
    let model_gb = model_gb_for(st, model_id);
    // Compute the new cumulative under the session lock, then drop it before awaiting node calls.
    let (channel_id, cumulative) = {
        let mut sessions = st.sessions.lock().await;
        if !sessions.contains_key(worker_node) {
            // Lock enough capacity for a long session: 10k request-units at the per-request floor.
            let capacity = Amount::from_base(st.price.per_request.base().saturating_mul(10_000).max(1));
            match st.wallet.open_channel(worker_node, capacity, 0).await {
                Ok(id) => {
                    sessions.insert(
                        worker_node.to_string(),
                        Session { channel_id: id, meter: Meter::new(st.price, model_gb, capacity) },
                    );
                }
                Err(e) => {
                    warn!(worker = worker_node, error = %e, "channel open failed — serving unbilled");
                    return None;
                }
            }
        }
        let session = sessions.get_mut(worker_node)?;
        let cumulative = session.meter.charge_request(tokens, seconds);
        (session.channel_id.clone(), cumulative)
    };
    sign_and_send_receipt(st, worker_node, &channel_id, cumulative, req_id).await
}

/// Charge accrued heartbeat intervals for a long streaming session and, if any whole interval
/// elapsed, sign + send an updated receipt. Returns the number of intervals charged.
async fn bill_heartbeats(
    st: &AppState,
    worker_node: &str,
    now_secs: u64,
    tokens_since_last: u64,
    req_id: &str,
) -> u64 {
    let (channel_id, cumulative, intervals) = {
        let mut sessions = st.sessions.lock().await;
        let Some(session) = sessions.get_mut(worker_node) else { return 0 };
        let n = session.meter.charge_heartbeats(now_secs, tokens_since_last);
        if n == 0 {
            return 0;
        }
        (session.channel_id.clone(), session.meter.cumulative(), n)
    };
    sign_and_send_receipt(st, worker_node, &channel_id, cumulative, req_id).await;
    intervals
}

/// Sign a receipt as the payer for `cumulative` on `channel_id` and deliver it to the worker on the
/// receipt topic so the host can advance its redeemable total.
async fn sign_and_send_receipt(
    st: &AppState,
    worker_node: &str,
    channel_id: &str,
    cumulative: Amount,
    req_id: &str,
) -> Option<ReceiptMsg> {
    let receipt = match st.wallet.sign_receipt(channel_id, worker_node, cumulative).await {
        Ok(r) => r,
        Err(e) => {
            warn!(worker = worker_node, error = %e, "receipt signing failed — serving unbilled");
            return None;
        }
    };
    let msg = ReceiptMsg {
        channel_id: receipt.channel_id,
        cumulative: receipt.cumulative,
        payer_sig: receipt.payer_sig,
        req_id: req_id.to_string(),
    };
    // Hand the receipt to the worker out of band (the request payload also carries the latest one).
    if let Ok(bytes) = serde_json::to_vec(&msg) {
        let _ = st.client.send_message(worker_node, &receipt_topic(req_id), &bytes).await;
    }
    Some(msg)
}

/// Non-streaming dispatch: try ranked workers in order until one returns a completion.
async fn unary_response(st: Arc<AppState>, infer: InferRequest) -> Response {
    let candidates = ranked_workers(&st, &infer.model_id, infer.op).await;
    if candidates.is_empty() {
        return error_json(StatusCode::SERVICE_UNAVAILABLE, &format!("no live worker for model '{}'", infer.model_id));
    }
    let payload = match serde_json::to_vec(&infer) {
        Ok(p) => p,
        Err(e) => return error_json(StatusCode::INTERNAL_SERVER_ERROR, &format!("encode request: {e}")),
    };
    for cand in &candidates {
        match st.client.request(&cand.node_id, TOPIC_INFER, &payload, REQUEST_TIMEOUT_MS).await {
            Ok(bytes) => match serde_json::from_slice::<InferReply>(&bytes) {
                Ok(reply) if reply.ok => {
                    record_result(&st, &cand.node_id, true).await;
                    // Bill the completed request: meter cost, sign a receipt, hand it to the worker.
                    let _ = bill_request(&st, &cand.node_id, &infer.model_id, reply.token_count, 0, &infer.req_id).await;
                    return Json(openai_chat_json(&infer.model_id, &reply.text, &reply.finish_reason, reply.token_count)).into_response();
                }
                Ok(reply) => {
                    // A denial/error from the worker is authoritative — do not retry on other workers.
                    record_result(&st, &cand.node_id, true).await;
                    let msg = reply.error.unwrap_or_else(|| "worker rejected request".into());
                    let code = if reply.finish_reason == "denied" {
                        StatusCode::FORBIDDEN
                    } else {
                        StatusCode::BAD_GATEWAY
                    };
                    return error_json(code, &msg);
                }
                Err(e) => {
                    warn!(node = %cand.node_id, error = %e, "bad worker reply — re-routing");
                    record_result(&st, &cand.node_id, false).await;
                }
            },
            Err(e) => {
                warn!(node = %cand.node_id, error = %e, "worker request failed — re-routing");
                record_result(&st, &cand.node_id, false).await;
            }
        }
    }
    error_json(StatusCode::BAD_GATEWAY, "all candidate workers failed")
}

/// Streaming dispatch: send the handshake request, then relay the worker's token deltas (received on
/// the per-request stream topic via the node's message ring) to the client as OpenAI SSE chunks.
async fn stream_response(st: Arc<AppState>, infer: InferRequest) -> Response {
    let candidates = ranked_workers(&st, &infer.model_id, infer.op).await;
    let Some(cand) = candidates.into_iter().next() else {
        return error_json(StatusCode::SERVICE_UNAVAILABLE, &format!("no live worker for model '{}'", infer.model_id));
    };
    let topic = stream_topic(&infer.req_id);
    // Subscribe so the node accepts/keeps deltas for this request. (Directed messages also arrive
    // without a subscription, but subscribing makes the topic explicit and future-proofs pubsub.)
    let _ = st.client.subscribe(&topic).await;

    let payload = match serde_json::to_vec(&infer) {
        Ok(p) => p,
        Err(e) => return error_json(StatusCode::INTERNAL_SERVER_ERROR, &format!("encode request: {e}")),
    };
    // Fire the handshake; the worker streams deltas on `topic` and returns a handshake reply.
    match st.client.request(&cand.node_id, TOPIC_INFER, &payload, REQUEST_TIMEOUT_MS).await {
        Ok(bytes) => {
            if let Ok(reply) = serde_json::from_slice::<InferReply>(&bytes)
                && !reply.ok
            {
                record_result(&st, &cand.node_id, true).await;
                return error_json(StatusCode::FORBIDDEN, &reply.error.unwrap_or_else(|| "denied".into()));
            }
            record_result(&st, &cand.node_id, true).await;
        }
        Err(e) => {
            record_result(&st, &cand.node_id, false).await;
            return error_json(StatusCode::BAD_GATEWAY, &format!("worker handshake failed: {e}"));
        }
    }

    let worker = cand.node_id.clone();
    let stream = delta_sse_stream(st.clone(), infer.req_id.clone(), infer.model_id.clone(), topic, worker);
    Sse::new(stream).into_response()
}

/// Build the SSE stream that consumes the node's push message stream (`messages_stream()`) for this
/// request's deltas, meters heartbeat billing as the generation runs, and emits OpenAI
/// `chat.completion.chunk` events, ending with `[DONE]`.
///
/// This is the SSE-helper path (replacing the old `messages()` poll). The pure relay lives in the
/// router library ([`ce_infer_router::relay_token_stream`]); here we wrap it with the node stream,
/// the heartbeat-billing callback, and the OpenAI chunk encoding. Channels keep deltas/chunks moving
/// between the relay task and the SSE response without blocking the relay on the client.
fn delta_sse_stream(
    st: Arc<AppState>,
    req_id: String,
    model_id: String,
    topic: String,
    worker: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    use ce_infer_router::RelayChunk;
    let deadline = ce_infer_core::now() + (REQUEST_TIMEOUT_MS / 1000) + 5;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<RelayChunk>();

    // Relay task: pull the worker's deltas off the node's SSE stream, bill heartbeats, forward.
    {
        let st = st.clone();
        let req_id = req_id.clone();
        let topic = topic.clone();
        let worker = worker.clone();
        tokio::spawn(async move {
            match st.client.messages_stream().await {
                Ok(messages) => {
                    let st2 = st.clone();
                    let worker2 = worker.clone();
                    let req2 = req_id.clone();
                    ce_infer_router::relay_token_stream(
                        messages,
                        &req_id,
                        &topic,
                        deadline,
                        ce_infer_core::now,
                        |now_secs, tokens_since_last| {
                            let st = st2.clone();
                            let worker = worker2.clone();
                            let req = req2.clone();
                            async move {
                                let _ = bill_heartbeats(&st, &worker, now_secs, tokens_since_last, &req).await;
                            }
                        },
                        |chunk| {
                            let tx = tx.clone();
                            async move {
                                let _ = tx.send(chunk);
                            }
                        },
                    )
                    .await;
                }
                Err(e) => warn!(error = %e, "messages_stream failed — no token relay"),
            }
        });
    }

    async_stream::stream! {
        while let Some(chunk) = rx.recv().await {
            match chunk {
                RelayChunk::Delta(text) => {
                    let json = openai_chunk_json(&model_id, &text, None);
                    yield Ok(Event::default().data(json.to_string()));
                }
                RelayChunk::Final(reason) => {
                    let json = openai_chunk_json(&model_id, "", Some(&reason));
                    yield Ok(Event::default().data(json.to_string()));
                    break;
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

// ----- OpenAI response builders -----

fn openai_chat_json(model: &str, content: &str, finish: &str, tokens: u64) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-ce",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": finish,
        }],
        "usage": { "completion_tokens": tokens, "prompt_tokens": 0, "total_tokens": tokens },
    })
}

fn openai_chunk_json(model: &str, delta: &str, finish: Option<&str>) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-ce",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": if finish.is_some() { serde_json::json!({}) } else { serde_json::json!({ "content": delta }) },
            "finish_reason": finish,
        }],
    })
}

fn error_json(code: StatusCode, message: &str) -> Response {
    (code, Json(serde_json::json!({ "error": { "message": message } }))).into_response()
}
