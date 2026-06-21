//! # ce-infer-worker — the per-node inference server.
//!
//! One per capable fleet node. Pattern = rdev/ce-pin server: a poll loop over mesh AppRequest
//! (`ce.messages()` / `ce.reply()`) authorizing every request with `ce_cap::authorize`, then
//! forwarding it to a local llama.cpp `llama-server` bound to loopback (or a deterministic mock
//! backend when no GGUF is published — so the routing path is testable end to end).
//!
//! Startup: probe -> resolve model from the registry by tier -> ensure weights present (pull the
//! GGUF over CE blobs by CID) -> launch the engine -> advertise the service on the DHT -> serve.
//!
//! Trust: capability-only. A request is honored only if its presented chain roots at this worker's
//! own key or a configured org root, attenuates correctly, grants the op's ability, and (for the
//! leaf) satisfies the `model_prefix` caveat. A denied attempt is STILL audited.

mod backend;

use anyhow::{Context, Result, anyhow};
use backend::Backend;
use ce_cap::{SignedCapability, decode_chain};
use ce_infer_core::audit::{AuditRecord, Outcome};
use ce_infer_core::billing::HighestReceipt;
use ce_infer_core::proto::{
    InferReply, InferRequest, ReceiptMsg, StreamDelta, TOPIC_INFER, stream_topic,
};
use ce_infer_core::registry::{ModelEntry, Registry};
use ce_infer_core::serve::{Decision, build_audit, decide, decision_outcome, decision_reply};
use ce_infer_core::{CapabilityProfile, load_roots, models_dir, node_id_from_hex, now, probe};
use ce_rs::{Amount, CeClient};
use clap::Parser;
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(name = "ce-infer-worker", about = "ce-infer per-node inference server")]
struct Args {
    /// Local CE node HTTP API base URL.
    #[arg(long, default_value = ce_rs::DEFAULT_BASE_URL)]
    node: String,
    /// Engine binary (llama.cpp server). Bundled per-platform by the installer.
    #[arg(long, default_value = "llama-server")]
    engine: String,
    /// Force the mock backend (no GGUF / no engine needed). Auto-enabled when no model is published.
    #[arg(long)]
    mock: bool,
    /// Override the assigned model id (otherwise chosen from the registry by tier).
    #[arg(long)]
    model: Option<String>,
    /// Path to a models.toml registry blob (otherwise the built-in default registry is used).
    #[arg(long)]
    registry: Option<PathBuf>,
    /// Continuous-batching parallel slots for llama-server.
    #[arg(long, default_value_t = 4)]
    parallel: u32,
    /// Per-request price in base units (the channel-receipt increment). Default ~0.001 credit.
    #[arg(long, default_value_t = (ce_rs::CREDIT / 1000) as u128)]
    price_base: u128,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let client = CeClient::new(&args.node);
    run(client, args).await
}

async fn run(client: CeClient, args: Args) -> Result<()> {
    let host_hex = client.status().await.context("query node status")?.node_id;
    let host_id = node_id_from_hex(&host_hex)?;
    let host_short = host_hex.chars().take(16).collect::<String>();

    // 1. Probe hardware and self-classify.
    let mut profile = probe();
    if !profile.tier.is_eligible() {
        info!(
            tier = profile.tier.as_str(),
            ram_mb = profile.ram_mb,
            "node is Ineligible for inference — meshing only, not serving as a worker"
        );
        return Ok(());
    }

    // 2. Resolve the assigned model from the registry by tier (or an admin override).
    let registry = load_registry(args.registry.as_deref())?;
    let assigned: ModelEntry = resolve_model(&registry, &profile, args.model.as_deref())?;
    profile.assigned_model = Some(assigned.id.clone());
    info!(
        tier = profile.tier.as_str(),
        model = %assigned.id,
        vram_mb = profile.vram_mb(),
        ram_mb = profile.ram_mb,
        "worker {host_short} assigned model"
    );

    // 3 + 4. Ensure weights present and launch the backend (mock if unavailable/forced).
    let backend = Arc::new(start_backend(&client, &args, &profile, &assigned).await?);

    // 5. Advertise the service on the DHT so routers discover us. (Self-tags for the atlas are set
    // by the node's own capability mechanism; ce-rs does not yet expose a push-tags call, so DHT
    // service advertisement is the discovery path — see TODO in serve_loop docs.)
    let service = format!("infer:{}", assigned.id);
    advertise(&client, &service).await;

    let roots = load_roots();
    let registry_version = registry.version;
    info!(
        "ce-infer-worker serving as {host_short} — model={} backend={} service={service} — {} root(s)",
        assigned.id,
        backend.label(),
        roots.len()
    );

    // Background re-advertisement so the provider record never expires.
    {
        let client2 = client.clone();
        let service2 = service.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(50)).await;
                advertise(&client2, &service2).await;
            }
        });
    }

    serve_loop(
        &client,
        &host_id,
        &host_short,
        &roots,
        backend,
        &assigned,
        registry_version,
        Amount::from_base(args.price_base as i128),
    )
    .await
}

/// Load the registry from a TOML blob path, or fall back to the built-in default.
fn load_registry(path: Option<&std::path::Path>) -> Result<Registry> {
    match path {
        Some(p) => {
            let s = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
            Registry::from_toml(&s)
        }
        None => Ok(Registry::builtin()),
    }
}

/// Resolve which model this worker serves: an explicit override (must exist), else the
/// tier-selected default.
fn resolve_model(
    registry: &Registry,
    profile: &CapabilityProfile,
    override_id: Option<&str>,
) -> Result<ModelEntry> {
    if let Some(id) = override_id {
        return registry.get(id).cloned().ok_or_else(|| anyhow!("model '{id}' not in registry"));
    }
    registry
        .select_for(profile.tier, profile.ram_mb, profile.vram_mb())
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "no published model fits tier {} (publish weights with `ce-infer models publish`)",
                profile.tier.as_str()
            )
        })
}

/// Ensure the GGUF is present locally and launch the engine. Falls back to the mock backend when
/// `--mock` is set, when the model has no published CID, or when fetching/launching the engine
/// fails (so a fresh fleet is testable before any weights exist).
async fn start_backend(
    client: &CeClient,
    args: &Args,
    profile: &CapabilityProfile,
    model: &ModelEntry,
) -> Result<Backend> {
    if args.mock || !model.is_available() {
        if !model.is_available() {
            warn!(model = %model.id, "no GGUF CID published — using deterministic mock backend");
        }
        return Ok(Backend::Mock);
    }
    let path = match ensure_weights(client, &model.gguf_object_cid).await {
        Ok(p) => p,
        Err(e) => {
            warn!(error = %e, model = %model.id, "weight fetch failed — falling back to mock backend");
            return Ok(Backend::Mock);
        }
    };
    let port = pick_loopback_port();
    match backend::LlamaServer::spawn(
        &args.engine,
        &path,
        port,
        model.ctx,
        args.parallel,
        profile.tier.is_gpu(),
    ) {
        Ok(srv) => {
            info!(port, model = %model.id, "launched llama-server on loopback");
            Ok(Backend::Llama(srv))
        }
        Err(e) => {
            warn!(error = %e, "engine launch failed — falling back to mock backend");
            Ok(Backend::Mock)
        }
    }
}

/// Ensure `<models_dir>/<cid>.gguf` exists, pulling the object over CE blobs (CID-verified by the
/// SDK) if absent. Returns the on-disk path.
async fn ensure_weights(client: &CeClient, cid: &str) -> Result<PathBuf> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let path = dir.join(format!("{cid}.gguf"));
    if path.exists() {
        info!(%cid, "weights already present locally");
        return Ok(path);
    }
    info!(%cid, "fetching GGUF over CE blobs…");
    let bytes = client.get_object(cid).await.with_context(|| format!("fetch object {cid}"))?;
    let tmp = dir.join(format!(".{cid}.gguf.partial"));
    std::fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("rename into {}", path.display()))?;
    info!(%cid, bytes = bytes.len(), "weights ready");
    Ok(path)
}

async fn advertise(client: &CeClient, service: &str) {
    if let Err(e) = client.advertise_service(service).await {
        warn!(error = %e, service, "service advertisement failed (will retry)");
    }
}

/// Pick a high loopback port for the engine. Collisions are tolerated: the OS bind fails, the
/// engine launch errors, and we fall back to the mock backend.
fn pick_loopback_port() -> u16 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    20_000 + (nanos % 20_000) as u16
}

/// Host-side billing state for one payment channel: the highest receipt the payer has signed for
/// this channel, which the worker redeems with `channel_close` when the session ends or the loop
/// shuts down. The router is the payer (it opens the channel and signs receipts); the worker is the
/// host (it accumulates the highest authorized cumulative and redeems exactly it).
struct Session {
    receipt: HighestReceipt,
}

/// The main request poll loop.
#[allow(clippy::too_many_arguments)]
async fn serve_loop(
    client: &CeClient,
    host_id: &[u8; 32],
    host_short: &str,
    roots: &[[u8; 32]],
    backend: Arc<Backend>,
    model: &ModelEntry,
    registry_version: u32,
    price: Amount,
) -> Result<()> {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut revoked: HashSet<([u8; 32], u64)> = HashSet::new();
    let mut channels: HashMap<String, Session> = HashMap::new();

    // Subscribe to inference + per-channel receipt traffic, then consume the node's push stream
    // (`messages_stream()` SSE) instead of polling `messages()`. The stream yields both unary
    // requests (with a reply_token) on TOPIC_INFER and out-of-band receipt messages on the
    // `infer/receipt/*` topics that a long streaming session signs.
    let _ = client.subscribe(TOPIC_INFER).await;
    let last_revoke_refresh = std::time::Instant::now();
    refresh_revoked(client, &mut revoked).await;

    // Outer loop reconnects the SSE stream if it drops; the worker keeps serving across reconnects.
    loop {
        let mut stream = match client.messages_stream().await {
            Ok(s) => Box::pin(s),
            Err(e) => {
                warn!(error = %e, "messages_stream open failed — retrying");
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };
        let mut last_refresh = last_revoke_refresh;
        loop {
            let next = tokio::time::timeout(Duration::from_secs(5), stream.next()).await;
            // Periodically refresh the revoked set and redeem accrued receipts.
            if last_refresh.elapsed() >= Duration::from_secs(30) {
                refresh_revoked(client, &mut revoked).await;
                redeem_all(client, &channels).await;
                last_refresh = std::time::Instant::now();
            }
            let m = match next {
                Ok(Some(Ok(m))) => m,
                Ok(Some(Err(e))) => {
                    warn!(error = %e, "message stream error — reconnecting");
                    break;
                }
                Ok(None) => break, // stream ended; reconnect.
                Err(_) => continue, // idle timeout; loop to do periodic work.
            };

            // Receipt traffic for a long streaming session (no reply_token).
            if m.topic.starts_with("infer/receipt/") {
                if let Ok(bytes) = m.payload()
                    && let Ok(receipt) = serde_json::from_slice::<ReceiptMsg>(&bytes)
                {
                    ingest_receipt(&mut channels, &receipt);
                }
                continue;
            }

            let Some(token) = m.reply_token else { continue };
            if m.topic != TOPIC_INFER || !seen.insert(token) {
                continue;
            }
            let reply = handle_request(
                client,
                host_id,
                host_short,
                roots,
                &revoked,
                &backend,
                model,
                registry_version,
                price,
                &mut channels,
                &m.from,
                &m.payload_hex,
            )
            .await;
            let bytes = serde_json::to_vec(&reply).unwrap_or_default();
            if let Err(e) = client.reply(token, &bytes).await {
                warn!(error = %e, "reply failed");
            }
        }
    }
}

async fn refresh_revoked(client: &CeClient, revoked: &mut HashSet<([u8; 32], u64)>) {
    if let Ok(pairs) = client.revoked().await {
        *revoked = pairs
            .into_iter()
            .filter_map(|(issuer, nonce)| node_id_from_hex(&issuer).ok().map(|i| (i, nonce)))
            .collect();
    }
}

/// Authorize, run, bill, and audit a single inference request; never returns Err to the loop.
#[allow(clippy::too_many_arguments)]
async fn handle_request(
    client: &CeClient,
    host_id: &[u8; 32],
    host_short: &str,
    roots: &[[u8; 32]],
    revoked: &HashSet<([u8; 32], u64)>,
    backend: &Backend,
    model: &ModelEntry,
    registry_version: u32,
    price: Amount,
    channels: &mut HashMap<String, Session>,
    from_hex: &str,
    payload_hex: &str,
) -> InferReply {
    match handle_inner(
        client,
        host_id,
        host_short,
        roots,
        revoked,
        backend,
        model,
        registry_version,
        price,
        channels,
        from_hex,
        payload_hex,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "request error");
            InferReply::error(e.to_string())
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_inner(
    client: &CeClient,
    host_id: &[u8; 32],
    host_short: &str,
    roots: &[[u8; 32]],
    revoked: &HashSet<([u8; 32], u64)>,
    backend: &Backend,
    model: &ModelEntry,
    registry_version: u32,
    price: Amount,
    channels: &mut HashMap<String, Session>,
    from_hex: &str,
    payload_hex: &str,
) -> Result<InferReply> {
    let req: InferRequest =
        serde_json::from_slice(&hex::decode(payload_hex).context("payload hex")?).context("payload json")?;
    let from = node_id_from_hex(from_hex)?;
    let model_label = format!("{}@v{registry_version}", model.id);

    let chain: Vec<SignedCapability> =
        decode_chain(&req.caps).map_err(|_| anyhow!("malformed capability chain"))?;
    let is_revoked = |issuer: &[u8; 32], nonce: u64| revoked.contains(&(*issuer, nonce));
    let cap_id = AuditRecord::capability_id_of(&ce_cap::encode_chain_bytes(&chain));

    // Authorize + validate via the shared pure decision logic (also unit-tested in ce-infer-core).
    let decision = decide(host_id, roots, now(), &from, &model.id, &req, &chain, &is_revoked);
    if !matches!(decision, Decision::Allow) {
        // A denied/error attempt is STILL audited (PHI-free).
        let outcome = decision_outcome(&decision);
        publish_audit(client, &build_audit(&req, from_hex, host_short, &model_label, &cap_id, outcome, 0)).await;
        if let Decision::Deny(reason) = &decision {
            warn!(reason = %reason, "request DENIED");
        }
        // Safe to unwrap-free: non-Allow always yields a reply.
        return Ok(decision_reply(&decision).unwrap_or_else(|| InferReply::error("rejected")));
    }

    let completion = match backend.complete(&req).await {
        Ok(c) => c,
        Err(e) => {
            publish_audit(client, &build_audit(&req, from_hex, host_short, &model_label, &cap_id, Outcome::Error, 0)).await;
            return Ok(InferReply::error(format!("inference failed: {e}")));
        }
    };

    // Ingest the payer's signed receipt for this request (host side); the router is the payer.
    if let Some(receipt) = &req.receipt {
        ingest_receipt(channels, receipt);
    }
    let _ = price; // pricing now lives on the router (payer); the worker only redeems receipts.
    publish_audit(
        client,
        &build_audit(&req, from_hex, host_short, &model_label, &cap_id, Outcome::Ok, completion.token_count),
    )
    .await;

    if req.stream {
        stream_back(client, from_hex, &req, &completion).await;
        return Ok(InferReply {
            ok: true,
            text: String::new(),
            token_count: completion.token_count,
            model_id: model.id.clone(),
            finish_reason: completion.finish_reason,
            error: None,
        });
    }

    Ok(InferReply {
        ok: true,
        text: completion.text,
        token_count: completion.token_count,
        model_id: model.id.clone(),
        finish_reason: completion.finish_reason,
        error: None,
    })
}

/// Send token deltas to the router's node on the per-request stream topic, terminated by a final
/// delta carrying the finish_reason. Best-effort: a delivery failure ends the stream.
async fn stream_back(client: &CeClient, to_hex: &str, req: &InferRequest, completion: &backend::Completion) {
    let topic = stream_topic(&req.req_id);
    let mut seq = 0u64;
    for tok in completion.tokens() {
        let delta = StreamDelta { req_id: req.req_id.clone(), seq, delta: tok, finish_reason: None };
        if send_delta(client, to_hex, &topic, &delta).await.is_err() {
            return;
        }
        seq += 1;
    }
    let final_delta = StreamDelta {
        req_id: req.req_id.clone(),
        seq,
        delta: String::new(),
        finish_reason: Some(completion.finish_reason.clone()),
    };
    let _ = send_delta(client, to_hex, &topic, &final_delta).await;
}

async fn send_delta(client: &CeClient, to_hex: &str, topic: &str, delta: &StreamDelta) -> Result<()> {
    let bytes = serde_json::to_vec(delta)?;
    client.send_message(to_hex, topic, &bytes).await
}

/// Ingest a payer-signed channel receipt: track the highest cumulative per channel so the worker can
/// redeem exactly it. Receipts arrive embedded in the request (and out of band on the receipt topic
/// for long streams). Only a strictly higher cumulative advances the redeemable total — replays and
/// out-of-order receipts are ignored. The receipt IS the economic + audit record of compute sold;
/// a missing receipt never blocks inference (the audit topic still records the op).
fn ingest_receipt(channels: &mut HashMap<String, Session>, receipt: &ReceiptMsg) {
    let session = channels
        .entry(receipt.channel_id.clone())
        .or_insert_with(|| Session { receipt: HighestReceipt::default() });
    if session.receipt.offer(receipt.cumulative, &receipt.payer_sig) {
        tracing::debug!(
            channel = %receipt.channel_id,
            cumulative = %receipt.cumulative.credits(),
            "advanced redeemable receipt"
        );
    }
}

/// Redeem every channel's highest receipt with one `channel_close`, settling the off-chain receipts
/// on-chain. Called when the worker shuts down (or could be called per idle channel). Best-effort:
/// a redemption failure is logged; the payer can still reclaim via expiry.
async fn redeem_all(client: &CeClient, channels: &HashMap<String, Session>) {
    for (channel_id, session) in channels {
        if !session.receipt.is_redeemable() {
            continue;
        }
        if let Err(e) = client
            .channel_close(channel_id, session.receipt.cumulative(), session.receipt.payer_sig())
            .await
        {
            warn!(channel = %channel_id, error = %e, "channel close (redeem) failed");
        } else {
            info!(
                channel = %channel_id,
                redeemed = %session.receipt.cumulative().credits(),
                "redeemed channel receipt on close"
            );
        }
    }
}

/// Publish a signed, PHI-free audit record on the audit topic. Redaction is asserted before
/// publishing; a tripped guard fails closed (nothing written).
async fn publish_audit(client: &CeClient, rec: &AuditRecord) {
    if let Err(e) = rec.assert_redacted() {
        error!(error = %e, "REFUSING to write audit record (redaction guard tripped)");
        return;
    }
    match rec.to_bytes() {
        Ok(bytes) => {
            if let Err(e) = client.publish(ce_infer_core::audit::TOPIC, &bytes).await {
                warn!(error = %e, "audit publish failed");
            }
        }
        Err(e) => error!(error = %e, "audit encode failed"),
    }
}
