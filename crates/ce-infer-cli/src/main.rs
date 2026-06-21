//! # ce-infer — ops CLI for the ce-infer app.
//!
//! Thin operator tooling over `ce-rs` + `ce-infer-core`:
//!   - `probe`              — print this node's tier + the model the registry would assign it.
//!   - `models pull <id>`   — fetch a model's GGUF over CE blobs into the local cache.
//!   - `models publish <f>` — upload a GGUF (`put_object` -> CID), update models.toml.
//!   - `status`             — atlas view of live ce-infer workers.
//!   - `audit export`       — pull the audit topic log + `/history` for OCR review (JSONL).
//!   - `grant`              — thin wrapper over `ce grant` issuing infer abilities + a model_prefix.

use anyhow::{Context, Result, anyhow, bail};
use ce_infer_core::audit::{AuditRecord, TOPIC};
use ce_infer_core::caps;
use ce_infer_core::registry::{ModelEntry, Registry, Role};
use ce_infer_core::{models_dir, now, probe};
use ce_rs::CeClient;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "ce-infer", about = "ce-infer operations CLI")]
struct Cli {
    /// Local CE node HTTP API base URL.
    #[arg(long, default_value = ce_rs::DEFAULT_BASE_URL, global = true)]
    node: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Print this node's tier and the model it would be assigned.
    Probe,
    /// Model registry operations.
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Atlas view of live ce-infer workers.
    Status,
    /// Export the audit log for OCR review.
    Audit {
        #[command(subcommand)]
        cmd: AuditCmd,
    },
    /// Issue an infer capability to a node (wraps `ce grant`).
    Grant {
        /// Recipient node id (hex).
        node_id: String,
        /// Comma-separated abilities (e.g. `infer:chat,infer:summarize`).
        #[arg(long)]
        can: String,
        /// Restrict to model ids under this prefix (e.g. `clinical-`).
        #[arg(long)]
        model_prefix: Option<String>,
        /// Expiry (passed through to `ce grant`, e.g. `30d`).
        #[arg(long, default_value = "30d")]
        expires: String,
    },
}

#[derive(Subcommand, Debug)]
enum ModelsCmd {
    /// Fetch a model's GGUF over CE blobs into the local cache.
    Pull {
        /// Model id (resolved via the registry) OR a raw object CID.
        model: String,
        /// Path to a models.toml registry (otherwise the built-in default).
        #[arg(long)]
        registry: Option<PathBuf>,
    },
    /// Publish a GGUF file: upload it (`put_object` -> CID) and write the CID into models.toml.
    Publish {
        /// Path to the .gguf file.
        file: PathBuf,
        /// Logical model id to attach the CID to.
        #[arg(long)]
        id: String,
        /// models.toml to update (created from the built-in default if absent).
        #[arg(long, default_value = "models.toml")]
        registry: PathBuf,
    },
    /// List the registry's models and which are published.
    List {
        #[arg(long)]
        registry: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum AuditCmd {
    /// Export audit records seen on the audit topic over the last `--since` hours, JSONL.
    Export {
        /// Look back this many hours.
        #[arg(long, default_value_t = 24)]
        since: u64,
        /// Output file (`-` for stdout).
        #[arg(short, long, default_value = "-")]
        out: String,
        /// Also include /history facts for this node id (reputation substrate).
        #[arg(long)]
        node: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let cli = Cli::parse();
    let client = CeClient::new(&cli.node);
    match cli.cmd {
        Cmd::Probe => cmd_probe(),
        Cmd::Models { cmd } => cmd_models(&client, cmd).await,
        Cmd::Status => cmd_status(&client).await,
        Cmd::Audit { cmd } => cmd_audit(&client, cmd).await,
        Cmd::Grant { node_id, can, model_prefix, expires } => {
            cmd_grant(&node_id, &can, model_prefix.as_deref(), &expires)
        }
    }
}

fn cmd_probe() -> Result<()> {
    let p = probe();
    let registry = Registry::builtin();
    let chosen = registry
        .select_for(p.tier, p.ram_mb, p.vram_mb())
        .map(|m| m.id.as_str())
        .unwrap_or("<none — publish weights first>");
    println!("os/arch     : {}/{}", p.os, p.arch);
    println!("cores       : {}", p.cores);
    println!("ram_mb      : {}", p.ram_mb);
    match &p.gpu {
        Some(g) => println!("gpu         : {} ({} MB VRAM)", g.vendor.as_str(), g.vram_mb),
        None => println!("gpu         : none"),
    }
    println!("tier        : {}", p.tier.as_str());
    println!("assigned    : {chosen}");
    println!("atlas tags  : {}", p.atlas_tags().join(", "));
    Ok(())
}

async fn cmd_models(client: &CeClient, cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::Pull { model, registry } => {
            let reg = load_or_builtin(registry.as_deref())?;
            // Accept either a logical id (resolve to CID) or a raw CID.
            let cid = match reg.get(&model) {
                Some(m) if m.is_available() => m.gguf_object_cid.clone(),
                Some(_) => bail!("model '{model}' has no published CID yet (run `models publish`)"),
                None => model.clone(), // treat as a raw CID
            };
            let dir = models_dir();
            std::fs::create_dir_all(&dir)?;
            let path = dir.join(format!("{cid}.gguf"));
            if path.exists() {
                println!("already present: {}", path.display());
                return Ok(());
            }
            println!("fetching {cid} over CE blobs…");
            let bytes = client.get_object(&cid).await.with_context(|| format!("fetch {cid}"))?;
            let tmp = dir.join(format!(".{cid}.partial"));
            std::fs::write(&tmp, &bytes)?;
            std::fs::rename(&tmp, &path)?;
            println!("wrote {} ({} bytes)", path.display(), bytes.len());
            Ok(())
        }
        ModelsCmd::Publish { file, id, registry } => {
            let bytes = std::fs::read(&file).with_context(|| format!("read {}", file.display()))?;
            println!("uploading {} ({} bytes) to CE blobs…", file.display(), bytes.len());
            let cid = client.put_object(&bytes).await.context("put_object")?;
            println!("object CID: {cid}");
            // Update (or create) the registry.
            let mut reg = if registry.exists() {
                Registry::from_toml(&std::fs::read_to_string(&registry)?)?
            } else {
                Registry::builtin()
            };
            match reg.get_mut(&id) {
                Some(m) => m.gguf_object_cid = cid.clone(),
                None => bail!("model id '{id}' is not declared in {} — add it first", registry.display()),
            }
            std::fs::write(&registry, reg.to_toml()?)?;
            println!("updated {} with {id} -> {cid}", registry.display());
            println!("(now run `ce-pin` / replicate the CID so it spreads across the LAN)");
            Ok(())
        }
        ModelsCmd::List { registry } => {
            let reg = load_or_builtin(registry.as_deref())?;
            for m in &reg.models {
                let status = if m.is_available() { "published" } else { "declared " };
                println!(
                    "{status}  {:<20} role={:<9} quant={:<7} ctx={:<6} ram_min={}MB vram_min={}MB",
                    m.id,
                    format!("{:?}", m.role).to_lowercase(),
                    m.quant,
                    m.ctx,
                    m.ram_min_mb,
                    m.vram_min_mb
                );
            }
            Ok(())
        }
    }
}

fn load_or_builtin(path: Option<&std::path::Path>) -> Result<Registry> {
    match path {
        Some(p) => Registry::from_toml(&std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?),
        None => Ok(Registry::builtin()),
    }
}

async fn cmd_status(client: &CeClient) -> Result<()> {
    let atlas = client.atlas().await.context("query atlas")?;
    let workers: Vec<_> = atlas.iter().filter(|e| e.has_tag("infer")).collect();
    if workers.is_empty() {
        println!("no live ce-infer workers in the atlas");
        return Ok(());
    }
    println!("{:<18} {:<10} {:<6} {:<8} models", "node", "tier", "jobs", "seen_s");
    for w in workers {
        let tier = w
            .tags
            .iter()
            .find(|t| t.starts_with("gpu-") || t.starts_with("cpu-"))
            .cloned()
            .unwrap_or_else(|| "?".into());
        let models: Vec<&str> =
            w.tags.iter().filter_map(|t| t.strip_prefix("model:")).collect();
        let short: String = w.node_id.chars().take(16).collect();
        println!(
            "{:<18} {:<10} {:<6} {:<8} {}",
            short,
            tier,
            w.running_jobs,
            w.last_seen_secs,
            models.join(",")
        );
    }
    Ok(())
}

async fn cmd_audit(client: &CeClient, cmd: AuditCmd) -> Result<()> {
    match cmd {
        AuditCmd::Export { since, out, node } => {
            // Ensure we're subscribed so the node's ring carries audit records.
            let _ = client.subscribe(TOPIC).await;
            let cutoff = now().saturating_sub(since * 3600);
            let msgs = client.messages().await.unwrap_or_default();
            let mut lines: Vec<String> = Vec::new();
            for m in msgs.iter().filter(|m| m.topic == TOPIC) {
                let Ok(payload) = m.payload() else { continue };
                let Ok(rec) = AuditRecord::from_bytes(&payload) else { continue };
                if rec.ts < cutoff {
                    continue;
                }
                // Redaction is asserted on the way out too — never export a record carrying PHI.
                if rec.assert_redacted().is_err() {
                    continue;
                }
                lines.push(serde_json::to_string(&rec)?);
            }
            if let Some(n) = node {
                let h = client.history(&n).await.context("query /history")?;
                lines.push(serde_json::to_string(&serde_json::json!({
                    "kind": "history",
                    "node_id": h.node_id,
                    "jobs_hosted": h.jobs_hosted,
                    "heartbeats_hosted": h.heartbeats_hosted,
                    "earned": h.earned.base().to_string(),
                    "spent": h.spent.base().to_string(),
                }))?);
            }
            let body = lines.join("\n");
            if out == "-" {
                println!("{body}");
            } else {
                std::fs::write(&out, format!("{body}\n"))?;
                println!("wrote {} audit record(s) to {out}", lines.len());
            }
            Ok(())
        }
    }
}

/// Wrap `ce grant`, mapping the requested abilities + optional model_prefix into the ce-cap chain
/// the node issues. The `infer:model_prefix:<p>` ability is appended so the resulting capability
/// carries the model restriction (enforced by the worker's leaf check).
fn cmd_grant(node_id: &str, can: &str, model_prefix: Option<&str>, expires: &str) -> Result<()> {
    let ops: Vec<&str> = can.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if ops.is_empty() {
        bail!("--can must list at least one ability (e.g. infer:chat)");
    }
    let abilities = caps::grant_abilities(&ops, model_prefix);
    let can_arg = abilities.join(",");
    println!("issuing capability to {node_id}: can={can_arg} expires={expires}");
    let status = std::process::Command::new("ce")
        .arg("grant")
        .arg(node_id)
        .arg("--can")
        .arg(&can_arg)
        .arg("--expires")
        .arg(expires)
        .status()
        .map_err(|e| anyhow!("failed to run `ce grant` (is the ce CLI on PATH?): {e}"))?;
    if !status.success() {
        bail!("`ce grant` exited with {status}");
    }
    Ok(())
}

// Keep ModelEntry/Role referenced so the imports are not flagged in trimmed builds.
#[allow(dead_code)]
fn _doc_refs(_m: &ModelEntry, _r: Role) {}
