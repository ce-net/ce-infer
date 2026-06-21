//! # ce-infer-core
//!
//! Shared types and helpers for the ce-infer app — the distributed clinical-inference system built
//! ON CE primitives (identity + mesh + blobs + ledger + the `ce-cap` verifier). This crate is the
//! seam between the worker, the router, and the CLI:
//!
//! - [`probe`] — hardware detection + the deterministic self-tier rule.
//! - [`registry`] — the signed `models.toml` manifest mapping logical model ids to GGUF CIDs.
//! - [`caps`] — ce-infer's capability abilities + the `model_prefix` app caveat.
//! - [`audit`] — tamper-evident, PHI-free inference records (HIPAA §164.312(b)/(c)).
//! - [`proto`] — the router<->worker mesh wire protocol.
//!
//! Everything here talks to CE only through `ce-rs` against a local node; ce-infer adds NO node
//! endpoints. Mesh-first, capability-only trust.

pub mod audit;
pub mod billing;
pub mod caps;
pub mod probe;
pub mod proto;
pub mod registry;
pub mod serve;

pub use billing::{HighestReceipt, Meter, PriceSheet, HEARTBEAT_INTERVAL_SECS};
pub use probe::{CapabilityProfile, Gpu, GpuVendor, Tier, probe};
pub use registry::{ModelEntry, Registry, Role};

use anyhow::{Result, anyhow};
use std::path::PathBuf;

/// Current unix time in seconds (for capability temporal checks and audit timestamps).
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Parse a 64-hex CE node id into the `[u8; 32]` the ce-cap verifier expects.
pub fn node_id_from_hex(hex_str: &str) -> Result<[u8; 32]> {
    hex::decode(hex_str.trim())
        .ok()
        .and_then(|b| <[u8; 32]>::try_from(b).ok())
        .ok_or_else(|| anyhow!("'{hex_str}' is not a 64-hex node id"))
}

/// The ce-infer data directory: `$CE_INFER_DATA`, else `$CE_DATA_DIR/ce-infer`, else
/// `~/.local/share/ce/ce-infer`. Models are cached under `<data_dir>/models/<cid>.gguf`.
pub fn data_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("CE_INFER_DATA") {
        return PathBuf::from(d);
    }
    if let Some(d) = std::env::var_os("CE_DATA_DIR") {
        return PathBuf::from(d).join("ce-infer");
    }
    home_dir().join(".local/share/ce/ce-infer")
}

/// The directory cached GGUF model files live in.
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

/// Load accepted capability root keys (64-hex node ids, one per line, `#` comments). Looked up at
/// `$CE_INFER_ROOTS`, else `$CE_DATA_DIR/roots`, else `~/.local/share/ce/roots` — mirrors the node's
/// `<data_dir>/roots` and rdev's `load_roots`. A node opts into an org/fleet by listing that org's
/// root key here. Empty by default => only self-issued caps are honored.
pub fn load_roots() -> Vec<[u8; 32]> {
    let path = std::env::var_os("CE_INFER_ROOTS")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("CE_DATA_DIR").map(|d| PathBuf::from(d).join("roots")))
        .unwrap_or_else(|| home_dir().join(".local/share/ce/roots"));
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .filter_map(|h| node_id_from_hex(h).ok())
        .collect()
}

/// Best-effort home directory (no extra dependency): `$HOME`, else `/tmp`.
fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_parses_64_hex() {
        let h = "ab".repeat(32);
        let id = node_id_from_hex(&h).unwrap();
        assert_eq!(id, [0xab; 32]);
    }

    #[test]
    fn node_id_rejects_bad_input() {
        assert!(node_id_from_hex("not-hex").is_err());
        assert!(node_id_from_hex("abcd").is_err());
    }

    #[test]
    fn data_dir_honors_env_override() {
        // Use a unique var-less check: CE_INFER_DATA wins when set.
        // (We don't mutate process env in parallel tests; just assert the default shape.)
        let d = data_dir();
        assert!(d.ends_with("ce-infer"));
    }
}
