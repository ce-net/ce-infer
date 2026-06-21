//! Model registry — the signed TOML manifest mapping logical model ids to quantized GGUF objects.
//!
//! The registry is content-addressed and distributed as a CE blob (`put_object` -> CID). It maps a
//! logical model id (e.g. `clinical-chat-8b`) to the GGUF object CID, quantization, context window,
//! and the resource floor (`ram_min_mb` / `vram_min_mb`) needed to run it. The probe uses the
//! registry to pick the **largest** model whose floor fits the node's [`Tier`], so a small CPU node
//! runs the 8B and a GPU node runs the 13B — same registry, tier-driven selection.
//!
//! Weights are NOT hardcoded: the default registry ships logical ids + roles + floors, and an
//! operator publishes the actual GGUF bytes with `ce-infer models publish`, which fills in the CID.
//! Until a CID is published a model is "declared but unavailable" (CID empty) — selection skips it.

use crate::Tier;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// What a model is for. Drives router op->model resolution and tier defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Interactive chat (also serves summarize unless a dedicated summarizer is configured).
    Chat,
    /// Summarization of clinical records.
    Summarize,
    /// Coding-assistant.
    Code,
    /// A small draft model for speculative decoding (v2; advisory in v1).
    Draft,
}

impl Role {
    /// The ce-infer ability string an op of this role maps to (see [`crate::caps`]).
    pub fn ability(self) -> &'static str {
        match self {
            Role::Chat => crate::caps::CHAT,
            Role::Summarize => crate::caps::SUMMARIZE,
            Role::Code => crate::caps::CODE,
            // Draft models are pulled as part of a chat session; gated by the chat ability.
            Role::Draft => crate::caps::CHAT,
        }
    }
}

/// One model entry. `gguf_object_cid` empty => declared but not yet published locally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Logical model id (the OpenAI `model` field clients send).
    pub id: String,
    /// CE object CID (manifest hash) of the GGUF blob. Empty until published.
    #[serde(default)]
    pub gguf_object_cid: String,
    /// Quantization label (e.g. `Q4_K_M`), informational.
    pub quant: String,
    /// Context window the worker launches llama-server with (`--ctx-size`).
    pub ctx: u32,
    /// Minimum system RAM (MB) to run this model on CPU.
    pub ram_min_mb: u64,
    /// Minimum GPU VRAM (MB) to run this model on GPU. `0` => CPU-runnable.
    #[serde(default)]
    pub vram_min_mb: u64,
    /// Primary role.
    pub role: Role,
    /// Optional draft model id for speculative decoding (v2).
    #[serde(default)]
    pub draft_model: Option<String>,
}

impl ModelEntry {
    /// Has a GGUF been published for this id (CID present)?
    pub fn is_available(&self) -> bool {
        !self.gguf_object_cid.trim().is_empty()
    }

    /// Approximate model residency in **whole GB**, used as the unit for per-GB-second billing of a
    /// session that pins this model. Derived from the resource floor (`ram_min_mb`, the proxy for a
    /// quantized GGUF's working size); at least 1 GB so a held model always meters something.
    pub fn model_gb(&self) -> u64 {
        (self.ram_min_mb / 1024).max(1)
    }

    /// Does this model fit a node of the given tier? A model may declare both a GPU floor
    /// (`vram_min_mb`) and a CPU floor (`ram_min_mb`) — e.g. the 13B runs on a GpuMid OR a CpuHigh.
    /// On a GPU tier we require enough VRAM; on a CPU tier we require enough RAM. A GPU-only model
    /// (`ram_min_mb == 0`) is never CPU-runnable. Selection only considers available (published)
    /// models.
    pub fn fits(&self, tier: Tier, ram_mb: u64, vram_mb: u64) -> bool {
        if tier.is_gpu() {
            // On a GPU node, a model with a VRAM floor uses it; a CPU-only model (no VRAM floor)
            // can still run on the GPU box's system RAM.
            if self.vram_min_mb > 0 {
                vram_mb >= self.vram_min_mb
            } else {
                ram_mb >= self.ram_min_mb
            }
        } else {
            // CPU tier: needs a CPU floor (ram_min_mb > 0) and enough RAM. GPU-only models don't fit.
            self.ram_min_mb > 0 && ram_mb >= self.ram_min_mb
        }
    }
}

/// The full registry — a versioned set of models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Registry {
    /// Manifest schema version.
    #[serde(default = "default_version")]
    pub version: u32,
    pub models: Vec<ModelEntry>,
}

fn default_version() -> u32 {
    1
}

impl Default for Registry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl Registry {
    /// Parse a `models.toml` manifest.
    pub fn from_toml(s: &str) -> Result<Registry> {
        toml::from_str(s).map_err(|e| anyhow!("invalid models.toml: {e}"))
    }

    /// Serialize to TOML (for `models publish` / distribution as a blob).
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| anyhow!("encode models.toml: {e}"))
    }

    /// Look up a model by its logical id.
    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Mutable lookup (used by `models publish` to fill in a CID).
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ModelEntry> {
        self.models.iter_mut().find(|m| m.id == id)
    }

    /// Choose the **primary assigned model** for a node of `tier` with `ram_mb`/`vram_mb`. A node's
    /// primary model is a conversational one (role Chat — it also serves summarize), so coding
    /// models are not picked as the default assignment even when they fit. Among the eligible Chat
    /// models we pick the **largest** that fits (largest = highest resource floor it satisfies),
    /// tie-broken by id for determinism. Returns `None` for an [`Tier::Ineligible`] node or when
    /// nothing chat-capable is published+fits.
    pub fn select_for(&self, tier: Tier, ram_mb: u64, vram_mb: u64) -> Option<&ModelEntry> {
        if tier == Tier::Ineligible {
            return None;
        }
        self.models
            .iter()
            .filter(|m| m.role == Role::Chat && m.is_available() && m.fits(tier, ram_mb, vram_mb))
            // "Largest that fits": rank by the binding floor (vram for GPU models, ram otherwise),
            // then by id so the choice is deterministic on ties.
            .max_by(|a, b| {
                let fa = if a.vram_min_mb > 0 { a.vram_min_mb } else { a.ram_min_mb };
                let fb = if b.vram_min_mb > 0 { b.vram_min_mb } else { b.ram_min_mb };
                fa.cmp(&fb).then_with(|| a.id.cmp(&b.id))
            })
    }

    /// All available models satisfying `role` (router op->model resolution for logical aliases).
    pub fn available_for_role(&self, role: Role) -> Vec<&ModelEntry> {
        self.models
            .iter()
            .filter(|m| m.is_available() && (m.role == role || role_serves(m.role, role)))
            .collect()
    }

    /// The built-in default registry (admin-configurable; CIDs empty until weights are published).
    /// Default clinical models per the spec — NOT bundled weights, only declarations + floors.
    pub fn builtin() -> Registry {
        Registry {
            version: 1,
            models: vec![
                ModelEntry {
                    id: "clinical-chat-8b".into(),
                    gguf_object_cid: String::new(),
                    quant: "Q4_K_M".into(),
                    ctx: 8192,
                    // ~4.5 GB Q4_K_M; fits CpuLow (8 GB) and up.
                    ram_min_mb: 6_000,
                    vram_min_mb: 0,
                    role: Role::Chat,
                    draft_model: None,
                },
                ModelEntry {
                    id: "clinical-chat-13b".into(),
                    gguf_object_cid: String::new(),
                    quant: "Q4_K_M".into(),
                    ctx: 8192,
                    // ~7.8 GB Q4_K_M; CpuHigh (>=24 GB RAM) or a GpuMid (>=10 GB VRAM).
                    ram_min_mb: 12_000,
                    vram_min_mb: 10_000,
                    role: Role::Chat,
                    draft_model: None,
                },
                ModelEntry {
                    id: "code-7b".into(),
                    gguf_object_cid: String::new(),
                    quant: "Q4_K_M".into(),
                    ctx: 16384,
                    ram_min_mb: 6_000,
                    vram_min_mb: 0,
                    role: Role::Code,
                    draft_model: None,
                },
                ModelEntry {
                    // v2 sharding target — only a GpuHeavy node runs it whole.
                    id: "clinical-34b".into(),
                    gguf_object_cid: String::new(),
                    quant: "Q4_K_M".into(),
                    ctx: 8192,
                    ram_min_mb: 40_000,
                    vram_min_mb: 22_000,
                    role: Role::Chat,
                    draft_model: None,
                },
            ],
        }
    }
}

/// A chat model also serves summarize requests (no dedicated summarizer required).
fn role_serves(model_role: Role, requested: Role) -> bool {
    matches!((model_role, requested), (Role::Chat, Role::Summarize))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A registry whose models are all "published" (CID set) for selection tests.
    fn published() -> Registry {
        let mut r = Registry::builtin();
        for m in &mut r.models {
            m.gguf_object_cid = format!("cid-{}", m.id);
        }
        r
    }

    #[test]
    fn unpublished_models_are_never_selected() {
        // The builtin registry has empty CIDs => nothing is available => no selection.
        let r = Registry::builtin();
        assert!(r.select_for(Tier::GpuHeavy, 64_000, 24_000).is_none());
    }

    #[test]
    fn cpu_low_node_gets_the_8b() {
        let r = published();
        let m = r.select_for(Tier::CpuLow, 8_000, 0).expect("a model fits CpuLow");
        assert_eq!(m.id, "clinical-chat-8b");
    }

    #[test]
    fn cpu_high_node_prefers_the_13b() {
        let r = published();
        // 24 GB RAM, no GPU: the 13B (ram_min 12_000) is the largest CPU-runnable that fits.
        let m = r.select_for(Tier::CpuHigh, 24_000, 0).expect("a model fits CpuHigh");
        assert_eq!(m.id, "clinical-chat-13b");
    }

    #[test]
    fn gpu_mid_runs_the_13b_not_the_34b() {
        let r = published();
        // 12 GB VRAM: 13B (vram_min 10_000) fits, 34B (vram_min 22_000) does not.
        let m = r.select_for(Tier::GpuMid, 16_000, 12_000).expect("a model fits GpuMid");
        assert_eq!(m.id, "clinical-chat-13b");
    }

    #[test]
    fn gpu_heavy_runs_the_34b() {
        let r = published();
        let m = r.select_for(Tier::GpuHeavy, 64_000, 24_000).expect("a model fits GpuHeavy");
        assert_eq!(m.id, "clinical-34b");
    }

    #[test]
    fn ineligible_node_gets_nothing() {
        let r = published();
        assert!(r.select_for(Tier::Ineligible, 4_000, 0).is_none());
    }

    #[test]
    fn toml_round_trips() {
        let r = published();
        let s = r.to_toml().unwrap();
        let back = Registry::from_toml(&s).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn chat_model_serves_summarize() {
        let r = published();
        let v = r.available_for_role(Role::Summarize);
        assert!(v.iter().any(|m| m.id == "clinical-chat-8b"));
    }
}
