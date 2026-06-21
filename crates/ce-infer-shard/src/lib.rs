//! # ce-infer-shard (v2, EXPERIMENTAL) — pipeline-parallel scaffold.
//!
//! **NOT wired into v1 routing.** This crate is the clearly-separated home for running models too
//! big for any single node by splitting the layer stack across nodes. It honors the research
//! constraint: **PIPELINE-parallel only, NEVER tensor-parallel over Ethernet** — only the boundary
//! activation tensor crosses the wire (~KB/token), never per-layer weight shards or all-reduces.
//!
//! The whole production surface is gated behind the `shard` feature (OFF by default). v1 ships
//! without it; `cargo build --features shard` compiles the experimental path. The types here are
//! interfaces + stubs with `// TODO` markers — there is no production execution path yet.
//!
//! Components:
//! - [`PlacementPlanner`] — reads `ce.atlas()` and assigns contiguous layer ranges proportional to
//!   advertised VRAM/RAM (the EXO memory-weighted ring), preferring high-history hosts; emits a
//!   signed [`PipelinePlan`] broadcast as an app message.
//! - [`ShardWorker`] — holds a layer range, pulls only its shard CIDs, runs that slice (llama.cpp
//!   `rpc-server` / `--rpc`), receives an activation tensor from the previous stage over a CE mesh
//!   STREAM addressed by node id, computes, forwards activations to the next stage.
//! - Rerouting — if a stage dies, re-request its layer range from another atlas peer holding the
//!   same shard CID.
//! - Capability — every stage is gated by [`SHARD_ABILITY`] (`infer:shard`).
//!
//! With the `shard` feature OFF (the default), this crate compiles to nothing — v1 carries no
//! sharding code path. The entire module below is `#[cfg(feature = "shard")]`.

#![cfg_attr(not(feature = "shard"), allow(unused))]

#[cfg(feature = "shard")]
mod inner {
use serde::{Deserialize, Serialize};

/// The capability ability a node must hold to participate as a pipeline stage.
pub const SHARD_ABILITY: &str = ce_infer_core::caps::SHARD;

/// The app pub/sub topic carrying signed pipeline plans.
pub const PLAN_TOPIC: &str = "infer/shard/plan/v1";

/// One contiguous layer range assigned to a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage {
    /// Hex node id of the stage host.
    pub node_id: String,
    /// First layer index (inclusive) this stage owns.
    pub layer_lo: u32,
    /// Last layer index (inclusive) this stage owns.
    pub layer_hi: u32,
    /// CE object CID of this stage's weight shard (only the layers in `[lo, hi]`).
    pub weight_shard_cid: String,
}

impl Stage {
    pub fn layer_count(&self) -> u32 {
        self.layer_hi.saturating_sub(self.layer_lo) + 1
    }
}

/// A signed pipeline plan: the full layer assignment for one model across an ordered ring of stages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelinePlan {
    pub model_id: String,
    /// Total layers in the model.
    pub total_layers: u32,
    /// Ordered stages (stage 0 holds the embedding/first layers; the last holds the head).
    pub stages: Vec<Stage>,
}

impl PipelinePlan {
    /// Validate the plan covers `[0, total_layers)` contiguously with no gaps or overlaps.
    pub fn is_contiguous(&self) -> bool {
        if self.stages.is_empty() {
            return false;
        }
        let mut expect = 0u32;
        for s in &self.stages {
            if s.layer_lo != expect {
                return false;
            }
            expect = s.layer_hi + 1;
        }
        expect == self.total_layers
    }
}

/// A node's advertised inference memory budget, distilled from the atlas for placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBudget {
    pub node_id: String,
    /// Memory weight used to size the layer range (VRAM for GPU hosts, else RAM), in MB.
    pub mem_mb: u64,
    /// Reputation (delivered work) — ties prefer higher-history hosts.
    pub reputation: u64,
}

/// Assigns contiguous layer ranges to hosts proportional to their memory budget (EXO
/// memory-weighted ring). Pure planning logic — no I/O — so it is unit-testable.
pub struct PlacementPlanner;

impl PlacementPlanner {
    /// Plan `total_layers` across `hosts`, giving each host a share of layers proportional to its
    /// `mem_mb`. Hosts are ordered by reputation (desc) then mem (desc) for a deterministic ring.
    /// Returns `None` if there are no hosts or zero total memory.
    ///
    /// NOTE (v2 STUB): this computes the *placement* only. Actual weight-shard CIDs are filled in by
    /// the publisher that slices the GGUF per range — left empty here. // TODO(shard): wire to a
    /// GGUF layer-slicer + `put_object` per shard.
    pub fn plan(model_id: &str, total_layers: u32, mut hosts: Vec<HostBudget>) -> Option<PipelinePlan> {
        if hosts.is_empty() || total_layers == 0 {
            return None;
        }
        let total_mem: u64 = hosts.iter().map(|h| h.mem_mb).sum();
        if total_mem == 0 {
            return None;
        }
        // Deterministic ring order.
        hosts.sort_by(|a, b| {
            b.reputation
                .cmp(&a.reputation)
                .then_with(|| b.mem_mb.cmp(&a.mem_mb))
                .then_with(|| a.node_id.cmp(&b.node_id))
        });
        let mut stages = Vec::with_capacity(hosts.len());
        let mut next_lo = 0u32;
        let n = hosts.len();
        for (i, h) in hosts.iter().enumerate() {
            let layers = if i + 1 == n {
                // Last host absorbs the remainder so coverage is exact.
                total_layers - next_lo
            } else {
                // Proportional share, at least 1 layer per host so every stage is real.
                let share = ((h.mem_mb as u128 * total_layers as u128) / total_mem as u128) as u32;
                share.max(1).min(total_layers.saturating_sub(next_lo).saturating_sub((n - i - 1) as u32))
            };
            if layers == 0 {
                continue;
            }
            let layer_hi = next_lo + layers - 1;
            stages.push(Stage {
                node_id: h.node_id.clone(),
                layer_lo: next_lo,
                layer_hi,
                weight_shard_cid: String::new(), // TODO(shard): fill from the GGUF slicer.
            });
            next_lo = layer_hi + 1;
        }
        let plan = PipelinePlan { model_id: model_id.to_string(), total_layers, stages };
        plan.is_contiguous().then_some(plan)
    }
}

/// A single pipeline stage worker (v2 STUB). Holds a layer range, exchanges boundary activations
/// with its neighbors over a CE mesh stream, and runs its slice via llama.cpp's RPC backend.
pub struct ShardWorker {
    pub stage: Stage,
    /// Hex node id of the next stage to forward activations to (`None` for the final stage).
    pub next: Option<String>,
}

impl ShardWorker {
    pub fn new(stage: Stage, next: Option<String>) -> Self {
        Self { stage, next }
    }

    /// Process one boundary activation tensor: run the local layer slice, returning the activations
    /// to send to [`Self::next`] (or the final logits on the last stage).
    ///
    /// STUB: the production path shells to llama.cpp `rpc-server` with `--rpc` and feeds the slice.
    /// Here it returns an error so callers can't mistake the scaffold for a working engine.
    // TODO(shard): launch `rpc-server`, load `weight_shard_cid`, run forward over [lo, hi].
    pub fn forward(&self, _activation: &[u8]) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("ce-infer-shard ShardWorker::forward is an unimplemented v2 scaffold")
    }
}

/// Rerouting helper (v2 STUB): pick a replacement host for a dead stage from atlas peers that hold
/// the same `weight_shard_cid`. Returns the first eligible candidate's node id.
// TODO(shard): consult ce.find_service("infer:shard:<cid>") and rank by history.
pub fn reroute_candidate(dead: &Stage, atlas_holders: &[String]) -> Option<String> {
    atlas_holders.iter().find(|id| *id != &dead.node_id).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(id: &str, mem: u64, rep: u64) -> HostBudget {
        HostBudget { node_id: id.into(), mem_mb: mem, reputation: rep }
    }

    #[test]
    fn plan_covers_all_layers_contiguously() {
        let hosts = vec![host("a", 24_000, 5), host("b", 12_000, 2), host("c", 12_000, 9)];
        let plan = PlacementPlanner::plan("clinical-34b", 48, hosts).expect("a plan");
        assert!(plan.is_contiguous());
        assert_eq!(plan.stages.iter().map(|s| s.layer_count()).sum::<u32>(), 48);
        // Highest reputation (c) leads the ring.
        assert_eq!(plan.stages[0].node_id, "c");
    }

    #[test]
    fn plan_is_proportional_to_memory() {
        // Two hosts, 3:1 memory => the big host gets ~3x the layers.
        let hosts = vec![host("big", 30_000, 0), host("small", 10_000, 0)];
        let plan = PlacementPlanner::plan("m", 40, hosts).unwrap();
        let big = plan.stages.iter().find(|s| s.node_id == "big").unwrap().layer_count();
        let small = plan.stages.iter().find(|s| s.node_id == "small").unwrap().layer_count();
        assert!(big > small, "big={big} small={small}");
        assert_eq!(big + small, 40);
    }

    #[test]
    fn plan_rejects_empty_inputs() {
        assert!(PlacementPlanner::plan("m", 0, vec![host("a", 1, 0)]).is_none());
        assert!(PlacementPlanner::plan("m", 10, vec![]).is_none());
    }

    #[test]
    fn non_contiguous_plan_is_detected() {
        let plan = PipelinePlan {
            model_id: "m".into(),
            total_layers: 10,
            stages: vec![
                Stage { node_id: "a".into(), layer_lo: 0, layer_hi: 3, weight_shard_cid: String::new() },
                // gap: 4 missing
                Stage { node_id: "b".into(), layer_lo: 5, layer_hi: 9, weight_shard_cid: String::new() },
            ],
        };
        assert!(!plan.is_contiguous());
    }

    #[test]
    fn forward_is_an_unimplemented_stub() {
        let w = ShardWorker::new(
            Stage { node_id: "a".into(), layer_lo: 0, layer_hi: 3, weight_shard_cid: String::new() },
            Some("b".into()),
        );
        assert!(w.forward(b"activation").is_err());
    }

    #[test]
    fn reroute_picks_another_holder() {
        let dead = Stage { node_id: "a".into(), layer_lo: 0, layer_hi: 3, weight_shard_cid: "cid".into() };
        let holders = vec!["a".to_string(), "b".to_string()];
        assert_eq!(reroute_candidate(&dead, &holders), Some("b".to_string()));
    }
}
} // mod inner

// Re-export the experimental surface at the crate root when the feature is on.
#[cfg(feature = "shard")]
pub use inner::*;
