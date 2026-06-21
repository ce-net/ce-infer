//! Hardware probe — detect what this node can run, and which inference tier it belongs to.
//!
//! [`probe`] inspects RAM, logical cores, and GPU (NVIDIA / Apple Metal / AMD ROCm) and emits a
//! [`CapabilityProfile`]. The [`Tier`] is assigned by a deterministic, documented rule (see
//! [`Tier::classify`]) so two nodes with the same hardware always self-classify identically — the
//! router relies on this. The profile is surfaced to the mesh as atlas self-tags (see
//! [`CapabilityProfile::atlas_tags`]) so the router reads it straight from `ce.atlas()`.

use serde::{Deserialize, Serialize};

/// GPU vendor classes the probe recognizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    Nvidia,
    Apple,
    Amd,
}

impl GpuVendor {
    pub fn as_str(self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Apple => "apple",
            GpuVendor::Amd => "amd",
        }
    }
}

/// A detected GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gpu {
    pub vendor: GpuVendor,
    /// Total VRAM in MB. For Apple unified memory this is an estimate of the GPU-addressable share.
    pub vram_mb: u64,
}

/// The inference tier a node self-classifies into. Ordered weakest -> strongest; the router prefers
/// GPU tiers for interactive chat and allows CPU tiers for async summarization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Tier {
    /// Cannot usefully run a quantized clinical model. Node still meshes, just not as a worker.
    Ineligible,
    CpuLow,
    CpuMid,
    CpuHigh,
    GpuSmall,
    GpuMid,
    GpuHeavy,
}

impl Tier {
    /// The canonical kebab-case tag string used in the atlas and `model:` self-tags.
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Ineligible => "ineligible",
            Tier::CpuLow => "cpu-low",
            Tier::CpuMid => "cpu-mid",
            Tier::CpuHigh => "cpu-high",
            Tier::GpuSmall => "gpu-small",
            Tier::GpuMid => "gpu-mid",
            Tier::GpuHeavy => "gpu-heavy",
        }
    }

    /// True for any GPU-backed tier.
    pub fn is_gpu(self) -> bool {
        matches!(self, Tier::GpuSmall | Tier::GpuMid | Tier::GpuHeavy)
    }

    /// True for any node that can act as a worker at all.
    pub fn is_eligible(self) -> bool {
        self != Tier::Ineligible
    }

    /// THE SELF-TIER RULE (deterministic, documented — keep this in sync with the module docs and
    /// the spec). GPU tiers are decided purely by VRAM; CPU tiers purely by system RAM.
    ///
    /// - `GpuHeavy`  if vram_mb >= 22000
    /// - `GpuMid`    if vram_mb in 10000..22000
    /// - `GpuSmall`  if vram_mb in 6000..10000
    /// - (a GPU with < 6000 MB VRAM is treated as CPU — too little to offload a useful model)
    /// - `CpuHigh`   if no usable gpu and ram_mb >= 24000
    /// - `CpuMid`    if ram_mb >= 12000
    /// - `CpuLow`    if ram_mb >= 8000
    /// - `Ineligible` below 8000
    pub fn classify(ram_mb: u64, gpu: Option<Gpu>) -> Tier {
        if let Some(g) = gpu {
            if g.vram_mb >= 22_000 {
                return Tier::GpuHeavy;
            }
            if g.vram_mb >= 10_000 {
                return Tier::GpuMid;
            }
            if g.vram_mb >= 6_000 {
                return Tier::GpuSmall;
            }
            // GPU too small to matter — fall through to CPU classification.
        }
        if ram_mb >= 24_000 {
            Tier::CpuHigh
        } else if ram_mb >= 12_000 {
            Tier::CpuMid
        } else if ram_mb >= 8_000 {
            Tier::CpuLow
        } else {
            Tier::Ineligible
        }
    }
}

/// A node's inference capability profile. Surfaced to the mesh via [`atlas_tags`](Self::atlas_tags).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityProfile {
    pub os: String,
    pub arch: String,
    pub cores: u32,
    pub ram_mb: u64,
    #[serde(default)]
    pub gpu: Option<Gpu>,
    pub tier: Tier,
    /// The model id the registry assigned for this tier, if any (filled by the worker; the bare
    /// probe leaves it `None`).
    #[serde(default)]
    pub assigned_model: Option<String>,
}

impl CapabilityProfile {
    /// VRAM in MB (0 if no GPU).
    pub fn vram_mb(&self) -> u64 {
        self.gpu.map(|g| g.vram_mb).unwrap_or(0)
    }

    /// The CE atlas self-tags this profile advertises. The router filters `ce.atlas()` on these:
    /// `infer`, `gpu`/`cpu`, the tier string, os, arch, and `model:<id>` once a model is assigned.
    pub fn atlas_tags(&self) -> Vec<String> {
        let mut tags = vec![
            "infer".to_string(),
            if self.tier.is_gpu() { "gpu".to_string() } else { "cpu".to_string() },
            self.tier.as_str().to_string(),
            self.os.clone(),
            self.arch.clone(),
        ];
        if let Some(m) = &self.assigned_model {
            tags.push(format!("model:{m}"));
        }
        tags
    }
}

/// Probe this machine and emit its [`CapabilityProfile`] (without an assigned model — the worker
/// fills that from the registry by tier). Never panics; on any detection failure it degrades to the
/// safe answer (no GPU, the RAM/cores it could read).
pub fn probe() -> CapabilityProfile {
    let (ram_mb, cores) = probe_ram_cores();
    let gpu = probe_gpu();
    let tier = Tier::classify(ram_mb, gpu);
    CapabilityProfile {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cores,
        ram_mb,
        gpu,
        tier,
        assigned_model: None,
    }
}

/// Total RAM (MB) and logical core count via sysinfo.
fn probe_ram_cores() -> (u64, u32) {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    // sysinfo reports bytes.
    let ram_mb = sys.total_memory() / (1024 * 1024);
    let cores = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1);
    (ram_mb, cores)
}

/// Detect a GPU, trying NVIDIA, then Apple Metal (macOS), then AMD ROCm. Returns the first hit.
fn probe_gpu() -> Option<Gpu> {
    if let Some(g) = probe_nvidia() {
        return Some(g);
    }
    #[cfg(target_os = "macos")]
    if let Some(g) = probe_apple() {
        return Some(g);
    }
    if let Some(g) = probe_amd() {
        return Some(g);
    }
    None
}

/// NVIDIA via `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` (MB). We take the
/// largest GPU's VRAM. No unwrap; any failure => not detected.
fn probe_nvidia() -> Option<Gpu> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let vram_mb = text
        .lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .max()?;
    if vram_mb == 0 {
        return None;
    }
    Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb })
}

/// Apple Metal: there is no dedicated VRAM, so we estimate the GPU-addressable share of unified
/// memory. macOS exposes total RAM via `sysctl hw.memsize` (bytes); Metal can address roughly
/// 70% of it for large models, so we use that as the effective VRAM figure.
#[cfg(target_os = "macos")]
fn probe_apple() -> Option<Gpu> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let bytes: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    let total_mb = bytes / (1024 * 1024);
    // Conservative GPU-addressable estimate for unified memory.
    let vram_mb = total_mb * 7 / 10;
    if vram_mb == 0 {
        return None;
    }
    Some(Gpu { vendor: GpuVendor::Apple, vram_mb })
}

/// AMD ROCm via `rocminfo` — we look for a `Pool` size line, best-effort. Many `rocminfo` builds
/// report pool sizes in KB; we parse the largest plausible VRAM figure. Any failure => not detected.
fn probe_amd() -> Option<Gpu> {
    let out = std::process::Command::new("rocminfo").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // Heuristic: scan "Size:" lines under GPU agents, pick the largest value, interpret as KB.
    let max_kb = text
        .lines()
        .filter(|l| l.contains("Size:"))
        .filter_map(|l| {
            l.split_whitespace()
                .find_map(|tok| tok.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<u64>().ok())
        })
        .max()?;
    let vram_mb = max_kb / 1024;
    if vram_mb < 1024 {
        return None;
    }
    Some(Gpu { vendor: GpuVendor::Amd, vram_mb })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Table-driven tier classification — the load-bearing deterministic rule.
    #[test]
    fn tier_classification_table() {
        let cases: &[(u64, Option<Gpu>, Tier)] = &[
            // GPU tiers by VRAM.
            (8_000, Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb: 24_000 }), Tier::GpuHeavy),
            (8_000, Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb: 22_000 }), Tier::GpuHeavy),
            (8_000, Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb: 21_999 }), Tier::GpuMid),
            (8_000, Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb: 16_000 }), Tier::GpuMid),
            (8_000, Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb: 10_000 }), Tier::GpuMid),
            (8_000, Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb: 9_999 }), Tier::GpuSmall),
            (8_000, Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb: 6_000 }), Tier::GpuSmall),
            // Tiny GPU falls through to CPU classification on RAM.
            (24_000, Some(Gpu { vendor: GpuVendor::Amd, vram_mb: 4_000 }), Tier::CpuHigh),
            (4_000, Some(Gpu { vendor: GpuVendor::Amd, vram_mb: 4_000 }), Tier::Ineligible),
            // CPU tiers by RAM.
            (64_000, None, Tier::CpuHigh),
            (24_000, None, Tier::CpuHigh),
            (23_999, None, Tier::CpuMid),
            (12_000, None, Tier::CpuMid),
            (11_999, None, Tier::CpuLow),
            (8_000, None, Tier::CpuLow),
            (7_999, None, Tier::Ineligible),
            (0, None, Tier::Ineligible),
        ];
        for (ram, gpu, want) in cases {
            let got = Tier::classify(*ram, *gpu);
            assert_eq!(got, *want, "classify(ram={ram}, gpu={gpu:?})");
        }
    }

    #[test]
    fn atlas_tags_include_infer_and_tier_and_model() {
        let p = CapabilityProfile {
            os: "linux".into(),
            arch: "x86_64".into(),
            cores: 16,
            ram_mb: 32_000,
            gpu: Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb: 24_000 }),
            tier: Tier::GpuHeavy,
            assigned_model: Some("clinical-chat-13b".into()),
        };
        let tags = p.atlas_tags();
        assert!(tags.contains(&"infer".to_string()));
        assert!(tags.contains(&"gpu".to_string()));
        assert!(tags.contains(&"gpu-heavy".to_string()));
        assert!(tags.contains(&"model:clinical-chat-13b".to_string()));
        assert!(!tags.contains(&"cpu".to_string()));
    }

    #[test]
    fn cpu_profile_tags_say_cpu() {
        let p = CapabilityProfile {
            os: "linux".into(),
            arch: "x86_64".into(),
            cores: 8,
            ram_mb: 16_000,
            gpu: None,
            tier: Tier::CpuMid,
            assigned_model: None,
        };
        let tags = p.atlas_tags();
        assert!(tags.contains(&"cpu".to_string()));
        assert!(!tags.iter().any(|t| t.starts_with("model:")));
    }

    #[test]
    fn probe_runs_and_self_classifies() {
        // Whatever this machine is, probe must not panic and must produce a consistent tier.
        let p = probe();
        assert_eq!(p.tier, Tier::classify(p.ram_mb, p.gpu));
        assert!(p.cores >= 1);
    }
}
