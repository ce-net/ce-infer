//! Targeted unit tests filling coverage gaps the in-module `#[cfg(test)]` suites leave: registry
//! role/fit/parse edge cases, probe helpers, serve seam corners, caps helpers, and billing
//! boundaries. These are deterministic example-based assertions (the broad invariants live in
//! `properties.rs`); the goal is every public fn has at least one happy-path and one failure-path
//! test, plus the boundary values the self-tier/selection rules pivot on.

use ce_infer_core::audit::Op;
use ce_infer_core::billing::{Meter, PriceSheet};
use ce_infer_core::caps::{self, CHAT, CODE, SUMMARIZE};
use ce_infer_core::probe::{CapabilityProfile, Gpu, GpuVendor, Tier};
use ce_infer_core::proto::InferReply;
use ce_infer_core::registry::{ModelEntry, Registry, Role};
use ce_infer_core::serve::{Decision, decision_outcome, decision_reply, op_or_chat};
use ce_rs::Amount;

// ---------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------

fn published() -> Registry {
    let mut r = Registry::builtin();
    for m in &mut r.models {
        m.gguf_object_cid = format!("cid-{}", m.id);
    }
    r
}

#[test]
fn from_toml_rejects_malformed_input() {
    assert!(Registry::from_toml("this is not toml ===").is_err());
    // Missing required fields (id) should fail to parse into a ModelEntry.
    assert!(Registry::from_toml("[[models]]\nquant = \"Q4\"\n").is_err());
}

#[test]
fn from_toml_defaults_version_when_absent() {
    let reg = Registry::from_toml("models = []\n").unwrap();
    assert_eq!(reg.version, 1, "version defaults to 1 when omitted");
}

#[test]
fn get_and_get_mut_round_trip_a_cid() {
    let mut r = Registry::builtin();
    assert!(r.get("clinical-chat-8b").unwrap().gguf_object_cid.is_empty());
    r.get_mut("clinical-chat-8b").unwrap().gguf_object_cid = "deadbeef".into();
    assert_eq!(r.get("clinical-chat-8b").unwrap().gguf_object_cid, "deadbeef");
    assert!(r.get("no-such-model").is_none());
    assert!(r.get_mut("no-such-model").is_none());
}

#[test]
fn available_for_role_filters_unpublished_and_maps_summarize_to_chat() {
    // Builtin (no CIDs) => nothing available.
    assert!(Registry::builtin().available_for_role(Role::Chat).is_empty());
    let r = published();
    // Chat role returns chat models (not code).
    let chat_ids: Vec<&str> = r.available_for_role(Role::Chat).iter().map(|m| m.id.as_str()).collect();
    assert!(chat_ids.contains(&"clinical-chat-8b"));
    assert!(!chat_ids.contains(&"code-7b"));
    // Summarize is served by chat models.
    let sum_ids: Vec<&str> = r.available_for_role(Role::Summarize).iter().map(|m| m.id.as_str()).collect();
    assert!(sum_ids.contains(&"clinical-chat-8b"));
    // Code role only returns code models.
    let code_ids: Vec<&str> = r.available_for_role(Role::Code).iter().map(|m| m.id.as_str()).collect();
    assert_eq!(code_ids, vec!["code-7b"]);
}

#[test]
fn role_ability_mapping_is_complete() {
    assert_eq!(Role::Chat.ability(), CHAT);
    assert_eq!(Role::Summarize.ability(), SUMMARIZE);
    assert_eq!(Role::Code.ability(), CODE);
    // Draft models ride the chat ability (pulled as part of a chat session).
    assert_eq!(Role::Draft.ability(), CHAT);
}

#[test]
fn fits_handles_gpu_only_and_cpu_only_models() {
    // A GPU-only model (ram_min_mb == 0) never fits a CPU tier.
    let gpu_only = ModelEntry {
        id: "gpu-only".into(),
        gguf_object_cid: "cid".into(),
        quant: "Q4".into(),
        ctx: 8192,
        ram_min_mb: 0,
        vram_min_mb: 16_000,
        role: Role::Chat,
        draft_model: None,
    };
    assert!(!gpu_only.fits(Tier::CpuHigh, 64_000, 0), "gpu-only must not fit a CPU tier");
    assert!(gpu_only.fits(Tier::GpuMid, 64_000, 16_000));
    assert!(!gpu_only.fits(Tier::GpuMid, 64_000, 15_999), "below the vram floor must not fit");

    // A CPU-runnable model with no vram floor can use a GPU box's system RAM.
    let cpu_model = ModelEntry { vram_min_mb: 0, ..gpu_only.clone() };
    let cpu_model = ModelEntry { ram_min_mb: 6_000, ..cpu_model };
    assert!(cpu_model.fits(Tier::GpuSmall, 8_000, 6_000), "cpu model runs on a GPU box's RAM");
    assert!(cpu_model.fits(Tier::CpuLow, 8_000, 0));
}

#[test]
fn select_for_is_deterministic_on_ties() {
    // Two chat models with the same floor — selection is tie-broken by id, deterministically.
    let mut r = Registry { version: 1, models: vec![] };
    for id in ["b-chat", "a-chat"] {
        r.models.push(ModelEntry {
            id: id.into(),
            gguf_object_cid: "cid".into(),
            quant: "Q4".into(),
            ctx: 8192,
            ram_min_mb: 6_000,
            vram_min_mb: 0,
            role: Role::Chat,
            draft_model: None,
        });
    }
    // Highest id wins on a floor tie ("b-chat" > "a-chat").
    let chosen = r.select_for(Tier::CpuLow, 8_000, 0).unwrap();
    assert_eq!(chosen.id, "b-chat");
    // Stable across calls.
    assert_eq!(r.select_for(Tier::CpuLow, 8_000, 0).unwrap().id, "b-chat");
}

#[test]
fn is_available_treats_whitespace_cid_as_unpublished() {
    let m = ModelEntry {
        id: "m".into(),
        gguf_object_cid: "   ".into(),
        quant: "Q4".into(),
        ctx: 8,
        ram_min_mb: 1,
        vram_min_mb: 0,
        role: Role::Chat,
        draft_model: None,
    };
    assert!(!m.is_available(), "a whitespace-only CID is not a published model");
}

#[test]
fn draft_model_field_round_trips_through_toml() {
    let mut r = published();
    r.get_mut("clinical-chat-8b").unwrap().draft_model = Some("draft-1b".into());
    let back = Registry::from_toml(&r.to_toml().unwrap()).unwrap();
    assert_eq!(back.get("clinical-chat-8b").unwrap().draft_model.as_deref(), Some("draft-1b"));
}

// ---------------------------------------------------------------------------
// probe
// ---------------------------------------------------------------------------

#[test]
fn tier_predicates_are_consistent() {
    assert!(Tier::GpuHeavy.is_gpu() && Tier::GpuHeavy.is_eligible());
    assert!(!Tier::CpuHigh.is_gpu() && Tier::CpuHigh.is_eligible());
    assert!(!Tier::Ineligible.is_eligible() && !Tier::Ineligible.is_gpu());
    // Ordering: weakest..strongest.
    assert!(Tier::Ineligible < Tier::CpuLow);
    assert!(Tier::CpuHigh < Tier::GpuSmall);
    assert!(Tier::GpuMid < Tier::GpuHeavy);
}

#[test]
fn vram_mb_helper_reports_zero_without_gpu() {
    let cpu = CapabilityProfile {
        os: "linux".into(), arch: "x86_64".into(), cores: 8, ram_mb: 16_000,
        gpu: None, tier: Tier::CpuMid, assigned_model: None,
    };
    assert_eq!(cpu.vram_mb(), 0);
    let gpu = CapabilityProfile {
        gpu: Some(Gpu { vendor: GpuVendor::Nvidia, vram_mb: 24_000 }), ..cpu.clone()
    };
    assert_eq!(gpu.vram_mb(), 24_000);
}

#[test]
fn gpu_vendor_strings_are_stable() {
    assert_eq!(GpuVendor::Nvidia.as_str(), "nvidia");
    assert_eq!(GpuVendor::Apple.as_str(), "apple");
    assert_eq!(GpuVendor::Amd.as_str(), "amd");
}

// ---------------------------------------------------------------------------
// serve seam
// ---------------------------------------------------------------------------

#[test]
fn op_or_chat_defaults_unknown_to_chat() {
    assert_eq!(op_or_chat("code"), Op::Code);
    assert_eq!(op_or_chat("SUMMARIZE"), Op::Summarize);
    assert_eq!(op_or_chat("garbage"), Op::Chat);
    assert_eq!(op_or_chat(""), Op::Chat);
}

#[test]
fn decision_reply_and_outcome_cover_every_arm() {
    use ce_infer_core::audit::Outcome;
    // Allow has no reply, outcome Ok.
    assert!(decision_reply(&Decision::Allow).is_none());
    assert_eq!(decision_outcome(&Decision::Allow), Outcome::Ok);
    // Deny -> denied reply + Denied outcome.
    let deny = Decision::Deny("nope".into());
    let r: InferReply = decision_reply(&deny).unwrap();
    assert!(!r.ok && r.finish_reason == "denied");
    assert_eq!(decision_outcome(&deny), Outcome::Denied);
    // WrongModel -> error reply + Error outcome.
    let wm = Decision::WrongModel { served: "a".into(), requested: "b".into() };
    let r: InferReply = decision_reply(&wm).unwrap();
    assert!(!r.ok && r.error.unwrap().contains("serves 'a'"));
    assert_eq!(decision_outcome(&wm), Outcome::Error);
}

#[test]
fn op_ability_round_trips_through_parse() {
    for op in [Op::Chat, Op::Summarize, Op::Code] {
        assert_eq!(Op::parse(op.as_str()).unwrap(), op);
    }
}

// ---------------------------------------------------------------------------
// caps helpers
// ---------------------------------------------------------------------------

#[test]
fn infer_resource_is_the_infer_tag() {
    use ce_cap::Resource;
    assert_eq!(caps::infer_resource(), Resource::Tag("infer".to_string()));
}

#[test]
fn expires_at_sets_only_not_after() {
    let c = caps::expires_at(12345);
    assert_eq!(c.not_after, 12345);
    assert_eq!(c.not_before, 0);
    assert!(c.max_credits.is_none());
}

#[test]
fn grant_abilities_without_prefix_has_no_prefix_ability() {
    let v = caps::grant_abilities(&[CHAT], None);
    assert_eq!(v, vec![CHAT.to_string()]);
    assert!(!v.iter().any(|a| a.starts_with(caps::MODEL_PREFIX)));
}

// ---------------------------------------------------------------------------
// billing boundaries
// ---------------------------------------------------------------------------

#[test]
fn meter_exhausted_boundary_is_inclusive() {
    let price = PriceSheet { per_token: Amount::ZERO, per_gb_second: Amount::ZERO, per_request: Amount::from_base(100) };
    let mut m = Meter::new(price, 1, Amount::from_base(100));
    assert!(!m.exhausted());
    m.charge_request(0, 0); // exactly hits 100 == capacity
    assert!(m.exhausted(), "cumulative == capacity counts as exhausted");
    assert_eq!(m.remaining().base(), 0);
}

#[test]
fn default_sheet_token_cost_is_nonzero_and_ordered() {
    let s = PriceSheet::default_sheet();
    // 1000 tokens cost strictly more than 1 token.
    assert!(s.cost(1000, 0, 0).base() > s.cost(1, 0, 0).base());
    // Each component contributes.
    assert!(s.cost(0, 0, 0).base() == s.per_request.base());
    assert!(s.cost(0, 8, 30).base() > s.per_request.base());
}
