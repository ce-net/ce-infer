//! Capability abilities for ce-infer, and the `model_prefix` app caveat.
//!
//! Abilities are opaque ce-cap strings — the node assigns them no meaning; ce-infer defines and
//! enforces them. We reuse ce-cap's chain verifier (`authorize`) exactly like rdev/replicator:
//! a request is honored only if its presented chain roots at the worker's own key or a configured
//! org root, attenuates correctly at every link, and grants the op's ability.
//!
//! ## The `model_prefix` caveat
//!
//! ce-cap's [`Caveats`](ce_cap::Caveats) struct is fixed (it has no `model_prefix` field) and is
//! not app-extensible. So ce-infer expresses "this capability may only invoke `clinical-*` models"
//! as a structured **ability string** of the form `infer:model_prefix:<prefix>` carried in the
//! same `abilities` vector. This rides ce-cap's existing attenuation for free: a child link's
//! abilities must be a subset of its parent's, so a child can only *narrow* (add a longer, more
//! specific prefix the parent doesn't have would fail the subset check; keeping/dropping the
//! parent's prefix is allowed and re-checked here). After `authorize` succeeds, the worker calls
//! [`enforce_model_prefix`] against the **leaf** capability to confirm the requested model id
//! satisfies every `model_prefix` constraint the leaf carries.

use ce_cap::{Caveats, Resource, SignedCapability};

/// Submit a chat/completion inference request.
pub const CHAT: &str = "infer:chat";
/// Submit a summarization job.
pub const SUMMARIZE: &str = "infer:summarize";
/// Submit a coding-assistant request.
pub const CODE: &str = "infer:code";
/// Manage worker config / model assignment.
pub const ADMIN: &str = "infer:admin";
/// Participate as a pipeline stage (v2).
pub const SHARD: &str = "infer:shard";

/// Prefix of the structured `model_prefix` ability string.
pub const MODEL_PREFIX: &str = "infer:model_prefix:";

/// All concrete op abilities (not the `model_prefix:` family, not admin/shard).
pub const OP_ABILITIES: &[&str] = &[CHAT, SUMMARIZE, CODE];

/// Build the structured ability string restricting a capability to model ids under `prefix`
/// (e.g. `model_prefix_ability("clinical-")` => `"infer:model_prefix:clinical-"`).
pub fn model_prefix_ability(prefix: &str) -> String {
    format!("{MODEL_PREFIX}{prefix}")
}

/// Extract every model-prefix constraint carried by a capability's abilities.
pub fn model_prefixes(abilities: &[String]) -> Vec<String> {
    abilities
        .iter()
        .filter_map(|a| a.strip_prefix(MODEL_PREFIX).map(|p| p.to_string()))
        .collect()
}

/// Enforce the `model_prefix` caveat against a leaf capability: if the leaf carries any
/// `infer:model_prefix:<p>` ability, then `model_id` must start with at least one such `<p>`.
/// A leaf with no model-prefix constraint imposes no restriction (returns `Ok`).
///
/// Returns `Err(reason)` (safe to surface/audit) when the requested model is out of prefix.
pub fn enforce_model_prefix(leaf: &SignedCapability, model_id: &str) -> Result<(), String> {
    let prefixes = model_prefixes(&leaf.cap.abilities);
    if prefixes.is_empty() {
        return Ok(());
    }
    if prefixes.iter().any(|p| model_id.starts_with(p.as_str())) {
        Ok(())
    } else {
        Err(format!(
            "model '{model_id}' is outside the capability's allowed prefixes {prefixes:?}"
        ))
    }
}

/// Convenience: the abilities vector for a clinician/router grant — the requested op abilities plus
/// an optional `model_prefix` restriction. Used by the `ce-infer grant` helper.
pub fn grant_abilities(ops: &[&str], model_prefix: Option<&str>) -> Vec<String> {
    let mut v: Vec<String> = ops.iter().map(|s| s.to_string()).collect();
    if let Some(p) = model_prefix {
        v.push(model_prefix_ability(p));
    }
    v
}

/// The default resource for an infer grant: any node advertising the `infer` self-tag. (Callers may
/// narrow to a specific node or tier tag.)
pub fn infer_resource() -> Resource {
    Resource::Tag("infer".to_string())
}

/// Caveats with just an expiry (the common case for a clinician grant).
pub fn expires_at(unix_secs: u64) -> Caveats {
    Caveats { not_after: unix_secs, ..Default::default() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ce_cap::{Resource, SignedCapability};
    use ce_identity::Identity;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn id(tag: &str) -> Identity {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("ce-infer-caps-{}-{n}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Identity::load_or_generate(&dir).unwrap()
    }

    fn leaf_with(abilities: Vec<String>) -> SignedCapability {
        let issuer = id("issuer");
        let aud = id("aud");
        SignedCapability::issue(
            &issuer,
            aud.node_id(),
            abilities,
            Resource::Any,
            Caveats::default(),
            1,
            None,
        )
    }

    #[test]
    fn no_prefix_means_unrestricted() {
        let leaf = leaf_with(vec![CHAT.to_string()]);
        assert!(enforce_model_prefix(&leaf, "anything-goes").is_ok());
    }

    #[test]
    fn in_prefix_model_is_allowed() {
        let leaf = leaf_with(vec![CHAT.to_string(), model_prefix_ability("clinical-")]);
        assert!(enforce_model_prefix(&leaf, "clinical-chat-8b").is_ok());
    }

    #[test]
    fn out_of_prefix_model_is_rejected() {
        let leaf = leaf_with(vec![CHAT.to_string(), model_prefix_ability("clinical-")]);
        let err = enforce_model_prefix(&leaf, "code-7b").unwrap_err();
        assert!(err.contains("outside the capability's allowed prefixes"));
    }

    #[test]
    fn multiple_prefixes_any_match_allows() {
        let leaf = leaf_with(vec![
            CHAT.to_string(),
            model_prefix_ability("clinical-"),
            model_prefix_ability("code-"),
        ]);
        assert!(enforce_model_prefix(&leaf, "code-7b").is_ok());
        assert!(enforce_model_prefix(&leaf, "clinical-chat-8b").is_ok());
        assert!(enforce_model_prefix(&leaf, "other-9b").is_err());
    }

    #[test]
    fn grant_abilities_builds_expected_vector() {
        let v = grant_abilities(&[CHAT, SUMMARIZE], Some("clinical-"));
        assert!(v.contains(&CHAT.to_string()));
        assert!(v.contains(&SUMMARIZE.to_string()));
        assert!(v.contains(&"infer:model_prefix:clinical-".to_string()));
    }
}
