//! Audit — tamper-evident, append-only inference records that satisfy HIPAA §164.312(b)/(c).
//!
//! Every inference event, **after authorization** (allowed OR denied), is recorded as a CE on-chain
//! interaction so it lands in `/history` (the hash-chained, append-only substrate) AND emitted as a
//! signed app message on the dedicated audit topic [`TOPIC`]. The record carries only metadata and a
//! caller-supplied SHA256 `record_ref` of the PHI record — **never raw PHI**, never prompt/response
//! text. [`AuditRecord::assert_redacted`] is the guard: an audit payload that smuggles prompt or
//! response text is rejected before it can be written.
//!
//! Economic + audit duality: the per-(payer,worker) payment-channel receipt is itself an audit
//! record (who paid whom, how much, when); this app message is the structured, queryable companion.

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The pub/sub topic carrying signed audit records. The node signs published messages, so every
/// subscriber can verify authorship — the record is non-repudiable.
pub const TOPIC: &str = "infer/audit/v1";

/// The operation an audit record describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    Chat,
    Summarize,
    Code,
}

impl Op {
    /// The ce-infer ability string this op authorizes against (see [`crate::caps`]).
    pub fn ability(self) -> &'static str {
        match self {
            Op::Chat => crate::caps::CHAT,
            Op::Summarize => crate::caps::SUMMARIZE,
            Op::Code => crate::caps::CODE,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Op::Chat => "chat",
            Op::Summarize => "summarize",
            Op::Code => "code",
        }
    }

    /// Parse from an `X-CE-Op` header / wire string.
    pub fn parse(s: &str) -> Result<Op> {
        match s.trim().to_ascii_lowercase().as_str() {
            "chat" => Ok(Op::Chat),
            "summarize" => Ok(Op::Summarize),
            "code" => Ok(Op::Code),
            other => Err(anyhow!("unknown op '{other}'")),
        }
    }
}

/// Whether the authorized request executed or was denied. A denied attempt is STILL audited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Ok,
    Denied,
    Error,
}

/// A single audit record. Contains ONLY the PHI record's hash (`record_ref`) — never the record,
/// the prompt, or the response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditRecord {
    /// Unix seconds when the worker recorded the event.
    pub ts: u64,
    /// Hex node id of the principal (clinician/router identity) who made the request.
    pub principal_node_id: String,
    /// Hex node id of the worker that served (or refused) it.
    pub worker_node_id: String,
    /// Logical model id plus the registry version it came from (`<id>@v<version>`).
    pub model_id: String,
    /// Hash of the presented capability chain — names *which authority* permitted this, without
    /// embedding the chain. (`sha256(encode_chain_bytes(chain))`, hex.)
    pub capability_id: String,
    /// Caller-supplied SHA256 (hex) of the PHI record this inference touched. NEVER the PHI itself.
    pub record_ref: String,
    /// The operation.
    pub op: Op,
    /// Tokens produced (0 for a denied request).
    pub token_count: u64,
    /// Allowed / denied / error.
    pub outcome: Outcome,
}

impl AuditRecord {
    /// `sha256` of a capability chain's wire bytes, hex — used for [`capability_id`](Self::capability_id).
    pub fn capability_id_of(chain_bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(chain_bytes);
        hex::encode(h.finalize())
    }

    /// Encode to the canonical JSON bytes published on [`TOPIC`].
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| anyhow!("encode audit record: {e}"))
    }

    /// Decode from the wire (for `audit export` / OCR review).
    pub fn from_bytes(b: &[u8]) -> Result<AuditRecord> {
        serde_json::from_slice(b).map_err(|e| anyhow!("decode audit record: {e}"))
    }

    /// REDACTION GUARD. The audit writer calls this before publishing. It enforces the HIPAA
    /// minimum-necessary invariant structurally:
    ///
    /// - `record_ref` must be a 64-hex SHA256 (a hash, not free text — so PHI can't ride in it);
    /// - the serialized record must contain no field that could carry prompt/response text.
    ///
    /// Because the struct has no prompt/response/messages/text field at all, the second check is a
    /// belt-and-suspenders scan of the serialized form for those key names. Any violation => error,
    /// and the record is not written.
    pub fn assert_redacted(&self) -> Result<()> {
        // record_ref must be a hash: exactly 64 lowercase/uppercase hex chars.
        let r = self.record_ref.trim();
        if r.len() != 64 || !r.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("record_ref is not a SHA256 hash (must be 64 hex chars) — possible raw PHI");
        }
        // Defensive: assert the serialized record has no PHI-bearing field names.
        let json = serde_json::to_string(self).map_err(|e| anyhow!("redaction check encode: {e}"))?;
        for banned in ["\"prompt\"", "\"response\"", "\"messages\"", "\"content\"", "\"text\""] {
            if json.contains(banned) {
                bail!("audit record carries a forbidden field {banned} — refusing to write PHI");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(record_ref: &str) -> AuditRecord {
        AuditRecord {
            ts: 1_700_000_000,
            principal_node_id: "aa".repeat(32),
            worker_node_id: "bb".repeat(32),
            model_id: "clinical-chat-8b@v1".into(),
            capability_id: "cc".repeat(32),
            record_ref: record_ref.to_string(),
            op: Op::Chat,
            token_count: 42,
            outcome: Outcome::Ok,
        }
    }

    #[test]
    fn valid_record_passes_redaction_and_round_trips() {
        let rec = sample(&"d".repeat(64));
        rec.assert_redacted().expect("a hash-only record is allowed");
        let back = AuditRecord::from_bytes(&rec.to_bytes().unwrap()).unwrap();
        assert_eq!(back, rec);
    }

    #[test]
    fn non_hash_record_ref_is_rejected_as_possible_phi() {
        // A free-text "record_ref" (e.g. an actual note) must be rejected.
        let rec = sample("Patient John Doe, MRN 12345, presents with chest pain");
        let err = rec.assert_redacted().unwrap_err().to_string();
        assert!(err.contains("not a SHA256"), "got: {err}");
    }

    #[test]
    fn short_or_nonhex_record_ref_is_rejected() {
        assert!(sample("deadbeef").assert_redacted().is_err());
        assert!(sample(&"z".repeat(64)).assert_redacted().is_err());
    }

    #[test]
    fn denied_attempt_is_still_a_valid_audit_record() {
        let mut rec = sample(&"e".repeat(64));
        rec.outcome = Outcome::Denied;
        rec.token_count = 0;
        rec.assert_redacted().expect("denied attempts are auditable");
    }

    #[test]
    fn capability_id_is_a_hash_of_the_chain() {
        let id = AuditRecord::capability_id_of(b"some-chain-bytes");
        assert_eq!(id.len(), 64);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
