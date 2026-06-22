//! The local inference backend the worker forwards requests to.
//!
//! ce-infer does NOT implement an inference engine. In production the worker shells out to
//! `llama-server` (llama.cpp) bound to **loopback only** and forwards requests to its
//! OpenAI-compatible `/v1/chat/completions`. For CI and for nodes without a published GGUF, a
//! deterministic [`Backend::Mock`] makes the whole routing/audit/streaming path testable end to end
//! without a real model.
//!
//! The trait surface is intentionally tiny — `complete` returns the full text + token count, and
//! `tokens` yields a deterministic token split so the worker can stream deltas uniformly regardless
//! of backend.

use anyhow::{Result, anyhow};
use ce_infer_core::proto::InferRequest;
use std::process::Child;
use std::time::Duration;

/// A completion produced by a backend.
#[derive(Debug, Clone)]
pub struct Completion {
    pub text: String,
    pub token_count: u64,
    /// `stop` | `length`.
    pub finish_reason: String,
}

impl Completion {
    /// Split the completion text into whitespace-preserving "tokens" for streaming. Real
    /// llama-server streaming would yield the model's own token deltas; here we approximate by
    /// word with trailing space, which is deterministic and good enough to drive the SSE relay.
    pub fn tokens(&self) -> Vec<String> {
        if self.text.is_empty() {
            return Vec::new();
        }
        // Keep spaces attached so concatenation reproduces the text exactly.
        let mut out = Vec::new();
        let mut cur = String::new();
        for ch in self.text.chars() {
            cur.push(ch);
            if ch == ' ' {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }
}

/// The backend the worker uses for inference.
pub enum Backend {
    /// A live `llama-server` child reachable at `http://127.0.0.1:<port>`.
    Llama(LlamaServer),
    /// A deterministic mock — no model required. Used in CI and as a graceful fallback.
    Mock,
}

impl Backend {
    /// Run an inference request and collect the full completion.
    pub async fn complete(&self, req: &InferRequest) -> Result<Completion> {
        match self {
            Backend::Llama(srv) => srv.complete(req).await,
            Backend::Mock => Ok(mock_complete(req)),
        }
    }

    /// A short human label for logs/tags.
    pub fn label(&self) -> &'static str {
        match self {
            Backend::Llama(_) => "llama-server",
            Backend::Mock => "mock",
        }
    }
}

/// A deterministic completion for the mock backend: echoes a canned answer that references the model
/// and the last user message length, so tests can assert on a stable, content-free string. NEVER
/// echoes PHI verbatim into the audit log — the audit log only records the `record_ref` hash; the
/// completion text itself stays on the LAN between router and worker.
pub fn mock_complete(req: &InferRequest) -> Completion {
    let last_user_len = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.chars().count())
        .unwrap_or(0);
    let text = format!(
        "[mock {model} op={op}] received {n} chars; this is a deterministic test completion.",
        model = req.model_id,
        op = req.op.as_str(),
        n = last_user_len
    );
    let token_count = text.split_whitespace().count() as u64;
    let cap = req.max_tokens.unwrap_or(u32::MAX) as u64;
    let (text, finish) = if token_count > cap {
        // Truncate to the token cap to model a `length` finish.
        let truncated: String =
            text.split_whitespace().take(cap as usize).collect::<Vec<_>>().join(" ");
        (truncated, "length")
    } else {
        (text, "stop")
    };
    Completion { token_count: text.split_whitespace().count() as u64, text, finish_reason: finish.into() }
}

/// A handle to a running `llama-server` child bound to loopback. Dropping it kills the child.
pub struct LlamaServer {
    child: Child,
    port: u16,
    http: reqwest::Client,
}

impl LlamaServer {
    /// Spawn `llama-server` for `model_path`, bound to `127.0.0.1:<port>` ONLY (never 0.0.0.0),
    /// with GPU offload when `gpu` is true. `parallel` sets continuous-batching slots. Returns the
    /// handle once the child is launched; readiness is probed lazily on first request.
    ///
    /// `bin` is the engine binary path (the installer bundles it per platform; default
    /// `llama-server` on `$PATH`).
    pub fn spawn(
        bin: &str,
        model_path: &std::path::Path,
        port: u16,
        ctx: u32,
        parallel: u32,
        gpu: bool,
    ) -> Result<LlamaServer> {
        let ngl = if gpu { "99" } else { "0" };
        let child = std::process::Command::new(bin)
            .arg("--model")
            .arg(model_path)
            .arg("--ctx-size")
            .arg(ctx.to_string())
            .arg("--parallel")
            .arg(parallel.to_string())
            .arg("-ngl")
            .arg(ngl)
            // Loopback only — the model is NEVER exposed to the LAN. All access is mesh-mediated
            // through the worker, which is the capability-enforcement point.
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| anyhow!("failed to launch '{bin}': {e} (is the engine installed?)"))?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| anyhow!("build http client: {e}"))?;
        Ok(LlamaServer { child, port, http })
    }

    /// Connect to an already-running llama-server on `127.0.0.1:<port>` without owning its
    /// lifecycle. Used in tests with a stub server, and when an operator runs the engine externally.
    #[allow(dead_code)] // operator/test entry point; not used on the default startup path.
    pub fn attach(port: u16) -> Result<LlamaServer> {
        // A dummy child that exits immediately so Drop is a no-op-ish kill of nothing meaningful.
        // On Windows a bare `cmd` opens an interactive shell that never exits, so we always pass
        // explicit "exit now" arguments per platform rather than spawning a bare binary.
        let (bin, args) = noop_child_cmd();
        let child = std::process::Command::new(bin)
            .args(args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| anyhow!("attach placeholder child: {e}"))?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| anyhow!("build http client: {e}"))?;
        Ok(LlamaServer { child, port, http })
    }

    fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Forward a request to the loopback OpenAI endpoint and collect the completion.
    async fn complete(&self, req: &InferRequest) -> Result<Completion> {
        let messages: Vec<serde_json::Value> = req
            .messages
            .iter()
            .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
            .collect();
        let mut body = serde_json::json!({
            "model": req.model_id,
            "messages": messages,
            "stream": false,
        });
        if let Some(mt) = req.max_tokens {
            body["max_tokens"] = serde_json::json!(mt);
        }
        let resp = self
            .http
            .post(format!("{}/v1/chat/completions", self.base()))
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("llama-server request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(anyhow!("llama-server returned {}", resp.status()));
        }
        let v: serde_json::Value = resp.json().await.map_err(|e| anyhow!("decode completion: {e}"))?;
        let text = v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string();
        let finish = v["choices"][0]["finish_reason"].as_str().unwrap_or("stop").to_string();
        let token_count = v["usage"]["completion_tokens"]
            .as_u64()
            .unwrap_or_else(|| text.split_whitespace().count() as u64);
        Ok(Completion { text, token_count, finish_reason: finish })
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        // Own the child's lifecycle: kill on drop so we never leak a model server.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A platform command + args that spawns a child which exits immediately, used by `attach` for the
/// placeholder child handle. On Windows `cmd /c exit` returns at once; a bare `cmd` would open an
/// interactive shell that never exits and would hang `Drop`. On unix `true` exits with success.
#[allow(dead_code)] // paired with `attach`.
fn noop_child_cmd() -> (&'static str, &'static [&'static str]) {
    if cfg!(windows) { ("cmd", &["/C", "exit"]) } else { ("true", &[]) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ce_infer_core::audit::Op;
    use ce_infer_core::proto::{ChatMessage, InferRequest};

    fn req(max: Option<u32>) -> InferRequest {
        InferRequest {
            req_id: "r".into(),
            op: Op::Chat,
            model_id: "clinical-chat-8b".into(),
            messages: vec![ChatMessage { role: "user".into(), content: "hello there".into() }],
            max_tokens: max,
            stream: false,
            caps: String::new(),
            record_ref: "a".repeat(64),
            receipt: None,
        }
    }

    #[test]
    fn mock_is_deterministic_and_phi_free() {
        let a = mock_complete(&req(None));
        let b = mock_complete(&req(None));
        assert_eq!(a.text, b.text);
        // The patient's words ("hello there") must not appear verbatim in the completion.
        assert!(!a.text.contains("hello there"));
        assert!(a.token_count > 0);
        assert_eq!(a.finish_reason, "stop");
    }

    #[test]
    fn mock_respects_max_tokens() {
        let c = mock_complete(&req(Some(3)));
        assert!(c.token_count <= 3);
        assert_eq!(c.finish_reason, "length");
    }

    #[test]
    fn tokens_reconstruct_text() {
        let c = mock_complete(&req(None));
        let joined: String = c.tokens().concat();
        assert_eq!(joined, c.text);
    }

    // ---- end-to-end: a stub llama-server (canned OpenAI completion) + LlamaServer::attach ----

    /// Spawn a tiny axum server that answers `/v1/chat/completions` with a canned completion, on a
    /// free loopback port. Returns the port.
    async fn spawn_stub_llama() -> u16 {
        use axum::{Json, Router, routing::post};

        async fn chat() -> Json<serde_json::Value> {
            Json(serde_json::json!({
                "choices": [{
                    "message": { "role": "assistant", "content": "canned reply from stub" },
                    "finish_reason": "stop",
                }],
                "usage": { "completion_tokens": 4 },
            }))
        }
        let app = Router::new().route("/v1/chat/completions", post(chat));
        // Bind to an OS-chosen free loopback port.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        // Give the server a moment to start accepting.
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        port
    }

    #[tokio::test]
    async fn forwards_to_stub_llama_server_end_to_end() {
        let port = spawn_stub_llama().await;
        let srv = LlamaServer::attach(port).expect("attach to stub");
        let completion = srv.complete(&req(None)).await.expect("completion");
        assert_eq!(completion.text, "canned reply from stub");
        assert_eq!(completion.token_count, 4);
        assert_eq!(completion.finish_reason, "stop");
        // The streamed tokens reconstruct the text exactly.
        assert_eq!(completion.tokens().concat(), completion.text);
    }
}
