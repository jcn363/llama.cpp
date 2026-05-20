//! HTTP server for llama inference.
//!
//! Usage: `llama-server -m model.gguf --host 0.0.0.0 --port 8080`

#![deny(missing_docs)]

use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

/// Command-line arguments for llama-server.
#[derive(Parser, Debug)]
#[command(name = "llama-server", about = "LLaMA HTTP server")]
struct Args {
    /// Path to the model file (GGUF format).
    #[arg(short, long)]
    model: String,

    /// Host to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on.
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Number of threads.
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
}

/// Shared server state.
#[derive(Clone)]
struct ServerState {
    #[allow(dead_code)]
    model_path: String,
}

/// Completion request body.
#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    #[allow(dead_code)]
    max_tokens: usize,
}

fn default_max_tokens() -> usize {
    128
}

/// Completion response body.
#[derive(Serialize)]
struct CompletionResponse {
    content: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    tracing::info!("Starting server on {}:{}", args.host, args.port);
    tracing::info!("Loading model from: {}", args.model);

    let state = ServerState {
        model_path: args.model,
    };

    let app = Router::new()
        .route("/completion", post(handle_completion))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_completion(
    State(_state): State<ServerState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    tracing::info!("Completion request: prompt_len={}", request.prompt.len());

    // TODO(#6): Implement actual inference
    Err((StatusCode::NOT_IMPLEMENTED, "inference not yet implemented".into()))
}
