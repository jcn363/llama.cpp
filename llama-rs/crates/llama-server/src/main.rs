//! HTTP server for llama inference.
//!
//! Usage: `llama-server -m model.gguf --host 0.0.0.0 --port 8080`

#![deny(missing_docs)]
#![allow(dead_code)]

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
};
use clap::Parser;
use futures::stream::Stream;
use futures::StreamExt;
use llama::{InferenceContext, Model, ModelConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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

    /// Context size.
    #[arg(short = 'c', long, default_value_t = 512)]
    ctx_size: usize,
}

/// Shared server state.
#[derive(Clone)]
struct ServerState {
    model: Arc<Model>,
    config: ModelConfig,
}

/// Completion request body.
#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

fn default_max_tokens() -> usize {
    128
}

fn default_temperature() -> f32 {
    0.8
}

/// Completion response body.
#[derive(Serialize)]
struct CompletionResponse {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

/// Streaming chunk response.
#[derive(Serialize)]
struct StreamChunk {
    content: String,
    stop: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    tracing::info!("Loading model from: {}", args.model);
    let model = Model::from_file(&args.model)?;
    tracing::info!("{}", model.summary());

    let config = ModelConfig {
        n_threads: if args.threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            args.threads
        },
        use_cuda: false,
        n_ctx: args.ctx_size,
        n_batch: args.ctx_size,
        ..Default::default()
    };

    let state = ServerState {
        model: Arc::new(model),
        config,
    };

    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/completion", post(handle_completion))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn handle_completion(
    State(state): State<ServerState>,
    Json(request): Json<CompletionRequest>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, String)> {
    tracing::info!(
        "Completion request: prompt_len={}, max_tokens={}, stream={}",
        request.prompt.len(),
        request.max_tokens,
        request.stream
    );

    if request.stream {
        return Ok(axum::response::IntoResponse::into_response(
            handle_streaming(state, request).await,
        ));
    }

    // Non-streaming: generate all tokens then return
    let model = Arc::clone(&state.model);

    let mut ctx = InferenceContext::new(model, state.config.clone());
    ctx.encode(&request.prompt);

    let generated = ctx.generate(request.max_tokens).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("inference error: {e}"),
        )
    })?;

    let content = generated
        .iter()
        .map(|&id| ctx.decode_from_id(id))
        .collect::<String>();

    Ok(axum::response::IntoResponse::into_response(Json(
        CompletionResponse {
            content,
            model: Some(state.model.summary()),
        },
    )))
}

async fn handle_streaming(
    state: ServerState,
    request: CompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let model = Arc::clone(&state.model);
    let config = state.config.clone();
    let prompt = request.prompt;
    let max_tokens = request.max_tokens;

    // Run inference in a blocking thread pool
    let stream = tokio::task::spawn_blocking(move || {
        let mut ctx = InferenceContext::new(model, config);
        ctx.encode(&prompt);

        let mut chunks = Vec::new();

        // Generate tokens one at a time
        for _ in 0..max_tokens {
            let generated = match ctx.generate(1) {
                Ok(tokens) => tokens,
                Err(_) => break,
            };

            if generated.is_empty() {
                break;
            }

            let token_id = generated[0];
            let content = ctx.decode_from_id(token_id);
            chunks.push(StreamChunk {
                content,
                stop: false,
            });
        }

        // Add final stop chunk
        chunks.push(StreamChunk {
            content: String::new(),
            stop: true,
        });

        chunks
    })
    .await
    .unwrap_or_default();

    // Convert to SSE stream
    let event_stream = futures::stream::iter(stream).map(|chunk| {
        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok(Event::default().data(data))
    });

    Sse::new(event_stream)
}
