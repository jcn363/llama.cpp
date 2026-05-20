//! Command-line interface for llama inference.
//!
//! Usage: `llama-cli -m model.gguf -p "Hello, world!"`

#![deny(missing_docs)]

use clap::Parser;
use tracing_subscriber::EnvFilter;

/// Command-line arguments for llama-cli.
#[derive(Parser, Debug)]
#[command(name = "llama-cli", about = "LLaMA inference CLI")]
struct Args {
    /// Path to the model file (GGUF format).
    #[arg(short, long)]
    model: String,

    /// Prompt text.
    #[arg(short, long, default_value = "")]
    prompt: String,

    /// Number of threads.
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,

    /// Context size.
    #[arg(short = 'c', long, default_value_t = 512)]
    ctx_size: usize,

    /// Maximum tokens to generate.
    #[arg(short = 'n', long, default_value_t = 128)]
    n_predict: usize,

    /// Enable verbose logging.
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(filter))
        .init();

    tracing::info!("Loading model from: {}", args.model);
    tracing::info!("Threads: {}, Context: {}", args.threads, args.ctx_size);

    if args.prompt.is_empty() {
        println!("Interactive mode — type your prompt (Ctrl+D to end):");
    } else {
        println!("Prompt: {}", args.prompt);
    }

    println!("TODO: inference not yet implemented");

    Ok(())
}
