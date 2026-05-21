//! Command-line interface for llama inference.
//!
//! Usage: `llama-cli -m model.gguf -p "Hello, world!"`

#![deny(missing_docs)]

use clap::Parser;
use llama::{InferenceContext, Model, ModelConfig};
use std::sync::Arc;

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
    eprintln!("Starting llama-cli...");
    let args = Args::parse();
    eprintln!("Parsed args: model={}, prompt={}, n_predict={}", args.model, args.prompt, args.n_predict);

    // let filter = if args.verbose { "debug" } else { "info" };
    // tracing_subscriber::fmt()
    //     .with_env_filter(EnvFilter::new(filter))
    //     .init();

    eprintln!("Loading model from: {}", args.model);
    let start = std::time::Instant::now();

    let model = Arc::new(Model::from_file(&args.model)?);
    let load_time = start.elapsed();
    eprintln!("Model loaded in {:.2}s", load_time.as_secs_f32());
    eprintln!("{}", model.summary());

    let config = ModelConfig {
        n_threads: if args.threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            args.threads
        },
        use_cuda: false,
        n_ctx: args.ctx_size,
        n_batch: args.ctx_size,
    };

    let ctx = InferenceContext::new(model, config);

    if args.prompt.is_empty() {
        println!("Interactive mode — type your prompt (Ctrl+D to end):");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        ctx.encode(input.trim());
    } else {
        println!("Prompt: {}", args.prompt);
        ctx.encode(&args.prompt);
    }

    println!("\nGenerating {} tokens...\n", args.n_predict);

    let generated = ctx.generate(&args.prompt, args.n_predict)?;

    // Print generated text
    for token_id in &generated {
        print!("{}", ctx.decode_from_id(*token_id));
    }
    println!();

    Ok(())
}
