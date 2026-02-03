//! Kolosal AutoML Server - Main Entry Point

use kolosal_server::{run_server, ServerConfig};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    let mut config = ServerConfig::default();
    
    // Parse simple args
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    config.port = args[i + 1].parse().unwrap_or(8080);
                    i += 1;
                }
            }
            "--host" | "-h" => {
                if i + 1 < args.len() {
                    config.host = args[i + 1].clone();
                    i += 1;
                }
            }
            "--data-dir" => {
                if i + 1 < args.len() {
                    config.data_dir = args[i + 1].clone();
                    i += 1;
                }
            }
            "--models-dir" => {
                if i + 1 < args.len() {
                    config.models_dir = args[i + 1].clone();
                    i += 1;
                }
            }
            "--help" => {
                print_help();
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    // Set static dir relative to binary
    let exe_dir = env::current_exe()?
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    
    // Try multiple paths for static files
    let static_paths = [
        exe_dir.join("kolosal-web/static"),
        exe_dir.join("../kolosal-web/static"),
        std::path::PathBuf::from("kolosal-web/static"),
        std::path::PathBuf::from("rust/kolosal-web/static"),
    ];
    
    for path in static_paths {
        if path.exists() {
            config.static_dir = Some(path.to_string_lossy().to_string());
            break;
        }
    }

    run_server(config).await
}

fn print_help() {
    println!(r#"
Kolosal AutoML Server

USAGE:
    kolosal-server [OPTIONS]

OPTIONS:
    -p, --port <PORT>       Server port (default: 8080)
    -h, --host <HOST>       Server host (default: 0.0.0.0)
    --data-dir <DIR>        Data directory (default: ./data)
    --models-dir <DIR>      Models directory (default: ./models)
    --help                  Print this help message

EXAMPLES:
    kolosal-server                    # Start on default port 8080
    kolosal-server --port 3000        # Start on port 3000
    kolosal-server -h 127.0.0.1       # Bind to localhost only
"#);
}
