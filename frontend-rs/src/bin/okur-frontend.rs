//! Reads JSON strings, one per line, on stdin; writes the letters for each as a JSON string per line.
use std::io::{BufRead, Write};

fn main() {
    let frontend = okur_frontend::Frontend::new();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for line in std::io::stdin().lock().lines() {
        let text: String = serde_json::from_str(&line.expect("stdin")).expect("one JSON string per line");
        writeln!(out, "{}", serde_json::to_string(&frontend.letters(&text)).expect("json")).expect("stdout");
    }
}
