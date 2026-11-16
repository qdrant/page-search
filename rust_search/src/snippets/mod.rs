mod handler;
mod models;

pub use handler::{
    index_overview, search_breakdown, search_handler, search_snippets, IndexOverview,
    ScoredSnippet, SearchBreakdown, ValueCount,
};
pub use models::Snippet;

/// Spell a language the way the snippet index does, accepting common
/// abbreviations. The docs MCP server does this; `/snippets/search` doesn't.
pub fn normalize_language(language: &str) -> String {
    let language = language.trim().to_lowercase();
    match language.as_str() {
        "py" => "python",
        "ts" => "typescript",
        "js" | "jsx" => "javascript",
        "rs" => "rust",
        "c#" | "cs" | "dotnet" | ".net" => "csharp",
        "golang" => "go",
        "rest" | "http request" => "http",
        "bash" | "sh" | "console" | "env" => "shell",
        other => other,
    }
    .to_string()
}
