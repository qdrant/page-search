//! The MCP server: four read-only tools over the content published on qdrant.tech.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::try_join_all;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{ScoredPoint, Value};
use qdrant_client::Qdrant;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, InitializeRequestParams, InitializeResult,
    ServerCapabilities, ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{schemars, tool, tool_handler, tool_router, ErrorData, RoleServer, ServerHandler};
use serde::{Deserialize, Serialize};

use super::config::Config;
use super::site::{
    check_enabled, normalize_path, similar_pages, source_of, DocPage, PageError, SiteClient,
};
use crate::embedding::QueryEncoder;
use crate::similar_paths::path_words;
use crate::snippets::Snippet;
use crate::{search, snippets};

fn source_description(source: &str) -> &'static str {
    match source {
        "documentation" => {
            "the official docs: concepts, API usage, deployment, Cloud, and integrations"
        }
        "articles" => "in-depth technical articles on vector search concepts and techniques",
        "blog" => "release announcements, product news, case studies, and talks",
        "course" => "Qdrant Academy: self-paced courses with lessons from beginner to advanced",
        _ => "",
    }
}

const INSTRUCTIONS: &str = "\
This server gives you official Qdrant content from qdrant.tech:

{sources}

Use it whenever a question involves Qdrant: the vector database, Qdrant Cloud,
client libraries, FastEmbed, or Qdrant Edge. Prefer it over answering from
memory, because APIs and defaults change between releases. The documentation
is the authority on current behavior; other sources can be older than the
release they describe.

Typical flow: search_docs to find the relevant pages, then get_page to read
them in full. Use list_docs to browse when search terms are unclear, and
search_snippets to find code examples in a given language. Documentation
pages link to full snippet pages with the same example in every client
language; get_page reads those too. Cite the page URLs you relied on.
";

/// A part of qdrant.tech that can be searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
#[schemars(inline)]
pub enum Source {
    Documentation,
    Articles,
    Blog,
    Course,
}

impl Source {
    pub const ALL: [Source; 4] = [
        Self::Documentation,
        Self::Articles,
        Self::Blog,
        Self::Course,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Documentation => "documentation",
            Self::Articles => "articles",
            Self::Blog => "blog",
            Self::Course => "course",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
#[schemars(inline)]
pub enum Partition {
    Develop,
    Deploy,
    Ecosystem,
    Learn,
}

impl Partition {
    const ALL: [Partition; 4] = [Self::Develop, Self::Deploy, Self::Ecosystem, Self::Learn];

    fn description(self) -> &'static str {
        match self {
            Self::Develop => "building apps and the API",
            Self::Deploy => "installation, Cloud, operations",
            Self::Ecosystem => "integrations and frameworks",
            Self::Learn => "guides to search quality and production operations",
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Develop => "develop",
            Self::Deploy => "deploy",
            Self::Ecosystem => "ecosystem",
            Self::Learn => "learn",
        }
    }
}

/// How a tool returns its result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
#[schemars(inline)]
pub enum Format {
    #[default]
    Markdown,
    Json,
}

/// The result as Markdown text, or as JSON in both the text and the
/// structured content.
fn respond<T: Serialize>(
    format: Format,
    value: &T,
    markdown: impl FnOnce(&T) -> String,
) -> Result<CallToolResult, String> {
    Ok(match format {
        Format::Markdown => CallToolResult::success(vec![ContentBlock::text(markdown(value))]),
        Format::Json => CallToolResult::structured(
            serde_json::to_value(value).map_err(|e| format!("Could not encode the result: {e}"))?,
        ),
    })
}

/// Most pages suggested for a missing page.
const MAX_SUGGESTIONS: usize = 5;

/// Matches kept per page: enough to show why it matched, without repeating
/// every heading of a long page.
const MAX_MATCHES: usize = 5;

fn default_search_limit() -> u64 {
    10
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchArgs {
    /// What to look for, in natural language or keywords.
    pub query: String,
    /// Which sources to search: documentation, articles, blog, course. Leave it out to search all of them.
    // Optional, not nullable: `type: [T, "null"]` is rejected by clients that
    // map tool schemas onto a single-type dialect, such as Gemini's.
    #[serde(default)]
    #[schemars(with = "Vec<Source>")]
    pub sources: Option<Vec<Source>>,
    /// Optionally restrict documentation results to one part of the docs: develop (building apps and the API), deploy (installation, Cloud, operations), ecosystem (integrations and frameworks), learn (guides to search quality and production operations). Other sources ignore it.
    #[serde(default)]
    #[schemars(with = "Partition")]
    pub partition: Option<Partition>,
    /// Maximum number of pages to return.
    #[serde(default = "default_search_limit")]
    #[schemars(range(min = 1, max = 50))]
    pub limit: u64,
    /// markdown (the default), or json for the same results as structured data.
    #[serde(default)]
    pub format: Format,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetPageArgs {
    /// The page to read: a qdrant.tech URL, its index.md URL, or a path such as /documentation/search/hybrid-queries/, /blog/qdrant-1.15.x/, or search/hybrid-queries (a path without a source is a documentation path).
    pub page: String,
}

fn default_list_limit() -> usize {
    50
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListDocsArgs {
    /// A section path such as search, /documentation/cloud/, or /blog/. Leave it out to list the top-level sections.
    #[serde(default)]
    #[schemars(with = "String")]
    pub section: Option<String>,
    /// How many pages of the section to skip.
    #[serde(default)]
    pub offset: usize,
    /// Maximum number of pages to list.
    #[serde(default = "default_list_limit")]
    #[schemars(range(min = 1, max = 200))]
    pub limit: usize,
    /// markdown (the default), or json for the same listing as structured data.
    #[serde(default)]
    pub format: Format,
}

fn default_snippet_limit() -> u64 {
    5
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchSnippetsArgs {
    /// What the code should do, in natural language or keywords, such as "create a collection with sparse vectors".
    pub query: String,
    /// The language of the code: python, typescript, rust, java, csharp, go, http (the REST API), shell, and so on. Leave it out to search all languages.
    #[serde(default)]
    #[schemars(with = "String")]
    pub language: Option<String>,
    /// Maximum number of snippets to return.
    #[serde(default = "default_snippet_limit")]
    #[schemars(range(min = 1, max = 20))]
    pub limit: u64,
    /// markdown (the default), or json for the same snippets as structured data.
    #[serde(default)]
    pub format: Format,
}

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub source: String,
    pub title: String,
    /// The headings and passages on the page that matched, best first.
    pub matches: Vec<String>,
    pub partition: String,
    pub url: String,
    pub markdown_url: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResults {
    pub results: Vec<SearchHit>,
}

#[derive(Debug, Serialize)]
pub struct SectionEntry {
    pub path: String,
    pub title: String,
    pub pages: usize,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub partition: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct SectionList {
    pub sections: Vec<SectionEntry>,
}

#[derive(Debug, Serialize)]
pub struct PageEntry {
    pub title: String,
    pub path: String,
    /// Nesting below the section, in sections listed in sidebar order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub partition: String,
    /// False for a section that only groups the pages that follow.
    #[serde(skip_serializing_if = "is_true")]
    pub has_page: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
}

fn is_true(value: &bool) -> bool {
    *value
}

#[derive(Debug, Serialize)]
pub struct PageList {
    pub section: String,
    pub total: usize,
    pub offset: usize,
    pub pages: Vec<PageEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_offset: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct SnippetHit {
    pub language: String,
    pub code: String,
    /// What the code does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The title of the page the snippet is on.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub title: String,
    pub url: String,
}

#[derive(Debug, Serialize)]
pub struct SnippetResults {
    pub results: Vec<SnippetHit>,
}

fn payload_str(payload: &HashMap<String, Value>, key: &str) -> Option<String> {
    match payload.get(key)?.kind.as_ref()? {
        Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    }
}

fn payload_first_str(payload: &HashMap<String, Value>, key: &str) -> Option<String> {
    match payload.get(key)?.kind.as_ref()? {
        Kind::ListValue(list) => list.values.iter().find_map(|v| match v.kind.as_ref()? {
            Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        }),
        Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    }
}

fn clean_title(title: &str) -> String {
    title
        .strip_suffix(" - Qdrant")
        .unwrap_or(title)
        .trim()
        .to_string()
}

/// Where `search_docs` gets its results.
#[derive(Clone)]
pub enum SearchBackend {
    /// The `site` collection, queried the same way as `/api/search`.
    Qdrant {
        client: Arc<Qdrant>,
        encoder: QueryEncoder,
    },
    /// Canned results per source, recording each request.
    #[cfg(test)]
    Fixed(Arc<tests::FixedSearch>),
}

#[derive(Clone)]
pub struct DocsServer {
    search: SearchBackend,
    site: Arc<SiteClient>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl DocsServer {
    pub fn new(config: Config, search: SearchBackend) -> Self {
        Self {
            search,
            site: Arc::new(SiteClient::new(config)),
            tool_router: Self::tool_router(),
        }
    }

    /// Search Qdrant content on qdrant.tech: the documentation, articles, the blog, and courses.
    ///
    /// Each result is a page, with its source and the headings or passages on it that matched. Read the page with get_page before relying on it: matches are short excerpts.
    #[tool(annotations(
        read_only_hint = true,
        destructive_hint = false,
        idempotent_hint = true,
        open_world_hint = true
    ))]
    async fn search_docs(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, String> {
        let query = args.query.trim();
        if query.is_empty() {
            return Err("The search query is empty.".to_string());
        }
        if let Some(requested) = &args.sources {
            for source in requested {
                check_enabled(&format!("/{}/", source.as_str()), self.site.sources())?;
            }
        }
        // The requested or else all served sources, in the canonical order.
        let served = |s: &Source| self.site.sources().iter().any(|e| e == s.as_str());
        let sources: Vec<Source> = Source::ALL
            .into_iter()
            .filter(|s| served(s) && args.sources.as_ref().is_none_or(|r| r.contains(s)))
            .collect();
        let limit = args.limit.clamp(1, 50);

        // One query per source, so one source can't crowd out the others.
        // Ask for more hits than pages, since several hits can share a page.
        let per_source = try_join_all(
            sources
                .iter()
                .map(|&source| self.search_source(query, source, args.partition, limit * 3)),
        )
        .await?;

        // Interleave by rank, so the best match from each source comes first,
        // and group matches by page, so one page doesn't take several slots.
        let mut hits: Vec<SearchHit> = Vec::new();
        let rounds = per_source.iter().map(Vec::len).max().unwrap_or(0);
        for rank in 0..rounds {
            for results in &per_source {
                let Some((hit, text)) = results.get(rank) else {
                    continue;
                };
                match hits.iter_mut().find(|h| h.url == hit.url) {
                    Some(existing) => {
                        if !text.is_empty()
                            && existing.matches.len() < MAX_MATCHES
                            && !existing.matches.contains(text)
                        {
                            existing.matches.push(text.clone());
                        }
                    }
                    None => hits.push(SearchHit {
                        source: hit.source.clone(),
                        title: hit.title.clone(),
                        matches: if text.is_empty() {
                            vec![]
                        } else {
                            vec![text.clone()]
                        },
                        partition: hit.partition.clone(),
                        url: hit.url.clone(),
                        markdown_url: hit.markdown_url.clone(),
                    }),
                }
            }
        }
        hits.truncate(limit as usize);
        respond(
            args.format,
            &SearchResults { results: hits },
            search_markdown,
        )
    }

    /// Find code examples in Qdrant content: the documentation, articles, the blog, and courses.
    ///
    /// Each result is a code snippet with a description of what it does and the page it is on. Pass a language to get examples for one client. Documentation pages usually show an example in one language and link to a full snippet page with it in every client language; read that with get_page.
    #[tool(annotations(
        read_only_hint = true,
        destructive_hint = false,
        idempotent_hint = true,
        open_world_hint = true
    ))]
    async fn search_snippets(
        &self,
        Parameters(args): Parameters<SearchSnippetsArgs>,
    ) -> Result<CallToolResult, String> {
        let query = args.query.trim();
        if query.is_empty() {
            return Err("The search query is empty.".to_string());
        }
        let language = args
            .language
            .as_deref()
            .map(normalize_language)
            .filter(|l| !l.is_empty());
        let limit = args.limit.clamp(1, 20) as usize;
        // Ask for more than needed: the same code can be on several pages,
        // and some pages may be in sources this server doesn't serve.
        let found: Vec<Snippet> = match &self.search {
            SearchBackend::Qdrant { client, .. } => {
                snippets::search_snippets(client, query, language.as_deref(), limit as u64 * 2)
                    .await
                    .map_err(|e| format!("Snippet search is unavailable: {e}"))?
            }
            #[cfg(test)]
            SearchBackend::Fixed(fixed) => fixed.search_snippets(language.as_deref()),
        };

        // Titles are a nicety: search still works if the page index doesn't load.
        let manifest = self.site.manifest().await.ok();
        let mut hits: Vec<SnippetHit> = Vec::new();
        for snippet in found {
            let Ok(path) = normalize_path(&snippet.source.url, None) else {
                continue;
            };
            let served = self.site.sources().iter().any(|s| s == source_of(&path));
            let code = snippet.code.trim_end().to_string();
            let repeated = hits
                .iter()
                .any(|h| h.language == snippet.language && h.code == code);
            if !served || repeated {
                continue;
            }
            let title = manifest
                .as_ref()
                .and_then(|m| m.pages.get(m.resolve(&path)))
                .map(|page| page.title.clone())
                .unwrap_or_default();
            hits.push(SnippetHit {
                language: snippet.language,
                code,
                description: snippet.description.map(|d| d.trim().to_string()),
                title,
                url: self.site.url(&path),
            });
            if hits.len() == limit {
                break;
            }
        }
        respond(
            args.format,
            &SnippetResults { results: hits },
            snippets_markdown,
        )
    }

    /// Read one page in full, as Markdown, including its code examples.
    ///
    /// For a missing page, the error suggests existing ones.
    #[tool(annotations(
        read_only_hint = true,
        destructive_hint = false,
        idempotent_hint = true,
        open_world_hint = true
    ))]
    async fn get_page(&self, Parameters(args): Parameters<GetPageArgs>) -> Result<String, String> {
        let (entry, url, body) = match self.site.get_page(&args.page).await {
            Ok(page) => page,
            Err(PageError::NotFound(path)) => return Err(self.not_found(&path).await),
            Err(PageError::Other(message)) => return Err(message),
        };
        let mut header = vec![format!("Source: {url}")];
        if let Some(entry) = entry {
            header.insert(0, format!("Title: {}", entry.title));
            if is_docs_with_partition(&entry) {
                header.push(format!("Partition: {}", entry.partition));
            }
            if let Some(date) = entry
                .date
                .as_ref()
                .filter(|_| entry.source() != "documentation")
            {
                header.push(format!("Published: {date}"));
            }
            if let Some(lastmod) = &entry.lastmod {
                header.push(format!("Last updated: {lastmod}"));
            }
        }
        Ok(format!("{}\n\n---\n\n{body}", header.join("\n")))
    }

    /// Browse the structure of the available content.
    ///
    /// Without a section, lists the top-level documentation sections and the other sources. With a section, lists its pages with a short description: documentation and the course in sidebar order, articles and the blog newest first. Long sections come in batches; the end of the output says how to get the next one.
    #[tool(annotations(
        read_only_hint = true,
        destructive_hint = false,
        idempotent_hint = true,
        open_world_hint = true
    ))]
    async fn list_docs(
        &self,
        Parameters(args): Parameters<ListDocsArgs>,
    ) -> Result<CallToolResult, String> {
        let section = args.section.as_deref().unwrap_or_default();
        if matches!(section.trim_matches('/'), "" | "documentation") {
            let sections = self
                .site
                .list_sections()
                .await?
                .into_iter()
                .map(|(page, count)| SectionEntry {
                    partition: if is_docs_with_partition(&page) {
                        page.partition
                    } else {
                        String::new()
                    },
                    path: page.path,
                    title: page.title,
                    pages: count,
                    description: page.description,
                })
                .collect();
            return respond(args.format, &SectionList { sections }, sections_markdown);
        }

        let (pages, nested) = self.site.list_pages(section).await?;
        let total = pages.len();
        if args.offset >= total {
            return Err(format!(
                "{section} has {total} pages; offset {} is past the end.",
                args.offset
            ));
        }
        let base_depth = pages
            .iter()
            .map(|p| p.path.matches('/').count())
            .min()
            .unwrap_or(0);
        let end = (args.offset + args.limit.clamp(1, 200)).min(total);
        let entries = pages[args.offset..end]
            .iter()
            .map(|p| PageEntry {
                title: p.title.clone(),
                path: p.path.clone(),
                depth: nested.then(|| p.path.matches('/').count() - base_depth),
                date: p.date.clone().filter(|_| !nested),
                partition: if p.source() == "documentation" {
                    p.partition.clone()
                } else {
                    String::new()
                },
                has_page: p.rendered,
                description: p.description.clone(),
            })
            .collect();
        let list = PageList {
            section: self.site.normalize(section)?,
            total,
            offset: args.offset,
            pages: entries,
            next_offset: (end < total).then_some(end),
        };
        respond(args.format, &list, pages_markdown)
    }
}

/// Collapse a passage onto one line, to keep it inside a list item.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn search_markdown(results: &SearchResults) -> String {
    if results.results.is_empty() {
        return "No results. Try other terms, or browse with list_docs.".to_string();
    }
    results
        .results
        .iter()
        .map(|hit| {
            let mut labels = vec![hit.source.as_str()];
            if !hit.partition.is_empty() {
                labels.push(&hit.partition);
            }
            let title = if hit.title.is_empty() {
                &hit.url
            } else {
                &hit.title
            };
            let mut block = format!("## {title}\n{} ({})", hit.url, labels.join(", "));
            for text in &hit.matches {
                block.push_str(&format!("\n- {}", one_line(text)));
            }
            block
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn sections_markdown(list: &SectionList) -> String {
    let rows: Vec<String> = list
        .sections
        .iter()
        .map(|s| {
            let mut row = section_row(&s.path, &s.title, s.pages);
            if !s.partition.is_empty() {
                row.push_str(&format!(" [{}]", s.partition));
            }
            if !s.description.is_empty() {
                row.push_str(&format!(". {}", s.description));
            }
            row
        })
        .collect();
    format!(
        "Top-level sections. Pass a section path to list_docs to see its pages.\n\n{}",
        rows.join("\n")
    )
}

fn pages_markdown(list: &PageList) -> String {
    let mut rows: Vec<String> = list
        .pages
        .iter()
        .map(|p| {
            let indent = "  ".repeat(p.depth.unwrap_or(0));
            let mut line = format!("{indent}- {}: {}", p.title, p.path);
            if let Some(date) = &p.date {
                line.push_str(&format!(" ({date})"));
            }
            if !p.partition.is_empty() {
                line.push_str(&format!(" [{}]", p.partition));
            }
            if !p.has_page {
                line.push_str(" (groups the pages that follow, no page of its own)");
            }
            if !p.description.is_empty() {
                line.push_str(&format!(". {}", p.description));
            }
            line
        })
        .collect();
    let end = list.offset + list.pages.len();
    if list.offset > 0 || list.next_offset.is_some() {
        let mut footer = format!("\nShowing {}-{end} of {}.", list.offset + 1, list.total);
        if let Some(next) = list.next_offset {
            footer.push_str(&format!(
                " Call list_docs with offset {next} for the next ones."
            ));
        }
        rows.push(footer);
    }
    rows.join("\n")
}

fn snippets_markdown(results: &SnippetResults) -> String {
    if results.results.is_empty() {
        return "No snippets found. Try other terms or another language, or search_docs."
            .to_string();
    }
    results
        .results
        .iter()
        .map(|hit| {
            let title = if hit.title.is_empty() {
                &hit.url
            } else {
                &hit.title
            };
            let mut block = format!("## {title} ({})\n{}\n", hit.language, hit.url);
            if let Some(description) = &hit.description {
                block.push_str(&format!("\n{description}\n"));
            }
            // A fence longer than any run of backticks in the code.
            let longest = hit
                .code
                .split(|c| c != '`')
                .map(str::len)
                .max()
                .unwrap_or(0);
            let fence = "`".repeat(longest.max(2) + 1);
            block.push_str(&format!("\n{fence}{}\n{}\n{fence}", hit.language, hit.code));
            block
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Spell a language the way the snippet index does, accepting common
/// abbreviations.
fn normalize_language(language: &str) -> String {
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

fn source_list(sources: &[String]) -> String {
    sources
        .iter()
        .map(|s| format!("- {s}: {}", source_description(s)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A top-level section as a list row, without its partition or description.
fn section_row(path: &str, title: &str, count: usize) -> String {
    format!(
        "- {path}: {title} ({count} {})",
        if count == 1 { "page" } else { "pages" }
    )
}

fn is_docs_with_partition(page: &DocPage) -> bool {
    page.source() == "documentation" && !page.partition.is_empty()
}

/// The top-level sections grouped by partition, so an agent can pass
/// `section` and `partition` without browsing first.
fn structure(sections: &[(DocPage, usize)]) -> String {
    let rows = |keep: &dyn Fn(&DocPage) -> bool| -> String {
        sections
            .iter()
            .filter(|(page, _)| keep(page))
            .map(|(page, count)| section_row(&page.path, &page.title, *count))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut groups: Vec<String> = Partition::ALL
        .iter()
        .map(|p| {
            let in_partition =
                |page: &DocPage| page.source() == "documentation" && page.partition == p.as_str();
            (
                format!("{} ({})", p.as_str(), p.description()),
                rows(&in_partition),
            )
        })
        .chain([
            (
                "Other documentation".to_string(),
                rows(&|page: &DocPage| {
                    page.source() == "documentation"
                        && !Partition::ALL.iter().any(|p| p.as_str() == page.partition)
                }),
            ),
            (
                "Other sources".to_string(),
                rows(&|page: &DocPage| page.source() != "documentation"),
            ),
        ])
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(heading, rows)| format!("{heading}:\n{rows}"))
        .collect();
    if groups.is_empty() {
        return String::new();
    }
    groups.insert(
        0,
        "Top-level sections, with the documentation grouped by partition. Pass a \
         section path to list_docs to see its pages, or a partition to search_docs \
         to search only that part of the documentation."
            .to_string(),
    );
    groups.join("\n\n")
}

/// A search result before grouping, with the heading or passage that matched.
struct RawHit {
    source: String,
    title: String,
    partition: String,
    url: String,
    markdown_url: String,
}

impl DocsServer {
    /// Error for a missing page: similar pages, then search results for the
    /// words of the path.
    async fn not_found(&self, path: &str) -> String {
        let mut suggestions: Vec<(String, String)> = Vec::new();
        let mut suggest = |title: String, url: String| {
            if suggestions.len() < MAX_SUGGESTIONS && !suggestions.iter().any(|(_, u)| *u == url) {
                suggestions.push((title, url));
            }
        };
        match self.site.manifest().await {
            Ok(manifest) => {
                for page in similar_pages(&manifest, path, MAX_SUGGESTIONS) {
                    suggest(page.title, self.site.url(&page.path));
                }
            }
            Err(e) => log::warn!("No suggestions from the page index: {e}"),
        }
        let source = Source::ALL
            .into_iter()
            .find(|s| s.as_str() == source_of(path));
        let query = path_words(path).join(" ");
        if let Some(source) = source.filter(|_| !query.is_empty()) {
            match self
                .search_source(&query, source, None, MAX_SUGGESTIONS as u64 * 3)
                .await
            {
                Ok(hits) => {
                    for (hit, _) in hits {
                        suggest(hit.title, hit.url);
                    }
                }
                Err(e) => log::warn!("No suggestions from search: {e}"),
            }
        }

        if suggestions.is_empty() {
            return format!(
                "No page at {path}. Use search_docs or list_docs to find the right one."
            );
        }
        let rows: Vec<String> = suggestions
            .into_iter()
            .map(|(title, url)| {
                if title.is_empty() {
                    format!("- {url}")
                } else {
                    format!("- {title}: {url}")
                }
            })
            .collect();
        format!(
            "No page at {path}. These existing pages may be the one you meant:\n\n{}\n\n\
             Read one with get_page, or use search_docs or list_docs to find others.",
            rows.join("\n")
        )
    }

    async fn search_source(
        &self,
        query: &str,
        source: Source,
        partition: Option<Partition>,
        limit: u64,
    ) -> Result<Vec<(RawHit, String)>, String> {
        let source = source.as_str();
        // Partitions only exist in the documentation; elsewhere they would
        // filter out every result.
        let partitions: Vec<String> = match partition {
            Some(p) if source == "documentation" => vec![p.as_str().to_string()],
            _ => vec![],
        };
        let points: Vec<ScoredPoint> = match &self.search {
            SearchBackend::Qdrant { client, encoder } => search::search(
                client,
                encoder,
                query,
                &[source.to_string()],
                &partitions,
                limit,
            )
            .await
            .map_err(|e| format!("Search is unavailable: {e}"))?,
            #[cfg(test)]
            SearchBackend::Fixed(fixed) => fixed.search(source, &partitions),
        };

        Ok(points
            .into_iter()
            .filter_map(|point| {
                let path = payload_str(&point.payload, "url")?;
                if source_of(&path) != source {
                    return None;
                }
                let hit = RawHit {
                    source: source.to_string(),
                    title: payload_first_str(&point.payload, "titles")
                        .map(|t| clean_title(&t))
                        .unwrap_or_default(),
                    partition: payload_str(&point.payload, "partition").unwrap_or_default(),
                    url: self.site.url(&path),
                    markdown_url: self.site.url(&format!("{path}index.md")),
                };
                let text = payload_str(&point.payload, "text")
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                Some((hit, text))
            })
            .collect())
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for DocsServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("qdrant-docs", env!("CARGO_PKG_VERSION"))
                    .with_title("Qdrant Documentation"),
            )
            .with_instructions(INSTRUCTIONS.replace("{sources}", &source_list(self.site.sources())))
    }

    /// Adds the site structure to the instructions. It comes from the page
    /// index, which needs a fetch, so it can't be part of `get_info`.
    async fn initialize(
        &self,
        request: InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, ErrorData> {
        context.peer.set_peer_info(request.clone());
        let mut info = self.negotiate_initialize(&request)?;
        let sections = self.site.list_sections().await.unwrap_or_else(|e| {
            log::warn!("Instructions without the section list: {e}");
            vec![]
        });
        let structure = structure(&sections);
        if let Some(instructions) = info.instructions.as_mut().filter(|_| !structure.is_empty()) {
            instructions.push_str(&format!("\n{structure}\n"));
        }
        Ok(info)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
