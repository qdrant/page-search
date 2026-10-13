//! The MCP server: three read-only tools over the content published on qdrant.tech.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::try_join_all;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{ScoredPoint, Value};
use qdrant_client::Qdrant;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerConfig};
use rmcp::{schemars, tool, tool_handler, tool_router, Json, ServerHandler};
use serde::{Deserialize, Serialize};

use super::config::Config;
use super::site::{check_enabled, source_of, SiteClient};
use crate::embedding::QueryEncoder;
use crate::search;

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
them in full. Use list_docs to browse when search terms are unclear.
Cite the page URLs you relied on.
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
    fn as_str(self) -> &'static str {
        match self {
            Self::Develop => "develop",
            Self::Deploy => "deploy",
            Self::Ecosystem => "ecosystem",
            Self::Learn => "learn",
        }
    }
}

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
    #[serde(default)]
    pub sources: Option<Vec<Source>>,
    /// Optionally restrict documentation results to one part of the docs: develop (building apps and the API), deploy (installation, Cloud, operations), ecosystem (integrations and frameworks), learn (tutorials). Other sources ignore it.
    #[serde(default)]
    pub partition: Option<Partition>,
    /// Maximum number of pages to return.
    #[serde(default = "default_search_limit")]
    #[schemars(range(min = 1, max = 50))]
    pub limit: u64,
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
    pub section: Option<String>,
    /// How many pages of the section to skip.
    #[serde(default)]
    pub offset: usize,
    /// Maximum number of pages to list.
    #[serde(default = "default_list_limit")]
    #[schemars(range(min = 1, max = 200))]
    pub limit: usize,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SearchHit {
    pub source: String,
    pub title: String,
    /// The headings and passages on the page that matched, best first.
    pub matches: Vec<String>,
    pub partition: String,
    pub url: String,
    pub markdown_url: String,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SearchResults {
    pub results: Vec<SearchHit>,
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
    ) -> Result<Json<SearchResults>, String> {
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
        Ok(Json(SearchResults { results: hits }))
    }

    /// Read one page in full, as Markdown, including its code examples.
    #[tool(annotations(
        read_only_hint = true,
        destructive_hint = false,
        idempotent_hint = true,
        open_world_hint = true
    ))]
    async fn get_page(&self, Parameters(args): Parameters<GetPageArgs>) -> Result<String, String> {
        let (entry, url, body) = self.site.get_page(&args.page).await?;
        let mut header = vec![format!("Source: {url}")];
        if let Some(entry) = entry {
            header.insert(0, format!("Title: {}", entry.title));
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
    ) -> Result<String, String> {
        let section = args.section.as_deref().unwrap_or_default();
        if matches!(section.trim_matches('/'), "" | "documentation") {
            let rows: Vec<String> = self
                .site
                .list_sections()
                .await?
                .into_iter()
                .map(|(page, count)| {
                    let mut row = format!(
                        "- {}: {} ({count} {})",
                        page.path,
                        page.title,
                        if count == 1 { "page" } else { "pages" }
                    );
                    if !page.description.is_empty() {
                        row.push_str(&format!(". {}", page.description));
                    }
                    row
                })
                .collect();
            return Ok(format!(
                "Top-level sections. Pass a section path to list_docs to see its pages.\n\n{}",
                rows.join("\n")
            ));
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
        let mut rows: Vec<String> = pages[args.offset..end]
            .iter()
            .map(|p| {
                let indent = if nested {
                    "  ".repeat(p.path.matches('/').count() - base_depth)
                } else {
                    String::new()
                };
                let mut line = format!("{indent}- {}: {}", p.title, p.path);
                if let Some(date) = p.date.as_ref().filter(|_| !nested) {
                    line.push_str(&format!(" ({date})"));
                }
                if !p.partition.is_empty() && p.source() == "documentation" {
                    line.push_str(&format!(" [{}]", p.partition));
                }
                if !p.rendered {
                    line.push_str(" (groups the pages that follow, no page of its own)");
                }
                if !p.description.is_empty() {
                    line.push_str(&format!(". {}", p.description));
                }
                line
            })
            .collect();
        if args.offset > 0 || end < total {
            let mut footer = format!("\nShowing {}-{end} of {total}.", args.offset + 1);
            if end < total {
                footer.push_str(&format!(
                    " Call list_docs with offset {end} for the next ones."
                ));
            }
            rows.push(footer);
        }
        Ok(rows.join("\n"))
    }
}

fn source_list(sources: &[String]) -> String {
    sources
        .iter()
        .map(|s| format!("- {s}: {}", source_description(s)))
        .collect::<Vec<_>>()
        .join("\n")
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
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
