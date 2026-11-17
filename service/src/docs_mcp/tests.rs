//! Tests that run the real server in-process, with an MCP client, against
//! fixed pages and canned search results.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use qdrant_client::qdrant::ScoredPoint;
use qdrant_client::Payload;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::RunningService;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{json, Value};

use super::{Backend, DocsServer, Source};
use crate::docs_mcp::config::{Config, SOURCES};
use crate::docs_mcp::library::SITE_URL;
use crate::feedback::tests::TempStore;
use crate::feedback::FeedbackFilter;
use crate::pages::{self, page_key, StoredPage};
use crate::similar_paths::MemoryIndex;
use crate::snippets::Snippet;

/// Canned search results per source, recording the source and partitions of
/// each request, canned snippets, recording the language of each request,
/// and pages, recording each page read.
#[derive(Default)]
pub struct FixedBackend {
    results: HashMap<String, Vec<ScoredPoint>>,
    pub requests: Mutex<Vec<(String, Vec<String>)>>,
    snippets: Vec<Value>,
    pub snippet_requests: Mutex<Vec<Option<String>>>,
    /// The page index. [`page_index`] if left empty.
    pub pages: Vec<StoredPage>,
    /// The pages with their own page, for suggestions.
    pub similar: MemoryIndex,
    /// Markdown by `page` key.
    markdown: HashMap<String, String>,
    pub markdown_requests: Mutex<Vec<String>>,
    /// The paths looked up with [`FixedBackend::find_page`].
    pub find_requests: Mutex<Vec<String>>,
}

impl FixedBackend {
    /// The pages `pages::find_page` would filter by, matched the same way.
    pub fn find_page(&self, page: &str) -> Option<StoredPage> {
        self.find_requests.lock().unwrap().push(page.to_string());
        let found = self
            .pages
            .iter()
            .filter(|p| p.page == page || p.meta.aliases.iter().any(|a| a == page))
            .cloned()
            .collect();
        pages::best_match(page, found)
    }

    pub fn page_markdown(&self, page: &str) -> Option<String> {
        self.markdown_requests
            .lock()
            .unwrap()
            .push(page.to_string());
        self.markdown.get(page).cloned()
    }

    pub fn search(&self, source: &str, partitions: &[String]) -> Vec<ScoredPoint> {
        self.requests
            .lock()
            .unwrap()
            .push((source.to_string(), partitions.to_vec()));
        self.results.get(source).cloned().unwrap_or_default()
    }

    pub fn search_snippets(&self, language: Option<&str>) -> Vec<Snippet> {
        self.snippet_requests
            .lock()
            .unwrap()
            .push(language.map(str::to_string));
        self.snippets
            .iter()
            .filter(|s| language.is_none_or(|l| s["language"] == l))
            .map(|s| serde_json::from_value(s.clone()).unwrap())
            .collect()
    }
}

fn snippet(url: &str, language: &str, code: &str) -> Value {
    json!({
        "code": code,
        "language": language,
        "version": "latest",
        "revision": 1,
        "package_name": "qdrant-client",
        "source": {"url": url, "hash": "abc", "lines": [10, 20]},
        "context": {"before": "", "after": ""},
        "description": format!("Runs a hybrid query in {language}."),
    })
}

fn snippets() -> Vec<Value> {
    let page = "https://qdrant.tech/documentation/search/hybrid-queries/";
    vec![
        snippet(page, "python", "client.query_points(prefetch=[...])\n"),
        snippet(
            page,
            "rust",
            "client.query(QueryPointsBuilder::new(\"c\")).await?;",
        ),
        // The same code on another page.
        snippet(
            "https://qdrant.tech/documentation/search/filtering/",
            "python",
            "client.query_points(prefetch=[...])",
        ),
        snippet("https://qdrant.tech/blog/new-post/", "python", "print(```)"),
    ]
}

fn hit(url: &str, title: &str, text: &str) -> ScoredPoint {
    let payload = Payload::try_from(json!({
        "url": url,
        "titles": [format!("{title} - Qdrant")],
        "text": text,
        "partition": if url.starts_with("/documentation/") { "develop" } else { "" },
    }))
    .unwrap();
    ScoredPoint {
        payload: payload.into(),
        ..Default::default()
    }
}

fn search_results() -> FixedBackend {
    FixedBackend {
        results: HashMap::from([
            (
                "documentation".to_string(),
                vec![
                    hit(
                        "/documentation/search/hybrid-queries/",
                        "Hybrid Queries",
                        "Hybrid Search",
                    ),
                    hit(
                        "/documentation/search/hybrid-queries/",
                        "Hybrid Queries",
                        "Fusion",
                    ),
                    hit(
                        "/documentation/search/hybrid-queries/",
                        "Hybrid Queries",
                        "Hybrid Search",
                    ),
                    // Tagged with the documentation section, but not a docs page.
                    hit("/blog/hybrid/", "Blog", "Hybrid"),
                    hit(
                        "/documentation/search/filtering/",
                        "Filtering",
                        "Filter and search",
                    ),
                ],
            ),
            (
                "blog".to_string(),
                vec![hit("/blog/new-post/", "New Post", "Hybrid is here")],
            ),
        ]),
        snippets: snippets(),
        ..FixedBackend::default()
    }
}

/// The page index, as the site's page index lists it.
fn page_index() -> Value {
    json!({
        "pages": [
            {"path": "/documentation/search/", "title": "Search", "description": "Search Qdrant.",
             "kind": "section", "partition": "develop", "weight": 2, "aliases": []},
            {"path": "/documentation/search/hybrid-queries/", "title": "Hybrid Queries",
             "description": "Fuse dense and sparse results.", "kind": "page", "partition": "develop",
             "weight": 3, "aliases": ["/documentation/concepts/hybrid-queries"], "lastmod": "2026-09-01"},
            {"path": "/documentation/search/filtering/", "title": "Filtering", "description": "Payload conditions.",
             "kind": "page", "partition": "develop", "weight": 1, "aliases": []},
            {"path": "/documentation/cloud/", "title": "Qdrant Cloud", "description": "Managed Qdrant.",
             "kind": "section", "partition": "deploy", "weight": 1, "aliases": []},
            {"path": "/documentation/faq/", "title": "FAQ", "kind": "section", "partition": "develop",
             "weight": 5, "aliases": [], "rendered": false},
            {"path": "/documentation/faq/basics/", "title": "Basics", "kind": "page", "weight": 1, "aliases": []},
            {"path": "/blog/", "title": "Blog", "description": "News.", "kind": "section", "weight": 0, "aliases": []},
            {"path": "/blog/old-post/", "title": "Old Post", "kind": "page", "weight": 1000.0, "date": "2023-01-01", "aliases": []},
            {"path": "/blog/new-post/", "title": "New Post", "kind": "page", "weight": 0, "date": "2026-01-01", "aliases": []},
        ],
    })
}

const PAGE_MD: &str = "\
> Explore Qdrant's agent skills catalog at https://skills.qdrant.tech/
> Search the documentation at https://skills.qdrant.tech/search?query=your+query+here
> Use this file to discover all available pages: https://qdrant.tech/llms.txt

# Hybrid and Multi-Stage Queries

Use `prefetch`.
";

/// Pages as the `pages` collection stores the entries of a page index.
fn stored(index: &Value) -> Vec<StoredPage> {
    index["pages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let mut entry = entry.clone();
            let path = entry["path"].as_str().unwrap().to_string();
            entry["page"] = json!(page_key(&path));
            let aliases: Vec<String> = entry["aliases"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|alias| page_key(alias.as_str().unwrap()))
                .collect();
            entry["aliases"] = json!(aliases);
            serde_json::from_value(entry).unwrap()
        })
        .collect()
}

struct Harness {
    client: RunningService<RoleClient, ()>,
    backend: Arc<FixedBackend>,
    feedback: TempStore,
}

impl Harness {
    async fn new() -> Self {
        Self::with_backend(search_results()).await
    }

    async fn with_backend(backend: FixedBackend) -> Self {
        Self::serving(&SOURCES, backend).await
    }

    /// A server configured to serve only `sources`.
    async fn serving(sources: &[&str], backend: FixedBackend) -> Self {
        let config = Config {
            sources: sources.iter().map(|s| s.to_string()).collect(),
        };
        let mut backend = backend;
        if backend.pages.is_empty() {
            backend.pages = stored(&page_index());
        }
        backend.markdown.insert(
            "documentation/search/hybrid-queries".to_string(),
            PAGE_MD.to_string(),
        );
        backend.similar = MemoryIndex(
            backend
                .pages
                .iter()
                .filter(|page| page.meta.rendered)
                .map(|page| page.page.clone())
                .collect(),
        );
        let backend = Arc::new(backend);
        let feedback = TempStore::new();
        let server = DocsServer::new(config, Backend::Fixed(backend.clone()))
            .with_feedback(feedback.store.clone());
        let (server_transport, client_transport) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            server
                .serve(server_transport)
                .await
                .unwrap()
                .waiting()
                .await
                .unwrap();
        });
        let client = ().serve(client_transport).await.unwrap();
        Self {
            client,
            backend,
            feedback,
        }
    }

    /// Everything `submit_feedback` stored, oldest first.
    fn stored_feedback(&self) -> Vec<crate::feedback::FeedbackRecord> {
        let mut items = self
            .feedback
            .store
            .list(&FeedbackFilter {
                include_dismissed: true,
                limit: 200,
                ..Default::default()
            })
            .unwrap()
            .items;
        items.reverse();
        items
    }

    async fn call(&self, tool: &'static str, arguments: Value) -> CallToolResult {
        self.client
            .call_tool(
                CallToolRequestParams::new(tool)
                    .with_arguments(arguments.as_object().unwrap().clone()),
            )
            .await
            .unwrap()
    }

    async fn text(&self, tool: &'static str, arguments: Value) -> String {
        text(&self.call(tool, arguments).await)
    }

    /// The pages whose Markdown was read.
    fn markdown_requests(&self) -> Vec<String> {
        self.backend.markdown_requests.lock().unwrap().clone()
    }
}

fn text(result: &CallToolResult) -> String {
    result.content[0].as_text().unwrap().text.clone()
}

fn is_error(result: &CallToolResult) -> bool {
    result.is_error == Some(true)
}

fn results(result: &CallToolResult) -> Vec<Value> {
    result.structured_content.as_ref().unwrap()["results"]
        .as_array()
        .unwrap()
        .clone()
}

#[tokio::test]
async fn search_groups_matches_by_page_and_keeps_only_docs() {
    let h = Harness::new().await;
    let result = h
        .call(
            "search_docs",
            json!({"query": "hybrid", "sources": ["documentation"], "partition": "develop", "format": "json"}),
        )
        .await;
    let hits = results(&result);
    let titles: Vec<&str> = hits.iter().map(|h| h["title"].as_str().unwrap()).collect();
    assert_eq!(titles, ["Hybrid Queries", "Filtering"]);
    assert_eq!(hits[0]["matches"], json!(["Hybrid Search", "Fusion"]));
    assert_eq!(hits[0]["source"], "documentation");
    assert_eq!(
        hits[0]["markdown_url"],
        format!("{}/documentation/search/hybrid-queries/index.md", SITE_URL)
    );
    assert_eq!(
        *h.backend.requests.lock().unwrap(),
        [("documentation".to_string(), vec!["develop".to_string()])]
    );
}

#[tokio::test]
async fn search_keeps_a_few_matches_per_page() {
    let mut search = search_results();
    let many = (0..8).map(|i| {
        hit(
            "/documentation/search/filtering/",
            "Filtering",
            &format!("Heading {i}"),
        )
    });
    search
        .results
        .insert("documentation".to_string(), many.collect());
    let h = Harness::with_backend(search).await;
    let result = h
        .call(
            "search_docs",
            json!({"query": "filter", "sources": ["documentation"], "format": "json"}),
        )
        .await;
    assert_eq!(results(&result)[0]["matches"].as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn search_queries_each_source_and_interleaves() {
    let h = Harness::new().await;
    let result = h
        .call(
            "search_docs",
            json!({"query": "hybrid", "partition": "develop", "format": "json"}),
        )
        .await;
    let hits: Vec<(String, String)> = results(&result)
        .iter()
        .map(|h| {
            (
                h["source"].as_str().unwrap().into(),
                h["title"].as_str().unwrap().into(),
            )
        })
        .collect();
    assert_eq!(
        hits,
        [
            ("documentation".into(), "Hybrid Queries".into()),
            ("blog".into(), "New Post".into()),
            ("documentation".into(), "Filtering".into()),
        ]
    );
    // The partition only narrows documentation results.
    let requests: HashMap<String, Vec<String>> =
        h.backend.requests.lock().unwrap().iter().cloned().collect();
    assert_eq!(requests["documentation"], ["develop"]);
    assert!(requests["blog"].is_empty());
}

#[tokio::test]
async fn search_can_pick_sources() {
    let h = Harness::new().await;
    let result = h
        .call(
            "search_docs",
            json!({"query": "hybrid", "sources": ["blog"], "format": "json"}),
        )
        .await;
    let titles: Vec<Value> = results(&result)
        .iter()
        .map(|h| h["title"].clone())
        .collect();
    assert_eq!(titles, [json!("New Post")]);
    assert_eq!(
        *h.backend.requests.lock().unwrap(),
        [("blog".to_string(), vec![])]
    );
}

#[tokio::test]
async fn sources_are_a_plain_enum_argument() {
    let h = Harness::new().await;
    let tools = h.client.list_all_tools().await.unwrap();
    let search = tools.iter().find(|t| t.name == "search_docs").unwrap();
    let items = &search.input_schema["properties"]["sources"]["items"];
    assert_eq!(items["enum"], json!(SOURCES), "{items}");

    let unknown = h
        .client
        .call_tool(
            CallToolRequestParams::new("search_docs").with_arguments(
                json!({"query": "hybrid", "sources": ["docs"]})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        )
        .await;
    assert!(unknown.is_err() || is_error(&unknown.unwrap()));
    assert!(h.backend.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn optional_arguments_have_a_single_type() {
    let h = Harness::new().await;
    for tool in h.client.list_all_tools().await.unwrap() {
        let schema = Value::Object((*tool.input_schema).clone());
        let required = schema["required"].as_array().cloned().unwrap_or_default();
        for (name, property) in schema["properties"].as_object().unwrap() {
            assert!(
                !property["type"].is_array() && property.get("anyOf").is_none(),
                "{}.{name}: {property}",
                tool.name
            );
            if !["query", "page", "feedback"].contains(&name.as_str()) {
                assert!(!required.contains(&json!(name)), "{}.{name}", tool.name);
            }
        }
    }
}

#[tokio::test]
async fn disabled_sources_are_refused() {
    let h = Harness::serving(&["documentation"], search_results()).await;

    // Without `sources`, only the served sources are searched.
    h.call("search_docs", json!({"query": "hybrid"})).await;
    assert_eq!(
        *h.backend.requests.lock().unwrap(),
        [("documentation".to_string(), vec![])]
    );

    let search = h
        .call(
            "search_docs",
            json!({"query": "hybrid", "sources": ["blog"]}),
        )
        .await;
    assert!(is_error(&search));
    assert!(text(&search).contains("doesn't serve /blog/ pages. It serves: documentation."));

    let page = h
        .call(
            "get_page",
            json!({"page": "https://qdrant.tech/blog/new-post/"}),
        )
        .await;
    assert!(is_error(&page));
    assert!(h.markdown_requests().is_empty());

    // The schema is the same whatever the server serves.
    let tools = h.client.list_all_tools().await.unwrap();
    let schema = &tools
        .iter()
        .find(|t| t.name == "search_docs")
        .unwrap()
        .input_schema;
    assert_eq!(
        schema["properties"]["sources"]["items"]["enum"],
        json!(SOURCES)
    );
}

#[test]
fn source_names_match_the_site_sections() {
    assert_eq!(Source::ALL.map(Source::as_str), SOURCES);
}

#[tokio::test]
async fn get_page_resolves_alias_and_strips_preamble() {
    let h = Harness::new().await;
    let body = h
        .text(
            "get_page",
            json!({"page": "https://qdrant.tech/documentation/concepts/hybrid-queries/"}),
        )
        .await;
    let expected = format!(
        "Title: Hybrid Queries\nSource: {}/documentation/search/hybrid-queries/",
        SITE_URL
    );
    assert!(body.starts_with(&expected), "{body}");
    assert!(body.contains("Last updated: 2026-09-01"));
    assert!(body.contains("Partition: develop"), "{body}");
    assert!(!body.contains("skills.qdrant.tech"));
    assert!(body.contains("# Hybrid and Multi-Stage Queries"));
    // Found with one filtered lookup of the old path, then read at the new one.
    assert_eq!(
        *h.backend.find_requests.lock().unwrap(),
        ["documentation/concepts/hybrid-queries"]
    );
    assert_eq!(
        h.markdown_requests(),
        ["documentation/search/hybrid-queries"]
    );
}

#[tokio::test]
async fn instructions_group_sections_by_partition() {
    let h = Harness::new().await;
    let instructions = h.client.peer_info().unwrap().instructions.clone().unwrap();
    assert!(instructions.contains("Typical flow"), "{instructions}");
    // Descriptions stay in list_docs, to keep the instructions short.
    let expected = "\
develop (building apps and the API):
- /documentation/search/: Search (3 pages)
- /documentation/faq/: FAQ (2 pages)

deploy (installation, Cloud, operations):
- /documentation/cloud/: Qdrant Cloud (1 page)

Other sources:
- /blog/: Blog (3 pages)
";
    assert!(instructions.ends_with(expected), "{instructions}");
}

#[tokio::test]
async fn instructions_without_documentation() {
    let h = Harness::serving(&["blog"], search_results()).await;
    let instructions = h.client.peer_info().unwrap().instructions.clone().unwrap();
    assert!(!instructions.contains("develop"), "{instructions}");
    assert!(instructions.ends_with("Other sources:\n- /blog/: Blog (3 pages)\n"));
}

#[tokio::test]
async fn instructions_with_unpartitioned_sections() {
    // A page without a partition, in a section the page index doesn't list.
    let h = Harness::with_backend(FixedBackend {
        pages: stored(&json!({"pages": [
            {"path": "/documentation/search/hybrid-queries/", "title": "Hybrid Queries"},
        ]})),
        ..FixedBackend::default()
    })
    .await;
    let instructions = h.client.peer_info().unwrap().instructions.clone().unwrap();
    assert!(
        instructions.contains("Other documentation:\n- /documentation/search/: Search (1 page)"),
        "{instructions}"
    );
}

#[tokio::test]
async fn get_page_reports_missing_page() {
    let h = Harness::with_backend(FixedBackend::default()).await;
    let result = h.call("get_page", json!({"page": "nope/nothing"})).await;
    assert!(is_error(&result));
    let text = text(&result);
    assert!(
        text.contains("No page at /documentation/nope/nothing/"),
        "{text}"
    );
    assert!(text.contains("Use search_docs or list_docs"), "{text}");
}

#[tokio::test]
async fn listed_page_without_sections_is_unavailable_not_missing() {
    let h = Harness::new().await;
    let result = h
        .call(
            "get_page",
            json!({"page": "/documentation/search/filtering/"}),
        )
        .await;
    assert!(is_error(&result));
    let text = text(&result);
    assert!(text.contains("can't be read right now"), "{text}");
    assert!(!text.contains("No page at"), "{text}");
}

#[tokio::test]
async fn missing_page_suggests_its_closest_ancestor() {
    let h = Harness::with_backend(FixedBackend::default()).await;
    let text = h
        .text("get_page", json!({"page": "search/nope/deeper"}))
        .await;
    let base = SITE_URL;
    assert!(
        text.contains(&format!("- Search: {base}/documentation/search/")),
        "{text}"
    );
}

#[tokio::test]
async fn missing_page_suggests_similar_paths_without_searching() {
    let h = Harness::new().await;
    let result = h.call("get_page", json!({"page": "concepts/filter"})).await;
    assert!(is_error(&result));
    let text = text(&result);
    let base = SITE_URL;
    assert!(
        text.starts_with("No page at /documentation/concepts/filter/."),
        "{text}"
    );
    assert!(
        text.contains(&format!(
            "- Filtering: {base}/documentation/search/filtering/"
        )),
        "{text}"
    );
    assert_eq!(
        text.matches("/documentation/search/filtering/").count(),
        1,
        "{text}"
    );
    assert!(!text.contains("/blog/"), "{text}");
    // Meaning is part of the page index query: search_docs isn't called.
    assert!(h.backend.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn missing_page_suggests_close_slugs_only() {
    let h = Harness::with_backend(FixedBackend::default()).await;
    let text = h
        .text("get_page", json!({"page": "/blog/new-posts/"}))
        .await;
    let base = SITE_URL;
    assert!(
        text.contains(&format!("- New Post: {base}/blog/new-post/")),
        "{text}"
    );
    assert!(!text.contains("/blog/old-post/"), "{text}");
}

#[tokio::test]
async fn list_docs_sections_and_sidebar_order() {
    let h = Harness::new().await;
    let sections = h.text("list_docs", json!({})).await;
    let pages = h.text("list_docs", json!({"section": "search"})).await;
    assert!(sections.find("/documentation/cloud/") < sections.find("/documentation/search/"));
    assert!(
        sections.contains("- /documentation/search/: Search (3 pages) [develop]. Search Qdrant."),
        "{sections}"
    );
    assert_eq!(
        pages.lines().collect::<Vec<_>>(),
        [
            "- Search: /documentation/search/ [develop]. Search Qdrant.",
            "  - Filtering: /documentation/search/filtering/ [develop]. Payload conditions.",
            "  - Hybrid Queries: /documentation/search/hybrid-queries/ [develop]. Fuse dense and sparse results.",
        ]
    );
}

#[tokio::test]
async fn section_without_a_page() {
    let h = Harness::new().await;
    let sections = h.text("list_docs", json!({})).await;
    let pages = h.text("list_docs", json!({"section": "faq"})).await;
    let result = h.call("get_page", json!({"page": "faq"})).await;
    assert!(
        sections.find("/documentation/search/")
            < sections.find("- /documentation/faq/: FAQ (2 pages) [develop]")
    );
    assert_eq!(
        pages.lines().next().unwrap(),
        "- FAQ: /documentation/faq/ [develop] (groups the pages that follow, no page of its own)"
    );
    assert!(is_error(&result));
    assert!(text(&result).contains("Call list_docs with section '/documentation/faq/'"));
    assert!(h.markdown_requests().is_empty());
}

#[tokio::test]
async fn list_docs_includes_other_sources_newest_first() {
    let h = Harness::new().await;
    let overview = h.text("list_docs", json!({})).await;
    let blog = h.text("list_docs", json!({"section": "/blog/"})).await;
    assert!(overview.find("/documentation/search/") < overview.find("- /blog/: Blog (3 pages)"));
    assert_eq!(
        blog.lines().collect::<Vec<_>>(),
        [
            "- Blog: /blog/. News.",
            "- New Post: /blog/new-post/ (2026-01-01)",
            "- Old Post: /blog/old-post/ (2023-01-01)",
        ]
    );
}

#[tokio::test]
async fn list_docs_pages_through_long_sections() {
    let h = Harness::new().await;
    let first = h
        .text("list_docs", json!({"section": "/blog/", "limit": 2}))
        .await;
    let last = h
        .text(
            "list_docs",
            json!({"section": "/blog/", "offset": 2, "limit": 2}),
        )
        .await;
    let past = h
        .call("list_docs", json!({"section": "/blog/", "offset": 3}))
        .await;
    assert_eq!(
        first.lines().last().unwrap(),
        "Showing 1-2 of 3. Call list_docs with offset 2 for the next ones."
    );
    assert_eq!(
        last.lines().collect::<Vec<_>>(),
        [
            "- Old Post: /blog/old-post/ (2023-01-01)",
            "",
            "Showing 3-3 of 3."
        ]
    );
    assert!(is_error(&past));
}

#[tokio::test]
async fn search_returns_markdown_by_default() {
    let h = Harness::new().await;
    let result = h
        .call(
            "search_docs",
            json!({"query": "hybrid", "sources": ["documentation"]}),
        )
        .await;
    assert!(result.structured_content.is_none());
    let expected = format!(
        "## Hybrid Queries\n{base}/documentation/search/hybrid-queries/ (documentation, develop)\n\
         - Hybrid Search\n- Fusion\n\n\
         ## Filtering\n{base}/documentation/search/filtering/ (documentation, develop)\n\
         - Filter and search",
        base = SITE_URL
    );
    assert_eq!(text(&result), expected);
}

#[tokio::test]
async fn search_without_results() {
    let h = Harness::with_backend(FixedBackend::default()).await;
    let body = h.text("search_docs", json!({"query": "nothing"})).await;
    assert!(body.starts_with("No results."), "{body}");
    let result = h
        .call("search_docs", json!({"query": "nothing", "format": "json"}))
        .await;
    assert_eq!(result.structured_content, Some(json!({"results": []})));
}

#[tokio::test]
async fn list_docs_as_json() {
    let h = Harness::new().await;
    let sections = h.call("list_docs", json!({"format": "json"})).await;
    let sections = &sections.structured_content.unwrap()["sections"];
    let search = sections
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["path"] == "/documentation/search/")
        .unwrap();
    assert_eq!(
        *search,
        json!({"path": "/documentation/search/", "title": "Search", "pages": 3,
               "partition": "develop", "description": "Search Qdrant."})
    );

    let faq = h
        .call("list_docs", json!({"section": "faq", "format": "json"}))
        .await;
    assert_eq!(
        faq.structured_content.unwrap(),
        json!({
            "section": "/documentation/faq/", "total": 2, "offset": 0,
            "pages": [
                {"title": "FAQ", "path": "/documentation/faq/", "depth": 0,
                 "partition": "develop", "has_page": false},
                {"title": "Basics", "path": "/documentation/faq/basics/", "depth": 1},
            ],
        })
    );

    let blog = h
        .call(
            "list_docs",
            json!({"section": "/blog/", "limit": 2, "format": "json"}),
        )
        .await;
    let blog = blog.structured_content.unwrap();
    assert_eq!(blog["next_offset"], 2);
    assert_eq!(blog["pages"][1]["date"], "2026-01-01");
    assert!(blog["pages"][1].get("depth").is_none());
}

#[tokio::test]
async fn search_snippets_in_markdown() {
    let h = Harness::new().await;
    let body = h
        .text(
            "search_snippets",
            json!({"query": "hybrid query", "language": "Py"}),
        )
        .await;
    assert_eq!(
        *h.backend.snippet_requests.lock().unwrap(),
        [Some("python".to_string())]
    );
    // The same code on another page is left out.
    let expected = format!(
        "## Hybrid Queries (python)\n{base}/documentation/search/hybrid-queries/\n\n\
         Runs a hybrid query in python.\n\n\
         ```python\nclient.query_points(prefetch=[...])\n```\n\n\
         ## New Post (python)\n{base}/blog/new-post/\n\n\
         Runs a hybrid query in python.\n\n\
         ````python\nprint(```)\n````",
        base = SITE_URL
    );
    assert_eq!(body, expected);
}

#[tokio::test]
async fn search_snippets_as_json_in_all_languages() {
    let h = Harness::new().await;
    let result = h
        .call(
            "search_snippets",
            json!({"query": "hybrid query", "limit": 2, "format": "json"}),
        )
        .await;
    assert_eq!(*h.backend.snippet_requests.lock().unwrap(), [None]);
    let hits = results(&result);
    assert_eq!(
        hits[1],
        json!({
            "language": "rust",
            "code": "client.query(QueryPointsBuilder::new(\"c\")).await?;",
            "description": "Runs a hybrid query in rust.",
            "title": "Hybrid Queries",
            "url": format!("{}/documentation/search/hybrid-queries/", SITE_URL),
        })
    );
    assert_eq!(hits.len(), 2);
}

#[tokio::test]
async fn search_snippets_only_from_served_sources() {
    let h = Harness::serving(&["blog"], search_results()).await;
    let result = h
        .call(
            "search_snippets",
            json!({"query": "hybrid", "format": "json"}),
        )
        .await;
    let urls: Vec<Value> = results(&result).iter().map(|h| h["url"].clone()).collect();
    assert_eq!(urls, [json!(format!("{}/blog/new-post/", SITE_URL))]);

    let empty = h
        .text(
            "search_snippets",
            json!({"query": "hybrid", "language": "go"}),
        )
        .await;
    assert!(empty.starts_with("No snippets found."), "{empty}");
}

#[tokio::test]
async fn format_is_a_plain_enum_argument() {
    let h = Harness::new().await;
    for tool in h.client.list_all_tools().await.unwrap() {
        let format = &tool.input_schema["properties"]["format"];
        if tool.name == "get_page" || tool.name == "submit_feedback" {
            assert!(format.is_null(), "{format}");
        } else {
            assert_eq!(format["enum"], json!(["markdown", "json"]), "{}", tool.name);
        }
    }
}

#[tokio::test]
async fn submit_feedback_records_feedback_on_the_canonical_page() {
    let h = Harness::new().await;
    let result = h
        .call(
            "submit_feedback",
            json!({"page": "https://qdrant.tech/documentation/concepts/hybrid-queries/",
                   "feedback": "  The RRF example uses a parameter removed in 1.15.  "}),
        )
        .await;
    assert!(!is_error(&result), "{}", text(&result));
    let reply = text(&result);
    assert!(reply.contains("Hybrid Queries"), "{reply}");
    let stored = h.stored_feedback();
    assert_eq!(stored.len(), 1);
    // Recorded at the page's current path, not the old one asked for.
    assert_eq!(stored[0].page, "/documentation/search/hybrid-queries/");
    assert_eq!(
        stored[0].url,
        format!("{SITE_URL}/documentation/search/hybrid-queries/")
    );
    assert_eq!(stored[0].title.as_deref(), Some("Hybrid Queries"));
    assert_eq!(
        stored[0].feedback,
        "The RRF example uses a parameter removed in 1.15."
    );
    assert!(!stored[0].dismissed);
    // The client named itself in initialize.
    assert!(stored[0].client.is_some());
}

#[tokio::test]
async fn submit_feedback_accepts_a_listed_page_without_markdown() {
    let h = Harness::new().await;
    let result = h
        .call(
            "submit_feedback",
            json!({"page": "search/filtering", "feedback": "Missing nested filters."}),
        )
        .await;
    assert!(!is_error(&result), "{}", text(&result));
    assert_eq!(
        h.stored_feedback()[0].page,
        "/documentation/search/filtering/"
    );
}

#[tokio::test]
async fn submit_feedback_refuses_missing_pages_and_empty_feedback() {
    let h = Harness::new().await;
    let missing = h
        .call(
            "submit_feedback",
            json!({"page": "/documentation/nope/", "feedback": "Wrong."}),
        )
        .await;
    assert!(is_error(&missing));
    assert!(text(&missing).contains("No page at /documentation/nope/"));

    let empty = h
        .call(
            "submit_feedback",
            json!({"page": "search/hybrid-queries", "feedback": "   "}),
        )
        .await;
    assert!(is_error(&empty));
    assert!(text(&empty).contains("empty"), "{}", text(&empty));

    let long = h
        .call(
            "submit_feedback",
            json!({"page": "search/hybrid-queries", "feedback": "x".repeat(5001)}),
        )
        .await;
    assert!(is_error(&long));

    let elsewhere = h
        .call(
            "submit_feedback",
            json!({"page": "https://example.com/docs/", "feedback": "Wrong."}),
        )
        .await;
    assert!(is_error(&elsewhere));
    assert!(h.stored_feedback().is_empty());
}

#[tokio::test]
async fn submit_feedback_is_not_read_only() {
    let h = Harness::new().await;
    let tools = h.client.list_all_tools().await.unwrap();
    let tool = tools
        .iter()
        .find(|t| t.name == "submit_feedback")
        .expect("submit_feedback is listed");
    let annotations = tool.annotations.as_ref().unwrap();
    assert_eq!(annotations.read_only_hint, Some(false));
    assert_eq!(annotations.destructive_hint, Some(false));
    let required = tool.input_schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("page")) && required.contains(&json!("feedback")));
    // Every other tool stays read-only.
    for other in tools.iter().filter(|t| t.name != "submit_feedback") {
        assert_eq!(
            other.annotations.as_ref().unwrap().read_only_hint,
            Some(true),
            "{}",
            other.name
        );
    }
}
