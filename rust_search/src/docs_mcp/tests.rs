//! Tests that run the real server in-process, with an MCP client, against a
//! fake qdrant.tech and canned search results.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use qdrant_client::qdrant::ScoredPoint;
use qdrant_client::Payload;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::RunningService;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::{DocsServer, SearchBackend, Source};
use crate::docs_mcp::config::{Config, SOURCES};
use crate::snippets::Snippet;

/// Canned search results per source, recording the source and partitions of
/// each request, and canned snippets, recording the language of each request.
#[derive(Default)]
pub struct FixedSearch {
    results: HashMap<String, Vec<ScoredPoint>>,
    pub requests: Mutex<Vec<(String, Vec<String>)>>,
    snippets: Vec<Value>,
    pub snippet_requests: Mutex<Vec<Option<String>>>,
}

impl FixedSearch {
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

fn search_results() -> FixedSearch {
    FixedSearch {
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
        ..FixedSearch::default()
    }
}

fn manifest() -> Value {
    json!({
        "version": 1,
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
            {"path": "/blog/new-post/", "title": "New Post", "kind": "page", "weight": 0, "date": "2026-01-01", "aliases": null},
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

const LLMS_TXT: &str = "\
# https://qdrant.tech/
## Documentation
- [Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/index.md): Fuse results.
- [Blog post](https://qdrant.tech/blog/x/index.md): Not docs.
";

/// A fake qdrant.tech, with or without the manifest.
async fn fake_site(with_manifest: bool) -> MockServer {
    let site = MockServer::start().await;
    let manifest = if with_manifest {
        ResponseTemplate::new(200).set_body_json(manifest())
    } else {
        ResponseTemplate::new(404)
    };
    Mock::given(method("GET"))
        .and(path("/docs-manifest.json"))
        .respond_with(manifest)
        .mount(&site)
        .await;
    Mock::given(method("GET"))
        .and(path("/llms.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_string(LLMS_TXT))
        .mount(&site)
        .await;
    Mock::given(method("GET"))
        .and(path("/documentation/search/hybrid-queries/index.md"))
        .respond_with(ResponseTemplate::new(200).set_body_string(PAGE_MD))
        .mount(&site)
        .await;
    site
}

struct Harness {
    client: RunningService<RoleClient, ()>,
    site: MockServer,
    search: Arc<FixedSearch>,
}

impl Harness {
    async fn new(with_manifest: bool) -> Self {
        Self::with_search(with_manifest, search_results()).await
    }

    async fn with_search(with_manifest: bool, search: FixedSearch) -> Self {
        Self::serving(&SOURCES, with_manifest, search).await
    }

    /// A server configured to serve only `sources`.
    async fn serving(sources: &[&str], with_manifest: bool, search: FixedSearch) -> Self {
        let site = fake_site(with_manifest).await;
        let config = Config {
            content_base_url: site.uri(),
            sources: sources.iter().map(|s| s.to_string()).collect(),
            ..Config::default()
        };
        let search = Arc::new(search);
        let server = DocsServer::new(config, SearchBackend::Fixed(search.clone()));
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
            site,
            search,
        }
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

    fn base(&self) -> String {
        self.site.uri()
    }

    async fn site_requests(&self) -> Vec<String> {
        let requests = self.site.received_requests().await.unwrap_or_default();
        requests.iter().map(|r| r.url.path().to_string()).collect()
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
    let h = Harness::new(true).await;
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
        format!("{}/documentation/search/hybrid-queries/index.md", h.base())
    );
    assert_eq!(
        *h.search.requests.lock().unwrap(),
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
    let h = Harness::with_search(true, search).await;
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
    let h = Harness::new(true).await;
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
        h.search.requests.lock().unwrap().iter().cloned().collect();
    assert_eq!(requests["documentation"], ["develop"]);
    assert!(requests["blog"].is_empty());
}

#[tokio::test]
async fn search_can_pick_sources() {
    let h = Harness::new(true).await;
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
        *h.search.requests.lock().unwrap(),
        [("blog".to_string(), vec![])]
    );
}

#[tokio::test]
async fn sources_are_a_plain_enum_argument() {
    let h = Harness::new(true).await;
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
    assert!(h.search.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn optional_arguments_have_a_single_type() {
    let h = Harness::new(true).await;
    for tool in h.client.list_all_tools().await.unwrap() {
        let schema = Value::Object((*tool.input_schema).clone());
        let required = schema["required"].as_array().cloned().unwrap_or_default();
        for (name, property) in schema["properties"].as_object().unwrap() {
            assert!(
                !property["type"].is_array() && property.get("anyOf").is_none(),
                "{}.{name}: {property}",
                tool.name
            );
            if name != "query" && name != "page" {
                assert!(!required.contains(&json!(name)), "{}.{name}", tool.name);
            }
        }
    }
}

#[tokio::test]
async fn disabled_sources_are_refused() {
    let h = Harness::serving(&["documentation"], true, search_results()).await;

    // Without `sources`, only the served sources are searched.
    h.call("search_docs", json!({"query": "hybrid"})).await;
    assert_eq!(
        *h.search.requests.lock().unwrap(),
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
    assert!(!h
        .site_requests()
        .await
        .iter()
        .any(|p| p.starts_with("/blog/")));

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
    let h = Harness::new(true).await;
    let body = h
        .text(
            "get_page",
            json!({"page": "https://qdrant.tech/documentation/concepts/hybrid-queries/"}),
        )
        .await;
    let expected = format!(
        "Title: Hybrid Queries\nSource: {}/documentation/search/hybrid-queries/",
        h.base()
    );
    assert!(body.starts_with(&expected), "{body}");
    assert!(body.contains("Last updated: 2026-09-01"));
    assert!(body.contains("Partition: develop"), "{body}");
    assert!(!body.contains("skills.qdrant.tech"));
    assert!(body.contains("# Hybrid and Multi-Stage Queries"));
}

#[tokio::test]
async fn instructions_group_sections_by_partition() {
    let h = Harness::new(true).await;
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
    let h = Harness::serving(&["blog"], true, search_results()).await;
    let instructions = h.client.peer_info().unwrap().instructions.clone().unwrap();
    assert!(!instructions.contains("develop"), "{instructions}");
    assert!(instructions.ends_with("Other sources:\n- /blog/: Blog (3 pages)\n"));
}

#[tokio::test]
async fn instructions_with_unpartitioned_sections() {
    // llms.txt has no partitions.
    let h = Harness::new(false).await;
    let instructions = h.client.peer_info().unwrap().instructions.clone().unwrap();
    assert!(
        instructions.contains("Other documentation:\n- /documentation/search/: Search (1 page)"),
        "{instructions}"
    );
}

#[tokio::test]
async fn get_page_reports_missing_page() {
    let h = Harness::new(true).await;
    let result = h.call("get_page", json!({"page": "search/nope"})).await;
    assert!(is_error(&result));
    assert!(text(&result).contains("No page at /documentation/search/nope/"));
}

#[tokio::test]
async fn list_docs_sections_and_sidebar_order() {
    let h = Harness::new(true).await;
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
    let h = Harness::new(true).await;
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
    assert!(!h
        .site_requests()
        .await
        .iter()
        .any(|p| p.ends_with("index.md")));
}

#[tokio::test]
async fn falls_back_to_llms_txt_without_manifest() {
    let h = Harness::new(false).await;
    let sections = h.text("list_docs", json!({})).await;
    let pages = h.text("list_docs", json!({"section": "search"})).await;
    // llms.txt has no section entries, so the section is named after its folder.
    assert!(
        sections.contains("- /documentation/search/: Search (1 page)"),
        "{sections}"
    );
    assert_eq!(
        pages,
        "- Hybrid Queries: /documentation/search/hybrid-queries/. Fuse results."
    );
}

#[tokio::test]
async fn list_docs_includes_other_sources_newest_first() {
    let h = Harness::new(true).await;
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
    let h = Harness::new(true).await;
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
    let h = Harness::new(true).await;
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
        base = h.base()
    );
    assert_eq!(text(&result), expected);
}

#[tokio::test]
async fn search_without_results() {
    let h = Harness::with_search(true, FixedSearch::default()).await;
    let body = h.text("search_docs", json!({"query": "nothing"})).await;
    assert!(body.starts_with("No results."), "{body}");
    let result = h
        .call("search_docs", json!({"query": "nothing", "format": "json"}))
        .await;
    assert_eq!(result.structured_content, Some(json!({"results": []})));
}

#[tokio::test]
async fn list_docs_as_json() {
    let h = Harness::new(true).await;
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
    let h = Harness::new(true).await;
    let body = h
        .text(
            "search_snippets",
            json!({"query": "hybrid query", "language": "Py"}),
        )
        .await;
    assert_eq!(
        *h.search.snippet_requests.lock().unwrap(),
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
        base = h.base()
    );
    assert_eq!(body, expected);
}

#[tokio::test]
async fn search_snippets_as_json_in_all_languages() {
    let h = Harness::new(true).await;
    let result = h
        .call(
            "search_snippets",
            json!({"query": "hybrid query", "limit": 2, "format": "json"}),
        )
        .await;
    assert_eq!(*h.search.snippet_requests.lock().unwrap(), [None]);
    let hits = results(&result);
    assert_eq!(
        hits[1],
        json!({
            "language": "rust",
            "code": "client.query(QueryPointsBuilder::new(\"c\")).await?;",
            "description": "Runs a hybrid query in rust.",
            "title": "Hybrid Queries",
            "url": format!("{}/documentation/search/hybrid-queries/", h.base()),
        })
    );
    assert_eq!(hits.len(), 2);
}

#[tokio::test]
async fn search_snippets_only_from_served_sources() {
    let h = Harness::serving(&["blog"], true, search_results()).await;
    let result = h
        .call(
            "search_snippets",
            json!({"query": "hybrid", "format": "json"}),
        )
        .await;
    let urls: Vec<Value> = results(&result).iter().map(|h| h["url"].clone()).collect();
    assert_eq!(urls, [json!(format!("{}/blog/new-post/", h.base()))]);

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
    let h = Harness::new(true).await;
    for tool in h.client.list_all_tools().await.unwrap() {
        let format = &tool.input_schema["properties"]["format"];
        if tool.name == "get_page" {
            assert!(format.is_null(), "{format}");
        } else {
            assert_eq!(format["enum"], json!(["markdown", "json"]), "{}", tool.name);
        }
    }
}

#[tokio::test]
async fn get_page_reads_snippet_pages() {
    let h = Harness::new(true).await;
    Mock::given(method("GET"))
        .and(path("/documentation/snippets/query-points/hybrid-rrf/index.md"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            "# Code snippet: query-points/hybrid-rrf\n\n## Qdrant Server\n\n```go\nclient.Query()\n```\n",
        ))
        .mount(&h.site)
        .await;
    // Snippet pages aren't in the page index, so there is no title.
    let body = h
        .text(
            "get_page",
            json!({"page": "https://qdrant.tech/documentation/snippets/query-points/hybrid-rrf/index.md"}),
        )
        .await;
    assert!(
        body.starts_with(&format!(
            "Source: {}/documentation/snippets/query-points/hybrid-rrf/",
            h.base()
        )),
        "{body}"
    );
    assert!(body.contains("```go\nclient.Query()\n```"));
}
