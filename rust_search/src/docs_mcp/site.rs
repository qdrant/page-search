//! Pages and the page index, read from the published site.
//!
//! Nothing here keeps its own copy of the content. Pages come from the Markdown
//! that qdrant.tech publishes for every page at `<url>/index.md`, and the page
//! list from `/docs-manifest.json`, which the landing_page repository builds
//! from `qdrant-landing/layouts/index.docsmanifest.json`.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use regex::Regex;
use serde::Deserialize;
use tokio::sync::Mutex;
use url::Url;

use super::config::{Config, SOURCES};

pub const DOCS_PREFIX: &str = "/documentation/";
/// Sources listed newest first rather than in sidebar order.
const DATED_SOURCES: [&str; 2] = ["articles", "blog"];
const PUBLIC_HOSTS: [&str; 2] = ["qdrant.tech", "www.qdrant.tech"];
/// Pointers that `single.markdown.md` prepends to every page. An agent already
/// connected to this server doesn't need them.
const PREAMBLE_MARKERS: [&str; 2] = ["skills.qdrant.tech", "llms.txt"];

static LLMS_ENTRY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^- \[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)(?::\s*(?P<description>.*))?$").unwrap()
});
static INDEX_FILE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/index\.(md|html)$").unwrap());
static REPEATED_SLASHES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/{2,}").unwrap());

/// One entry of the page index.
#[derive(Clone, Debug, Deserialize)]
pub struct DocPage {
    pub path: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "page_kind")]
    pub kind: String,
    #[serde(default)]
    pub partition: String,
    #[serde(default)]
    pub weight: i64,
    #[serde(default)]
    pub aliases: Option<Vec<String>>,
    pub date: Option<String>,
    pub lastmod: Option<String>,
    pub external_url: Option<String>,
    /// False for a section that only groups pages and has no page of its own.
    #[serde(default = "yes")]
    pub rendered: bool,
}

fn page_kind() -> String {
    "page".to_string()
}

fn yes() -> bool {
    true
}

impl DocPage {
    fn new(path: String, title: String) -> Self {
        Self {
            path,
            title,
            description: String::new(),
            kind: page_kind(),
            partition: String::new(),
            weight: 0,
            aliases: None,
            date: None,
            lastmod: None,
            external_url: None,
            rendered: true,
        }
    }

    pub fn source(&self) -> &str {
        source_of(&self.path)
    }
}

pub struct Manifest {
    pub pages: HashMap<String, DocPage>,
    pub aliases: HashMap<String, String>,
}

impl Manifest {
    fn resolve<'a>(&'a self, path: &'a str) -> &'a str {
        self.aliases.get(path).map(String::as_str).unwrap_or(path)
    }
}

/// The first segment of a site path: `documentation`, `blog`, and so on.
pub fn source_of(path: &str) -> &str {
    path.trim_matches('/').split('/').next().unwrap_or("")
}

/// Turn a URL, path, or `index.md` link into a site path such as
/// `/documentation/search/`.
///
/// Paths that don't start with a known source, such as "search/hybrid-queries",
/// are taken to be documentation paths. Only qdrant.tech and `extra_host` URLs
/// are accepted, so a tool argument can never make the server fetch from
/// another host.
pub fn normalize_path(reference: &str, extra_host: Option<&str>) -> Result<String, String> {
    let reference = reference.trim();
    if reference.is_empty() {
        return Err("Empty page reference.".to_string());
    }
    let url = if reference.contains("://") {
        Url::parse(reference)
    } else {
        Url::parse(&format!(
            "https://qdrant.tech/{}",
            reference.trim_start_matches('/')
        ))
    }
    .map_err(|e| format!("Not a valid page reference: {e}."))?;

    let host = url.host_str().unwrap_or_default();
    if !PUBLIC_HOSTS.contains(&host) && Some(host) != extra_host {
        return Err(format!("Only qdrant.tech pages can be read, not {host}."));
    }

    let mut path = INDEX_FILE.replace(url.path(), "/").into_owned();
    if !path.ends_with('/') {
        path.push('/');
    }
    let mut path = format!(
        "/{}",
        REPEATED_SLASHES
            .replace_all(&path, "/")
            .trim_start_matches('/')
    );
    if !SOURCES.contains(&source_of(&path)) {
        path = format!("{DOCS_PREFIX}{}", path.trim_start_matches('/'));
    }
    Ok(path)
}

/// Refuse a path in a source this server doesn't serve.
pub fn check_enabled(path: &str, enabled: &[String]) -> Result<(), String> {
    let source = source_of(path);
    if enabled.iter().any(|s| s == source) {
        Ok(())
    } else {
        Err(format!(
            "This server doesn't serve /{source}/ pages. It serves: {}.",
            enabled.join(", ")
        ))
    }
}

pub fn strip_preamble(markdown: &str) -> String {
    let lines: Vec<&str> = markdown.lines().collect();
    let start = lines
        .iter()
        .position(|line| {
            let preamble =
                line.starts_with('>') && PREAMBLE_MARKERS.iter().any(|m| line.contains(m));
            !(line.trim().is_empty() || preamble)
        })
        .unwrap_or(lines.len());
    format!("{}\n", lines[start..].join("\n").trim())
}

/// A section missing from the page index, as in the llms.txt fallback or a
/// manifest from before unrendered sections were listed. Name it after its
/// folder and place it with its first page.
fn unlisted_section(manifest: &Manifest, path: &str) -> DocPage {
    let slug = path
        .trim_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or_default();
    let mut title = slug.replace('-', " ");
    if let Some(first) = title.get(..1) {
        title = first.to_uppercase() + &title[1..];
    }
    let mut page = DocPage::new(path.to_string(), title);
    page.kind = "section".to_string();
    page.weight = manifest
        .pages
        .values()
        .filter(|p| p.path.starts_with(path))
        .map(|p| p.weight)
        .min()
        .unwrap_or_default();
    page
}

pub struct SiteClient {
    config: Config,
    http: reqwest::Client,
    base_host: Option<String>,
    manifest: Mutex<Option<(Instant, Arc<Manifest>)>>,
}

impl SiteClient {
    pub fn new(config: Config) -> Self {
        let http = reqwest::Client::builder()
            .timeout(config.http_timeout)
            .user_agent(concat!("qdrant-docs-mcp/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("failed to build the HTTP client");
        let base_host = Url::parse(&config.content_base_url)
            .ok()
            .and_then(|url| url.host_str().map(str::to_string));
        Self {
            config,
            http,
            base_host,
            manifest: Mutex::new(None),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.config.content_base_url, path)
    }

    pub fn sources(&self) -> &[String] {
        &self.config.sources
    }

    pub fn normalize(&self, reference: &str) -> Result<String, String> {
        let path = normalize_path(reference, self.base_host.as_deref())?;
        check_enabled(&path, &self.config.sources)?;
        Ok(path)
    }

    // Manifest

    pub async fn manifest(&self) -> Result<Arc<Manifest>, String> {
        // Holding the lock while loading means concurrent requests wait for one
        // fetch instead of each starting their own.
        let mut cached = self.manifest.lock().await;
        if let Some((loaded_at, manifest)) = cached.as_ref() {
            if loaded_at.elapsed() < self.config.manifest_ttl {
                return Ok(manifest.clone());
            }
        }
        let manifest = Arc::new(
            self.load_manifest()
                .await
                .map_err(|e| format!("Could not load the page index: {e}"))?,
        );
        *cached = Some((Instant::now(), manifest.clone()));
        Ok(manifest)
    }

    async fn load_manifest(&self) -> reqwest::Result<Manifest> {
        let response = self
            .http
            .get(self.url("/docs-manifest.json"))
            .send()
            .await?;
        if response.status() != reqwest::StatusCode::NOT_FOUND {
            #[derive(Deserialize)]
            struct ManifestFile {
                pages: Vec<DocPage>,
            }
            let file: ManifestFile = response.error_for_status()?.json().await?;
            return Ok(self.parse_manifest(file.pages));
        }
        // The manifest is new; sites built before it was added only have llms.txt.
        let text = self
            .http
            .get(self.url("/llms.txt"))
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        Ok(self.parse_llms_txt(&text))
    }

    fn parse_manifest(&self, entries: Vec<DocPage>) -> Manifest {
        let mut pages = HashMap::new();
        let mut aliases = HashMap::new();
        for page in entries {
            if !self.config.sources.iter().any(|s| s == page.source()) {
                continue;
            }
            for alias in page.aliases.iter().flatten() {
                aliases.insert(
                    format!("{}/", alias.trim_end_matches('/')),
                    page.path.clone(),
                );
            }
            pages.insert(page.path.clone(), page);
        }
        Manifest { pages, aliases }
    }

    fn parse_llms_txt(&self, text: &str) -> Manifest {
        let mut pages = HashMap::new();
        for line in text.lines() {
            let Some(entry) = LLMS_ENTRY.captures(line.trim()) else {
                continue;
            };
            let Ok(url) = Url::parse(&entry["url"]) else {
                continue;
            };
            let path = url.path().trim_end_matches("index.md").to_string();
            if !self.config.sources.iter().any(|s| s == source_of(&path)) {
                continue;
            }
            let mut page = DocPage::new(path.clone(), entry["title"].to_string());
            page.description = entry
                .name("description")
                .map(|d| d.as_str().trim().to_string())
                .unwrap_or_default();
            page.weight = pages.len() as i64;
            pages.insert(path, page);
        }
        Manifest {
            pages,
            aliases: HashMap::new(),
        }
    }

    // Tools

    /// The index entry (if any), the canonical URL, and the Markdown body of a page.
    pub async fn get_page(
        &self,
        reference: &str,
    ) -> Result<(Option<DocPage>, String, String), String> {
        let path = self.normalize(reference)?;
        let manifest = self.manifest().await?;
        let path = manifest.resolve(&path).to_string();
        let page = manifest.pages.get(&path).cloned();
        if let Some(page) = &page {
            if !page.rendered {
                return Err(format!(
                    "{path} is a section without a page of its own. \
                     Call list_docs with section '{path}' to see its pages."
                ));
            }
            if let Some(external) = &page.external_url {
                return Err(format!(
                    "{} is hosted outside qdrant.tech: {external}",
                    page.title
                ));
            }
        }
        let fetch_error = |e: reqwest::Error| format!("Could not fetch the page: {e}");
        let response = self
            .http
            .get(self.url(&format!("{path}index.md")))
            .send()
            .await
            .map_err(fetch_error)?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(format!(
                "No page at {path}. Use search_docs or list_docs to find the right one."
            ));
        }
        let body = response
            .error_for_status()
            .map_err(fetch_error)?
            .text()
            .await
            .map_err(fetch_error)?;
        Ok((page, self.url(&path), strip_preamble(&body)))
    }

    /// Top-level documentation sections, then each other source, with page counts.
    pub async fn list_sections(&self) -> Result<Vec<(DocPage, usize)>, String> {
        let manifest = self.manifest().await?;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for path in manifest.pages.keys() {
            let segments: Vec<&str> = path.trim_matches('/').split('/').collect();
            let top = if segments[0] == "documentation" {
                if segments.len() < 2 {
                    continue;
                }
                format!("/{}/", segments[..2].join("/"))
            } else {
                format!("/{}/", segments[0])
            };
            *counts.entry(top).or_default() += 1;
        }

        let source_index = |path: &str| {
            SOURCES
                .iter()
                .position(|s| *s == source_of(path))
                .unwrap_or(usize::MAX)
        };
        let mut sections: Vec<(DocPage, usize)> = counts
            .into_iter()
            .map(|(path, count)| {
                let page = manifest
                    .pages
                    .get(&path)
                    .cloned()
                    .unwrap_or_else(|| unlisted_section(&manifest, &path));
                (page, count)
            })
            .collect();
        sections.sort_by(|(a, _), (b, _)| {
            (source_index(&a.path), a.weight, &a.path).cmp(&(
                source_index(&b.path),
                b.weight,
                &b.path,
            ))
        });
        Ok(sections)
    }

    /// Pages under a section, and whether they are nested (sidebar order) or
    /// flat (newest first).
    pub async fn list_pages(&self, section: &str) -> Result<(Vec<DocPage>, bool), String> {
        let prefix = self.normalize(section)?;
        let manifest = self.manifest().await?;
        let mut pages: Vec<DocPage> = manifest
            .pages
            .values()
            .filter(|p| p.path.starts_with(&prefix))
            .cloned()
            .collect();
        if pages.is_empty() {
            return Err(format!(
                "No section at {prefix}. Call list_docs without arguments to see the sections."
            ));
        }

        if DATED_SOURCES.contains(&source_of(&prefix)) {
            // Section pages first, then posts newest first. Both sorts are stable.
            pages.sort_by(|a, b| b.date.cmp(&a.date).then_with(|| a.path.cmp(&b.path)));
            pages.sort_by_key(|p| p.kind != "section");
            return Ok((pages, false));
        }

        // Order like the sidebar: by weight among siblings, children after their parent.
        let sidebar_key = |page: &DocPage| -> Vec<(i64, String)> {
            let segments: Vec<&str> = page.path.trim_matches('/').split('/').collect();
            (1..=segments.len())
                .map(|i| {
                    let ancestor = format!("/{}/", segments[..i].join("/"));
                    let weight = manifest.pages.get(&ancestor).map(|p| p.weight).unwrap_or(0);
                    (weight, segments[i - 1].to_string())
                })
                .collect()
        };
        pages.sort_by_cached_key(sidebar_key);
        Ok((pages, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_paths() {
        for (reference, expected) in [
            (
                "https://qdrant.tech/documentation/search/",
                "/documentation/search/",
            ),
            (
                "https://qdrant.tech/documentation/search/index.md",
                "/documentation/search/",
            ),
            ("/documentation/search#filters", "/documentation/search/"),
            (
                "search/hybrid-queries",
                "/documentation/search/hybrid-queries/",
            ),
            ("https://qdrant.tech/blog//x", "/blog/x/"),
        ] {
            assert_eq!(
                normalize_path(reference, None).unwrap(),
                expected,
                "{reference}"
            );
        }
    }

    #[test]
    fn rejects_other_hosts() {
        let error = normalize_path("https://example.com/documentation/", None).unwrap_err();
        assert!(error.contains("not example.com"), "{error}");
        assert!(normalize_path("http://localhost:1313/blog/", Some("localhost")).is_ok());
    }

    #[test]
    fn strips_only_the_preamble() {
        assert_eq!(
            strip_preamble("> Use https://qdrant.tech/llms.txt\n\n> A real note\n\nBody"),
            "> A real note\n\nBody\n"
        );
    }
}
