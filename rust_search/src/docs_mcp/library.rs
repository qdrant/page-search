//! The pages the server lists and reads, from Qdrant: the page index from
//! the `pages` collection, and each page's Markdown from its sections in the
//! `sections` collection. Nothing here requests the site.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use regex::Regex;
use tokio::sync::Mutex;
use url::Url;

use super::config::{Config, SOURCES};
use super::server::Backend;
use crate::pages::{page_key, StoredPage};

/// Where the pages are published.
pub const SITE_URL: &str = "https://qdrant.tech";
pub const DOCS_PREFIX: &str = "/documentation/";
/// Sources listed newest first rather than in sidebar order.
const DATED_SOURCES: [&str; 2] = ["articles", "blog"];
const PUBLIC_HOSTS: [&str; 2] = ["qdrant.tech", "www.qdrant.tech"];
/// Pointers that `single.markdown.md` prepends to every page. An agent already
/// connected to this server doesn't need them.
const PREAMBLE_MARKERS: [&str; 2] = ["skills.qdrant.tech", "llms.txt"];
/// How long the page index is kept before it's read again. Qdrant has a new
/// one after every nightly indexing.
const PAGE_INDEX_TTL: Duration = Duration::from_secs(300);

static INDEX_FILE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/index\.(md|html)$").unwrap());
static REPEATED_SLASHES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/{2,}").unwrap());

/// Why a page couldn't be read.
#[derive(Debug)]
pub enum PageError {
    /// Nothing is published at this path.
    NotFound(String),
    Other(String),
}

impl From<String> for PageError {
    fn from(message: String) -> Self {
        Self::Other(message)
    }
}

/// One entry of the page index.
#[derive(Clone, Debug)]
pub struct DocPage {
    /// Such as `/documentation/search/filtering/`.
    pub path: String,
    pub title: String,
    pub description: String,
    pub kind: String,
    pub partition: String,
    pub weight: i64,
    pub aliases: Vec<String>,
    pub date: Option<String>,
    pub lastmod: Option<String>,
    pub external_url: Option<String>,
    /// False for a section that only groups pages and has no page of its own.
    pub rendered: bool,
}

impl DocPage {
    fn new(path: String, title: String) -> Self {
        Self::from(StoredPage {
            page: page_key(&path),
            title,
            meta: Default::default(),
        })
    }

    pub fn source(&self) -> &str {
        source_of(&self.path)
    }
}

impl From<StoredPage> for DocPage {
    fn from(stored: StoredPage) -> Self {
        let meta = stored.meta;
        Self {
            path: site_path(&stored.page),
            title: stored.title,
            description: meta.description,
            kind: meta.kind,
            partition: meta.partition,
            weight: meta.weight,
            aliases: meta.aliases.iter().map(|a| site_path(a)).collect(),
            date: meta.date,
            lastmod: meta.lastmod,
            external_url: meta.external_url,
            rendered: meta.rendered,
        }
    }
}

/// A `page` payload value as a site path: `/documentation/search/`.
fn site_path(page: &str) -> String {
    format!("/{page}/")
}

/// The page index, by path and by old path.
pub struct PageIndex {
    pub pages: HashMap<String, DocPage>,
    pub aliases: HashMap<String, String>,
}

impl PageIndex {
    pub fn resolve<'a>(&'a self, path: &'a str) -> &'a str {
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
/// are taken to be documentation paths. Only qdrant.tech URLs are accepted.
pub fn normalize_path(reference: &str) -> Result<String, String> {
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
    if !PUBLIC_HOSTS.contains(&host) {
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

/// A section missing from the page index. Name it after its folder and place
/// it with its first page.
fn unlisted_section(index: &PageIndex, path: &str) -> DocPage {
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
    if let Some(first) = index
        .pages
        .values()
        .filter(|p| p.path.starts_with(path))
        .min_by_key(|p| (p.weight, &p.path))
    {
        page.weight = first.weight;
        page.partition = first.partition.clone();
    }
    page
}

pub struct Library {
    sources: Vec<String>,
    backend: Backend,
    index: Mutex<Option<(Instant, Arc<PageIndex>)>>,
}

impl Library {
    pub fn new(config: Config, backend: Backend) -> Self {
        Self {
            sources: config.sources,
            backend,
            index: Mutex::new(None),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{SITE_URL}{path}")
    }

    pub fn sources(&self) -> &[String] {
        &self.sources
    }

    pub fn normalize(&self, reference: &str) -> Result<String, String> {
        let path = normalize_path(reference)?;
        check_enabled(&path, &self.sources)?;
        Ok(path)
    }

    // Page index

    pub async fn page_index(&self) -> Result<Arc<PageIndex>, String> {
        // Holding the lock while loading means concurrent requests wait for one
        // read instead of each starting their own.
        let mut cached = self.index.lock().await;
        if let Some((loaded_at, index)) = cached.as_ref() {
            if loaded_at.elapsed() < PAGE_INDEX_TTL {
                return Ok(index.clone());
            }
        }
        let pages = self
            .backend
            .pages()
            .await
            .map_err(|e| format!("Could not load the page index: {e}"))?;
        let index = Arc::new(self.build_index(pages));
        *cached = Some((Instant::now(), index.clone()));
        Ok(index)
    }

    fn build_index(&self, stored: Vec<StoredPage>) -> PageIndex {
        let mut pages = HashMap::new();
        let mut aliases = HashMap::new();
        for page in stored.into_iter().map(DocPage::from) {
            if !self.sources.iter().any(|s| s == page.source()) {
                continue;
            }
            for alias in &page.aliases {
                aliases.insert(alias.clone(), page.path.clone());
            }
            pages.insert(page.path.clone(), page);
        }
        PageIndex { pages, aliases }
    }

    // Tools

    /// The index entry (if any), the canonical URL, and the Markdown body of a page.
    pub async fn get_page(
        &self,
        reference: &str,
    ) -> Result<(Option<DocPage>, String, String), PageError> {
        let path = self.normalize(reference)?;
        // One filtered query rather than the cached page index, so a page is
        // readable as soon as it's indexed.
        let page = self
            .backend
            .find_page(&page_key(&path))
            .await
            .map_err(|e| PageError::Other(format!("Could not look up the page: {e}")))?
            .map(DocPage::from)
            .filter(|page| self.sources.iter().any(|s| s == page.source()));
        let path = page.as_ref().map_or(path, |page| page.path.clone());
        if let Some(page) = &page {
            if !page.rendered {
                return Err(PageError::Other(format!(
                    "{path} is a section without a page of its own. \
                     Call list_docs with section '{path}' to see its pages."
                )));
            }
            if let Some(external) = &page.external_url {
                return Err(PageError::Other(format!(
                    "{} is hosted outside qdrant.tech: {external}",
                    page.title
                )));
            }
        }
        let body = self
            .backend
            .page_markdown(&page_key(&path))
            .await
            .map_err(|e| PageError::Other(format!("Could not read the page: {e}")))?;
        let body = match (body, &page) {
            (Some(body), _) => body,
            // `pages` and `sections` are indexed separately, so a listed page
            // can be missing from `sections` for a while, such as during the
            // nightly indexing.
            (None, Some(page)) => {
                return Err(PageError::Other(format!(
                    "{} ({path}) exists but can't be read right now. Try again later.",
                    page.title
                )))
            }
            (None, None) => return Err(PageError::NotFound(path)),
        };
        Ok((page, self.url(&path), strip_preamble(&body)))
    }

    /// Top-level documentation sections, then each other source, with page counts.
    pub async fn list_sections(&self) -> Result<Vec<(DocPage, usize)>, String> {
        let index = self.page_index().await?;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for path in index.pages.keys() {
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
                let page = index
                    .pages
                    .get(&path)
                    .cloned()
                    .unwrap_or_else(|| unlisted_section(&index, &path));
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
        let index = self.page_index().await?;
        let mut pages: Vec<DocPage> = index
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
                    let weight = index.pages.get(&ancestor).map(|p| p.weight).unwrap_or(0);
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
            assert_eq!(normalize_path(reference).unwrap(), expected, "{reference}");
        }
    }

    #[test]
    fn rejects_other_hosts() {
        let error = normalize_path("https://example.com/documentation/").unwrap_err();
        assert!(error.contains("not example.com"), "{error}");
    }

    #[test]
    fn strips_only_the_preamble() {
        assert_eq!(
            strip_preamble("> Use https://qdrant.tech/llms.txt\n\n> A real note\n\nBody"),
            "> A real note\n\nBody\n"
        );
    }
}
