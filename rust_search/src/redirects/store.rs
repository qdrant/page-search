//! Loading and refreshing the redirect table.
//!
//! `qdrant.tech/redirects.txt` is generated at build time by
//! `automation/generate-redirects-table.py` in `landing_page`. It merges the
//! `_redirects` file, the `netlify.toml` rules and Hugo's `aliases:` front
//! matter into one first-match-wins table, none of which this service can see
//! otherwise — both `/_redirects` and `/netlify.toml` 404 from the CDN.
//!
//! One generated table with one owner is the whole point: never hand-maintain
//! rules here, or the two halves will diverge.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use super::rules::{parse_table, Rule};

const DEFAULT_URL: &str = "https://qdrant.tech/redirects.txt";
const DEFAULT_CACHE_NAME: &str = "qdrant-redirects.txt";
const DEFAULT_REFRESH_SECS: u64 = 3600;

fn table_url() -> String {
    std::env::var("REDIRECTS_URL").unwrap_or_else(|_| DEFAULT_URL.to_string())
}

/// The disk cache path, always absolute.
///
/// A relative default would resolve against the process cwd, which in a
/// container is often `/` or read-only — the write would fail, the fallback copy
/// would never materialise, and "degrades to yesterday's redirects" would
/// quietly not be a guarantee. So the default is absolute, a configured relative
/// path is resolved against the cwd rather than left ambiguous, and the resolved
/// path is logged and probed at startup so a bad one is visible immediately
/// instead of at the first fetch failure.
fn cache_path() -> PathBuf {
    let configured = match std::env::var("REDIRECTS_CACHE_PATH") {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        _ => return std::env::temp_dir().join(DEFAULT_CACHE_NAME),
    };

    if configured.is_absolute() {
        return configured;
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(configured),
        Err(_) => configured,
    }
}

fn refresh_interval() -> Duration {
    let secs = std::env::var("REDIRECTS_REFRESH_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_REFRESH_SECS);
    Duration::from_secs(secs)
}

static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn shared_client() -> &'static reqwest::Client {
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("reqwest client")
    })
}

/// The parsed table, swapped in place while the service keeps serving.
pub struct RedirectStore {
    rules: RwLock<Arc<Vec<Rule>>>,
    url: String,
    cache: PathBuf,
}

impl RedirectStore {
    pub fn new() -> Self {
        Self {
            rules: RwLock::new(Arc::new(Vec::new())),
            url: table_url(),
            cache: cache_path(),
        }
    }

    /// The current table. Cheap enough to call per request.
    pub fn rules(&self) -> Arc<Vec<Rule>> {
        self.rules.read().expect("redirect table lock").clone()
    }

    fn swap(&self, rules: Vec<Rule>) {
        *self.rules.write().expect("redirect table lock") = Arc::new(rules);
    }

    /// Report the resolved cache path and whether it is actually writable.
    ///
    /// Probes a sibling file rather than the cache itself, so a cache left by a
    /// previous run survives the check. Called once at startup: without it, an
    /// unwritable path only shows up as a warning at the first refresh, and the
    /// missing fallback stays invisible until the day it is needed.
    fn report_cache(&self) {
        let probe = self.cache.with_extension("probe");
        match std::fs::write(&probe, b"") {
            Ok(()) => {
                let _ = std::fs::remove_file(&probe);
                log::info!("redirect table cache: {}", self.cache.display());
            }
            Err(e) => log::error!(
                "redirect table cache {} is not writable ({}). No copy will be kept, so a \
                 fetch failure before the first successful load will 404 /md/ misses instead of \
                 falling back. Set REDIRECTS_CACHE_PATH to a writable absolute path.",
                self.cache.display(),
                e
            ),
        }
    }

    /// Load the copy on disk, if there is one. Local and fast — no network.
    fn load_cached(&self) -> usize {
        match std::fs::read_to_string(&self.cache) {
            Ok(text) => {
                let rules = parse_table(&text);
                let count = rules.len();
                if count > 0 {
                    log::info!(
                        "loaded {} redirect rules from the cache in {}",
                        count,
                        self.cache.display()
                    );
                    self.swap(rules);
                }
                count
            }
            Err(_) => 0,
        }
    }

    /// Fetch the table and swap it in.
    ///
    /// A fetch error leaves the previous table in place; with no previous table
    /// it falls back to the copy on disk, so a failure degrades to yesterday's
    /// redirects rather than to none. With neither, `/md/` misses keep 404ing
    /// exactly as they did before this existed.
    pub async fn refresh(&self) {
        let err = match self.fetch().await {
            Ok(text) => {
                let rules = parse_table(&text);
                if rules.is_empty() {
                    log::warn!(
                        "redirect table at {} parsed to 0 rules, keeping the previous table",
                        self.url
                    );
                    return;
                }
                if let Err(e) = std::fs::write(&self.cache, &text) {
                    // Loud: this is the fallback silently not existing.
                    log::error!(
                        "could not cache the redirect table to {}: {}. The disk fallback will \
                         not exist for the next restart.",
                        self.cache.display(),
                        e
                    );
                }
                log::info!("loaded {} redirect rules from {}", rules.len(), self.url);
                self.swap(rules);
                return;
            }
            Err(e) => e,
        };

        let held = self.rules().len();
        if held > 0 {
            log::warn!(
                "redirect table fetch from {} failed ({}), keeping the {} rules already loaded",
                self.url,
                err,
                held
            );
            return;
        }

        if self.load_cached() > 0 {
            log::warn!(
                "redirect table fetch from {} failed ({}), fell back to the disk cache",
                self.url,
                err
            );
        } else {
            log::warn!(
                "no redirect table: fetch from {} failed ({}) and no usable cache in {}. \
                 /md/ misses will 404 as before.",
                self.url,
                err,
                self.cache.display()
            );
        }
    }

    async fn fetch(&self) -> anyhow::Result<String> {
        Ok(shared_client()
            .get(&self.url)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?)
    }
}

/// Load whatever is on disk, then fetch and keep refreshing in the background.
///
/// Deliberately does **not** await the first network fetch: boot must not depend
/// on qdrant.tech being reachable. Reading the cache is local and instant, so a
/// restarted service serves redirects from its first request; a cold start with
/// no cache has a brief window where `/md/` misses 404, which is exactly the
/// behaviour this feature replaces, so nothing regresses while it closes.
///
/// Refreshing matters because a landing-page deploy does not restart this
/// service, so a load-once table would go stale the first time someone moves a
/// page. A Netlify deploy webhook would be tighter, but hourly is ample.
pub fn spawn_refresh(store: Arc<RedirectStore>) {
    store.report_cache();
    store.load_cached();

    let interval = refresh_interval();
    actix_web::rt::spawn(async move {
        store.refresh().await;
        loop {
            actix_web::rt::time::sleep(interval).await;
            store.refresh().await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Refuses connections immediately, so the fetch fails without a timeout.
    const UNREACHABLE: &str = "http://127.0.0.1:1/redirects.txt";

    fn store_with(url: &str, cache: &str) -> RedirectStore {
        RedirectStore {
            rules: RwLock::new(Arc::new(Vec::new())),
            url: url.to_string(),
            cache: cache.into(),
        }
    }

    /// Serve `body` once over HTTP on a loopback port, and return its URL.
    fn serve_once(body: &'static str) -> String {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let url = format!("http://{}/redirects.txt", listener.local_addr().expect("addr"));

        std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("accept");
            let _ = socket.read(&mut [0_u8; 1024]);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes());
        });

        url
    }

    #[actix_web::test]
    async fn a_successful_fetch_parses_to_an_ordered_table_and_caches_it() {
        let table = "# generated\n/a/  /one/  301\n/b/  /two/  301\n/c/  /three/  301\n";
        let cache = std::env::temp_dir().join("page-search-redirects-fetched.txt");
        let _ = std::fs::remove_file(&cache);

        let store = store_with(&serve_once(table), cache.to_str().unwrap());
        store.refresh().await;

        let rules = store.rules();
        assert_eq!(rules.len(), 3, "the table must parse to a non-empty list");
        let order: Vec<&str> = rules.iter().map(|r| r.frm.as_str()).collect();
        assert_eq!(order, ["/a/", "/b/", "/c/"], "file order is load-bearing");
        assert_eq!(
            std::fs::read_to_string(&cache).expect("cached copy"),
            table,
            "the raw file is persisted so a later fetch failure has something to fall back to"
        );

        let _ = std::fs::remove_file(&cache);
    }

    #[actix_web::test]
    async fn a_fetch_failure_leaves_the_previous_table_in_place() {
        let store = store_with(UNREACHABLE, "/nonexistent/redirects.txt");
        store.swap(parse_table("/a/  /b/  301\n/c/  /d/  301\n"));

        store.refresh().await;

        let rules = store.rules();
        assert_eq!(rules.len(), 2, "the loaded table must survive a failed fetch");
        assert_eq!(rules[0].frm, "/a/");
    }

    #[actix_web::test]
    async fn a_fetch_failure_with_no_table_falls_back_to_the_disk_cache() {
        let cache = std::env::temp_dir().join("page-search-redirects-test.txt");
        std::fs::write(&cache, "# cached\n/a/  /b/  301\n").expect("write cache");

        let store = store_with(UNREACHABLE, cache.to_str().unwrap());
        store.refresh().await;

        assert_eq!(store.rules().len(), 1);
        let _ = std::fs::remove_file(&cache);
    }

    #[actix_web::test]
    async fn a_fetch_failure_with_neither_leaves_an_empty_table() {
        let store = store_with(UNREACHABLE, "/nonexistent/redirects.txt");
        store.refresh().await;
        assert!(store.rules().is_empty(), "no table means 404 as before, not a panic");
    }
}
