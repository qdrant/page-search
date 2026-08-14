//! Cross-host agreement between qdrant.tech and the `/md/` mirror.
//!
//! The regression to guard against is the two redirect tables drifting apart
//! again, so this tests them against each other rather than against a fixture:
//! for a sample of `from` paths in the published table, `search.qdrant.tech/md/`
//! and `qdrant.tech/…/index.md` must resolve to the same final path.
//!
//! Ignored by default — it needs the network and both hosts live, and
//! `qdrant.tech/redirects.txt` only exists once the landing_page PR deploys.
//! Run it deliberately:
//!
//! ```bash
//! cargo test --test cross_host_agreement -- --ignored --nocapture
//! ```
//!
//! Point it elsewhere with `REDIRECTS_URL` and `MIRROR_BASE_URL` — e.g. at a
//! local copy of the table and a locally running service.

use std::time::Duration;

const DEFAULT_TABLE_URL: &str = "https://qdrant.tech/redirects.txt";
const DEFAULT_MIRROR: &str = "https://search.qdrant.tech";
const SITE: &str = "https://qdrant.tech";

/// How many `from` paths to check. Keep it modest: each one is two requests
/// against production.
const SAMPLE: usize = 40;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// The literal `from` paths in the table — the ones a fixed request can exercise.
/// Wildcard and `:placeholder` rules need a concrete path substituted in, which
/// the table does not carry, so they are out of scope here.
fn literal_sources(table: &str) -> Vec<String> {
    table
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let frm = fields.next()?;
            let to = fields.next()?;
            let internal = to.starts_with('/') && !to.starts_with("//");
            let literal = !frm.contains('*') && !frm.contains(':') && frm.starts_with('/');
            (internal && literal).then(|| frm.to_string())
        })
        .collect()
}

/// Reduce a path to what "the same page" means across the two hosts: no `/md`
/// prefix, no `index.md`, no trailing slash, no fragment.
///
/// Deliberately **case-sensitive**. Case is the subtlest behaviour in this system
/// and the likeliest thing to drift — the CDN matches rules case-sensitively,
/// looks assets up case-insensitively, and lowercases directory-style requests
/// but not the `index.md` form. Folding case here would apply to both sides and
/// make exactly that divergence undetectable, which is the one comparison this
/// test exists to make.
fn canonical(path: &str) -> String {
    let path = path.split('#').next().unwrap_or(path);
    let path = path.strip_prefix("/md").unwrap_or(path);
    let path = path.strip_suffix("index.md").unwrap_or(path);
    path.trim_end_matches('/').to_string()
}

async fn final_path(client: &reqwest::Client, url: &str) -> Option<(u16, String)> {
    let response = client.get(url).send().await.ok()?;
    let status = response.status().as_u16();
    Some((status, canonical(response.url().path())))
}

#[tokio::test]
#[ignore = "hits qdrant.tech and search.qdrant.tech"]
async fn mirror_and_site_resolve_redirects_to_the_same_page() {
    let table_url = env_or("REDIRECTS_URL", DEFAULT_TABLE_URL);
    let mirror = env_or("MIRROR_BASE_URL", DEFAULT_MIRROR);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .expect("reqwest client");

    let table = client
        .get(&table_url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .unwrap_or_else(|e| panic!("could not fetch the redirect table from {table_url}: {e}"))
        .text()
        .await
        .expect("redirect table body");

    let sources = literal_sources(&table);
    assert!(
        !sources.is_empty(),
        "the table parsed to no literal rules, so nothing was actually compared"
    );

    // Spread the sample across the table rather than taking the first N, so
    // specific rules and catch-all neighbours both get covered.
    let step = (sources.len() / SAMPLE).max(1);
    let sample: Vec<&String> = sources.iter().step_by(step).take(SAMPLE).collect();
    println!("comparing {} of {} literal rules", sample.len(), sources.len());

    let mut disagreements = Vec::new();
    let mut compared = 0usize;

    for frm in sample {
        let trimmed = frm.trim_matches('/');
        let site_url = format!("{SITE}/{trimmed}/index.md");
        let mirror_url = format!("{mirror}/md/{trimmed}");

        let (Some((site_status, site_path)), Some((mirror_status, mirror_path))) = (
            final_path(&client, &site_url).await,
            final_path(&client, &mirror_url).await,
        ) else {
            disagreements.push(format!("{frm}: a request failed outright"));
            continue;
        };

        // Where qdrant.tech itself has no markdown at the destination, the mirror
        // is right to 404 too — that is the known divergence in the CDN half, not
        // drift between the tables. Only compare where the site actually serves.
        if site_status != 200 {
            println!("skip {frm}: qdrant.tech answers {site_status} for the .md form");
            continue;
        }

        compared += 1;
        if mirror_status != 200 || site_path != mirror_path {
            disagreements.push(format!(
                "{frm}: qdrant.tech -> {site_path} (200), mirror -> {mirror_path} ({mirror_status})"
            ));
        }
    }

    assert!(
        compared > 0,
        "every sampled rule was skipped, so the invariant went untested"
    );
    assert!(
        disagreements.is_empty(),
        "{} of {compared} comparable rules disagree across hosts:\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );
}
