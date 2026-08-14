//! Matching for the qdrant.tech redirect table (`_redirects` syntax).
//!
//! `match_rule`, `match_placeholders` and `first_match` are a port of the
//! reference implementation in `automation/generate-redirects-table.py` in
//! `landing_page`, which the generator also uses in its own validation pass.
//! Keep the port faithful rather than re-deriving it: rule order and the order
//! the four forms are tested in are both load-bearing.

use std::collections::HashSet;

/// Follow at most this many hops when flattening a chain.
const MAX_HOPS: usize = 10;

/// One parsed line of the table: `from  to  status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub frm: String,
    pub to: String,
    pub status: u16,
    /// Whether the status carried a trailing `!`, meaning Netlify applies the
    /// rule even when a file exists at that path.
    #[allow(dead_code)]
    pub force: bool,
}

/// Parse the table into rules, in file order. Blank lines and `#` comments are
/// skipped; a line without both a `from` and a `to` is not a rule.
pub fn parse_table(text: &str) -> Vec<Rule> {
    let mut rules = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut fields = line.split_whitespace();
        let (Some(frm), Some(to)) = (fields.next(), fields.next()) else {
            continue;
        };

        // Read the status rather than assuming a redirect: `200` is a rewrite,
        // and answering `301` for one would diverge from the live site.
        let (status, force) = match fields.next() {
            Some(field) => match field.strip_suffix('!') {
                Some(digits) => (digits.parse().unwrap_or(301), true),
                None => (field.parse().unwrap_or(301), false),
            },
            None => (301, false),
        };

        rules.push(Rule {
            frm: frm.to_string(),
            to: to.to_string(),
            status,
            force,
        });
    }

    rules
}

/// Return `rule`'s target for `path`, or `None`. Trailing slashes are ignored.
fn match_rule(rule: &Rule, path: &str) -> Option<String> {
    let frm = rule.frm.as_str();

    if frm.ends_with("/*") {
        let prefix = &frm[..frm.len() - 1]; // "/a/b/*" -> "/a/b/"
        if let Some(splat) = path.strip_prefix(prefix) {
            return Some(rule.to.replace(":splat", splat));
        }
        if path.trim_end_matches('/') == prefix.trim_end_matches('/') {
            return Some(rule.to.replace(":splat", ""));
        }
        return None;
    }

    if let Some(prefix) = frm.strip_suffix('*') {
        return path
            .strip_prefix(prefix)
            .map(|splat| rule.to.replace(":splat", splat));
    }

    if frm.contains(':') {
        return match_placeholders(rule, path);
    }

    if path.trim_end_matches('/') == frm.trim_end_matches('/') {
        return Some(rule.to.clone());
    }

    None
}

/// Match a rule with `:named` segments. Each matches exactly one segment.
fn match_placeholders(rule: &Rule, path: &str) -> Option<String> {
    let want: Vec<&str> = rule.frm.split('/').filter(|s| !s.is_empty()).collect();
    let got: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if want.len() != got.len() {
        return None;
    }

    let mut captures = Vec::new();
    for (w, g) in want.iter().zip(got.iter()) {
        if let Some(name) = w.strip_prefix(':') {
            captures.push((name, *g));
        } else if w != g {
            return None;
        }
    }

    let mut to = rule.to.clone();
    for (name, value) in captures {
        to = to.replace(&format!(":{}", name), value);
    }
    Some(to)
}

/// First-match-wins, in table order. The table deliberately places specific
/// rules above catch-alls, so the order must not be disturbed.
fn first_match<'a>(rules: &'a [Rule], path: &str) -> Option<(&'a Rule, String)> {
    rules
        .iter()
        .find_map(|rule| match_rule(rule, path).map(|to| (rule, to)))
}

/// `first_match`, retried against the lowercased path.
///
/// Rule matching on the CDN is case-sensitive, but asset lookup is not, and
/// qdrant.tech 301s a mixed-case *directory* path to its lowercase form before
/// applying rules — a hop the `index.md` form never gets, and the `index.md`
/// form is the only one this mirror serves. So the table does no case folding
/// for us and mixed-case requests would 404 here while qdrant.tech serves them.
///
/// Trying the path as received first, then lowercased, agrees with production on
/// every measured case: it keeps the one rule spelled with capitals
/// (`running-with-GPU`) reachable for the mixed-case request that matches it,
/// and reproduces the lowercasing 301 for everything else — `BULK-UPLOAD/` and
/// its kind, which is real traffic. Lowercasing only would lose the former.
/// Mixed-case paths 404 today, so this is additive.
///
/// It does **not** fix a lowercase request that falls through to a catch-all
/// with a dead landing (the `running-with-gpu` class): the first pass already
/// matches, so the fallback never fires, and the destination check turns it into
/// a logged 404. That needs a lowercase rule in `_redirects`, on the CDN side.
///
/// The CDN applies its lowercasing 301 to the incoming request only, so
/// `resolve` uses this for the first hop and plain `first_match` after it.
fn first_match_any_case<'a>(rules: &'a [Rule], path: &str) -> Option<(&'a Rule, String)> {
    if let Some(hit) = first_match(rules, path) {
        return Some(hit);
    }
    let lowered = path.to_lowercase();
    if lowered != path {
        return first_match(rules, &lowered);
    }
    None
}

/// Where a resolved path points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// A path on the mirror, without the `/md` prefix and without a fragment.
    Internal {
        path: String,
        fragment: Option<String>,
    },
    /// An absolute or protocol-relative URL. Never followed.
    External(String),
}

/// The answer the table gives for a request path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    pub target: Target,
    /// The status of the first rule that matched. `200` means rewrite: serve the
    /// target's content at the requested URL without changing it.
    pub status: u16,
}

impl Resolution {
    pub fn is_rewrite(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// Resolve `path` against the table, flattening chains to a fixed point.
///
/// Netlify does not chain internally — it answers one redirect and the client
/// re-requests. Flattening lands in the same place and is friendlier to agent
/// HTTP clients that do not follow redirects. Stops at an external target, at a
/// rewrite (a `200` cannot be collapsed into a redirect that precedes it), after
/// `MAX_HOPS`, or on a cycle. The status answered is always the first hop's.
pub fn resolve(rules: &[Rule], path: &str) -> Option<Resolution> {
    let mut seen = HashSet::new();
    let mut current = path.to_string();
    let mut answer: Option<Resolution> = None;

    for hop in 0..MAX_HOPS {
        if !seen.insert(current.clone()) {
            break; // cycle
        }

        // The case fallback models the CDN's lowercasing 301, which applies to
        // the incoming request only — not to a path it has already rewritten.
        let matched = if hop == 0 {
            first_match_any_case(rules, &current)
        } else {
            first_match(rules, &current)
        };
        let Some((rule, to)) = matched else {
            break;
        };
        let status = answer.as_ref().map_or(rule.status, |a| a.status);

        if is_external(&to) {
            return Some(Resolution {
                target: Target::External(to),
                status,
            });
        }

        let rewrite = (200..300).contains(&rule.status);
        if rewrite && answer.is_some() {
            break; // cannot collapse a rewrite into the redirect before it
        }

        let (target_path, fragment) = split_fragment(&to);
        let target_path = strip_index_md(target_path);
        answer = Some(Resolution {
            target: Target::Internal {
                path: target_path.clone(),
                fragment,
            },
            status,
        });

        if rewrite {
            break;
        }
        current = target_path;
    }

    answer
}

/// External targets are never followed. Protocol-relative targets (`//host/path`)
/// count as external too, or one would read as a site path.
fn is_external(target: &str) -> bool {
    target.starts_with("//") || target.contains("://") || target.starts_with("mailto:")
}

fn split_fragment(target: &str) -> (&str, Option<String>) {
    match target.split_once('#') {
        Some((path, "")) => (path, None),
        Some((path, fragment)) => (path, Some(fragment.to_string())),
        None => (target, None),
    }
}

/// `/a/b/index.md` -> `/a/b/`. The mirror addresses pages directory-style.
fn strip_index_md(path: &str) -> String {
    path.strip_suffix("index.md").unwrap_or(path).to_string()
}

/// Turn the path captured by the `/md/{path:.*}` route into a lookup key for the
/// table: leading slash, no `index.md` suffix, no fragment.
///
/// Case is deliberately left alone — `resolve` tries the path as received before
/// falling back to lowercase, which needs the original.
pub fn lookup_path(captured: &str) -> String {
    let path = captured.split('#').next().unwrap_or(captured);
    let path = path.trim_start_matches('/');
    let path = path.strip_suffix("index.md").unwrap_or(path);
    format!("/{}", path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(frm: &str, to: &str) -> Rule {
        Rule {
            frm: frm.into(),
            to: to.into(),
            status: 301,
            force: false,
        }
    }

    fn target_of(rules: &[Rule], path: &str) -> Option<String> {
        match resolve(rules, path)?.target {
            Target::Internal { path, .. } => Some(path),
            Target::External(url) => Some(url),
        }
    }

    // ── parsing ─────────────────────────────────────────────────────

    #[test]
    fn parses_rules_in_file_order() {
        let table = "\
# a comment

/documentation/a/  /documentation/one/  301
/documentation/b/  /documentation/two/  301
";
        let rules = parse_table(table);
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].frm, "/documentation/a/");
        assert_eq!(rules[1].to, "/documentation/two/");
    }

    #[test]
    fn parses_status_and_force_flag() {
        let rules = parse_table(
            "/a/  https://example.com/  301!\n/b/  /c/  200\n/d/  /e/\n/f/  /g/  302\n",
        );
        assert_eq!((rules[0].status, rules[0].force), (301, true));
        assert_eq!((rules[1].status, rules[1].force), (200, false));
        assert_eq!((rules[2].status, rules[2].force), (301, false), "no status defaults to 301");
        assert_eq!((rules[3].status, rules[3].force), (302, false));
    }

    #[test]
    fn skips_comments_blanks_and_incomplete_lines() {
        let rules = parse_table("# header\n\n   \n/only-a-from\n/a/  /b/\n");
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].frm, "/a/");
    }

    // ── the four match forms ────────────────────────────────────────

    #[test]
    fn exact_match() {
        let rules = vec![rule("/documentation/scroll/", "/documentation/points/")];
        assert_eq!(
            target_of(&rules, "/documentation/scroll/").as_deref(),
            Some("/documentation/points/")
        );
        assert_eq!(target_of(&rules, "/documentation/other/"), None);
    }

    #[test]
    fn exact_match_ignores_trailing_slashes() {
        let rules = vec![rule("/documentation/scroll/", "/documentation/points/")];
        assert_eq!(
            target_of(&rules, "/documentation/scroll").as_deref(),
            Some("/documentation/points/")
        );

        let rules = vec![rule("/documentation/scroll", "/documentation/points/")];
        assert_eq!(
            target_of(&rules, "/documentation/scroll/").as_deref(),
            Some("/documentation/points/")
        );
    }

    #[test]
    fn slash_splat_prefix_match() {
        let rules = vec![rule("/documentation/guides/*", "/documentation/new/:splat")];
        assert_eq!(
            target_of(&rules, "/documentation/guides/quickstart/").as_deref(),
            Some("/documentation/new/quickstart/")
        );
    }

    #[test]
    fn slash_splat_matches_bare_prefix_with_empty_splat() {
        let rules = vec![rule("/documentation/guides/*", "/documentation/new/:splat")];
        assert_eq!(
            target_of(&rules, "/documentation/guides/").as_deref(),
            Some("/documentation/new/")
        );
        assert_eq!(
            target_of(&rules, "/documentation/guides").as_deref(),
            Some("/documentation/new/")
        );
    }

    #[test]
    fn slash_splat_does_not_match_a_sibling() {
        let rules = vec![rule("/documentation/guides/*", "/documentation/new/:splat")];
        assert_eq!(target_of(&rules, "/documentation/guidelines/"), None);
    }

    #[test]
    fn bare_splat_has_no_slash_boundary() {
        let rules = vec![rule("/documentation/x*", "/documentation/y:splat")];
        assert_eq!(
            target_of(&rules, "/documentation/xyz/").as_deref(),
            Some("/documentation/yyz/")
        );
        assert_eq!(
            target_of(&rules, "/documentation/x").as_deref(),
            Some("/documentation/y")
        );
    }

    #[test]
    fn placeholder_matches_exactly_one_segment() {
        let rules = vec![rule(
            "/documentation/platforms/:slug/",
            "/documentation/integrations/:slug/",
        )];
        assert_eq!(
            target_of(&rules, "/documentation/platforms/airflow/").as_deref(),
            Some("/documentation/integrations/airflow/")
        );
        assert_eq!(
            target_of(&rules, "/documentation/platforms/airflow/setup/"),
            None,
            "two segments must not match one placeholder"
        );
        assert_eq!(target_of(&rules, "/documentation/platforms/"), None);
    }

    #[test]
    fn first_match_wins_in_table_order() {
        let rules = vec![
            rule("/documentation/operations/running-with-GPU/", "/ops/gpu/"),
            rule("/documentation/operations/*", "/deploy-intro/:splat"),
        ];
        assert_eq!(
            target_of(&rules, "/documentation/operations/running-with-GPU/").as_deref(),
            Some("/ops/gpu/"),
            "the specific rule sits above the catch-all"
        );
        assert_eq!(
            target_of(&rules, "/documentation/operations/something-else/").as_deref(),
            Some("/deploy-intro/something-else/")
        );
    }

    // ── case handling ───────────────────────────────────────────────

    #[test]
    fn lowercase_request_falls_through_to_the_catch_all() {
        // Measured on production: the rule spells it `running-with-GPU`, and real
        // traffic arriving lowercase takes the catch-all instead.
        let rules = vec![
            rule("/documentation/operations/running-with-GPU/", "/ops/gpu/"),
            rule("/documentation/operations/*", "/deploy-intro/:splat"),
        ];
        assert_eq!(
            target_of(&rules, "/documentation/operations/running-with-gpu/").as_deref(),
            Some("/deploy-intro/running-with-gpu/")
        );
    }

    #[test]
    fn the_case_fallback_applies_to_the_request_only_not_to_later_hops() {
        let rules = vec![rule("/a/", "/B/"), rule("/b/", "/c/")];
        let resolved = resolve(&rules, "/a/").expect("resolves");
        assert_eq!(
            resolved.target,
            Target::Internal { path: "/B/".into(), fragment: None },
            "the CDN lowercases the incoming request, not a path it already rewrote"
        );
    }

    #[test]
    fn mixed_case_request_falls_back_to_lowercase() {
        let rules = vec![rule("/documentation/tutorials/bulk-upload/", "/documentation/send-data/")];
        assert_eq!(
            target_of(&rules, "/documentation/tutorials/BULK-UPLOAD/").as_deref(),
            Some("/documentation/send-data/"),
            "qdrant.tech 301s the directory form to lowercase before applying rules"
        );
    }

    // ── resolution ──────────────────────────────────────────────────

    #[test]
    fn flattens_a_chain_keeping_the_first_status() {
        let rules = vec![
            Rule { status: 302, ..rule("/a/", "/b/") },
            rule("/b/", "/c/"),
            rule("/c/", "/d/"),
        ];
        let resolved = resolve(&rules, "/a/").expect("resolves");
        assert_eq!(
            resolved.target,
            Target::Internal { path: "/d/".into(), fragment: None }
        );
        assert_eq!(resolved.status, 302);
    }

    #[test]
    fn stops_on_a_cycle() {
        let rules = vec![rule("/a/", "/b/"), rule("/b/", "/a/")];
        let resolved = resolve(&rules, "/a/").expect("resolves to the last hop before the cycle");
        assert_eq!(
            resolved.target,
            Target::Internal { path: "/a/".into(), fragment: None }
        );
    }

    #[test]
    fn stops_at_the_hop_cap() {
        // A catch-all that rewrites itself forever.
        let rules = vec![rule("/a/*", "/a/x:splat")];
        assert!(resolve(&rules, "/a/").is_some(), "does not hang or panic");
    }

    #[test]
    fn stops_at_an_external_target_without_following_it() {
        let rules = vec![
            rule("/documentation/hybrid-cloud/", "https://hybrid-cloud.qdrant.tech/"),
            rule("/documentation/protocol-relative/", "//qdrant.to/somewhere"),
        ];
        assert_eq!(
            resolve(&rules, "/documentation/hybrid-cloud/").unwrap().target,
            Target::External("https://hybrid-cloud.qdrant.tech/".into())
        );
        assert_eq!(
            resolve(&rules, "/documentation/protocol-relative/").unwrap().target,
            Target::External("//qdrant.to/somewhere".into()),
            "a protocol-relative target must not be read as a site path"
        );
    }

    #[test]
    fn a_rewrite_is_not_collapsed_into_a_preceding_redirect() {
        let rules = vec![rule("/a/", "/b/"), Rule { status: 200, ..rule("/b/", "/c/") }];
        let resolved = resolve(&rules, "/a/").expect("resolves");
        assert_eq!(
            resolved.target,
            Target::Internal { path: "/b/".into(), fragment: None }
        );
        assert_eq!(resolved.status, 301);
        assert!(!resolved.is_rewrite());
    }

    #[test]
    fn a_leading_rewrite_is_reported_as_one() {
        let rules = vec![Rule { status: 200, ..rule("/a/", "/b/") }];
        let resolved = resolve(&rules, "/a/").expect("resolves");
        assert!(resolved.is_rewrite());
        assert_eq!(
            resolved.target,
            Target::Internal { path: "/b/".into(), fragment: None }
        );
    }

    #[test]
    fn splits_a_fragment_off_the_target() {
        let rules = vec![rule(
            "/documentation/scroll/",
            "/documentation/manage-data/points/#scroll-points",
        )];
        assert_eq!(
            resolve(&rules, "/documentation/scroll/").unwrap().target,
            Target::Internal {
                path: "/documentation/manage-data/points/".into(),
                fragment: Some("scroll-points".into()),
            }
        );
    }

    #[test]
    fn strips_index_md_from_a_target() {
        let rules = vec![rule("/a/", "/documentation/b/index.md")];
        assert_eq!(
            resolve(&rules, "/a/").unwrap().target,
            Target::Internal { path: "/documentation/b/".into(), fragment: None }
        );
    }

    #[test]
    fn unmatched_path_resolves_to_nothing() {
        let rules = vec![rule("/a/", "/b/")];
        assert!(resolve(&rules, "/documentation/unrelated/").is_none());
    }

    // ── lookup key ──────────────────────────────────────────────────

    #[test]
    fn lookup_path_adds_a_leading_slash() {
        assert_eq!(lookup_path("documentation/guides/"), "/documentation/guides/");
        assert_eq!(lookup_path("/documentation/guides/"), "/documentation/guides/");
    }

    #[test]
    fn lookup_path_strips_index_md_and_fragments() {
        assert_eq!(
            lookup_path("documentation/guides/index.md"),
            "/documentation/guides/"
        );
        assert_eq!(lookup_path("documentation/guides/#setup"), "/documentation/guides/");
    }

    #[test]
    fn lookup_path_preserves_case_for_the_two_pass_lookup() {
        assert_eq!(
            lookup_path("documentation/operations/running-with-GPU/index.md"),
            "/documentation/operations/running-with-GPU/"
        );
    }

    #[test]
    fn lookup_path_of_the_mirror_root() {
        assert_eq!(lookup_path(""), "/");
    }
}
