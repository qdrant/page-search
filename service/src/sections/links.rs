use std::sync::LazyLock;

use regex::{Captures, Regex};

/// The address the indexed pages write into their text, as in
/// "Search the documentation at https://search.qdrant.tech/md/...".
static INDEXED_BASE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"https?://search\.qdrant\.tech([^A-Za-z0-9.\-:@]|$)").unwrap()
});

/// Point links to `https://search.qdrant.tech` in `content` at `base_url`,
/// the address this page was requested from (`{scheme}://{host}`), so a page
/// read through another host links back to that host.
pub fn rebase_service_links(content: &str, base_url: &str) -> String {
    INDEXED_BASE
        .replace_all(content, |caps: &Captures| format!("{base_url}{}", &caps[1]))
        .into_owned()
}

/// Rewrite markdown links in `content` for the `/md/` service.
///
/// 1. Strip `index.md` from the link path
/// 2. Convert `#fragment` into `?s=fragment`
/// 3. Add `/md` prefix only for absolute paths (starting with `/`)
/// 4. Point links to qdrant.tech pages at the same page under
///    `https://search.qdrant.tech/md/`, which [`rebase_service_links`] then
///    points at the requested host
/// 5. Leave relative links relative; leave other external links unchanged
pub fn rewrite_links(content: &str) -> String {
    let re = Regex::new(r"\[([^\]]*)\]\(([^)]+)\)").unwrap();

    re.replace_all(content, |caps: &Captures| {
        let text = &caps[1];
        let raw = caps[2].trim();

        // Separate an optional markdown title: [txt](url "title")
        let (url, title) = match raw.find(" \"") {
            Some(idx) => (&raw[..idx], Some(&raw[idx..])),
            None => (raw, None),
        };

        let new_url = transform_link(url);

        match title {
            Some(t) => format!("[{}]({}{})", text, new_url, t),
            None => format!("[{}]({})", text, new_url),
        }
    })
    .to_string()
}

fn is_external(url: &str) -> bool {
    url.contains("://") || url.starts_with("mailto:")
}

/// The site the indexed pages come from.
const DOCS_SITE: &str = "https://qdrant.tech";

/// The address of this service that the indexed pages link to.
const SERVICE_SITE: &str = "https://search.qdrant.tech";

/// For a link to a page on qdrant.tech, such as
/// `https://qdrant.tech/documentation/foo/index.md#bar` or
/// `https://qdrant.tech/documentation/foo/`, the root-relative link
/// (`/documentation/foo/index.md#bar`). `None` for other links, the site
/// root, links with a query, and files such as `llms.txt` or images.
fn docs_page_link(link: &str) -> Option<&str> {
    let rest = link.strip_prefix(DOCS_SITE)?;
    if !rest.starts_with('/') {
        return None; // another host, such as qdrant.tech.example.com
    }
    let path = rest.split_once('#').map_or(rest, |(p, _)| p);
    if path.contains('?') {
        return None;
    }
    let last = path.rsplit('/').next().unwrap_or("");
    if !(last.is_empty() || last == "index.md" || !last.contains('.')) {
        return None;
    }
    let page = path.strip_suffix("index.md").unwrap_or(path).trim_matches('/');
    if page.is_empty() {
        return None;
    }
    Some(rest)
}

fn transform_link(link: &str) -> String {
    if let Some(rest) = docs_page_link(link) {
        return format!("{SERVICE_SITE}{}", transform_link(rest));
    }
    if is_external(link) {
        return link.to_string();
    }

    let (path_part, fragment) = match link.split_once('#') {
        Some((p, f)) => (p, Some(f)),
        None => (link, None),
    };

    // Strip index.md, then trim trailing slash
    let path = path_part
        .strip_suffix("index.md")
        .unwrap_or(path_part)
        .trim_end_matches('/');

    // Add /md prefix for absolute paths
    let path = if path.starts_with('/') {
        format!("/md{}", path)
    } else {
        path.to_string()
    };

    // Empty relative path with no fragment → "."
    let path = if path.is_empty() && fragment.is_none() {
        ".".to_string()
    } else {
        path
    };

    match fragment {
        Some(f) if !f.is_empty() => format!("{}?s={}", path, f),
        _ => path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── external links are not touched ──────────────────────────────

    #[test]
    fn external_https_unchanged() {
        let md = "[site](https://example.com/foo)";
        assert_eq!(rewrite_links(md), md);
    }

    #[test]
    fn external_http_unchanged() {
        let md = "[site](http://example.com)";
        assert_eq!(rewrite_links(md), md);
    }

    #[test]
    fn mailto_unchanged() {
        let md = "[email](mailto:user@example.com)";
        assert_eq!(rewrite_links(md), md);
    }

    // ── fragment-only links ─────────────────────────────────────────

    #[test]
    fn fragment_only() {
        assert_eq!(
            rewrite_links("[s](#overview)"),
            "[s](?s=overview)"
        );
    }

    #[test]
    fn fragment_only_faq() {
        assert_eq!(
            rewrite_links("[f](#faq)"),
            "[f](?s=faq)"
        );
    }

    // ── absolute internal links ─────────────────────────────────────

    #[test]
    fn absolute_internal() {
        assert_eq!(
            rewrite_links("[a](/documentation/foo)"),
            "[a](/md/documentation/foo)"
        );
    }

    #[test]
    fn absolute_internal_with_index_md() {
        assert_eq!(
            rewrite_links("[a](/documentation/foo/index.md)"),
            "[a](/md/documentation/foo)"
        );
    }

    #[test]
    fn absolute_internal_with_fragment() {
        assert_eq!(
            rewrite_links("[a](/documentation/foo#bar)"),
            "[a](/md/documentation/foo?s=bar)"
        );
    }

    #[test]
    fn absolute_internal_index_md_and_fragment() {
        assert_eq!(
            rewrite_links("[a](/documentation/foo/index.md#setup)"),
            "[a](/md/documentation/foo?s=setup)"
        );
    }

    // ── relative links (stay relative) ──────────────────────────────

    #[test]
    fn relative_plain() {
        assert_eq!(
            rewrite_links("[r](child)"),
            "[r](child)"
        );
    }

    #[test]
    fn relative_dot_slash() {
        assert_eq!(
            rewrite_links("[r](./child)"),
            "[r](./child)"
        );
    }

    #[test]
    fn relative_parent() {
        assert_eq!(
            rewrite_links("[r](../filtering)"),
            "[r](../filtering)"
        );
    }

    #[test]
    fn relative_parent_with_index_md() {
        assert_eq!(
            rewrite_links("[r](../filtering/index.md)"),
            "[r](../filtering)"
        );
    }

    #[test]
    fn relative_with_fragment() {
        assert_eq!(
            rewrite_links("[r](../filtering#tips)"),
            "[r](../filtering?s=tips)"
        );
    }

    #[test]
    fn relative_deep() {
        assert_eq!(
            rewrite_links("[r](sub/page)"),
            "[r](sub/page)"
        );
    }

    // ── index.md removal edge cases ─────────────────────────────────

    #[test]
    fn bare_index_md() {
        assert_eq!(
            rewrite_links("[r](index.md)"),
            "[r](.)"
        );
    }

    #[test]
    fn index_md_with_fragment() {
        assert_eq!(
            rewrite_links("[r](index.md#faq)"),
            "[r](?s=faq)"
        );
    }

    #[test]
    fn non_index_md_kept() {
        assert_eq!(
            rewrite_links("[r](setup.md)"),
            "[r](setup.md)"
        );
    }

    // ── multiple links in one block ─────────────────────────────────

    #[test]
    fn multiple_links() {
        let md = "See [a](../foo) and [b](https://ext.com) and [c](#sec)";
        let expected = "See [a](../foo) and [b](https://ext.com) and [c](?s=sec)";
        assert_eq!(rewrite_links(md), expected);
    }

    // ── link with title ─────────────────────────────────────────────

    #[test]
    fn link_with_title() {
        assert_eq!(
            rewrite_links("[text](../foo/index.md \"some title\")"),
            "[text](../foo \"some title\")"
        );
    }

    // ── transform_link unit tests ───────────────────────────────────

    #[test]
    fn transform_external() {
        assert_eq!(
            transform_link("https://example.com/docs"),
            "https://example.com/docs"
        );
    }

    #[test]
    fn transform_fragment_only() {
        assert_eq!(
            transform_link("#install"),
            "?s=install"
        );
    }

    #[test]
    fn transform_absolute_index_md_fragment() {
        assert_eq!(
            transform_link("/guides/intro/index.md#setup"),
            "/md/guides/intro?s=setup"
        );
    }

    #[test]
    fn transform_relative_index_md_fragment() {
        assert_eq!(
            transform_link("intro/index.md#setup"),
            "intro?s=setup"
        );
    }

    // ── links to qdrant.tech pages ──────────────────────────────────

    #[test]
    fn docs_index_md_with_fragment() {
        assert_eq!(
            rewrite_links(
                "[Upsert points in batches](https://qdrant.tech/documentation/manage-data/points/index.md#upload-points)"
            ),
            "[Upsert points in batches](https://search.qdrant.tech/md/documentation/manage-data/points?s=upload-points)"
        );
    }

    #[test]
    fn docs_html_page() {
        assert_eq!(
            rewrite_links("[a](https://qdrant.tech/articles/filterable-hnsw/)"),
            "[a](https://search.qdrant.tech/md/articles/filterable-hnsw)"
        );
        assert_eq!(
            rewrite_links("[a](https://qdrant.tech/documentation/manage-data/indexing/#full-text-index)"),
            "[a](https://search.qdrant.tech/md/documentation/manage-data/indexing?s=full-text-index)"
        );
        assert_eq!(
            rewrite_links("[a](https://qdrant.tech/documentation/security)"),
            "[a](https://search.qdrant.tech/md/documentation/security)"
        );
    }

    #[test]
    fn docs_non_pages_unchanged() {
        let md = "[root](https://qdrant.tech/) [root](https://qdrant.tech) \
                  [llms](https://qdrant.tech/llms.txt) \
                  [img](https://qdrant.tech/docs/gettingstarted/x.png) \
                  [q](https://qdrant.tech/documentation/?q=x) \
                  [api](https://api.qdrant.tech/api-reference) \
                  [other](https://qdrant.tech.example.com/documentation/) \
                  [http](http://qdrant.tech/documentation/)";
        assert_eq!(rewrite_links(md), md);
    }

    #[test]
    fn docs_links_follow_the_requested_host() {
        let md = rewrite_links("[a](https://qdrant.tech/documentation/manage-data/points/index.md#upload-points)");
        assert_eq!(
            rebase_service_links(&md, "https://skills.qdrant.tech"),
            "[a](https://skills.qdrant.tech/md/documentation/manage-data/points?s=upload-points)"
        );
    }

    // ── rebase_service_links ────────────────────────────────────────

    const SKILLS: &str = "https://skills.qdrant.tech";

    #[test]
    fn rebase_plain_and_markdown_links() {
        let md = "> Search at https://search.qdrant.tech/md/documentation?q=x\n\
                  [Filter](https://search.qdrant.tech/md/documentation/search/filtering)";
        assert_eq!(
            rebase_service_links(md, SKILLS),
            "> Search at https://skills.qdrant.tech/md/documentation?q=x\n\
             [Filter](https://skills.qdrant.tech/md/documentation/search/filtering)"
        );
    }

    #[test]
    fn rebase_bare_host() {
        assert_eq!(
            rebase_service_links("see https://search.qdrant.tech", SKILLS),
            "see https://skills.qdrant.tech"
        );
        assert_eq!(
            rebase_service_links("(https://search.qdrant.tech)", SKILLS),
            "(https://skills.qdrant.tech)"
        );
    }

    #[test]
    fn rebase_leaves_other_hosts() {
        let md = "https://qdrant.tech/documentation/ \
                  https://skills.qdrant.tech/ \
                  https://search.qdrant.technology/ \
                  https://search.qdrant.tech.example.com/";
        assert_eq!(rebase_service_links(md, "http://localhost:8005"), md);
    }

    #[test]
    fn rebase_same_host_is_unchanged() {
        let md = "[a](https://search.qdrant.tech/md/a) https://search.qdrant.tech/md/b";
        assert_eq!(rebase_service_links(md, "https://search.qdrant.tech"), md);
    }
}
