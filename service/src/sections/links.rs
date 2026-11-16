use regex::{Captures, Regex};

/// Rewrite markdown links in `content` for the `/md/` service.
///
/// 1. Strip `index.md` from the link path
/// 2. Convert `#fragment` into `?s=fragment`
/// 3. Add `/md` prefix only for absolute paths (starting with `/`)
/// 4. Leave relative links relative; leave external links unchanged
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

fn transform_link(link: &str) -> String {
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
            transform_link("https://qdrant.tech/docs"),
            "https://qdrant.tech/docs"
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
}
