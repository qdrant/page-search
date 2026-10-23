//! Find existing pages whose paths resemble a missing one.

/// The words of a path below its source, such as `["search", "hybrid", "queries"]`
/// for `/documentation/search/hybrid-queries/`.
pub fn path_words(path: &str) -> Vec<String> {
    path.trim_matches('/')
        .split('/')
        .skip(1)
        .flat_map(|segment| segment.split(['-', '_', '.']))
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn slug(path: &str) -> &str {
    path.trim_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or_default()
}

fn slug_words(path: &str) -> Vec<String> {
    path_words(&format!("/x/{}/", slug(path)))
}

fn source(path: &str) -> &str {
    path.trim_matches('/').split('/').next().unwrap_or_default()
}

/// Endings of other forms of a word.
const ENDINGS: [&str; 4] = ["s", "es", "ing", "ed"];

/// Equal words or forms of one word, such as "query" and "queries".
fn same_word(a: &str, b: &str) -> bool {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    match long.strip_prefix(short) {
        Some(ending) => ending.is_empty() || ENDINGS.contains(&ending),
        None => {
            short
                .strip_suffix('y')
                .and_then(|stem| long.strip_prefix(stem))
                == Some("ies")
        }
    }
}

/// Ancestors of `path` below its source, closest first: `/docs/a/b/`,
/// `/docs/a/` for `/docs/a/b/c/`.
fn ancestors(path: &str) -> Vec<String> {
    let lead = if path.starts_with('/') { "/" } else { "" };
    let trail = if path.ends_with('/') { "/" } else { "" };
    let segments: Vec<&str> = path.trim_matches('/').split('/').collect();
    (2..segments.len())
        .rev()
        .map(|n| format!("{lead}{}{trail}", segments[..n].join("/")))
        .collect()
}

/// Candidates resembling `path`, best first: those sharing words with its
/// last segment, then its closest ancestor.
pub fn similar_paths<'a>(
    path: &str,
    candidates: impl IntoIterator<Item = &'a str>,
    limit: usize,
) -> Vec<&'a str> {
    let missing_slug = slug(path);
    let missing_slug_words = slug_words(path);
    let words = path_words(path);
    let candidates: Vec<&str> = candidates.into_iter().collect();
    let mut scored: Vec<(usize, &str)> = candidates
        .iter()
        .copied()
        .filter_map(|candidate| {
            let candidate_slug_words = slug_words(candidate);
            let slug_matches = missing_slug_words
                .iter()
                .filter(|w| candidate_slug_words.iter().any(|c| same_word(w, c)))
                .count();
            if slug_matches == 0 {
                return None;
            }
            let candidate_words = path_words(candidate);
            let matches = words
                .iter()
                .filter(|w| candidate_words.iter().any(|c| same_word(w, c)))
                .count();
            let same_slug = slug(candidate) == missing_slug;
            let same_source = source(candidate) == source(path);
            Some((
                10 * usize::from(same_slug) + 2 * slug_matches + matches + usize::from(same_source),
                candidate,
            ))
        })
        .collect();
    scored.sort_by(|(a, x), (b, y)| b.cmp(a).then_with(|| x.cmp(y)));
    scored.dedup_by(|(_, x), (_, y)| x == y);
    let mut similar: Vec<&str> = scored
        .into_iter()
        .take(limit)
        .map(|(_, candidate)| candidate)
        .collect();
    let ancestor = ancestors(path)
        .into_iter()
        .find_map(|a| candidates.iter().copied().find(|c| *c == a));
    if let Some(ancestor) = ancestor.filter(|a| limit > 0 && !similar.contains(a)) {
        similar.truncate(limit - 1);
        similar.push(ancestor);
    }
    similar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_paths_into_words_below_the_source() {
        assert_eq!(
            path_words("/documentation/search/hybrid-queries/"),
            ["search", "hybrid", "queries"]
        );
        assert_eq!(path_words("blog/qdrant-1.15.x"), ["qdrant", "1", "15", "x"]);
    }

    #[test]
    fn matches_other_forms_of_a_word() {
        assert!(same_word("query", "queries"));
        assert!(same_word("filter", "filtering"));
        assert!(same_word("collection", "collections"));
        assert!(!same_word("api", "app"));
        assert!(same_word("index", "indexing"));
        assert!(!same_word("search", "sparse"));
        assert!(!same_word("three", "threshold"));
        assert!(!same_word("point", "pointer"));
    }

    #[test]
    fn lists_ancestors_below_the_source() {
        assert_eq!(
            ancestors("/documentation/a/b/c/"),
            ["/documentation/a/b/", "/documentation/a/"]
        );
        assert_eq!(ancestors("documentation/a/b"), ["documentation/a"]);
        assert!(ancestors("documentation/a").is_empty());
    }

    #[test]
    fn suggests_the_closest_existing_ancestor() {
        let pages = [
            "documentation/manage-data",
            "documentation/manage-data/points",
            "documentation/manage-data/vectors",
        ];
        assert_eq!(
            similar_paths("documentation/manage-data/points/one/two/three", pages, 5),
            ["documentation/manage-data/points"]
        );
    }

    #[test]
    fn ancestor_comes_after_similar_paths() {
        let pages = ["documentation/concepts", "documentation/search/filtering"];
        assert_eq!(
            similar_paths("documentation/concepts/filtering", pages, 5),
            ["documentation/search/filtering", "documentation/concepts"]
        );
        assert_eq!(
            similar_paths("documentation/concepts/filtering", pages, 1),
            ["documentation/concepts"]
        );
    }

    #[test]
    fn ranks_moved_pages_first() {
        let pages = [
            "documentation/search/filtering",
            "documentation/concepts/collections",
            "documentation/guides/filtering-tips",
            "documentation/concepts/points",
        ];
        assert_eq!(
            similar_paths("documentation/concepts/filtering", pages, 5),
            [
                "documentation/search/filtering",
                "documentation/guides/filtering-tips"
            ]
        );
    }

    #[test]
    fn needs_a_word_of_the_last_segment() {
        let pages = ["documentation/concepts/collections"];
        assert!(similar_paths("documentation/concepts/nope", pages, 5).is_empty());
    }
}
