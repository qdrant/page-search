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

/// The parent of a path: `docs/search` for `/docs/search/filtering/`.
fn parent(path: &str) -> &str {
    path.trim_matches('/')
        .rsplit_once('/')
        .map_or("", |(parent, _)| parent)
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

/// Edits between `a` and `b`, counting a swap of neighboring letters as one.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut d: Vec<Vec<usize>> = (0..=a.len())
        .map(|i| {
            (0..=b.len())
                .map(|j| {
                    if i == 0 {
                        j
                    } else if j == 0 {
                        i
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect();
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let substitution = d[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = substitution.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[a.len()][b.len()]
}

/// One word misspelled as the other: one edit in words of 5 to 8 letters, two
/// in longer ones.
fn misspelled(a: &str, b: &str) -> bool {
    let (a_len, b_len) = (a.chars().count(), b.chars().count());
    let allowed = match a_len.min(b_len) {
        0..=4 => return false,
        5..=8 => 1,
        _ => 2,
    };
    a_len.abs_diff(b_len) <= allowed && edit_distance(a, b) <= allowed
}

/// How well two words match: 2 for forms of one word, 1 for a misspelling.
fn word_match(a: &str, b: &str) -> usize {
    if same_word(a, b) {
        2
    } else {
        usize::from(misspelled(a, b))
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

/// A section's main page: `/docs/search/search/` for `/docs/search/`.
fn main_page(section: &str) -> String {
    let name = slug(section);
    if section.ends_with('/') {
        format!("{section}{name}/")
    } else {
        format!("{section}/{name}")
    }
}

/// Candidates resembling `path`, best first: those matching words of its last
/// segment, then its closest ancestor and that ancestor's main page.
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
            let slug_matches: usize = missing_slug_words
                .iter()
                .map(|w| {
                    candidate_slug_words
                        .iter()
                        .map(|c| word_match(w, c))
                        .max()
                        .unwrap_or(0)
                })
                .sum();
            if slug_matches == 0 {
                return None;
            }
            let candidate_words = path_words(candidate);
            let matches = words
                .iter()
                .filter(|w| candidate_words.iter().any(|c| word_match(w, c) > 0))
                .count();
            let same_slug = slug(candidate) == missing_slug;
            let same_source = source(candidate) == source(path);
            let same_parent = parent(candidate) == parent(path);
            let score = 10 * usize::from(same_slug)
                + slug_matches
                + matches
                + usize::from(same_source)
                + usize::from(same_parent);
            Some((score, candidate))
        })
        .collect();
    // Ties go to the slug with fewer words.
    scored.sort_by_cached_key(|(score, candidate)| {
        (
            std::cmp::Reverse(*score),
            slug_words(candidate).len(),
            *candidate,
        )
    });
    scored.dedup_by(|(_, x), (_, y)| x == y);
    let mut similar: Vec<&str> = scored
        .into_iter()
        .take(limit)
        .map(|(_, candidate)| candidate)
        .collect();
    let closest: Vec<&str> = ancestors(path)
        .into_iter()
        .map(|a| {
            let main = main_page(&a);
            [a, main]
                .into_iter()
                .filter_map(|p| candidates.iter().copied().find(|c| *c == p))
                .filter(|c| !similar.contains(c))
                .collect::<Vec<_>>()
        })
        .find(|found| !found.is_empty())
        .unwrap_or_default();
    let closest: Vec<&str> = closest.into_iter().take(limit).collect();
    similar.truncate(limit - closest.len());
    similar.extend(closest);
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
    fn matches_misspellings_of_longer_words() {
        assert!(misspelled("fitlering", "filtering"));
        assert!(misspelled("colections", "collections"));
        assert!(misspelled("hybird", "hybrid"));
        assert!(misspelled("quantisation", "quantization"));
        assert!(!misspelled("data", "date"));
        assert!(!misspelled("points", "pints-of-beer"));
        assert!(!misspelled("search", "sparse"));
    }

    #[test]
    fn suggests_pages_for_misspelled_slugs() {
        let pages = [
            "documentation/search/filtering",
            "documentation/manage-data/collections",
            "documentation/search/search",
        ];
        assert_eq!(
            similar_paths("documentation/search/fitlering", pages, 5),
            [
                "documentation/search/filtering",
                "documentation/search/search"
            ]
        );
        assert_eq!(
            similar_paths("documentation/manage-data/colections", pages, 5),
            ["documentation/manage-data/collections"]
        );
    }

    #[test]
    fn ranks_pages_in_the_same_section_first() {
        let pages = [
            "documentation/search-patterns/vector-search-filtering",
            "documentation/search/filtering",
        ];
        assert_eq!(
            similar_paths("documentation/search/fitlering", pages, 1),
            ["documentation/search/filtering"]
        );
    }

    #[test]
    fn ranks_spelled_words_above_misspelled_ones() {
        let pages = ["documentation/a/hybrid-queries", "documentation/a/hybird"];
        assert_eq!(
            similar_paths("documentation/a/hybird-queries", pages, 2),
            ["documentation/a/hybrid-queries", "documentation/a/hybird"]
        );
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
    fn suggests_the_main_page_of_the_closest_section() {
        let pages = [
            "/documentation/search/",
            "/documentation/search/search/",
            "/documentation/search/filtering/",
        ];
        assert_eq!(
            similar_paths("/documentation/search/aaa/", pages, 5),
            ["/documentation/search/", "/documentation/search/search/"]
        );
        // A section without its own page still points to its main page.
        assert_eq!(
            similar_paths("/documentation/search/aaa/", pages[1..].iter().copied(), 5),
            ["/documentation/search/search/"]
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
