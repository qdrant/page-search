//! Find existing pages whose paths resemble a missing one.
//!
//! The `pages` collection has a point per page with a sparse vector of the
//! letter trigrams of its slug, so Qdrant finds slugs that share pieces of
//! words, misspelled ones included, and its IDF weighting makes rare pieces
//! count more. The best hits are then ranked here by whole words.

use std::collections::BTreeMap;
use std::future::Future;

use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    Condition, Filter, PayloadIncludeSelector, Query, QueryPointsBuilder, ScrollPointsBuilder,
    Value, VectorInput,
};
use qdrant_client::Qdrant;

use crate::common::PAGE_COLLECTION_NAME;

/// The sparse vector of a page's slug in the `pages` collection.
pub const SLUG_VECTOR: &str = "slug";
/// Hits from Qdrant to rank for a missing path.
const CANDIDATES: u64 = 50;

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

fn trigram_letter(c: char) -> u32 {
    match c {
        'a'..='z' => c as u32 - 'a' as u32 + 1,
        '0'..='9' => c as u32 - '0' as u32 + 27,
        '^' => 37,
        '$' => 38,
        _ => 0,
    }
}

/// The sparse vector of a slug: each word of the last segment of `path`,
/// padded as `^word$`, split into overlapping trigrams. A trigram's index is
/// its letters in base 39, where a-z are 1-26, 0-9 are 27-36, `^` is 37, `$`
/// is 38, and anything else is 0. Its value is how often it occurs.
///
/// `site_search/page_paths.py` builds the same vectors for the index.
pub fn slug_trigrams(path: &str) -> (Vec<u32>, Vec<f32>) {
    let mut counts: BTreeMap<u32, f32> = BTreeMap::new();
    for word in slug_words(path) {
        let letters: Vec<u32> = format!("^{word}$").chars().map(trigram_letter).collect();
        for trigram in letters.windows(3) {
            *counts
                .entry(trigram[0] * 39 * 39 + trigram[1] * 39 + trigram[2])
                .or_default() += 1.0;
        }
    }
    counts.into_iter().unzip()
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

fn words_match(a: &str, b: &str) -> bool {
    same_word(a, b) || misspelled(a, b)
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

/// Whether `page` is below `section`: `docs/search/filtering` is below `docs/search`.
fn is_under(page: &str, section: &str) -> bool {
    page.trim_matches('/')
        .strip_prefix(section.trim_matches('/'))
        .is_some_and(|rest| rest.starts_with('/'))
}

/// Where pages are looked up, written like `documentation/search/filtering`.
pub trait PageIndex {
    /// Pages whose slug shares trigrams with the slug of `path`, with their
    /// scores, best first. Empty `sources` means all of them, and `within`
    /// keeps only pages under that path.
    fn search(
        &self,
        path: &str,
        sources: &[String],
        within: Option<&str>,
        limit: u64,
    ) -> impl Future<Output = anyhow::Result<Vec<(String, f32)>>> + Send;

    /// Those of `pages` that exist.
    fn existing(
        &self,
        pages: &[String],
    ) -> impl Future<Output = anyhow::Result<Vec<String>>> + Send;
}

fn page_of(payload: &std::collections::HashMap<String, Value>) -> Option<String> {
    match payload.get("page")?.kind.as_ref()? {
        Kind::StringValue(page) => Some(page.clone()),
        _ => None,
    }
}

fn page_selector() -> PayloadIncludeSelector {
    PayloadIncludeSelector {
        fields: vec!["page".to_string()],
    }
}

impl PageIndex for Qdrant {
    async fn search(
        &self,
        path: &str,
        sources: &[String],
        within: Option<&str>,
        limit: u64,
    ) -> anyhow::Result<Vec<(String, f32)>> {
        let (indices, values) = slug_trigrams(path);
        if indices.is_empty() {
            return Ok(vec![]);
        }
        let mut conditions = vec![];
        if !sources.is_empty() {
            conditions.push(Condition::matches("source", sources.to_vec()));
        }
        if let Some(within) = within {
            conditions.push(Condition::matches(
                "ancestors",
                within.trim_matches('/').to_string(),
            ));
        }
        let query = QueryPointsBuilder::new(PAGE_COLLECTION_NAME)
            .query(Query::new_nearest(VectorInput::new_sparse(indices, values)))
            .using(SLUG_VECTOR)
            .filter(Filter::must(conditions))
            .limit(limit)
            .with_payload(page_selector());
        Ok(self
            .query(query)
            .await?
            .result
            .into_iter()
            .filter_map(|point| Some((page_of(&point.payload)?, point.score)))
            .collect())
    }

    async fn existing(&self, pages: &[String]) -> anyhow::Result<Vec<String>> {
        if pages.is_empty() {
            return Ok(vec![]);
        }
        let response = self
            .scroll(
                ScrollPointsBuilder::new(PAGE_COLLECTION_NAME)
                    .filter(Filter::must([Condition::matches("page", pages.to_vec())]))
                    .limit(pages.len() as u32)
                    .with_payload(page_selector())
                    .with_vectors(false),
            )
            .await?;
        Ok(response
            .result
            .iter()
            .filter_map(|point| page_of(&point.payload))
            .collect())
    }
}

/// Search hits resembling `path`, best first. A hit must share a word, or a
/// misspelling of one, with the last segment of `path`, since trigrams also
/// match unrelated words. Same source, same section, words of the parent
/// path, and being below `prefix`, the closest existing ancestor, raise a hit.
fn rank(path: &str, hits: &[(String, f32)], prefix: Option<&str>) -> Vec<String> {
    let top = hits.iter().map(|(_, score)| *score).fold(0.0, f32::max);
    let missing_slug_words = slug_words(path);
    let parent_words = path_words(parent(path));
    let mut ranked: Vec<(f32, &str)> = hits
        .iter()
        .filter(|(_, score)| *score > 0.0)
        .filter_map(|(page, score)| {
            let page_slug_words = slug_words(page);
            let shares_a_word = missing_slug_words
                .iter()
                .any(|w| page_slug_words.iter().any(|p| words_match(w, p)));
            if !shares_a_word {
                return None;
            }
            let page_words = path_words(page);
            let shared_parent_words = parent_words
                .iter()
                .filter(|w| page_words.iter().any(|p| words_match(w, p)))
                .count();
            let flag = |condition: bool| f32::from(u8::from(condition));
            let rank = score / top
                + flag(slug(page) == slug(path))
                + 0.5 * flag(source(page) == source(path))
                + 0.25 * flag(parent(page) == parent(path))
                + 0.25 * shared_parent_words as f32 / parent_words.len().max(1) as f32
                + 0.5 * flag(prefix.is_some_and(|prefix| is_under(page, prefix)));
            Some((rank, page.as_str()))
        })
        .collect();
    // Ties go to the slug with fewer words.
    ranked.sort_by(|(a, x), (b, y)| {
        b.total_cmp(a)
            .then_with(|| slug_words(x).len().cmp(&slug_words(y).len()))
            .then_with(|| x.cmp(y))
    });
    let mut pages: Vec<String> = ranked
        .into_iter()
        .map(|(_, page)| page.to_string())
        .collect();
    pages.dedup();
    pages
}

/// Pages to suggest for a missing `path`, best first: pages with a similar
/// slug, preferring those below the closest existing ancestor, then that
/// ancestor and its main page.
pub async fn suggest(
    index: &impl PageIndex,
    path: &str,
    sources: &[String],
    limit: usize,
) -> anyhow::Result<Vec<String>> {
    let levels: Vec<[String; 2]> = ancestors(path)
        .into_iter()
        .map(|ancestor| {
            let main = main_page(&ancestor);
            [ancestor, main]
        })
        .collect();
    let existing = index.existing(&levels.concat()).await?;
    let closest = levels
        .into_iter()
        .find(|level| level.iter().any(|page| existing.contains(page)));
    let prefix = closest.as_ref().map(|[ancestor, _]| ancestor.as_str());

    // Searched separately, so pages below the prefix aren't crowded out of
    // the hits by similar slugs elsewhere.
    let within = async {
        match prefix {
            Some(prefix) => index.search(path, sources, Some(prefix), CANDIDATES).await,
            None => Ok(vec![]),
        }
    };
    let (mut hits, within) =
        futures::try_join!(index.search(path, sources, None, CANDIDATES), within)?;
    for hit in within {
        if !hits.iter().any(|(page, _)| *page == hit.0) {
            hits.push(hit);
        }
    }
    let mut similar = rank(path, &hits, prefix);
    similar.truncate(limit);

    let closest: Vec<String> = closest
        .into_iter()
        .flatten()
        .filter(|page| existing.contains(page) && !similar.contains(page))
        .take(limit)
        .collect();
    similar.truncate(limit - closest.len());
    similar.extend(closest);
    Ok(similar)
}

/// Pages kept in memory, scored by the trigrams they share with the query.
#[cfg(test)]
#[derive(Default)]
pub struct MemoryIndex(pub Vec<String>);

#[cfg(test)]
impl PageIndex for MemoryIndex {
    async fn search(
        &self,
        path: &str,
        sources: &[String],
        within: Option<&str>,
        limit: u64,
    ) -> anyhow::Result<Vec<(String, f32)>> {
        let (query, _) = slug_trigrams(path);
        let mut hits: Vec<(String, f32)> = self
            .0
            .iter()
            .filter(|page| sources.is_empty() || sources.iter().any(|s| s == source(page)))
            .filter(|page| within.is_none_or(|w| is_under(page, w)))
            .map(|page| {
                let (trigrams, _) = slug_trigrams(page);
                let shared = trigrams.iter().filter(|t| query.contains(t)).count();
                (page.clone(), shared as f32)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();
        hits.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        hits.truncate(limit as usize);
        Ok(hits)
    }

    async fn existing(&self, pages: &[String]) -> anyhow::Result<Vec<String>> {
        Ok(pages
            .iter()
            .filter(|page| self.0.contains(page))
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(pages: &[&str]) -> MemoryIndex {
        MemoryIndex(pages.iter().map(|p| p.to_string()).collect())
    }

    async fn suggestions(pages: &[&str], path: &str, limit: usize) -> Vec<String> {
        suggest(&index(pages), path, &[], limit).await.unwrap()
    }

    #[test]
    fn splits_paths_into_words_below_the_source() {
        assert_eq!(
            path_words("/documentation/search/hybrid-queries/"),
            ["search", "hybrid", "queries"]
        );
        assert_eq!(path_words("blog/qdrant-1.15.x"), ["qdrant", "1", "15", "x"]);
    }

    /// The same vectors as `site_search/page_paths.py` builds.
    #[test]
    fn encodes_slugs_as_trigrams() {
        assert_eq!(
            slug_trigrams("documentation/hy-1"),
            (vec![13181, 56614, 57407], vec![1.0, 1.0, 1.0])
        );
        assert_eq!(
            slug_trigrams("/blog/Aa_aa/"),
            (vec![1598, 56317], vec![2.0, 2.0])
        );
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
    fn lists_ancestors_below_the_source() {
        assert_eq!(
            ancestors("/documentation/a/b/c/"),
            ["/documentation/a/b/", "/documentation/a/"]
        );
        assert_eq!(ancestors("documentation/a/b"), ["documentation/a"]);
        assert!(ancestors("documentation/a").is_empty());
    }

    #[tokio::test]
    async fn ranks_moved_pages_first() {
        let pages = [
            "documentation/search/filtering",
            "documentation/concepts/collections",
            "documentation/guides/filtering-tips",
            "documentation/concepts/points",
        ];
        assert_eq!(
            suggestions(&pages, "documentation/concepts/filtering", 5).await,
            [
                "documentation/search/filtering",
                "documentation/guides/filtering-tips"
            ]
        );
    }

    #[tokio::test]
    async fn needs_a_word_of_the_last_segment() {
        let pages = [
            "documentation/concepts/collections",
            "documentation/a/nothing",
        ];
        assert!(suggestions(&pages, "documentation/b/nope-ing", 5)
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn suggests_pages_for_misspelled_slugs() {
        let pages = [
            "documentation/search/filtering",
            "documentation/manage-data/collections",
            "documentation/search/search",
        ];
        assert_eq!(
            suggestions(&pages, "documentation/search/fitlering", 5).await,
            [
                "documentation/search/filtering",
                "documentation/search/search"
            ]
        );
        assert_eq!(
            suggestions(&pages, "documentation/manage-data/colections", 5).await,
            ["documentation/manage-data/collections"]
        );
    }

    #[test]
    fn higher_scores_rank_first() {
        // Qdrant scores rare trigrams higher: "quantization" outweighs "vector".
        let hits = [
            ("articles/what-is-vector-quantization".to_string(), 10.0),
            ("documentation/manage-data/vectors".to_string(), 4.0),
            ("documentation/manage-data/quantization".to_string(), 7.0),
        ];
        assert_eq!(
            rank(
                "documentation/data-management/vector-quantisation",
                &hits,
                None
            )[0],
            "documentation/manage-data/quantization"
        );
    }

    #[test]
    fn ranks_pages_in_the_same_section_first() {
        let hits = [
            (
                "documentation/search-patterns/vector-search-filtering".to_string(),
                5.0,
            ),
            ("documentation/search/filtering".to_string(), 5.0),
        ];
        assert_eq!(
            rank("documentation/search/fitlering", &hits, None)[0],
            "documentation/search/filtering"
        );
    }

    #[test]
    fn checks_pages_below_a_section() {
        assert!(is_under(
            "documentation/search/filtering",
            "documentation/search"
        ));
        assert!(is_under(
            "/documentation/search/a/b/",
            "/documentation/search/"
        ));
        assert!(!is_under(
            "documentation/search-tuning/x",
            "documentation/search"
        ));
        assert!(!is_under("documentation/search", "documentation/search"));
    }

    #[test]
    fn prefers_pages_below_the_prefix() {
        let hits = [
            ("documentation/other/airbyte".to_string(), 1.0),
            (
                "documentation/data-management/connectors/airbyte".to_string(),
                0.6,
            ),
        ];
        let path = "documentation/data-management/airbyt";
        assert_eq!(rank(path, &hits, None)[0], "documentation/other/airbyte");
        assert_eq!(
            rank(path, &hits, Some("documentation/data-management"))[0],
            "documentation/data-management/connectors/airbyte"
        );
    }

    #[tokio::test]
    async fn searches_below_the_closest_existing_ancestor() {
        let pages = [
            "documentation/data-management",
            "documentation/data-management/connectors/airbyte-sync",
            "documentation/other/airbyte",
        ];
        assert_eq!(
            suggestions(&pages, "documentation/data-management/airbyt", 5).await,
            [
                "documentation/data-management/connectors/airbyte-sync",
                "documentation/other/airbyte",
                "documentation/data-management"
            ]
        );
    }

    #[tokio::test]
    async fn suggests_the_closest_existing_ancestor() {
        let pages = [
            "documentation/manage-data",
            "documentation/manage-data/points",
            "documentation/manage-data/vectors",
        ];
        assert_eq!(
            suggestions(&pages, "documentation/manage-data/points/one/two/three", 5).await,
            ["documentation/manage-data/points"]
        );
    }

    #[tokio::test]
    async fn ancestor_comes_after_similar_paths() {
        let pages = ["documentation/concepts", "documentation/search/filtering"];
        assert_eq!(
            suggestions(&pages, "documentation/concepts/filtering", 5).await,
            ["documentation/search/filtering", "documentation/concepts"]
        );
        assert_eq!(
            suggestions(&pages, "documentation/concepts/filtering", 1).await,
            ["documentation/concepts"]
        );
    }

    #[tokio::test]
    async fn suggests_the_main_page_of_the_closest_section() {
        let pages = [
            "/documentation/search/",
            "/documentation/search/search/",
            "/documentation/search/filtering/",
        ];
        assert_eq!(
            suggestions(&pages, "/documentation/search/aaa/", 5).await,
            ["/documentation/search/", "/documentation/search/search/"]
        );
        // A section without its own page still points to its main page.
        assert_eq!(
            suggestions(&pages[1..], "/documentation/search/aaa/", 5).await,
            ["/documentation/search/search/"]
        );
    }

    #[tokio::test]
    async fn searches_only_the_given_sources() {
        let index = index(&["blog/filtering", "documentation/search/filtering"]);
        let sources = ["documentation".to_string()];
        assert_eq!(
            suggest(&index, "blog/filtering-tips", &sources, 5)
                .await
                .unwrap(),
            ["documentation/search/filtering"]
        );
    }
}
