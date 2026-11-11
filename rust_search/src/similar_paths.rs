//! Find existing pages that resemble a missing one.
//!
//! The `pages` collection has a point per page with three vectors, and Qdrant
//! does the ranking in one query:
//!
//! - `slug`: the letter bigrams and trigrams of the slug, scaled to unit
//!   length, so the dot product is their cosine similarity. It finds
//!   misspellings such as `fitlering` and `serach`.
//! - `words`: `qdrant/bm25` of the slug words and the title, embedded by
//!   Qdrant itself, with the IDF modifier. Qdrant's English stemmer and
//!   stopwords make `indexes` find `indexing`, and leave out words like `and`.
//! - `meaning`: the page's title, slug words, and headings embedded with
//!   [`NEURAL_ENCODER`], so rephrased paths find pages by meaning.
//!
//! Each vector is searched with a minimum score, the hits are fused with RRF,
//! and a formula raises pages in the same source and below the same
//! ancestors as the missing path. A path that resembles nothing gets nothing.
//!
//! The `index_pages` binary builds the collection with [`page_point`], so pages
//! and queries are encoded by the same code.

use std::collections::BTreeMap;
use std::future::Future;

use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    Condition, Document, Expression, Filter, FormulaBuilder, MultExpression, NamedVectors,
    PayloadIncludeSelector, PointStruct, PrefetchQueryBuilder, Query, QueryBatchPointsBuilder,
    QueryPoints, QueryPointsBuilder, RrfBuilder, ScoredPoint, SumExpression, Vector, VectorInput,
};
use qdrant_client::{Payload, Qdrant};

use crate::embedding::QueryEncoder;
#[cfg(doc)]
use crate::embedding::NEURAL_ENCODER;
use crate::pages::{PageMeta, RENDERED_FIELD};
use crate::qdrant::PAGE_COLLECTION_NAME;

/// Letter n-grams of the slug, for misspellings.
pub const SLUG_VECTOR: &str = "slug";
/// BM25 of the slug words and title, for the same words in other forms.
pub const WORDS_VECTOR: &str = "words";
/// Dense embedding of the title, slug words, and headings, for rephrasings.
pub const MEANING_VECTOR: &str = "meaning";
/// Qdrant's built-in BM25 model: tokenizes, stems, and drops English stopwords.
pub const BM25_MODEL: &str = "qdrant/bm25";

/// Hits taken from each vector before fusion.
const CANDIDATES: u64 = 50;
/// Slugs less similar than this share only pieces of words.
const MIN_SLUG_SIMILARITY: f32 = 0.45;
/// Pages less similar in meaning than this are unrelated.
const MIN_MEANING_SIMILARITY: f32 = 0.45;
/// RRF's k: higher values flatten the difference between ranks.
const RRF_K: u32 = 10;
/// Added for sharing the deepest ancestor of the missing path, less for
/// shallower ones.
const ANCESTOR_BOOST: f32 = 0.2;
/// Added for being in the same source as the missing path.
const SOURCE_BOOST: f32 = 0.1;

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

fn letter(c: char) -> u32 {
    match c {
        'a'..='z' => c as u32 - 'a' as u32 + 1,
        '0'..='9' => c as u32 - '0' as u32 + 27,
        '^' => 37,
        '$' => 38,
        _ => 0,
    }
}

/// Bigram indices start after the trigram ones.
const BIGRAM_OFFSET: u32 = 39 * 39 * 39;

/// The sparse vector of a slug: each word of the last segment of `path`,
/// padded as `^word$`, split into overlapping trigrams and bigrams. An
/// n-gram's index is its letters in base 39, where a-z are 1-26, 0-9 are
/// 27-36, `^` is 37, `$` is 38, and anything else is 0; bigrams are offset by
/// 39³. Values count occurrences, scaled so the vector has unit length.
pub fn slug_ngrams(path: &str) -> (Vec<u32>, Vec<f32>) {
    let mut counts: BTreeMap<u32, f32> = BTreeMap::new();
    for word in slug_words(path) {
        let letters: Vec<u32> = format!("^{word}$").chars().map(letter).collect();
        for t in letters.windows(3) {
            *counts.entry(t[0] * 39 * 39 + t[1] * 39 + t[2]).or_default() += 1.0;
        }
        for b in letters.windows(2) {
            *counts.entry(BIGRAM_OFFSET + b[0] * 39 + b[1]).or_default() += 1.0;
        }
    }
    let norm = counts.values().map(|v| v * v).sum::<f32>().sqrt();
    counts
        .into_iter()
        .map(|(index, count)| (index, count / norm))
        .unzip()
}

/// The paths above `page`: `["docs", "docs/search"]` for `docs/search/filtering`.
fn page_ancestors(page: &str) -> Vec<String> {
    let segments: Vec<&str> = page.trim_matches('/').split('/').collect();
    (1..segments.len())
        .map(|n| segments[..n].join("/"))
        .collect()
}

/// A page as `index_pages` reads it from the `sections` collection.
pub struct PageText {
    pub page: String,
    pub title: String,
    pub headings: Vec<String>,
}

impl PageText {
    /// What `words` is built from: the slug words and the title.
    pub fn words(&self) -> String {
        format!("{} {}", slug_words(&self.page).join(" "), self.title)
    }

    /// What `meaning` is built from: the title, slug words, and headings.
    pub fn meaning(&self) -> String {
        format!(
            "{}. {}. {}",
            self.title,
            slug_words(&self.page).join(" "),
            self.headings.join(", ")
        )
    }
}

/// The point of a page in the `pages` collection, with its metadata.
/// `meaning` is the embedding of [`PageText::meaning`]; Qdrant computes
/// `words` itself.
pub fn page_point(id: u64, page: &PageText, meta: &PageMeta, meaning: Vector) -> PointStruct {
    let path = page.page.trim_matches('/');
    let (indices, values) = slug_ngrams(path);
    let mut payload = serde_json::json!({
        "page": path,
        "title": page.title,
        "source": source(path),
        "ancestors": page_ancestors(path),
    });
    let serde_json::Value::Object(meta) = serde_json::to_value(meta).expect("PageMeta serializes")
    else {
        unreachable!("PageMeta serializes to an object")
    };
    payload
        .as_object_mut()
        .expect("the payload is a JSON object")
        .extend(meta);
    let payload = Payload::try_from(payload).expect("the payload is a JSON object");
    PointStruct::new(
        id,
        NamedVectors::default()
            .add_vector(SLUG_VECTOR, Vector::new_sparse(indices, values))
            .add_vector(WORDS_VECTOR, Document::new(page.words(), BM25_MODEL))
            .add_vector(MEANING_VECTOR, meaning),
        payload,
    )
}

/// The query for pages resembling `path`, ranked by Qdrant. Without
/// `meaning`, the dense vector isn't searched.
pub fn similar_query(
    path: &str,
    sources: &[String],
    meaning: Option<VectorInput>,
    limit: usize,
) -> QueryPoints {
    let path = path.trim_matches('/');
    let mut filter = readable();
    if !sources.is_empty() {
        filter.must = vec![Condition::matches("source", sources.to_vec())];
    }
    let candidates = |query: Query, using: &str| {
        PrefetchQueryBuilder::default()
            .query(query)
            .using(using)
            .limit(CANDIDATES)
            .filter(filter.clone())
    };

    let (indices, values) = slug_ngrams(path);
    let mut prefetch = vec![
        candidates(
            Query::new_nearest(VectorInput::new_sparse(indices, values)),
            SLUG_VECTOR,
        )
        .score_threshold(MIN_SLUG_SIMILARITY)
        .build(),
        candidates(
            Query::new_nearest(Document::new(slug_words(path).join(" "), BM25_MODEL)),
            WORDS_VECTOR,
        )
        .build(),
    ];
    if let Some(meaning) = meaning {
        prefetch.push(
            candidates(Query::new_nearest(meaning), MEANING_VECTOR)
                .score_threshold(MIN_MEANING_SIMILARITY)
                .build(),
        );
    }
    let fused = PrefetchQueryBuilder::default()
        .prefetch(prefetch)
        .query(Query::new_rrf(RrfBuilder::with_k(RRF_K)))
        .limit(CANDIDATES);

    // Raise pages that share the source and the ancestors of the missing
    // path, the deepest ancestor most.
    let shared = page_ancestors(path).split_off(1);
    let mut sum = vec![Expression::score()];
    for (depth, ancestor) in shared.iter().enumerate() {
        let weight = ANCESTOR_BOOST * (depth + 1) as f32 / shared.len() as f32;
        sum.push(boost(
            weight,
            Condition::matches("ancestors", ancestor.clone()),
        ));
    }
    sum.push(boost(
        SOURCE_BOOST,
        Condition::matches("source", source(path).to_string()),
    ));

    QueryPointsBuilder::new(PAGE_COLLECTION_NAME)
        .add_prefetch(fused)
        .query(Query::new_formula(FormulaBuilder::new(SumExpression {
            sum,
        })))
        .limit(limit as u64)
        .with_payload(page_selector())
        .build()
}

fn boost(weight: f32, condition: Condition) -> Expression {
    MultExpression {
        mult: vec![
            Expression::constant(weight),
            Expression::condition(condition),
        ],
    }
    .into()
}

/// Pages that can be read: not sections without a page of their own.
fn readable() -> Filter {
    Filter::must_not([Condition::matches(RENDERED_FIELD, false)])
}

/// The query for those of `pages` that exist.
fn existing_query(pages: &[String]) -> QueryPoints {
    let mut filter = readable();
    filter.must = vec![Condition::matches("page", pages.to_vec())];
    QueryPointsBuilder::new(PAGE_COLLECTION_NAME)
        .filter(filter)
        .limit(pages.len().max(1) as u64)
        .with_payload(page_selector())
        .build()
}

/// A page suggested in place of a missing one.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    /// Written like `documentation/search/filtering`.
    pub page: String,
    /// Empty if unknown.
    pub title: String,
}

/// Where pages are looked up.
pub trait PageIndex {
    /// Pages resembling `path`, best first, and those of `pages` that exist,
    /// in one request.
    fn lookup(
        &self,
        path: &str,
        sources: &[String],
        meaning: Option<VectorInput>,
        limit: usize,
        pages: &[String],
    ) -> impl Future<Output = anyhow::Result<(Vec<Suggestion>, Vec<String>)>> + Send;
}

fn payload_str(point: &ScoredPoint, key: &str) -> Option<String> {
    match point.payload.get(key)?.kind.as_ref()? {
        Kind::StringValue(value) => Some(value.clone()),
        _ => None,
    }
}

fn page_selector() -> PayloadIncludeSelector {
    PayloadIncludeSelector {
        fields: vec!["page".to_string(), "title".to_string()],
    }
}

impl PageIndex for Qdrant {
    async fn lookup(
        &self,
        path: &str,
        sources: &[String],
        meaning: Option<VectorInput>,
        limit: usize,
        pages: &[String],
    ) -> anyhow::Result<(Vec<Suggestion>, Vec<String>)> {
        let mut queries = vec![similar_query(path, sources, meaning, limit)];
        if !pages.is_empty() {
            queries.push(existing_query(pages));
        }
        let mut results = self
            .query_batch(QueryBatchPointsBuilder::new(PAGE_COLLECTION_NAME, queries))
            .await?
            .result
            .into_iter()
            .map(|batch| batch.result);
        let similar = results
            .next()
            .unwrap_or_default()
            .iter()
            .filter_map(|point| {
                Some(Suggestion {
                    page: payload_str(point, "page")?,
                    title: payload_str(point, "title").unwrap_or_default(),
                })
            })
            .collect();
        let existing = results
            .next()
            .unwrap_or_default()
            .iter()
            .filter_map(|point| payload_str(point, "page"))
            .collect();
        Ok((similar, existing))
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

/// Pages to suggest for a missing `path`, best first: pages that resemble
/// it, as ranked by Qdrant, then its closest existing ancestor and that
/// ancestor's main page. With an `encoder`, pages are also matched by meaning.
pub async fn suggest(
    index: &impl PageIndex,
    encoder: Option<&QueryEncoder>,
    path: &str,
    sources: &[String],
    limit: usize,
) -> anyhow::Result<Vec<Suggestion>> {
    let levels: Vec<[String; 2]> = ancestors(path)
        .into_iter()
        .map(|ancestor| {
            let main = main_page(&ancestor);
            [ancestor, main]
        })
        .collect();
    let words = path_words(path).join(" ");
    let meaning = match encoder {
        Some(encoder) if !words.is_empty() => match encoder.encode(&words).await {
            Ok(meaning) => Some(meaning),
            Err(e) => {
                log::warn!("Suggesting without meaning: {e}");
                None
            }
        },
        _ => None,
    };
    let (mut similar, existing) = index
        .lookup(path, sources, meaning, limit, &levels.concat())
        .await?;

    let closest: Vec<Suggestion> = levels
        .into_iter()
        .find(|level| level.iter().any(|page| existing.contains(page)))
        .into_iter()
        .flatten()
        .filter(|page| existing.contains(page) && !similar.iter().any(|s| s.page == *page))
        .map(|page| Suggestion {
            page,
            title: String::new(),
        })
        .take(limit)
        .collect();
    similar.truncate(limit - closest.len());
    similar.extend(closest);
    Ok(similar)
}

/// Pages kept in memory, matched by slug similarity only, for tests without
/// Qdrant.
#[cfg(test)]
#[derive(Default)]
pub struct MemoryIndex(pub Vec<String>);

#[cfg(test)]
impl PageIndex for MemoryIndex {
    async fn lookup(
        &self,
        path: &str,
        sources: &[String],
        _meaning: Option<VectorInput>,
        limit: usize,
        pages: &[String],
    ) -> anyhow::Result<(Vec<Suggestion>, Vec<String>)> {
        let query: BTreeMap<u32, f32> = {
            let (indices, values) = slug_ngrams(path);
            indices.into_iter().zip(values).collect()
        };
        let mut hits: Vec<(f32, &String)> = self
            .0
            .iter()
            .filter(|page| sources.is_empty() || sources.iter().any(|s| s == source(page)))
            .filter_map(|page| {
                let (indices, values) = slug_ngrams(page);
                let similarity: f32 = indices
                    .iter()
                    .zip(values)
                    .filter_map(|(i, v)| Some(query.get(i)? * v))
                    .sum();
                (similarity >= MIN_SLUG_SIMILARITY).then_some((similarity, page))
            })
            .collect();
        hits.sort_by(|(a, x), (b, y)| b.total_cmp(a).then_with(|| x.cmp(y)));
        let similar = hits
            .into_iter()
            .take(limit)
            .map(|(_, page)| Suggestion {
                page: page.clone(),
                title: String::new(),
            })
            .collect();
        let existing = pages
            .iter()
            .filter(|page| self.0.contains(page))
            .cloned()
            .collect();
        Ok((similar, existing))
    }
}

#[cfg(test)]
mod tests {
    use qdrant_client::qdrant::{expression, query, Rrf};

    use super::*;

    fn index(pages: &[&str]) -> MemoryIndex {
        MemoryIndex(pages.iter().map(|p| p.to_string()).collect())
    }

    async fn suggestions(pages: &[&str], path: &str, limit: usize) -> Vec<String> {
        suggest(&index(pages), None, path, &[], limit)
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.page)
            .collect()
    }

    fn similarity(a: &str, b: &str) -> f32 {
        let (ai, av) = slug_ngrams(a);
        let (bi, bv) = slug_ngrams(b);
        let b: BTreeMap<u32, f32> = bi.into_iter().zip(bv).collect();
        ai.iter()
            .zip(av)
            .filter_map(|(i, v)| Some(b.get(i)? * v))
            .sum()
    }

    #[test]
    fn splits_paths_into_words_below_the_source() {
        assert_eq!(
            path_words("/documentation/search/hybrid-queries/"),
            ["search", "hybrid", "queries"]
        );
        assert_eq!(path_words("blog/qdrant-1.15.x"), ["qdrant", "1", "15", "x"]);
    }

    #[test]
    fn encodes_slugs_as_unit_ngram_vectors() {
        let (indices, values) = slug_ngrams("documentation/hy-1");
        // Trigrams ^hy hy$ ^1$, then bigrams ^h hy y$ ^1 1$.
        assert_eq!(indices[..3], [13181, 56614, 57407]);
        assert_eq!(indices.len(), 8);
        assert!(indices[3..].iter().all(|&i| i > BIGRAM_OFFSET));
        let length: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-6, "{length}");
    }

    #[test]
    fn misspelled_slugs_are_similar_and_others_are_not() {
        for (typo, page) in [
            ("search/fitlering", "search/filtering"),
            ("x/serach", "x/search"),
            ("x/colections", "x/collections"),
            ("x/quantisation", "x/quantization"),
            ("x/airbyt", "x/airbyte"),
        ] {
            let s = similarity(typo, page);
            assert!(s >= MIN_SLUG_SIMILARITY, "{typo} {page} {s}");
        }
        for (path, page) in [
            ("x/xyzzy-plugh", "x/why-similarity-is-not-enough"),
            ("x/backups-and-restore", "x/tutorials-and-examples"),
            ("x/disaster", "x/discovery"),
        ] {
            let s = similarity(path, page);
            assert!(s < MIN_SLUG_SIMILARITY, "{path} {page} {s}");
        }
    }

    #[test]
    fn fewer_extra_words_are_more_similar() {
        assert!(
            similarity("x/filtering", "x/filtering")
                > similarity("x/filtering", "x/text-filtering")
        );
        assert!(
            similarity("x/filtering", "x/text-filtering")
                > similarity("x/filtering", "x/pre-filtering-vs-post-filtering-tips")
        );
    }

    #[test]
    fn builds_points_with_ancestors_and_title() {
        let page = PageText {
            page: "/documentation/data-management/airbyte/".to_string(),
            title: "Airbyte".to_string(),
            headings: vec!["Setup".to_string(), "Usage".to_string()],
        };
        assert_eq!(page.words(), "airbyte Airbyte");
        assert_eq!(page.meaning(), "Airbyte. airbyte. Setup, Usage");
        let meta = PageMeta {
            partition: "ecosystem".to_string(),
            weight: 10,
            aliases: vec!["documentation/airbyte".to_string()],
            ..PageMeta::default()
        };
        let point = page_point(7, &page, &meta, Vector::new_dense(vec![0.0; 3]));
        assert_eq!(
            serde_json::Value::from(Payload::from(point.payload)),
            serde_json::json!({
                "page": "documentation/data-management/airbyte",
                "title": "Airbyte",
                "source": "documentation",
                "ancestors": ["documentation", "documentation/data-management"],
                "description": "",
                "kind": "page",
                "partition": "ecosystem",
                "weight": 10,
                "aliases": ["documentation/airbyte"],
                "rendered": true,
            })
        );
    }

    #[test]
    fn queries_three_vectors_fused_and_boosted_in_qdrant() {
        let query = similar_query(
            "documentation/data-management/connectors/airbyt",
            &["documentation".to_string()],
            Some(VectorInput::new_dense(vec![0.5; 3])),
            5,
        );
        assert_eq!(query.limit, Some(5));
        let fused = &query.prefetch[0];
        assert!(matches!(
            fused.query.as_ref().and_then(|q| q.variant.as_ref()),
            Some(query::Variant::Rrf(Rrf { k: Some(RRF_K), .. }))
        ));
        let usings: Vec<_> = fused
            .prefetch
            .iter()
            .map(|p| p.using.clone().unwrap())
            .collect();
        assert_eq!(usings, [SLUG_VECTOR, WORDS_VECTOR, MEANING_VECTOR]);
        assert!(fused.prefetch.iter().all(|p| p.filter.is_some()));
        assert_eq!(fused.prefetch[0].score_threshold, Some(MIN_SLUG_SIMILARITY));
        assert_eq!(
            fused.prefetch[2].score_threshold,
            Some(MIN_MEANING_SIMILARITY)
        );

        // $score, two ancestors below the source, and the source.
        let Some(query::Variant::Formula(formula)) =
            query.query.as_ref().and_then(|q| q.variant.as_ref())
        else {
            panic!("{query:?}")
        };
        let Some(expression::Variant::Sum(sum)) =
            formula.expression.as_ref().and_then(|e| e.variant.as_ref())
        else {
            panic!("{formula:?}")
        };
        assert_eq!(sum.sum.len(), 4);
    }

    #[test]
    fn searches_meaning_only_with_an_encoder() {
        let query = similar_query("documentation/x", &[], None, 5);
        assert_eq!(query.prefetch[0].prefetch.len(), 2);
        // Without sources, only sections without a page are left out.
        assert!(query.prefetch[0]
            .prefetch
            .iter()
            .all(|p| p.filter == Some(readable())));
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

    #[tokio::test]
    async fn suggests_nothing_for_unrelated_paths() {
        let pages = [
            "documentation/concepts/collections",
            "documentation/a/nothing",
        ];
        assert!(suggestions(&pages, "documentation/b/xyzzy-plugh", 5)
            .await
            .is_empty());
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
}
