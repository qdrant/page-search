use actix_web::web::{Data, Query};
use actix_web::{get, HttpRequest, HttpResponse};
use qdrant_client::qdrant::facet_value::Variant as FacetVariant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, Document, FacetCountsBuilder, Filter, PointId,
    PrefetchQueryBuilder, QueryPointsBuilder, Rrf, ScoredPoint, VectorInput,
};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};

use super::models::{Snippet, SnippetSearchResult};
use crate::qdrant::SNIPPET_COLLECTION_NAME;

const SNIPPET_ENCODER: &str = "mixedbread-ai/mxbai-embed-large-v1";

fn default_limit() -> u64 {
    3
}

#[derive(Deserialize)]
struct SnippetSearch {
    query: String,
    language: String,
    #[serde(default = "default_limit")]
    limit: u64,
    format: Option<String>,
}

fn parse_snippets(points: Vec<ScoredPoint>) -> Vec<Snippet> {
    points
        .into_iter()
        .filter_map(|p| Snippet::from_payload(p.payload))
        .collect()
}

/// The dense query: an embedding of the description.
fn dense_query(query: &str) -> VectorInput {
    VectorInput::from(Document::new(query, SNIPPET_ENCODER))
}

/// The sparse query: BM25 over the code.
fn sparse_query(query: &str) -> VectorInput {
    let mut bm25_doc = Document::new(query, "qdrant/bm25");
    bm25_doc.options.insert(
        "language".to_string(),
        qdrant_client::qdrant::Value {
            kind: Some(Kind::StringValue("none".to_string())),
        },
    );
    VectorInput::from(bm25_doc)
}

fn language_filter(language: &str) -> Filter {
    Filter::must(vec![Condition::matches(
        "language",
        MatchValue::Keyword(language.to_string()),
    )])
}

/// How many candidates each prefetch gives the fusion.
fn prefetch_limit(limit: u64) -> u64 {
    limit.max(20)
}

fn hybrid_request(query: &str, language: Option<&str>, limit: u64) -> QueryPointsBuilder {
    let prefetch_limit = prefetch_limit(limit);
    let mut request = QueryPointsBuilder::new(SNIPPET_COLLECTION_NAME)
        .add_prefetch(
            PrefetchQueryBuilder::default()
                .query(dense_query(query))
                .using("dense")
                .limit(prefetch_limit)
                .build(),
        )
        .add_prefetch(
            PrefetchQueryBuilder::default()
                .query(sparse_query(query))
                .using("sparse")
                .limit(prefetch_limit)
                .build(),
        )
        .query(qdrant_client::qdrant::Query::new_rrf(Rrf {
            k: Some(1),
            ..Default::default()
        }))
        .limit(limit)
        .with_payload(true);
    if let Some(language) = language {
        request = request.filter(language_filter(language));
    }
    request
}

/// Hybrid search over the snippets: the dense vector embeds the description,
/// the sparse one the code. Without a language, snippets in every language.
pub async fn search_snippets(
    client: &Qdrant,
    query: &str,
    language: Option<&str>,
    limit: u64,
) -> anyhow::Result<Vec<Snippet>> {
    let result = client.query(hybrid_request(query, language, limit)).await?;
    Ok(parse_snippets(result.result))
}

/// A snippet with its score, for the admin panel's search debugger.
#[derive(Debug, Serialize)]
pub struct ScoredSnippet {
    pub id: String,
    pub score: f32,
    #[serde(flatten)]
    pub snippet: Snippet,
}

/// What each half of the hybrid search finds, and what fusing them gives.
#[derive(Debug, Serialize)]
pub struct SearchBreakdown {
    /// The search as `/snippets/search` and the MCP server run it.
    pub fused: Vec<ScoredSnippet>,
    /// The dense prefetch alone, as many as the fusion sees.
    pub dense: Vec<ScoredSnippet>,
    /// The sparse prefetch alone, as many as the fusion sees.
    pub sparse: Vec<ScoredSnippet>,
    pub prefetch_limit: u64,
}

fn point_id(id: Option<PointId>) -> String {
    match id.and_then(|id| id.point_id_options) {
        Some(PointIdOptions::Num(n)) => n.to_string(),
        Some(PointIdOptions::Uuid(u)) => u,
        None => String::new(),
    }
}

fn scored(points: Vec<ScoredPoint>) -> Vec<ScoredSnippet> {
    points
        .into_iter()
        .filter_map(|p| {
            Some(ScoredSnippet {
                score: p.score,
                snippet: Snippet::from_payload(p.payload)?,
                id: point_id(p.id),
            })
        })
        .collect()
}

/// Runs the hybrid search and each of its prefetches on its own.
pub async fn search_breakdown(
    client: &Qdrant,
    query: &str,
    language: Option<&str>,
    limit: u64,
) -> anyhow::Result<SearchBreakdown> {
    let prefetch_limit = prefetch_limit(limit);
    let single = |input: VectorInput, using: &str| {
        let mut request = QueryPointsBuilder::new(SNIPPET_COLLECTION_NAME)
            .query(input)
            .using(using)
            .limit(prefetch_limit)
            .with_payload(true);
        if let Some(language) = language {
            request = request.filter(language_filter(language));
        }
        request
    };
    let (fused, dense, sparse) = futures::try_join!(
        client.query(hybrid_request(query, language, limit)),
        client.query(single(dense_query(query), "dense")),
        client.query(single(sparse_query(query), "sparse")),
    )?;
    Ok(SearchBreakdown {
        fused: scored(fused.result),
        dense: scored(dense.result),
        sparse: scored(sparse.result),
        prefetch_limit,
    })
}

#[derive(Debug, Serialize)]
pub struct ValueCount {
    pub value: String,
    pub count: u64,
}

/// What the snippet collection holds.
#[derive(Debug, Serialize)]
pub struct IndexOverview {
    pub total: u64,
    pub languages: Vec<ValueCount>,
    pub versions: Vec<ValueCount>,
}

/// Snippet counts in total, by language and by version.
pub async fn index_overview(client: &Qdrant) -> anyhow::Result<IndexOverview> {
    let facet = |key: &str| {
        client.facet(
            FacetCountsBuilder::new(SNIPPET_COLLECTION_NAME, key)
                .limit(500)
                .exact(true),
        )
    };
    let (total, languages, versions) = futures::try_join!(
        client.count(CountPointsBuilder::new(SNIPPET_COLLECTION_NAME).exact(true)),
        facet("language"),
        facet("version"),
    )?;
    let values = |response: qdrant_client::qdrant::FacetResponse| {
        response
            .hits
            .into_iter()
            .filter_map(|hit| {
                let value = match hit.value?.variant? {
                    FacetVariant::StringValue(s) => s,
                    FacetVariant::IntegerValue(i) => i.to_string(),
                    FacetVariant::BoolValue(b) => b.to_string(),
                };
                Some(ValueCount {
                    value,
                    count: hit.count,
                })
            })
            .collect()
    };
    Ok(IndexOverview {
        total: total.result.map_or(0, |r| r.count),
        languages: values(languages),
        versions: values(versions),
    })
}

fn resolve_format(format_param: Option<&str>, accept: Option<&str>) -> &'static str {
    if let Some(f) = format_param {
        match f.trim().to_lowercase().as_str() {
            "json" => return "json",
            "markdown" => return "markdown",
            _ => {}
        }
    }
    if let Some(accept) = accept {
        for part in accept.split(',') {
            let mime = part.split(';').next().unwrap_or("").trim().to_lowercase();
            if mime == "application/json" {
                return "json";
            }
            if mime == "text/markdown" {
                return "markdown";
            }
        }
    }
    "markdown"
}

#[get("/snippets/search")]
pub async fn search_handler(
    req: HttpRequest,
    query: Query<SnippetSearch>,
    qdrant: Data<Qdrant>,
) -> HttpResponse {
    let SnippetSearch {
        query: q,
        language,
        limit,
        format,
    } = query.into_inner();
    let accept = req.headers().get("accept").and_then(|v| v.to_str().ok());
    let fmt = resolve_format(format.as_deref(), accept);

    match search_snippets(qdrant.get_ref(), &q, Some(&language), limit)
        .await
        .map(SnippetSearchResult)
    {
        Ok(result) => {
            log::info!("snippets={}", result.0.len());
            if fmt == "json" {
                HttpResponse::Ok()
                    .content_type("application/json")
                    .body(result.to_json())
            } else {
                HttpResponse::Ok()
                    .content_type("text/markdown; charset=utf-8")
                    .body(result.to_markdown())
            }
        }
        Err(e) => {
            log::error!("Snippet search error: {}", e);
            HttpResponse::InternalServerError().body(e.to_string())
        }
    }
}
