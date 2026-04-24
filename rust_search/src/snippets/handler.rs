use actix_web::web::{Data, Query};
use actix_web::{get, HttpRequest, HttpResponse};
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    Condition, Document, FacetCountsBuilder, Filter, PrefetchQueryBuilder, QueryPointsBuilder, Rrf,
    ScoredPoint, VectorInput,
};
use qdrant_client::Qdrant;
use serde::Deserialize;

use super::models::{Snippet, SnippetSearchResult};

const SNIPPET_COLLECTION_NAME: &str = "snippet-search";
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

async fn find_latest_revision(client: &Qdrant) -> anyhow::Result<i64> {
    let result = client
        .facet(FacetCountsBuilder::new(SNIPPET_COLLECTION_NAME, "revision").limit(1_000_000))
        .await?;
    let max = result
        .hits
        .into_iter()
        .filter_map(|hit| {
            hit.value.and_then(|v| match v.variant? {
                qdrant_client::qdrant::facet_value::Variant::IntegerValue(n) => Some(n),
                _ => None,
            })
        })
        .max()
        .unwrap_or(0);
    Ok(max)
}

fn parse_snippets(points: Vec<ScoredPoint>) -> Vec<Snippet> {
    points
        .into_iter()
        .filter_map(|p| Snippet::from_payload(p.payload))
        .collect()
}

async fn search_snippets(
    client: &Qdrant,
    query: &str,
    language: &str,
    limit: u64,
) -> anyhow::Result<SnippetSearchResult> {
    let revision = find_latest_revision(client).await?;

    let mut bm25_doc = Document::new(query, "qdrant/bm25");
    bm25_doc.options.insert(
        "language".to_string(),
        qdrant_client::qdrant::Value {
            kind: Some(Kind::StringValue("none".to_string())),
        },
    );

    let result = client
        .query(
            QueryPointsBuilder::new(SNIPPET_COLLECTION_NAME)
                .add_prefetch(
                    PrefetchQueryBuilder::default()
                        .query(VectorInput::from(Document::new(query, SNIPPET_ENCODER)))
                        .using("dense")
                        .limit(20u64)
                        .build(),
                )
                .add_prefetch(
                    PrefetchQueryBuilder::default()
                        .query(VectorInput::from(bm25_doc))
                        .using("sparse")
                        .limit(20u64)
                        .build(),
                )
                .query(qdrant_client::qdrant::Query::new_rrf(Rrf {
                    k: Some(1),
                    ..Default::default()
                }))
                .filter(Filter::must(vec![
                    Condition::matches("language", MatchValue::Keyword(language.to_string())),
                    Condition::matches("revision", MatchValue::Integer(revision)),
                ]))
                .limit(limit)
                .with_payload(true),
        )
        .await?;

    Ok(SnippetSearchResult(parse_snippets(result.result)))
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

    match search_snippets(qdrant.get_ref(), &q, &language, limit).await {
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
