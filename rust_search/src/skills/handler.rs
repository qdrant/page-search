use actix_web::web::{Data, Query};
use actix_web::{get, HttpRequest, HttpResponse};
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    Condition, Document, Filter, QueryPointsBuilder, ScoredPoint, VectorInput,
};
use qdrant_client::Qdrant;
use serde::Deserialize;

use super::models::{Skill, SkillSearchResult};

fn slugify_heading(title: &str) -> String {
    let s = title.to_lowercase();
    let s = s.trim().to_string();
    let re_spaces = regex::Regex::new(r"[\s\-_]+").unwrap();
    let s = re_spaces.replace_all(&s, "-").to_string();
    let re_nonword = regex::Regex::new(r"[^\w\-]").unwrap();
    re_nonword.replace_all(&s, "").to_string()
}

const SKILLS_COLLECTION_NAME: &str = "skills";
const SKILLS_ENCODER: &str = "mixedbread-ai/mxbai-embed-large-v1";

fn skills_exact_limit() -> u64 {
    std::env::var("SKILLS_EXACT_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100)
}

fn skills_search_limit() -> u64 {
    std::env::var("SKILLS_SEARCH_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

#[derive(Deserialize)]
struct SkillSearch {
    q: Option<String>,
}

fn parse_skills(points: Vec<ScoredPoint>) -> Vec<Skill> {
    points
        .into_iter()
        .filter_map(|p| Skill::from_payload(p.payload))
        .collect()
}

fn build_conditions(path: &str, query: Option<&str>) -> Vec<Condition> {
    if query.is_some() && !path.is_empty() {
        vec![Condition::matches(
            "parent_pages",
            MatchValue::Keyword(path.to_string()),
        )]
    } else if !path.is_empty() {
        vec![Condition::matches(
            "page",
            MatchValue::Keyword(path.to_string()),
        )]
    } else {
        vec![]
    }
}

async fn query_by_filter(
    client: &Qdrant,
    conditions: Vec<Condition>,
    limit: u64,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let mut builder = QueryPointsBuilder::new(SKILLS_COLLECTION_NAME)
        .limit(limit)
        .with_payload(true);
    if !conditions.is_empty() {
        builder = builder.filter(Filter::must(conditions));
    }
    Ok(client.query(builder).await?.result)
}

async fn query_by_document(
    client: &Qdrant,
    query: &str,
    conditions: Vec<Condition>,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let mut builder = QueryPointsBuilder::new(SKILLS_COLLECTION_NAME)
        .query(VectorInput::from(Document::new(query, SKILLS_ENCODER)))
        .limit(skills_search_limit())
        .with_payload(true);
    if !conditions.is_empty() {
        builder = builder.filter(Filter::must(conditions));
    }
    Ok(client.query(builder).await?.result)
}

async fn search_skills(
    client: &Qdrant,
    query: Option<&str>,
    path: &str,
) -> anyhow::Result<SkillSearchResult> {
    let clean_path = path.trim_matches('/');
    let conditions = build_conditions(clean_path, query);

    if let Some(q) = query {
        // Try exact name match first
        let slug = slugify_heading(q);
        let mut exact_conditions = conditions.clone();
        exact_conditions.push(Condition::matches("name", MatchValue::Keyword(slug)));
        let points = query_by_filter(client, exact_conditions, skills_exact_limit()).await?;
        if !points.is_empty() {
            return Ok(SkillSearchResult(parse_skills(points)));
        }
        // Fallback to vector search
        let points = query_by_document(client, q, conditions).await?;
        return Ok(SkillSearchResult(parse_skills(points)));
    }

    // No query — browse by page or all
    let points = query_by_filter(client, conditions, skills_exact_limit()).await?;
    Ok(SkillSearchResult(parse_skills(points)))
}

#[get("/skills/{path:.*}")]
pub async fn skills_handler(
    path: actix_web::web::Path<String>,
    _req: HttpRequest,
    query: Query<SkillSearch>,
    qdrant: Data<Qdrant>,
) -> HttpResponse {
    let path_str = path.into_inner();

    match search_skills(qdrant.get_ref(), query.q.as_deref(), &path_str).await {
        Ok(result) => {
            log::info!("skills={}", result.0.len());
            HttpResponse::Ok()
                .content_type("text/markdown; charset=utf-8")
                .body(result.to_markdown())
        }
        Err(e) => {
            log::error!("Skills search error: {}", e);
            HttpResponse::InternalServerError().body(e.to_string())
        }
    }
}
