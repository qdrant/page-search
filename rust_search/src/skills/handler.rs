use std::time::Instant;

use actix_web::web::{Data, Query};
use actix_web::{get, HttpRequest, HttpResponse};
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    Condition, Document, Filter, PrefetchQueryBuilder, QueryPointsBuilder, ScoredPoint, VectorInput,
};
use qdrant_client::Qdrant;
use serde::Deserialize;

use super::fusion::{self, Strategy};
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
        .unwrap_or(1)
}

#[derive(Deserialize)]
struct SkillSearch {
    #[serde(alias = "query")]
    q: Option<String>,
}

fn parse_skills(points: Vec<ScoredPoint>) -> Vec<Skill> {
    points
        .into_iter()
        .filter_map(|p| Skill::from_payload(p.payload))
        .collect()
}

fn build_conditions(path: &str, query: Option<&str>) -> Vec<Condition> {
    if query.is_some() {
        if path.is_empty() {
            return vec![];
        }
        vec![Condition::matches(
            "parent_pages",
            MatchValue::Keyword(path.to_string()),
        )]
    } else {
        vec![Condition::matches(
            "page",
            MatchValue::Keyword(path.to_string()),
        )]
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

async fn query_hybrid(
    client: &Qdrant,
    query: &str,
    conditions: Vec<Condition>,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let mut builder = QueryPointsBuilder::new(SKILLS_COLLECTION_NAME)
        .add_prefetch(
            PrefetchQueryBuilder::default()
                .query(VectorInput::from(Document::new(query, SKILLS_ENCODER)))
                .using("dense")
                .limit(20u64)
                .build(),
        )
        .add_prefetch(
            PrefetchQueryBuilder::default()
                .query(VectorInput::from(Document::new(query, "qdrant/bm25")))
                .using("sparse")
                .limit(20u64)
                .build(),
        )
        .query(qdrant_client::qdrant::Fusion::Rrf)
        .limit(skills_search_limit())
        .with_payload(true);
    if !conditions.is_empty() {
        builder = builder.filter(Filter::must(conditions));
    }
    Ok(client.query(builder).await?.result)
}

async fn query_dense(
    client: &Qdrant,
    query: &str,
    conditions: Vec<Condition>,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let mut builder = QueryPointsBuilder::new(SKILLS_COLLECTION_NAME)
        .query(VectorInput::from(Document::new(query, SKILLS_ENCODER)))
        .using("dense")
        .limit(skills_search_limit())
        .with_payload(true);
    if !conditions.is_empty() {
        builder = builder.filter(Filter::must(conditions));
    }
    Ok(client.query(builder).await?.result)
}

async fn query_bm25(
    client: &Qdrant,
    query: &str,
    conditions: Vec<Condition>,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let mut builder = QueryPointsBuilder::new(SKILLS_COLLECTION_NAME)
        .query(VectorInput::from(Document::new(query, "qdrant/bm25")))
        .using("sparse")
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
        // Run exact-name lookup in parallel with the fusion guidance calls;
        // when exact misses (common case) we don't pay them sequentially.
        let slug = slugify_heading(q);
        let mut exact_conditions = conditions.clone();
        exact_conditions.push(Condition::matches("name", MatchValue::Keyword(slug)));

        let t_par = Instant::now();
        let (exact_res, score_res, tokens_res) = tokio::join!(
            query_by_filter(client, exact_conditions, skills_exact_limit()),
            fusion::classify(q),
            fusion::extract(q),
        );
        let par_ms = t_par.elapsed().as_millis();

        let exact_points = exact_res?;
        log::debug!(
            "exact+fusion took={}ms exact_hits={}",
            par_ms,
            exact_points.len()
        );
        if !exact_points.is_empty() {
            return Ok(SkillSearchResult::with_strategy(
                parse_skills(exact_points),
                "exact-name",
            ));
        }

        let score = score_res
            .map_err(|e| log::warn!("fusion.classify failed: {e}"))
            .ok();
        let tokens = tokens_res
            .map_err(|e| log::warn!("fusion.extract failed: {e}"))
            .unwrap_or_default();

        log::debug!("fusion: q={:?} score={:?} tokens={:?}", q, score, tokens);

        // Tokens present → require them in content + dense search; fallback to hybrid.
        if !tokens.is_empty() {
            let mut filtered = conditions.clone();
            for tok in &tokens {
                filtered.push(Condition::matches("content", MatchValue::Text(tok.clone())));
            }
            let token_strategy = format!("tokens+dense (tokens={tokens:?})");
            let t = Instant::now();
            let points = query_dense(client, q, filtered).await?;
            log::debug!(
                "strategy={token_strategy} took={}ms hits={}",
                t.elapsed().as_millis(),
                points.len()
            );
            if !points.is_empty() {
                return Ok(SkillSearchResult::with_strategy(
                    parse_skills(points),
                    token_strategy,
                ));
            }
            let t = Instant::now();
            let points = query_hybrid(client, q, conditions).await?;
            log::debug!(
                "fallback strategy=hybrid took={}ms hits={}",
                t.elapsed().as_millis(),
                points.len()
            );
            let fallback = format!("{token_strategy} → hybrid-fallback");
            return Ok(SkillSearchResult::with_strategy(
                parse_skills(points),
                fallback,
            ));
        }

        // No tokens → pick by classifier score, default to hybrid on missing score.
        let strategy = score.map(Strategy::from).unwrap_or(Strategy::Hybrid);
        let t = Instant::now();
        let (points, label) = match strategy {
            Strategy::Dense => (
                query_dense(client, q, conditions).await?,
                format!("dense (score={score:?})"),
            ),
            Strategy::Hybrid => (
                query_hybrid(client, q, conditions).await?,
                format!("hybrid-rrf (score={score:?})"),
            ),
            Strategy::Bm25 => (
                query_bm25(client, q, conditions).await?,
                format!("bm25 (score={score:?})"),
            ),
        };
        log::debug!(
            "strategy={label} took={}ms hits={}",
            t.elapsed().as_millis(),
            points.len()
        );
        return Ok(SkillSearchResult::with_strategy(parse_skills(points), label));
    }

    // No query — browse by page or all
    let t = Instant::now();
    let points = query_by_filter(client, conditions, skills_exact_limit()).await?;
    log::debug!("browse took={}ms hits={}", t.elapsed().as_millis(), points.len());
    Ok(SkillSearchResult::with_strategy(parse_skills(points), "browse"))
}

#[get("/skills{path:/?.*}")]
pub async fn skills_handler(
    path: actix_web::web::Path<String>,
    _req: HttpRequest,
    query: Query<SkillSearch>,
    qdrant: Data<Qdrant>,
) -> HttpResponse {
    let path_str = path.into_inner();

    let t_total = Instant::now();
    match search_skills(qdrant.get_ref(), query.q.as_deref(), &path_str).await {
        Ok(result) => {
            log::debug!(
                "skills={} strategy={:?} total={}ms",
                result.len(),
                result.strategy,
                t_total.elapsed().as_millis()
            );
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
