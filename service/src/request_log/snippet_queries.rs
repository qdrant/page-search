//! Reports on snippet searches, from `/snippets/search` and the docs MCP
//! server's `search_snippets` tool, for debugging snippet search.

use rusqlite::types::Value;
use rusqlite::{params_from_iter, Connection, OptionalExtension};
use serde::Serialize;

use super::queries::IS_NOISE;
use super::store::{row_to_record, RequestRecord};
use super::{IS_ERROR, MCP_SNIPPETS_ENDPOINT, SNIPPETS_ENDPOINT};
use crate::snippets::normalize_language;

/// Where a snippet search came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// `/snippets/search`
    Rest,
    /// The docs MCP server's `search_snippets` tool.
    Mcp,
}

impl Source {
    pub fn parse(source: &str) -> Option<Self> {
        match source {
            "rest" => Some(Self::Rest),
            "mcp" => Some(Self::Mcp),
            _ => None,
        }
    }

    fn of_endpoint(endpoint: &str) -> Self {
        if endpoint == MCP_SNIPPETS_ENDPOINT {
            Self::Mcp
        } else {
            Self::Rest
        }
    }

    fn endpoint(self) -> &'static str {
        match self {
            Self::Rest => SNIPPETS_ENDPOINT,
            Self::Mcp => MCP_SNIPPETS_ENDPOINT,
        }
    }

    /// The language the search filters the index by: the MCP server
    /// normalizes it, `/snippets/search` takes it as it is.
    pub fn searched_as(self, language: &str) -> String {
        match self {
            Self::Rest => language.to_string(),
            Self::Mcp => normalize_language(language),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct SnippetFilter {
    pub since: i64,
    pub source: Option<Source>,
    /// The language as requested; empty for searches without one.
    pub language: Option<String>,
    /// Text the query has to contain.
    pub contains: Option<String>,
    pub errors_only: bool,
}

impl SnippetFilter {
    /// The `WHERE` clause and its values.
    fn clause(&self) -> (String, Vec<Value>) {
        // Scanners' payloads say nothing about snippet search.
        let mut sql = format!("ts >= ? AND NOT {IS_NOISE}");
        let mut values: Vec<Value> = vec![self.since.into()];
        match self.source {
            Some(source) => {
                sql.push_str(" AND endpoint = ?");
                values.push(source.endpoint().to_string().into());
            }
            None => {
                sql.push_str(" AND endpoint IN (?, ?)");
                values.push(SNIPPETS_ENDPOINT.to_string().into());
                values.push(MCP_SNIPPETS_ENDPOINT.to_string().into());
            }
        }
        if let Some(language) = &self.language {
            sql.push_str(" AND IFNULL(language, '') = ?");
            values.push(language.clone().into());
        }
        if let Some(text) = &self.contains {
            sql.push_str(" AND instr(lower(query), lower(?)) > 0");
            values.push(text.clone().into());
        }
        if self.errors_only {
            sql.push_str(&format!(" AND {IS_ERROR}"));
        }
        (sql, values)
    }
}

#[derive(Debug, Serialize)]
pub struct SourceStats {
    pub source: Source,
    pub count: i64,
    pub errors: i64,
    pub avg_ms: f64,
    pub p95_ms: f64,
}

/// Searches for one language from one source.
#[derive(Debug, Serialize)]
pub struct LanguageStats {
    /// As requested; empty for searches in every language.
    pub language: String,
    pub source: Source,
    /// What the index is filtered by, after the source's normalization.
    pub searched_as: String,
    pub count: i64,
    pub distinct_queries: i64,
    pub errors: i64,
    pub avg_ms: f64,
    pub last_ts: i64,
}

/// Searches for one requested language on one UTC day.
#[derive(Debug, Serialize)]
pub struct DailyLanguage {
    /// `YYYY-MM-DD`
    pub day: String,
    pub language: String,
    pub count: i64,
}

#[derive(Debug, Serialize)]
pub struct SnippetStats {
    pub since: i64,
    pub total: i64,
    pub errors: i64,
    /// Searches in every language, without one.
    pub without_language: i64,
    pub distinct_queries: i64,
    pub sources: Vec<SourceStats>,
    pub daily: Vec<DailyLanguage>,
}

#[derive(Debug, Serialize)]
pub struct SnippetTopQuery {
    pub query: String,
    pub language: String,
    pub source: Source,
    pub count: i64,
    pub errors: i64,
    /// The latest request with this query, to replay. The query is spelled
    /// as in that request.
    pub last_id: i64,
}

pub fn stats(conn: &Connection, filter: &SnippetFilter) -> rusqlite::Result<SnippetStats> {
    let (clause, values) = filter.clause();

    let (total, errors, without_language, distinct_queries) = conn.query_row(
        &format!(
            "SELECT COUNT(*), IFNULL(SUM({IS_ERROR}), 0), IFNULL(SUM(IFNULL(language, '') = ''), 0),
                    COUNT(DISTINCT lower(query))
             FROM requests WHERE {clause}"
        ),
        params_from_iter(values.iter()),
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;

    let mut sources = conn
        .prepare(&format!(
            "SELECT endpoint, COUNT(*), SUM({IS_ERROR}), AVG(duration_ms)
             FROM requests WHERE {clause} GROUP BY endpoint ORDER BY COUNT(*) DESC"
        ))?
        .query_map(params_from_iter(values.iter()), |row| {
            Ok(SourceStats {
                source: Source::of_endpoint(&row.get::<_, String>(0)?),
                count: row.get(1)?,
                errors: row.get(2)?,
                avg_ms: row.get(3)?,
                p95_ms: 0.0,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for s in &mut sources {
        let only_source = SnippetFilter {
            source: Some(s.source),
            ..filter.clone()
        };
        let (clause, mut values) = only_source.clause();
        values.push((((s.count - 1) as f64 * 0.95).floor() as i64).into());
        s.p95_ms = conn
            .query_row(
                &format!(
                    "SELECT duration_ms FROM requests WHERE {clause}
                     ORDER BY duration_ms LIMIT 1 OFFSET ?"
                ),
                params_from_iter(values.iter()),
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or_default();
    }

    let daily = conn
        .prepare(&format!(
            "SELECT date(ts / 1000, 'unixepoch') AS day, IFNULL(language, ''), COUNT(*)
             FROM requests WHERE {clause} GROUP BY day, 2 ORDER BY day"
        ))?
        .query_map(params_from_iter(values.iter()), |row| {
            Ok(DailyLanguage {
                day: row.get(0)?,
                language: row.get(1)?,
                count: row.get(2)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(SnippetStats {
        since: filter.since,
        total,
        errors,
        without_language,
        distinct_queries,
        sources,
        daily,
    })
}

/// The searches since a time by requested language and source, for
/// comparing with the languages in the index.
pub fn languages(conn: &Connection, since: i64) -> rusqlite::Result<Vec<LanguageStats>> {
    let filter = SnippetFilter {
        since,
        ..Default::default()
    };
    let (clause, values) = filter.clause();
    conn.prepare(&format!(
        "SELECT IFNULL(language, ''), endpoint, COUNT(*), COUNT(DISTINCT lower(query)),
                SUM({IS_ERROR}), AVG(duration_ms), MAX(ts)
         FROM requests WHERE {clause}
         GROUP BY 1, endpoint ORDER BY COUNT(*) DESC"
    ))?
    .query_map(params_from_iter(values.iter()), |row| {
        let language: String = row.get(0)?;
        let source = Source::of_endpoint(&row.get::<_, String>(1)?);
        Ok(LanguageStats {
            searched_as: source.searched_as(&language),
            language,
            source,
            count: row.get(2)?,
            distinct_queries: row.get(3)?,
            errors: row.get(4)?,
            avg_ms: row.get(5)?,
            last_ts: row.get(6)?,
        })
    })?
    .collect()
}

/// The most frequent queries, by language and source. The same query in
/// another case counts as the same.
pub fn top_queries(
    conn: &Connection,
    filter: &SnippetFilter,
    limit: u32,
) -> rusqlite::Result<Vec<SnippetTopQuery>> {
    let (clause, mut values) = filter.clause();
    values.push(i64::from(limit).into());
    conn.prepare(&format!(
        "SELECT query, IFNULL(language, ''), endpoint, COUNT(*) AS c, SUM({IS_ERROR}), MAX(id) AS last
         FROM requests WHERE {clause} AND query IS NOT NULL AND query != ''
         GROUP BY lower(query), 2, endpoint ORDER BY c DESC, last DESC LIMIT ?"
    ))?
    .query_map(params_from_iter(values.iter()), |row| {
        Ok(SnippetTopQuery {
            query: row.get(0)?,
            language: row.get(1)?,
            source: Source::of_endpoint(&row.get::<_, String>(2)?),
            count: row.get(3)?,
            errors: row.get(4)?,
            last_id: row.get(5)?,
        })
    })?
    .collect()
}

/// A column to sort the request list by.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SortBy {
    #[default]
    Time,
    Source,
    Language,
    Query,
    Status,
    Duration,
}

impl SortBy {
    pub fn parse(column: &str) -> Option<Self> {
        Some(match column {
            "time" => Self::Time,
            "source" => Self::Source,
            "language" => Self::Language,
            "query" => Self::Query,
            "status" => Self::Status,
            "duration" => Self::Duration,
            _ => return None,
        })
    }

    fn expression(self) -> &'static str {
        match self {
            Self::Time => "id",
            Self::Source => "endpoint",
            Self::Language => "IFNULL(language, '')",
            Self::Query => "lower(IFNULL(query, ''))",
            Self::Status => "status",
            Self::Duration => "duration_ms",
        }
    }
}

/// The order of the request list: a column, and which way.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sort {
    pub by: SortBy,
    pub ascending: bool,
}

/// A page of the matching requests, in the given order. Ties go newest first.
pub fn list(
    conn: &Connection,
    filter: &SnippetFilter,
    sort: Sort,
    offset: u32,
    limit: u32,
) -> rusqlite::Result<Vec<RequestRecord>> {
    let (clause, mut values) = filter.clause();
    values.push(i64::from(limit).into());
    values.push(i64::from(offset).into());
    let direction = if sort.ascending { "ASC" } else { "DESC" };
    conn.prepare(&format!(
        "SELECT * FROM requests WHERE {clause}
         ORDER BY {} {direction}, id DESC LIMIT ? OFFSET ?",
        sort.by.expression()
    ))?
    .query_map(params_from_iter(values.iter()), row_to_record)?
    .collect()
}

#[cfg(test)]
mod tests {
    use rusqlite::params;

    use super::*;

    fn snippet_search(
        conn: &Connection,
        ts: i64,
        endpoint: &str,
        query: &str,
        language: &str,
        status: u16,
    ) {
        conn.execute(
            "INSERT INTO requests (ts, endpoint, query, method, path, query_string, status, duration_ms, language)
             VALUES (?1, ?2, ?3, 'GET', '/snippets/search', '', ?4, ?5, ?6)",
            params![ts, endpoint, query, status, ts as f64, language],
        )
        .unwrap();
    }

    fn log() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(crate::request_log::store::SCHEMA)
            .unwrap();
        snippet_search(&conn, 1, "snippets", "upsert", "python", 200);
        snippet_search(&conn, 2, "snippets", "Upsert", "python", 200);
        snippet_search(&conn, 3, "snippets", "upsert", "py", 200);
        snippet_search(&conn, 4, "mcp:search_snippets", "upsert", "ts", 200);
        snippet_search(&conn, 5, "mcp:search_snippets", "filter", "", 200);
        snippet_search(&conn, 6, "snippets", "filter", "", 400);
        // Not a snippet search.
        conn.execute(
            "INSERT INTO requests (ts, endpoint, query, method, path, query_string, status, duration_ms)
             VALUES (7, 'search', 'upsert', 'GET', '/api/search', '', 200, 1.0)",
            [],
        )
        .unwrap();
        conn
    }

    #[test]
    fn stats_by_language_and_source() {
        let conn = log();
        let all = stats(&conn, &SnippetFilter::default()).unwrap();
        assert_eq!(all.total, 6);
        assert_eq!(all.errors, 1);
        assert_eq!(all.without_language, 2);
        assert_eq!(all.distinct_queries, 2);
        let rows = languages(&conn, 0).unwrap();
        let languages: Vec<_> = rows
            .iter()
            .map(|l| {
                (
                    l.language.as_str(),
                    l.source,
                    l.searched_as.as_str(),
                    l.count,
                    l.distinct_queries,
                )
            })
            .collect();
        assert_eq!(languages[0], ("python", Source::Rest, "python", 2, 1));
        // The MCP server normalizes languages, /snippets/search doesn't.
        assert!(languages.contains(&("py", Source::Rest, "py", 1, 1)));
        assert!(languages.contains(&("ts", Source::Mcp, "typescript", 1, 1)));

        let sources: Vec<_> = all
            .sources
            .iter()
            .map(|s| (s.source, s.count, s.errors))
            .collect();
        assert_eq!(sources, [(Source::Rest, 4, 1), (Source::Mcp, 2, 0)]);
        assert_eq!(all.sources[0].p95_ms, 3.0);

        let python = stats(
            &conn,
            &SnippetFilter {
                language: Some("python".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(python.total, 2);
        assert_eq!(python.without_language, 0);
    }

    #[test]
    fn top_queries_and_lists_filter() {
        let conn = log();
        let top = top_queries(&conn, &SnippetFilter::default(), 10).unwrap();
        assert_eq!(
            (
                top[0].query.as_str(),
                top[0].language.as_str(),
                top[0].count
            ),
            ("Upsert", "python", 2)
        );

        let mcp = SnippetFilter {
            source: Some(Source::Mcp),
            ..Default::default()
        };
        assert_eq!(list(&conn, &mcp, Sort::default(), 0, 50).unwrap().len(), 2);
        let without_language = SnippetFilter {
            language: Some(String::new()),
            ..Default::default()
        };
        assert_eq!(
            list(&conn, &without_language, Sort::default(), 0, 50)
                .unwrap()
                .len(),
            2
        );
        let errors = SnippetFilter {
            errors_only: true,
            ..Default::default()
        };
        let errors = list(&conn, &errors, Sort::default(), 0, 50).unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].status, 400);
        let upsert = SnippetFilter {
            contains: Some("UPS".into()),
            ..Default::default()
        };
        let newest_first: Vec<i64> = list(&conn, &upsert, Sort::default(), 0, 50)
            .unwrap()
            .iter()
            .map(|r| r.ts)
            .collect();
        assert_eq!(newest_first, [4, 3, 2, 1]);
        let paged: Vec<i64> = list(&conn, &upsert, Sort::default(), 1, 2)
            .unwrap()
            .iter()
            .map(|r| r.ts)
            .collect();
        assert_eq!(paged, [3, 2]);

        // Errors first, then the newest.
        let by_status = Sort {
            by: SortBy::Status,
            ascending: false,
        };
        let statuses: Vec<(u16, i64)> = list(&conn, &SnippetFilter::default(), by_status, 0, 3)
            .unwrap()
            .iter()
            .map(|r| (r.status, r.ts))
            .collect();
        assert_eq!(statuses, [(400, 6), (200, 5), (200, 4)]);
        let fastest = Sort {
            by: SortBy::Duration,
            ascending: true,
        };
        let first = &list(&conn, &SnippetFilter::default(), fastest, 0, 1).unwrap()[0];
        assert_eq!(first.ts, 1);
        let by_language = Sort {
            by: SortBy::Language,
            ascending: true,
        };
        let languages: Vec<String> = list(&conn, &SnippetFilter::default(), by_language, 0, 50)
            .unwrap()
            .into_iter()
            .map(|r| r.language.unwrap_or_default())
            .collect();
        assert_eq!(languages, ["", "", "py", "python", "python", "ts"]);
    }
}
