use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;

use super::store::{row_to_record, RequestRecord};
use super::IS_ERROR;

/// SQL for requests that are noise: unknown MCP tools and scanners.
pub const IS_NOISE: &str = "category IN ('unknown', 'scanner')";

/// Which requests the panel's traffic view counts.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrafficFilter {
    /// Leave out MCP sessions and discovery, such as `initialize`.
    pub hide_protocol: bool,
    /// Leave out unknown MCP tools and scanners.
    pub hide_noise: bool,
    /// Only failed requests: HTTP errors and MCP errors.
    pub errors_only: bool,
}

impl TrafficFilter {
    /// `AND` conditions for the filter, to append to a `WHERE` clause.
    fn sql(&self) -> String {
        let mut sql = String::new();
        if self.hide_protocol {
            sql.push_str(" AND category != 'protocol'");
        }
        if self.hide_noise {
            sql.push_str(&format!(" AND NOT {IS_NOISE}"));
        }
        if self.errors_only {
            sql.push_str(&format!(" AND {IS_ERROR}"));
        }
        sql
    }
}

/// Request counts and latencies of one endpoint.
#[derive(Debug, Serialize)]
pub struct EndpointStats {
    pub endpoint: String,
    pub count: i64,
    pub errors: i64,
    /// Of `count`, requests that are noise.
    pub noise: i64,
    /// Of `count`, MCP protocol requests.
    pub protocol: i64,
    pub avg_ms: f64,
    pub p95_ms: f64,
}

/// Requests of one endpoint group on one UTC day.
#[derive(Debug, Serialize)]
pub struct DailyCount {
    /// `YYYY-MM-DD`
    pub day: String,
    /// The endpoint, with all MCP methods and tools as `mcp`.
    pub group: String,
    pub count: i64,
}

#[derive(Debug, Serialize)]
pub struct Stats {
    pub since: i64,
    pub total: i64,
    pub errors: i64,
    /// MCP tool calls of tools the server has.
    pub mcp_tool_calls: i64,
    /// MCP `initialize` requests, whatever the filter.
    pub mcp_sessions: i64,
    /// MCP protocol requests in the range, whatever the filter.
    pub protocol: i64,
    /// Noise in the range, whatever the filter.
    pub noise: i64,
    /// Failed noise requests in the range, whatever the filter.
    pub noise_errors: i64,
    pub endpoints: Vec<EndpointStats>,
    pub daily: Vec<DailyCount>,
}

#[derive(Debug, Serialize)]
pub struct TopQuery {
    pub endpoint: String,
    pub query: String,
    pub count: i64,
    /// The latest request with this query, to replay.
    pub last_id: i64,
}

#[derive(Debug, Default)]
pub struct RequestFilter {
    pub since: i64,
    pub endpoint: Option<String>,
    /// Text the query has to contain.
    pub contains: Option<String>,
    /// Only requests older than this id, for paging.
    pub before_id: Option<i64>,
    pub limit: u32,
    pub traffic: TrafficFilter,
}

/// `endpoint = ?` or, for `mcp`, any MCP endpoint.
fn endpoint_clause(endpoint: &str) -> (&'static str, String) {
    if endpoint == "mcp" {
        ("endpoint LIKE ?", "mcp:%".to_string())
    } else {
        ("endpoint = ?", endpoint.to_string())
    }
}

pub fn stats(conn: &Connection, since: i64, filter: TrafficFilter) -> rusqlite::Result<Stats> {
    let and = filter.sql();
    let mut endpoints = conn
        .prepare(&format!(
            "SELECT endpoint, COUNT(*), SUM({IS_ERROR}), SUM({IS_NOISE}),
                    SUM(category = 'protocol'), AVG(duration_ms)
             FROM requests WHERE ts >= ?1{and} GROUP BY endpoint ORDER BY COUNT(*) DESC"
        ))?
        .query_map([since], |row| {
            Ok(EndpointStats {
                endpoint: row.get(0)?,
                count: row.get(1)?,
                errors: row.get(2)?,
                noise: row.get(3)?,
                protocol: row.get(4)?,
                avg_ms: row.get(5)?,
                p95_ms: 0.0,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut p95 = conn.prepare(&format!(
        "SELECT duration_ms FROM requests WHERE endpoint = ?1 AND ts >= ?2{and}
         ORDER BY duration_ms LIMIT 1 OFFSET ?3"
    ))?;
    for e in &mut endpoints {
        let offset = ((e.count - 1) as f64 * 0.95).floor() as i64;
        e.p95_ms = p95
            .query_row(params![e.endpoint, since, offset], |row| row.get(0))
            .optional()?
            .unwrap_or_default();
    }

    let daily = conn
        .prepare(&format!(
            "SELECT date(ts / 1000, 'unixepoch') AS day,
                    CASE WHEN endpoint LIKE 'mcp:%' THEN 'mcp' ELSE endpoint END AS grp,
                    COUNT(*)
             FROM requests WHERE ts >= ?1{and} GROUP BY day, grp ORDER BY day"
        ))?
        .query_map([since], |row| {
            Ok(DailyCount {
                day: row.get(0)?,
                group: row.get(1)?,
                count: row.get(2)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let (mcp_tool_calls,): (i64,) = conn.query_row(
        &format!(
            "SELECT IFNULL(SUM(endpoint LIKE 'mcp:%' AND category = 'use'), 0)
             FROM requests WHERE ts >= ?1{and}"
        ),
        [since],
        |row| Ok((row.get(0)?,)),
    )?;
    let (mcp_sessions, protocol, noise, noise_errors) = conn.query_row(
        &format!(
            "SELECT IFNULL(SUM(endpoint = 'mcp:initialize' AND NOT {IS_NOISE}), 0),
                    IFNULL(SUM(category = 'protocol'), 0), IFNULL(SUM({IS_NOISE}), 0),
                    IFNULL(SUM({IS_NOISE} AND {IS_ERROR}), 0)
             FROM requests WHERE ts >= ?1"
        ),
        [since],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;

    Ok(Stats {
        since,
        total: endpoints.iter().map(|e| e.count).sum(),
        errors: endpoints.iter().map(|e| e.errors).sum(),
        mcp_tool_calls,
        mcp_sessions,
        protocol,
        noise,
        noise_errors,
        endpoints,
        daily,
    })
}

/// How soon after a search a longer one that starts with it makes it a
/// partial query, typed on the way to the longer one.
const PARTIAL_QUERY_WINDOW_MS: i64 = 10_000;

/// The most frequent queries, leaving out partial queries: the search box
/// searches as you type, so "qu" and "quan" come before "quantization".
/// The log has no client IPs, so the same user agent stands in for the
/// same client.
pub fn top_queries(
    conn: &Connection,
    since: i64,
    endpoint: Option<&str>,
    filter: TrafficFilter,
    limit: u32,
) -> rusqlite::Result<Vec<TopQuery>> {
    let mut sql = "SELECT endpoint, query, COUNT(*) AS c, MAX(id) FROM requests AS r
                   WHERE ts >= ? AND query IS NOT NULL AND query != ''
                   AND NOT (endpoint = 'search' AND EXISTS (
                       SELECT 1 FROM requests AS n
                       WHERE n.endpoint = 'search'
                         AND n.ts BETWEEN r.ts AND r.ts + ?
                         AND n.user_agent IS r.user_agent
                         AND length(n.query) > length(r.query)
                         AND lower(substr(n.query, 1, length(r.query))) = lower(r.query)))"
        .to_string();
    sql.push_str(&filter.sql());
    let mut values: Vec<rusqlite::types::Value> =
        vec![since.into(), PARTIAL_QUERY_WINDOW_MS.into()];
    if let Some(endpoint) = endpoint {
        let (clause, value) = endpoint_clause(endpoint);
        sql.push_str(&format!(" AND {clause}"));
        values.push(value.into());
    }
    sql.push_str(" GROUP BY endpoint, query ORDER BY c DESC LIMIT ?");
    values.push(i64::from(limit).into());

    conn.prepare(&sql)?
        .query_map(rusqlite::params_from_iter(values), |row| {
            Ok(TopQuery {
                endpoint: row.get(0)?,
                query: row.get(1)?,
                count: row.get(2)?,
                last_id: row.get(3)?,
            })
        })?
        .collect()
}

pub fn list(conn: &Connection, filter: &RequestFilter) -> rusqlite::Result<Vec<RequestRecord>> {
    let mut sql = format!(
        "SELECT * FROM requests WHERE ts >= ?{}",
        filter.traffic.sql()
    );
    let mut values: Vec<rusqlite::types::Value> = vec![filter.since.into()];
    if let Some(endpoint) = &filter.endpoint {
        let (clause, value) = endpoint_clause(endpoint);
        sql.push_str(&format!(" AND {clause}"));
        values.push(value.into());
    }
    if let Some(text) = &filter.contains {
        sql.push_str(" AND instr(lower(query), lower(?)) > 0");
        values.push(text.clone().into());
    }
    if let Some(id) = filter.before_id {
        sql.push_str(" AND id < ?");
        values.push(id.into());
    }
    sql.push_str(" ORDER BY id DESC LIMIT ?");
    values.push(i64::from(filter.limit).into());

    conn.prepare(&sql)?
        .query_map(rusqlite::params_from_iter(values), row_to_record)?
        .collect()
}

pub fn get(conn: &Connection, id: i64) -> rusqlite::Result<Option<RequestRecord>> {
    conn.query_row("SELECT * FROM requests WHERE id = ?1", [id], row_to_record)
        .optional()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search(conn: &Connection, ts: i64, query: &str, user_agent: &str) {
        conn.execute(
            "INSERT INTO requests (ts, endpoint, query, method, path, query_string, status, duration_ms, user_agent)
             VALUES (?1, 'search', ?2, 'GET', '/api/search', '', 200, 1.0, ?3)",
            params![ts, query, user_agent],
        )
        .unwrap();
    }

    #[test]
    fn top_queries_leave_out_partial_queries() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(crate::request_log::store::SCHEMA)
            .unwrap();
        // Typed on the way to "quantization".
        search(&conn, 1_000, "qu", "firefox");
        search(&conn, 1_400, "Quan", "firefox");
        search(&conn, 2_000, "quantization", "firefox");
        // A whole query that others extend later, or from another client.
        search(&conn, 50_000, "hnsw", "firefox");
        search(&conn, 70_000, "hnsw index", "firefox");
        search(&conn, 80_000, "hnsw", "chrome");
        search(&conn, 81_000, "hnsw index", "safari");
        // The same query twice is no partial query.
        search(&conn, 90_000, "filters", "chrome");
        search(&conn, 90_500, "filters", "chrome");

        let mut top: Vec<_> = top_queries(&conn, 0, None, TrafficFilter::default(), 50)
            .unwrap()
            .into_iter()
            .map(|t| (t.query, t.count))
            .collect();
        top.sort();
        assert_eq!(
            top,
            [
                ("filters".into(), 2),
                ("hnsw".into(), 2),
                ("hnsw index".into(), 2),
                ("quantization".into(), 1),
            ]
        );
    }
}
