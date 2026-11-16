use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender, TrySendError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OpenFlags};

use super::middleware::snippet_language;
use super::{MCP_SNIPPETS_ENDPOINT, SNIPPETS_ENDPOINT};

/// How many records may wait for the writer before new ones are dropped.
const QUEUE_SIZE: usize = 10_000;
/// How often the writer deletes records past the retention period.
const PRUNE_INTERVAL: Duration = Duration::from_secs(3600);

pub(crate) const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    ts INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    query TEXT,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    query_string TEXT NOT NULL,
    body TEXT,
    status INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    user_agent TEXT,
    language TEXT
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS requests_endpoint_ts ON requests(endpoint, ts);
";

/// Brings a log written by an older version up to [`SCHEMA`].
fn migrate(conn: &Connection) -> rusqlite::Result<()> {
    let has_language = conn
        .prepare("SELECT 1 FROM pragma_table_info('requests') WHERE name = 'language'")?
        .exists([])?;
    if !has_language {
        conn.execute_batch("ALTER TABLE requests ADD COLUMN language TEXT")?;
    }
    backfill_languages(conn)
}

/// Fills in the language of snippet searches logged before it was kept.
fn backfill_languages(conn: &Connection) -> rusqlite::Result<()> {
    let rows = conn
        .prepare(
            "SELECT id, endpoint, query_string, body FROM requests
             WHERE language IS NULL AND endpoint IN (?1, ?2)",
        )?
        .query_map([SNIPPETS_ENDPOINT, MCP_SNIPPETS_ENDPOINT], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if rows.is_empty() {
        return Ok(());
    }
    let tx = conn.unchecked_transaction()?;
    {
        let mut update = tx.prepare("UPDATE requests SET language = ?1 WHERE id = ?2")?;
        for (id, endpoint, query_string, body) in &rows {
            let language =
                snippet_language(endpoint, query_string, body.as_deref().map(str::as_bytes));
            update.execute(params![language, id])?;
        }
    }
    tx.commit()?;
    log::info!(
        "Filled in the language of {} logged snippet searches",
        rows.len()
    );
    Ok(())
}

/// One request to the service, as the admin panel shows and replays it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RequestRecord {
    pub id: i64,
    /// Unix time in milliseconds.
    pub ts: i64,
    /// What was called: `search`, `md`, `skills`, `snippets`, or `mcp:<tool or method>`.
    pub endpoint: String,
    /// The text searched for or the page asked for, if any.
    pub query: Option<String>,
    pub method: String,
    pub path: String,
    pub query_string: String,
    /// The request body, only kept for the MCP server.
    pub body: Option<String>,
    pub status: u16,
    pub duration_ms: f64,
    pub user_agent: Option<String>,
    /// The language a snippet search asked for, empty for none.
    /// `None` for other requests.
    pub language: Option<String>,
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or_default()
}

/// A SQLite log of the requests of the last `retention_days`.
///
/// Requests are written by a thread of their own, so serving a request never
/// waits for the disk. Readers open their own connections.
#[derive(Clone)]
pub struct RequestLog {
    path: PathBuf,
    sender: SyncSender<RequestRecord>,
}

impl RequestLog {
    pub fn open(path: impl AsRef<Path>, retention_days: u32) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(&path)?;
        // Incremental vacuum has to be on before the first table exists.
        conn.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(SCHEMA)?;
        migrate(&conn)?;

        let retention_ms = i64::from(retention_days) * 24 * 3600 * 1000;
        let (sender, receiver) = mpsc::sync_channel::<RequestRecord>(QUEUE_SIZE);
        std::thread::Builder::new()
            .name("request-log".into())
            .spawn(move || {
                let mut conn = conn;
                let mut last_prune: Option<Instant> = None;
                loop {
                    if last_prune.map_or(true, |t| t.elapsed() >= PRUNE_INTERVAL) {
                        if let Err(e) = prune(&conn, now_ms() - retention_ms) {
                            log::error!("Pruning the request log failed: {e}");
                        }
                        last_prune = Some(Instant::now());
                    }
                    let first = match receiver.recv_timeout(PRUNE_INTERVAL) {
                        Ok(record) => record,
                        Err(RecvTimeoutError::Timeout) => continue,
                        Err(RecvTimeoutError::Disconnected) => return,
                    };
                    let batch: Vec<_> = std::iter::once(first)
                        .chain(receiver.try_iter().take(1000))
                        .collect();
                    if let Err(e) = insert(&mut conn, &batch) {
                        log::error!("Writing {} requests to the log failed: {e}", batch.len());
                    }
                }
            })?;
        Ok(Self { path, sender })
    }

    /// Queues a record for writing. Its `id` is assigned by the database.
    pub fn record(&self, record: RequestRecord) {
        if let Err(TrySendError::Full(_)) = self.sender.try_send(record) {
            log::warn!("Request log queue is full, dropping a record");
        }
    }

    /// A read-only connection, for blocking code.
    pub fn reader(&self) -> rusqlite::Result<Connection> {
        Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
    }
}

fn insert(conn: &mut Connection, batch: &[RequestRecord]) -> rusqlite::Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO requests (ts, endpoint, query, method, path, query_string, body, status, duration_ms, user_agent, language)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )?;
        for r in batch {
            stmt.execute(params![
                r.ts,
                r.endpoint,
                r.query,
                r.method,
                r.path,
                r.query_string,
                r.body,
                r.status,
                r.duration_ms,
                r.user_agent,
                r.language,
            ])?;
        }
    }
    tx.commit()
}

fn prune(conn: &Connection, before_ms: i64) -> rusqlite::Result<()> {
    let deleted = conn.execute("DELETE FROM requests WHERE ts < ?1", [before_ms])?;
    if deleted > 0 {
        log::info!("Deleted {deleted} requests past retention from the request log");
        conn.execute_batch("PRAGMA incremental_vacuum;")?;
    }
    Ok(())
}

pub(crate) fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<RequestRecord> {
    Ok(RequestRecord {
        id: row.get("id")?,
        ts: row.get("ts")?,
        endpoint: row.get("endpoint")?,
        query: row.get("query")?,
        method: row.get("method")?,
        path: row.get("path")?,
        query_string: row.get("query_string")?,
        body: row.get("body")?,
        status: row.get("status")?,
        duration_ms: row.get("duration_ms")?,
        user_agent: row.get("user_agent")?,
        language: row.get("language")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_logs_without_languages() {
        let path =
            std::env::temp_dir().join(format!("log-migrate-{}.sqlite", rand::random::<u64>()));
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE requests (
                    id INTEGER PRIMARY KEY, ts INTEGER NOT NULL, endpoint TEXT NOT NULL, query TEXT,
                    method TEXT NOT NULL, path TEXT NOT NULL, query_string TEXT NOT NULL, body TEXT,
                    status INTEGER NOT NULL, duration_ms REAL NOT NULL, user_agent TEXT);
                 INSERT INTO requests (ts, endpoint, query, method, path, query_string, body, status, duration_ms) VALUES
                    (1, 'snippets', 'upsert', 'GET', '/snippets/search', 'query=upsert&language=rust', NULL, 200, 1),
                    (2, 'mcp:search_snippets', 'upsert', 'POST', '/mcp', '',
                     '{\"method\":\"tools/call\",\"params\":{\"name\":\"search_snippets\",\"arguments\":{\"query\":\"upsert\"}}}', 200, 1),
                    (3, 'search', 'upsert', 'GET', '/api/search', 'q=upsert', NULL, 200, 1);",
            )
            .unwrap();
        }
        let log = RequestLog::open(&path, 3650 * 100).unwrap();
        // Twice, as on every start.
        drop(RequestLog::open(&path, 3650 * 100).unwrap());
        let languages: Vec<Option<String>> = log
            .reader()
            .unwrap()
            .prepare("SELECT language FROM requests ORDER BY id")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(languages, [Some("rust".into()), Some("".into()), None]);
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }
}
