//! Feedback on pages that AI agents send with the docs MCP server's
//! `submit_feedback` tool, kept for the admin panel.
//!
//! Unlike the request log, feedback isn't pruned: it stays until an admin
//! deletes the file. Admins dismiss what they've dealt with instead.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde::Serialize;

use crate::request_log::now_ms;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    ts INTEGER NOT NULL,
    page TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    feedback TEXT NOT NULL,
    client TEXT,
    dismissed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS feedback_ts ON feedback(ts);
CREATE INDEX IF NOT EXISTS feedback_page ON feedback(page);
";

/// Feedback as the tool hands it over.
#[derive(Debug, Clone)]
pub struct NewFeedback {
    /// The site path of the page, such as `/documentation/search/filtering/`.
    pub page: String,
    pub url: String,
    pub title: Option<String>,
    pub feedback: String,
    /// The MCP client, such as `claude-code/2.1`, if it said.
    pub client: Option<String>,
}

/// One piece of stored feedback.
#[derive(Debug, Clone, Serialize)]
pub struct FeedbackRecord {
    pub id: i64,
    /// Unix time in milliseconds.
    pub ts: i64,
    pub page: String,
    pub url: String,
    pub title: Option<String>,
    pub feedback: String,
    pub client: Option<String>,
    pub dismissed: bool,
}

/// What to list.
#[derive(Debug, Clone, Default)]
pub struct FeedbackFilter {
    pub include_dismissed: bool,
    /// Only feedback whose page or text contains this.
    pub contains: Option<String>,
    /// Only feedback older than this id, to page through.
    pub before_id: Option<i64>,
    pub limit: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeedbackList {
    pub items: Vec<FeedbackRecord>,
    /// Pass as `before_id` for the next batch.
    pub next_before: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PageCount {
    pub page: String,
    pub url: String,
    pub title: Option<String>,
    pub open: i64,
}

/// A SQLite store of feedback. Calls block: run them off the async workers.
#[derive(Clone)]
pub struct FeedbackStore {
    path: PathBuf,
    writer: Arc<Mutex<Connection>>,
}

impl FeedbackStore {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(&path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            path,
            writer: Arc::new(Mutex::new(conn)),
        })
    }

    fn writer(&self) -> std::sync::MutexGuard<'_, Connection> {
        // A panic while writing leaves nothing half done: SQLite rolls back.
        self.writer.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn reader(&self) -> rusqlite::Result<Connection> {
        Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
    }

    /// Stores feedback, returning its id.
    pub fn add(&self, feedback: &NewFeedback) -> rusqlite::Result<i64> {
        let conn = self.writer();
        conn.execute(
            "INSERT INTO feedback (ts, page, url, title, feedback, client)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                now_ms(),
                feedback.page,
                feedback.url,
                feedback.title,
                feedback.feedback,
                feedback.client,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Marks feedback as dealt with, or not. False if there's no such feedback.
    pub fn set_dismissed(&self, id: i64, dismissed: bool) -> rusqlite::Result<bool> {
        let changed = self.writer().execute(
            "UPDATE feedback SET dismissed = ?1 WHERE id = ?2",
            params![dismissed, id],
        )?;
        Ok(changed > 0)
    }

    pub fn get(&self, id: i64) -> rusqlite::Result<Option<FeedbackRecord>> {
        self.reader()?
            .query_row("SELECT * FROM feedback WHERE id = ?1", [id], row_to_record)
            .optional()
    }

    /// Newest first.
    pub fn list(&self, filter: &FeedbackFilter) -> rusqlite::Result<FeedbackList> {
        let mut sql = "SELECT * FROM feedback WHERE id < ?1".to_string();
        if !filter.include_dismissed {
            sql.push_str(" AND dismissed = 0");
        }
        if filter.contains.is_some() {
            sql.push_str(
                " AND (instr(lower(page), lower(?3)) > 0 OR instr(lower(feedback), lower(?3)) > 0)",
            );
        } else {
            sql.push_str(" AND ?3 IS NULL");
        }
        sql.push_str(" ORDER BY id DESC LIMIT ?2");
        let limit = filter.limit.clamp(1, 200);
        let conn = self.reader()?;
        let items: Vec<FeedbackRecord> = conn
            .prepare(&sql)?
            .query_map(
                params![
                    filter.before_id.unwrap_or(i64::MAX),
                    limit + 1,
                    filter.contains
                ],
                row_to_record,
            )?
            .collect::<rusqlite::Result<_>>()?;
        let mut items = items;
        let next_before = if items.len() > limit as usize {
            items.truncate(limit as usize);
            items.last().map(|r| r.id)
        } else {
            None
        };
        Ok(FeedbackList { items, next_before })
    }

    /// Pages by how much open feedback they have, most first.
    pub fn pages(&self, limit: u32) -> rusqlite::Result<Vec<PageCount>> {
        let conn = self.reader()?;
        let rows = conn
            .prepare(
                "SELECT page, max(url), max(title), count(*) AS open FROM feedback
                 WHERE dismissed = 0 GROUP BY page ORDER BY open DESC, max(ts) DESC LIMIT ?1",
            )?
            .query_map([limit], |row| {
                Ok(PageCount {
                    page: row.get(0)?,
                    url: row.get(1)?,
                    title: row.get(2)?,
                    open: row.get(3)?,
                })
            })?
            .collect();
        rows
    }
}

fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<FeedbackRecord> {
    Ok(FeedbackRecord {
        id: row.get("id")?,
        ts: row.get("ts")?,
        page: row.get("page")?,
        url: row.get("url")?,
        title: row.get("title")?,
        feedback: row.get("feedback")?,
        client: row.get("client")?,
        dismissed: row.get("dismissed")?,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A store in a file of its own, deleted when dropped.
    pub(crate) struct TempStore {
        pub store: FeedbackStore,
        path: PathBuf,
    }

    impl TempStore {
        pub fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("feedback-{}.sqlite", rand::random::<u64>()));
            Self {
                store: FeedbackStore::open(&path).unwrap(),
                path,
            }
        }
    }

    impl Drop for TempStore {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{suffix}", self.path.display()));
            }
        }
    }

    fn feedback(page: &str, text: &str) -> NewFeedback {
        NewFeedback {
            page: page.to_string(),
            url: format!("https://qdrant.tech{page}"),
            title: None,
            feedback: text.to_string(),
            client: Some("test/1".to_string()),
        }
    }

    #[test]
    fn adds_lists_and_dismisses() {
        let t = TempStore::new();
        let a = t
            .store
            .add(&feedback("/documentation/a/", "Outdated"))
            .unwrap();
        let b = t
            .store
            .add(&feedback("/documentation/b/", "Wrong param"))
            .unwrap();
        let c = t
            .store
            .add(&feedback("/documentation/a/", "Confusing"))
            .unwrap();

        let all = t
            .store
            .list(&FeedbackFilter {
                limit: 50,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            all.items.iter().map(|r| r.id).collect::<Vec<_>>(),
            [c, b, a]
        );
        assert_eq!(all.next_before, None);

        let pages = t.store.pages(10).unwrap();
        assert_eq!(pages[0].page, "/documentation/a/");
        assert_eq!(pages[0].open, 2);

        assert!(t.store.set_dismissed(a, true).unwrap());
        assert!(!t.store.set_dismissed(9999, true).unwrap());
        let open = t
            .store
            .list(&FeedbackFilter {
                limit: 50,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(open.items.iter().map(|r| r.id).collect::<Vec<_>>(), [c, b]);
        assert!(t.store.get(a).unwrap().unwrap().dismissed);

        let found = t
            .store
            .list(&FeedbackFilter {
                limit: 50,
                include_dismissed: true,
                contains: Some("PARAM".to_string()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(found.items.iter().map(|r| r.id).collect::<Vec<_>>(), [b]);

        let first = t
            .store
            .list(&FeedbackFilter {
                limit: 1,
                include_dismissed: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(first.items[0].id, c);
        assert_eq!(first.next_before, Some(c));
        let next = t
            .store
            .list(&FeedbackFilter {
                limit: 1,
                include_dismissed: true,
                before_id: first.next_before,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(next.items[0].id, b);
    }
}
