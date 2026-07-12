use crate::Database;
use chrono::Utc;
use mnemosyne_core::Error;
use rusqlite::params;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub model_id: String,
    pub local_path: String,
    pub version: Option<String>,
    pub downloaded_at: Option<i64>,
}

pub struct ModelRepo<'a> {
    pub db: &'a Database,
}

impl<'a> ModelRepo<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    pub fn register(
        &self,
        model_id: &str,
        local_path: &str,
        version: Option<&str>,
    ) -> Result<(), Error> {
        let now = Utc::now().timestamp();
        let conn = self.db.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO model_registry (model_id, local_path, version, downloaded_at)
               VALUES (?1, ?2, ?3, ?4)
               ON CONFLICT(model_id) DO UPDATE SET
                   local_path    = excluded.local_path,
                   version       = excluded.version,
                   downloaded_at = excluded.downloaded_at"#,
            params![model_id, local_path, version, now],
        )
        .map_err(|e| Error::storage(e.to_string()))?;
        Ok(())
    }

    pub fn get(&self, model_id: &str) -> Result<Option<ModelRecord>, Error> {
        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT model_id, local_path, version, downloaded_at
                 FROM model_registry WHERE model_id = ?1",
            )
            .map_err(|e| Error::storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![model_id], |row| {
                Ok(ModelRecord {
                    model_id: row.get(0)?,
                    local_path: row.get(1)?,
                    version: row.get(2)?,
                    downloaded_at: row.get(3)?,
                })
            })
            .map_err(|e| Error::storage(e.to_string()))?;

        rows.next()
            .transpose()
            .map_err(|e| Error::storage(e.to_string()))
    }

    pub fn list(&self) -> Result<Vec<ModelRecord>, Error> {
        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT model_id, local_path, version, downloaded_at
                 FROM model_registry ORDER BY model_id",
            )
            .map_err(|e| Error::storage(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(ModelRecord {
                    model_id: row.get(0)?,
                    local_path: row.get(1)?,
                    version: row.get(2)?,
                    downloaded_at: row.get(3)?,
                })
            })
            .map_err(|e| Error::storage(e.to_string()))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error::storage(e.to_string()))
    }
}
