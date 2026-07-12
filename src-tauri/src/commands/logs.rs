use crate::state::{AppState, LogEntry};
use tauri::State;

/// Return all buffered log entries (newest last).
#[tauri::command]
pub async fn get_logs(state: State<'_, AppState>) -> Result<Vec<LogEntry>, String> {
    Ok(match state.log_buffer.lock() {
        Ok(buf) => buf.iter().cloned().collect(),
        Err(_) => vec![],
    })
}

/// Clear the in-memory log buffer.
#[tauri::command]
pub async fn clear_logs(state: State<'_, AppState>) -> Result<(), String> {
    if let Ok(mut buf) = state.log_buffer.lock() {
        buf.clear();
    }
    Ok(())
}
