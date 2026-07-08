#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    mnemosyne_tauri_lib::run();
}
