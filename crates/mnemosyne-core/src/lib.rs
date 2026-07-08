//! Core types, traits, and error definitions for Mnemosyne.
//!
//! This crate has no internal dependencies and is the foundation
//! for all other Mnemosyne crates.

pub mod error;
pub mod traits;
pub mod types;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;
