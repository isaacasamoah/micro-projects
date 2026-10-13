//! Inspectable self-attention, shared by the native example and browser.
//! This is not yet a complete transformer or a trained decision model.

pub mod attention;
pub mod example;
mod math;

#[cfg(feature = "browser")]
mod browser;
