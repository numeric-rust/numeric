//! Numeric Rust provides a foundation for doing scientific computing with Rust. It aims to be for
//! Rust what Numpy is for Python.
//!
//! Its Tensor object uses OpenBLAS for fast matrix muliplications and other operations.
extern crate libc;
extern crate blas_sys;

pub mod tensor;
pub mod tests;
