//! Numeric Rust provides a foundation for doing scientific computing with Rust. It aims to be for
//! Rust what Numpy is for Python.
//!
//! Its Tensor object uses OpenBLAS for fast matrix muliplications and other operations.
extern crate libc;
extern crate blas_sys;
extern crate num;
extern crate rand;

use num::traits::{Num, NumCast};

/// Numeric is a short-hand for all traits that need to be implemented for `T` in the `Tensor<T>`
/// struct.
pub trait TensorType: Copy + PartialOrd {}
impl<T: Copy + PartialOrd> TensorType for T {}

pub trait Numeric: Copy + Num + NumCast + PartialOrd {}
impl<T: Copy + Num + NumCast + PartialOrd> Numeric for T {}

pub mod tensor;
pub mod math;
pub mod random;

mod tests;

/// Many of the things in tensor is lifted into numeric since it is so common.
pub use tensor::{Tensor, AxisIndex};
pub use tensor::{SingleTensor, DoubleTensor};
