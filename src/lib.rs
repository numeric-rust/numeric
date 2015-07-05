//! Numeric Rust provides a foundation for doing scientific computing with Rust. It aims to be for
//! Rust what Numpy is for Python.
//!
//! OpenBLAS/LAPACK is used to make things like matrix muliplications and solving linear equations
//! fast.
extern crate libc;
extern crate blas_sys;
extern crate lapack_sys;
extern crate num;
extern crate rand;

use num::traits::{Num, NumCast};

// TODO: Sometimes TensorType is used, sometimes Copy. Make a decision and stick with it.
/// This is the basic trait that must be satisfied for basic elements used in `Tensor`.
pub trait TensorType: Copy {}
impl<T: Copy> TensorType for T {}

/// `Numeric` extends `TensorType` to all the numeric types supported by `Tensor` 
/// (e.g. `u8` and `f32`).
pub trait Numeric: TensorType + Num + NumCast + PartialOrd {}
impl<T: TensorType + Num + NumCast + PartialOrd> Numeric for T {}

pub mod tensor;
pub mod math;
pub mod random;
pub mod linalg;

mod tests;

/// Many of the things in tensor is lifted into numeric since it is so common.
pub use tensor::{Tensor, AxisIndex};
pub use tensor::{SingleTensor, DoubleTensor};
