//! Numeric Rust provides a foundation for doing scientific computing with Rust. It aims to be for
//! Rust what Numpy is for Python.
//!
//! OpenBLAS/LAPACK is used to make things like matrix muliplications and solving linear equations
//! fast.
extern crate blas;
extern crate lapack;
extern crate num;
extern crate rand;
extern crate hdf5_sys;
extern crate libc;

pub mod traits;
pub mod tensor;
pub mod math;
pub mod random;
pub mod linalg;
pub mod io;

/// Many of the things in tensor is lifted into numeric since they are so common.
pub use tensor::{Tensor, AxisIndex, Ellipsis, StridedSlice, Index, Full, NewAxis};
pub use tensor::{SingleTensor, DoubleTensor};

/// Many of the functions in math are lifted into numeric since they are so common.
pub use math::{log, ln, log10, log2, sin, cos, tan, asin, acos, atan, exp_m1, exp, exp2,
               ln_1p, sinh, cosh, tanh, asinh, acosh, atanh, atan2, sqrt,
               floor, ceil, round, trunc, fract, abs, signum, powf, powi,
               is_nan, is_finite, is_infinite, is_normal,
               is_sign_positive, is_sign_negative};
