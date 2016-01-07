//! Traits used by Tensor.

use num::traits::{Num, NumCast};

/// This is the basic trait that must be satisfied for basic elements used in `Tensor`.
pub trait TensorTrait: Copy {}
impl<T: Copy> TensorTrait for T {}

/// `NumericTrait` extends `TensorTrait` to all the numeric types supported by `Tensor`
/// (e.g. `u8` and `f32`).
pub trait NumericTrait: TensorTrait + Num + NumCast + PartialOrd {}
impl<T: TensorTrait + Num + NumCast + PartialOrd> NumericTrait for T {}
