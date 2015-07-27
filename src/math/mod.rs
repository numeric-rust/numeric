//! Contains mathematical functions that operate on tensors. These functions are largely modelled
//! after what is available natively in Rust.

use tensor::Tensor;
use Numeric;
use num::traits::Float;

macro_rules! add_impl {
    ($($f:ident)*) => ($(
        pub fn $f<T: Numeric + Float>(x: Tensor<T>) -> Tensor<T> {
            //let mut y = Tensor::zeros(&x.shape());
            let mut y = x;
            for i in 0..y.size() {
                y[i] = y[i].$f();
            }
            y
        }
    )*)
}

add_impl! { ln log10 log2 sin cos tan asin acos atan exp_m1 exp exp2
            ln_1p sinh cosh tanh asinh acosh atanh sqrt
            floor ceil round trunc fract abs signum }

macro_rules! add_impl_to_bool {
    ($($f:ident)*) => ($(
        pub fn $f<T: Numeric + Float>(x: &Tensor<T>) -> Tensor<bool> {
            let mut y = Tensor::empty(&x.shape());
            for i in 0..y.size() {
                y[i] = x[i].$f();
            }
            y
        }
    )*)
}

add_impl_to_bool! { is_nan is_finite is_infinite is_normal is_sign_positive is_sign_negative }

pub fn log<T: Numeric + Float>(x: Tensor<T>, base: T) -> Tensor<T> {
    let mut y = x;
    for i in 0..y.size() {
        y[i] = y[i].log(base);
    }
    y
}

/// Calculates atan(y/x).
pub fn atan2<T: Numeric + Float>(y: &Tensor<T>, x: &Tensor<T>) -> Tensor<T> {
    assert!(x.shape() == y.shape(), "Shapes must match");
    let mut z = Tensor::empty(&x.shape());
    for i in 0..x.size() {
        z[i] = y[i].atan2(x[i]);
    }
    z
}

pub fn powf<T: Numeric + Float>(y: &Tensor<T>, x: &Tensor<T>) -> Tensor<T> {
    assert!(x.shape() == y.shape(), "Shapes must match");
    let mut z = Tensor::empty(&x.shape());
    for i in 0..x.size() {
        z[i] = y[i].powf(x[i]);
    }
    z
}

pub fn powi<T: Numeric + Float>(y: &Tensor<T>, x: &Tensor<i32>) -> Tensor<T> {
    assert!(x.shape() == y.shape(), "Shapes must match");
    let mut z = Tensor::empty(&x.shape());
    for i in 0..x.size() {
        z[i] = y[i].powi(x[i]);
    }
    z
}
