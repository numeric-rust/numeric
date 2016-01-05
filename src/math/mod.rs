//! Contains mathematical functions that operate on tensors. These functions are largely modelled
//! after what is available natively in Rust.

use num::traits::Float;
use tensor::Tensor;
use traits::NumericTrait;

macro_rules! add_impl {
    ($($f:ident)*) => ($(
        pub fn $f<T: NumericTrait + Float>(x: Tensor<T>) -> Tensor<T> {
            let mut y = x;
            y.canonize_inplace();
            {
                let n = y.size();
                let mut data = y.slice_mut();
                for i in 0..n {
                    data[i] = data[i].$f();
                }
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
        pub fn $f<T: NumericTrait + Float>(x: &Tensor<T>) -> Tensor<bool> {
            let mut y: Tensor<bool> = Tensor::empty(&x.shape());
            {
                let mut data = y.slice_mut();
                for (i, v) in x.iter().enumerate() {
                    data[i] = v.$f();
                }
            }
            y
        }
    )*)
}

add_impl_to_bool! { is_nan is_finite is_infinite is_normal is_sign_positive is_sign_negative }

pub fn log<T: NumericTrait + Float>(x: Tensor<T>, base: T) -> Tensor<T> {
    let mut y = x;
    y.canonize_inplace();
    {
        let n = y.size();
        let mut data = y.slice_mut();
        for i in 0..n {
            data[i] = data[i].log(base);
        }
    }
    y
}

/// Calculates atan(y/x).
pub fn atan2<T: NumericTrait + Float>(y: &Tensor<T>, x: &Tensor<T>) -> Tensor<T> {
    assert!(x.shape() == y.shape(), "Shapes must match");
    let mut z = Tensor::empty(&x.shape());
    {
        let mut data = z.slice_mut();
        for (i, (v1, v2)) in y.iter().zip(x.iter()).enumerate() {
            data[i] = v1.atan2(v2);
        }
    }
    z
}

pub fn powf<T: NumericTrait + Float>(y: &Tensor<T>, x: &Tensor<T>) -> Tensor<T> {
    assert!(x.shape() == y.shape(), "Shapes must match");
    let mut z = Tensor::empty(&x.shape());
    {
        let mut data = z.slice_mut();
        for (i, (v1, v2)) in y.iter().zip(x.iter()).enumerate() {
            data[i] = v1.powf(v2);
        }
    }
    z
}

pub fn powi<T: NumericTrait + Float>(y: &Tensor<T>, x: &Tensor<i32>) -> Tensor<T> {
    assert!(x.shape() == y.shape(), "Shapes must match");
    let mut z = Tensor::empty(&x.shape());
    {
        let mut data = z.slice_mut();
        for (i, (v1, v2)) in y.iter().zip(x.iter()).enumerate() {
            data[i] = v1.powi(v2);
        }
    }
    z
}
