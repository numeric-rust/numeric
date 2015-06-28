//! Contains mathematical functions that operate on tensors. These functions are largely modelled
//! after what is available natively in Rust.

use tensor::Tensor;
use {Numeric, TensorType};
use num::traits::Float;

macro_rules! add_impl0 {
    ($($f:ident)*) => ($(
        pub fn $f<T: Numeric + Float>(x: &Tensor<T>) -> Tensor<T> {
            let mut y = Tensor::zeros(&x.shape());
            for i in 0..x.size() {
                y[i] = x[i].$f();
            }
            y
        }
    )*)
}

add_impl0! { ln log10 log2 sin cos tan asin acos atan exp_m1
             ln_1p sinh cosh tanh asinh acosh atanh }

pub fn log<T: Numeric + Float>(x: &Tensor<T>, base: T) -> Tensor<T> {
    let mut y = Tensor::zeros(&x.shape());
    for i in 0..x.size() {
        y[i] = x[i].log(base);
    }
    y
}

pub fn gt<T: TensorType>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<bool> {
    assert_eq!(lhs.shape(), rhs.shape());
    let mut y = Tensor::empty(&lhs.shape());
    for i in 0..lhs.size() {
        y[i] = lhs[i] > rhs[i];
    }
    y
}
