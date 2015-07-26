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

add_impl! { ln log10 log2 sin cos tan asin acos atan exp_m1
            ln_1p sinh cosh tanh asinh acosh atanh sqrt }

pub fn log<T: Numeric + Float>(x: Tensor<T>, base: T) -> Tensor<T> {
    let mut y = x;
    for i in 0..y.size() {
        y[i] = y[i].log(base);
    }
    y
}
