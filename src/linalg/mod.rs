//! Linear algebra functions.
//!
//! The functions are often implemented as member functions of `Tensor`, since it offers better
//! handling of generics.
//!
//! # Solving a linear equation
//!
//! ```
//! use numeric::Tensor;
//!
//! let a = Tensor::new(vec![1.0_f64, 0.5, 1.5, -1.0]).reshape(&[2, 2]);
//! let b = Tensor::ones(&[2]);
//!
//! let x = a.solve(&b);
//! ```

use tensor::Tensor;
use std::cmp::min;
use num::traits::Zero;

mod solve;
mod svd;

/// If passed a vector, creates a diagonal matrix with the vector as its diagonal.
/// If passed a matrix, the diagonal is extracted and returned.
pub fn diag<T: Copy + Zero>(a: &Tensor<T>) -> Tensor<T> {
    assert!(a.ndim() == 1 || a.ndim() == 2, "Can only run diag for vectors and matrices");
    if a.ndim() == 1 {
        let mut b = Tensor::zeros(&[a.size(), a.size()]);
        for i in 0..a.size() {
            b[(i, i)] = a[(i,)];
        }
        b
    } else {
        let mn = min(a.dim(0), a.dim(1));
        let mut b = Tensor::zeros(&[mn]);
        for i in 0..mn {
            b[(i,)] = a[(i, i)];
        }
        b
    }
}
