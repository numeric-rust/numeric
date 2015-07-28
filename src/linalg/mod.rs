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
use lapack;

macro_rules! add_impl {
    ($t:ty, $gesv:ident) => (
        impl Tensor<$t> {
            /// Solves the linear equation `Ax = b` and returns `x`. The matrix `A` is `self` and
            /// must be a square matrix. The input `b` must be a vector.
            ///
            /// Panics if matrix is singular.
            pub fn solve(&self, b: &Tensor<$t>) -> Tensor<$t> {
                assert!(self.ndim() == 2, "`A` must be a matrix (2D)");
                assert!(self.shape()[0] == self.shape()[1], "`A` must be a square matrix");
                assert!(b.ndim() == 1, "`b` must be a vector (1D)");
                assert!(self.shape()[0] == b.size(), "`A` and `b` must match");

                // A must be transposed, since LAPACK is column-major.
                let mut a_ = self.transpose();
                let mut b_ = b.clone();
                let mut info = 0;

                let n = self.shape()[0];
                let mut ipiv: Tensor<i32> = Tensor::empty(&[n]);
                lapack::$gesv(n, 1, a_.data_mut(), n, ipiv.data_mut(), b_.data_mut(), n,
                              &mut info);
                // TODO: Change this to a recoverable failure instead of a panic?
                if info < 0 {
                    panic!("Illegal input");
                } else if info > 0 {
                    panic!("Singular matrix");
                }
                b_
            }
        }
    )
}

add_impl!(f64, dgesv);
add_impl!(f32, sgesv);
