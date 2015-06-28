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
//! let a = Tensor::new(vec![1.0_f64, 0.5, 1.5, -1.0]).reshaped(&[2, 2]);
//! let b = Tensor::ones(&[2]);
//!
//! let x = a.solve(&b);
//! ```
use tensor::Tensor;
use libc::c_int;
use lapack_sys;

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

                // A must be tranposed, since LAPACK is column-major.
                let mut a_ = self.transpose();
                let mut b_ = b.clone();
                let mut info: c_int = 0;

                // TODO: This should not be necessary, but the lapack_sys currently specifies
                //       everything as a mut
                let mut n: c_int = self.shape()[0] as c_int;
                let mut nrhs: c_int = 1;
                let mut lda: c_int = n;
                let mut ldb: c_int = n;
                let mut ipiv: Tensor<i32> = Tensor::empty(&[n as usize]);
                unsafe {
                    lapack_sys::$gesv(&mut n,                       // n
                                      &mut nrhs,                    // a
                                      a_.data_mut().as_mut_ptr(),   // nrhs
                                      &mut lda,                     // lda
                                      ipiv.data_mut().as_mut_ptr(), // ipiv
                                      b_.data_mut().as_mut_ptr(),   // b
                                      &mut ldb,                     // ldb
                                      &mut info                     // info
                                      );
                }
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

add_impl!(f64, dgesv_);
add_impl!(f32, sgesv_);
