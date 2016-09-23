use tensor::Tensor;
use lapack;
use std::cmp::{min, max};

macro_rules! add_svd_impl {
    ($t:ty, $gesdd:ident) => (
        impl Tensor<$t> {
            /// Performs a singular value decomposition on the matrix.
            pub fn svd(&self, full_matrices: bool) -> (Tensor<$t>, Tensor<$t>, Tensor<$t>) {
                assert!(self.ndim() == 2, "`A` must be a matrix (2D)");
                let m = self.dim(0);
                let n = self.dim(1);
                let k = min(m, n);
                let mn = min(m, n);
                let mx = max(m, n);
                let (jobz, ldu, ldvt, ucol, lwork) = if full_matrices {
                    (b'A', m, n, m, 4*mn*mx + 6*mn + mx)
                } else {
                    (b'S', m, mn, mn, 4*mn*mx + 7*mn)
                };

                let mut a = self.clone().transpose().canonize();
                let mut work: Tensor<$t> = Tensor::empty(&[lwork]);
                let mut s: Tensor<$t> = Tensor::empty(&[k]);
                let mut ut: Tensor<$t> = Tensor::empty(&[ucol, ldu]);
                let mut v: Tensor<$t> = Tensor::empty(&[n, ldvt]);
                let mut iwork: Tensor<i32> = Tensor::empty(&[8*mn]);
                let mut info = 0;

                lapack::$gesdd(jobz, m, n, a.slice_mut(), m, s.slice_mut(),
                               ut.slice_mut(), ldu, v.slice_mut(), ldvt,
                               work.slice_mut(), lwork as isize, iwork.slice_mut(),
                               &mut info);

                if info < 0 {
                    panic!("Illegal input ({})", -info);
                } else if info > 0 {
                    panic!("Did not converge");
                }
                (ut.transpose(), s, v.transpose())
            }
        }
    )
}

add_svd_impl!(f64, dgesdd);
add_svd_impl!(f32, sgesdd);
