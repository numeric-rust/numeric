use std::ops::Mul;
use tensor::Tensor;
use blas;

macro_rules! add_impl {
    ($t:ty, $tbmv:ident, $sbmv:ident, $scal:ident) => (
        impl Mul<Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn mul(mut self, rhs: Tensor<$t>) -> Tensor<$t> {
                assert_eq!(self.shape, rhs.shape);
                if cfg!(noblas) {
                    for i in 0..self.size() {
                        self.data[i] *= rhs.data[i];
                    }
                } else {
                    blas::$tbmv(b'L', b'T', b'N', self.size(), 0, &rhs.data, 1, &mut self.data, 1);
                }
                self
            }
        }

        impl<'a> Mul<&'a Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn mul(mut self, rhs: &Tensor<$t>) -> Tensor<$t> {
                assert_eq!(self.shape, rhs.shape);
                if cfg!(noblas) {
                    for i in 0..self.size() {
                        self.data[i] *= rhs.data[i];
                    }
                } else {
                    blas::$tbmv(b'L', b'T', b'N', self.size(), 0, &rhs.data, 1, &mut self.data, 1);
                }
                self
            }
        }

        // TODO: Change to separate lifetimes
        impl<'a> Mul<&'a Tensor<$t>> for &'a Tensor<$t> {
            type Output = Tensor<$t>;
            fn mul(self, rhs: &Tensor<$t>) -> Tensor<$t> {
                assert_eq!(self.shape, rhs.shape);
                let mut t = self.clone();
                if cfg!(noblas) {
                    for i in 0..self.size() {
                        t.data[i] *= rhs.data[i];
                    }
                } else {
                    blas::$sbmv(b'L', self.size(), 0, 1.0, &self.data, 1, &rhs.data, 1, 0.0,
                                &mut t.data, 1);
                }
                t
            }
        }

        // T * S
        impl Mul<$t> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn mul(mut self, rhs: $t) -> Tensor<$t> {
                if cfg!(noblas) {
                    for i in 0..self.size() {
                        self.data[i] *= rhs;
                    }
                } else {
                    blas::$scal(self.size(), rhs, &self.data, 1);
                }
                self
            }
        }
    )
}

add_impl!(f32, stbmv, ssbmv, sscal);
add_impl!(f64, dtbmv, dsbmv, dscal);
