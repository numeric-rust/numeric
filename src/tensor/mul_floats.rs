use std::ops::Mul;
use tensor::Tensor;
use libc::{c_int, c_char};
use blas_sys;

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
                    unsafe {
                        blas_sys::$tbmv(&('L' as c_char),
                                         &('T' as c_char),
                                         &('N' as c_char),
                                         &(self.size() as c_int),
                                         &0,
                                         rhs.data.as_ptr(),
                                         &1,
                                         self.data.as_mut_ptr(),
                                         &1);
                    }
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
                    unsafe {
                        blas_sys::$tbmv(&('L' as c_char),
                                         &('T' as c_char),
                                         &('N' as c_char),
                                         &(self.size() as c_int),
                                         &0,
                                         rhs.data.as_ptr(),
                                         &1,
                                         self.data.as_mut_ptr(),
                                         &1);
                    }
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
                    unsafe {
                        blas_sys::$sbmv(&('L' as c_char),
                                         &(self.size() as c_int),
                                         &0,
                                         &1.0,
                                         self.data.as_ptr(),
                                         &1,
                                         rhs.data.as_ptr(),
                                         &1,
                                         &0.0,
                                         t.data.as_mut_ptr(),
                                         &1);
                    }
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
                    unsafe {
                        blas_sys::$scal(&(self.size() as c_int),
                                         &rhs,
                                         self.data.as_mut_ptr(),
                                         &1);
                    }
                }
                self
            }
        }
    )
}

add_impl!(f32, stbmv_, ssbmv_, sscal_);
add_impl!(f64, dtbmv_, dsbmv_, dscal_);
