use std::ops::Sub;
use tensor::Tensor;
use libc::c_int;
use blas_sys;

macro_rules! add_impl {
    ($t:ty, $bfunc:ident) => (
        impl Sub<Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn sub(mut self, rhs: Tensor<$t>) -> Tensor<$t> {
                assert_eq!(self.shape, rhs.shape);
                if cfg!(noblas) {
                    for i in 0..self.size() {
                        self.data[i] -= rhs.data[i];
                    }
                } else {
                    unsafe {
                        blas_sys::$bfunc(&(self.data.len() as c_int),
                                         &-1.0,
                                         rhs.data.as_ptr(),
                                         &1,
                                         self.data.as_mut_ptr(),
                                         &1);
                    }
                }
                self
            }
        }

        impl<'a> Sub<&'a Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn sub(mut self, rhs: &Tensor<$t>) -> Tensor<$t> {
                assert_eq!(self.shape, rhs.shape);
                if cfg!(noblas) {
                    for i in 0..self.size() {
                        self.data[i] -= rhs.data[i];
                    }
                } else {
                    unsafe {
                        blas_sys::$bfunc(&(self.data.len() as c_int),
                                         &-1.0,
                                         rhs.data.as_ptr(),
                                         &1,
                                         self.data.as_mut_ptr(),
                                         &1);
                    }
                }
                self
            }
        }

        impl<'a> Sub for &'a Tensor<$t> {
            type Output = Tensor<$t>;
            fn sub(self, rhs: &Tensor<$t>) -> Tensor<$t> {
                assert_eq!(self.shape, rhs.shape);
                let mut t = self.clone();
                if cfg!(noblas) {
                    for i in 0..self.size() {
                        t.data[i] -= rhs.data[i];
                    }
                } else {
                    unsafe {
                        blas_sys::$bfunc(&(self.data.len() as c_int),
                                         &-1.0,
                                         rhs.data.as_ptr(),
                                         &1,
                                         t.data.as_mut_ptr(),
                                         &1);
                    }
                }
                t
            }
        }

        impl Sub<$t> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn sub(mut self, rhs: $t) -> Tensor<$t> {
                if cfg!(noblas) {
                    for i in 0..self.size() {
                        self.data[i] -= rhs;
                    }
                } else {
                    unsafe {
                        blas_sys::$bfunc(&(self.data.len() as c_int),
                                         &-1.0,
                                         &rhs,
                                         &0,
                                         self.data.as_mut_ptr(),
                                         &1);
                    }
                }
                self
            }
        }
    )
}

add_impl!(f32, saxpy_);
add_impl!(f64, daxpy_);
