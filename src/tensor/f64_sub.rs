use std::ops::Sub;
use tensor::Tensor;
use libc::c_int;
use blas_sys;

impl Sub<Tensor<f64>> for Tensor<f64> {
    type Output = Tensor<f64>;
    fn sub(mut self, rhs: Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, rhs.shape);
        if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] -= rhs.data[i];
            }
        } else {
            unsafe {
                blas_sys::daxpy_(&(self.data.len() as c_int),
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

impl<'a> Sub<&'a Tensor<f64>> for Tensor<f64> {
    type Output = Tensor<f64>;
    fn sub(mut self, rhs: &Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, rhs.shape);
        if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] -= rhs.data[i];
            }
        } else {
            unsafe {
                blas_sys::daxpy_(&(self.data.len() as c_int),
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

impl<'a> Sub for &'a Tensor<f64> {
    type Output = Tensor<f64>;
    fn sub(self, rhs: &Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, rhs.shape);
        let mut t = self.clone();
        if cfg!(noblas) {
            for i in 0..self.size() {
                t.data[i] -= rhs.data[i];
            }
        } else {
            unsafe {
                blas_sys::daxpy_(&(self.data.len() as c_int),
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

impl Sub<f64> for Tensor<f64> {
    type Output = Tensor<f64>;
    fn sub(mut self, rhs: f64) -> Tensor<f64> {
        if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] -= rhs;
            }
        } else {
            unsafe {
                blas_sys::daxpy_(&(self.data.len() as c_int),
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
