use std::ops::Mul;
use tensor::Tensor;
use libc::{c_int, c_char};
use blas_sys;

impl Mul<Tensor<f64>> for Tensor<f64> {
    type Output = Tensor<f64>;
    fn mul(mut self, rhs: Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, rhs.shape);
        if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] *= rhs.data[i];
            }
        } else {
            unsafe {
                blas_sys::dtbmv_(&('L' as c_char),
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

impl<'a> Mul<&'a Tensor<f64>> for Tensor<f64> {
    type Output = Tensor<f64>;
    fn mul(mut self, rhs: &Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, rhs.shape);
        if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] *= rhs.data[i];
            }
        } else {
            unsafe {
                blas_sys::dtbmv_(&('L' as c_char),
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
impl<'a> Mul<&'a Tensor<f64>> for &'a Tensor<f64> {
    type Output = Tensor<f64>;
    fn mul(self, rhs: &Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, rhs.shape);
        let mut t = self.clone();
        if cfg!(noblas) {
            for i in 0..self.size() {
                t.data[i] *= rhs.data[i];
            }
        } else {
            unsafe {
                blas_sys::dsbmv_(&('L' as c_char),
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
impl Mul<f64> for Tensor<f64> {
    type Output = Tensor<f64>;
    fn mul(mut self, rhs: f64) -> Tensor<f64> {
        if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] *= rhs;
            }
        } else {
            unsafe {
                blas_sys::dscal_(&(self.size() as c_int),
                                 &rhs,
                                 self.data.as_mut_ptr(),
                                 &1);
            }
        }
        self
    }
}
