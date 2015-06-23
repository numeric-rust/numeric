use tensor::Tensor;
use std::ops::Add;
use num::traits::Num;

impl<T: Copy + Num> Add<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;
    fn add(mut self, rhs: Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.size() {
            self.data[i] += rhs.data[i];
        }
        self
    }
}

/*
impl<'a> Add<&'a Tensor> for Tensor {
    type Output = Tensor;
    fn add(mut self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] += rhs.data[i];
            }
        } else {
            unsafe {
                blas_sys::daxpy_(&(self.data.len() as c_int),
                                 &1.0,
                                 rhs.data.as_ptr(),
                                 &1,
                                 self.data.as_mut_ptr(),
                                 &1);
            }
        }
        self
    }
}

impl<'a> Add for &'a Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        let mut t = self.clone();
        if cfg!(noblas) {
            for i in 0..self.size() {
                t.data[i] += rhs.data[i];
            }
        } else {
            unsafe {
                blas_sys::daxpy_(&(self.data.len() as c_int),
                                 &1.0,
                                 rhs.data.as_ptr(),
                                 &1,
                                 t.data.as_mut_ptr(),
                                 &1);
            }
        }
        t
    }
}

impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(mut self, rhs: f64) -> Tensor {
        if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] += rhs;
            }
        } else {
            unsafe {
                blas_sys::daxpy_(&(self.data.len() as c_int),
                                 &1.0,
                                 &rhs,
                                 &0,
                                 self.data.as_mut_ptr(),
                                 &1);
            }
        }
        self
    }
}
*/
