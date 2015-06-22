use std::vec::Vec;
use std::ops::{Add ,Sub, Mul, Div};
use libc::{c_int, c_char};
use std::cmp::{PartialEq, Eq};
use blas_sys;
use std::fmt;

pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new() -> Tensor {
        Tensor{data: Vec::new(), shape: vec![0]}
    }

    pub fn new_data(data: Vec<f64>) -> Tensor {
        let len = data.len();
        Tensor{data: data, shape: vec![len]}
    }

    pub fn new_range(size: usize) -> Tensor {
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(i as f64);
        }
        Tensor{data: data, shape: vec![size]}
    }

    pub fn new_filled(shape: &[usize], v: f64) -> Tensor {
        let size = shape_product(shape);
        let sh = shape.to_vec();

        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(v);
        }
        Tensor{data: data, shape: sh}
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::new_filled(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::new_filled(shape, 1.0)
    }

    // Converts a shape that allows -1 to one with actual sizes
    fn convert_shape(&self, shape: &[isize]) -> Vec<usize> {
        let mut missing_index: Option<usize> = None;
        let mut total = 1;
        let mut sh = Vec::with_capacity(shape.len());

        for i in 0..shape.len() {
            if shape[i] == -1 {
                assert!(missing_index == None, "Can only specify one axis as -1");
                missing_index = Some(i);
                sh.push(0);
            } else {
                let v = shape[i] as usize;
                total *= v;
                sh.push(v);
            }
        }

        if let Some(i) = missing_index {
            sh[i] = self.size() / total;
        }
        sh
    }

    // Reshapes in-place
    /*
    pub fn reshape(&mut self, shape: &[isize]) {
        let proper_shape = self.convert_shape(shape);
        let s = proper_shape.iter().fold(1, |acc, &item| acc * item);
        assert_eq!(self.size(), s);
        self.shape = proper_shape;
    }
    */

    // Moves data
    pub fn reshaped(self, shape: &[isize]) -> Tensor {
        let proper_shape = self.convert_shape(shape);
        let s = proper_shape.iter().fold(1, |acc, &item| acc * item);
        assert_eq!(self.size(), s);
        Tensor{data: self.data, shape: proper_shape}
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.shape[1] + j]
    }

    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.shape[1] + j] = v;
    }

    /*
    fn get1(&self, i: usize) -> f64 {
        self.data[i]
    }

    fn set1(&mut self, i: usize, v: f64) {
        self.data[i] = v;
    }
    */

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn dot(t1: &Tensor, t2: &Tensor) -> Tensor {
        if t1.ndim() == 2 && t2.ndim() == 1 {
            assert_eq!(t1.shape[1], t2.shape[0]);
            let mut t3 = Tensor::zeros(&[t1.shape[0]]);
            if cfg!(noblas) {
                // Naive implementation, BLAS will be much faster
                for i in 0..t1.shape[0] {
                    let mut v = 0.0;
                    for k in 0..t1.shape[1] {
                        v += t1.get(i, k) * t2.data[k];
                    }
                    t3.data[i] = v;
                }
            } else {
                unsafe {
                    blas_sys::dgemv_(
                        &('T' as c_char),
                        &(t1.shape[1] as c_int),
                        &(t1.shape[0] as c_int),
                        &1.0,
                        t1.data.as_ptr(),
                        &(t1.shape[1] as c_int),
                        t2.data.as_ptr(),
                        &1,
                        &0.0,
                        t3.data.as_mut_ptr(),
                        &1
                    );
                }
            }
            t3
        } else if t1.ndim() == 2 && t2.ndim() == 2 {
            assert_eq!(t1.shape[1], t2.shape[0]);
            let mut t3 = Tensor::zeros(&[t1.shape[0], t2.shape[1]]);
            if cfg!(noblas) {
                // Naive implementation, BLAS will be much faster
                for i in 0..t1.shape[0] {
                    for j in 0..t2.shape[1] {
                        let mut v = 0.0;
                        for k in 0..t1.shape[1] {
                            v += t1.get(i, k) * t2.get(k, j);
                        }
                        t3.set(i, j, v);
                    }
                }
            } else {
                unsafe {
                    // Note: dgemm assumes column-major while we have row-major,
                    //       so we have to re-arrange things a bit
                    blas_sys::dgemm_(&('N' as c_char),
                                     &('N' as c_char),
                                     &(t2.shape[1] as c_int),
                                     &(t1.shape[0] as c_int),
                                     &(t2.shape[0] as c_int),
                                     &1.0,
                                     t2.data.as_ptr(),
                                     &(t2.shape[1] as c_int),
                                     t1.data.as_ptr(),
                                     &(t2.shape[0] as c_int),
                                     &0.0,
                                     t3.data.as_mut_ptr(),
                                     &(t2.shape[1] as c_int)
                                     );
                }
            }
            t3
        } else if t1.ndim() == 1 && t2.ndim() == 1 { // scalar product
            assert_eq!(t1.size(), t2.size());
            let mut v = 0.0;
            if cfg!(noblas) {
                // Naive implementation, BLAS will be much faster
                for k in 0..t1.shape[0] {
                    v += t1.data[k] * t2.data[k];
                }
            } else {
                let n = t1.size() as c_int;
                unsafe {
                    v = blas_sys::ddot_(&n,
                                        t1.data.as_ptr(),
                                        &1,
                                        t2.data.as_ptr(),
                                        &1);
                }
            }
            Tensor::new_data(vec![v])
        } else {
            panic!("Dot product is not supported for the matrix dimensions provided");
        }
    }
}

fn shape_product(shape: &[usize]) -> usize {
    shape.iter().fold(1, |acc, &v| acc * v)
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The `f` value implements the `Write` trait, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.
        //write!(f, "Tensor({:?})", self.shape)

        let mv = &self.data[..];
        let mut s = "".to_string();
        if self.ndim() == 1 {
            s.push_str("[");
            for i in 0..self.shape[0] {
                s = format!("{} {}", s, mv[i]);
            }
            s.push_str("]");
        } else if self.ndim() == 2 {
            s.push_str("[[");
            for i in 0..self.shape[0] {
                if i > 0 {
                    s.push_str(" [");
                }
                for j in 0..self.shape[1] {
                    if j > 0 {
                        s.push_str(" ");
                    }
                    s = format!("{}{:6.2}", s, self.get(i, j));
                }
                if i == self.shape[0] - 1 {
                    s.push_str("]]");
                } else {
                    s.push_str("]\n");
                }
            }
        }
        write!(f, "{}", s)
    }
}

//////////////
// ADDITION //
//////////////
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(mut self, rhs: Tensor) -> Tensor {
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

/////////////////
// SUBTRACTION //
/////////////////
impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(mut self, rhs: Tensor) -> Tensor {
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

impl<'a> Sub<&'a Tensor> for Tensor {
    type Output = Tensor;
    fn sub(mut self, rhs: &Tensor) -> Tensor {
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

impl<'a> Sub for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
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

impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(mut self, rhs: f64) -> Tensor {
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

////////////////////
// MULTIPLICATION //
////////////////////
impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(mut self, rhs: Tensor) -> Tensor {
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

impl<'a> Mul<&'a Tensor> for Tensor {
    type Output = Tensor;
    fn mul(mut self, rhs: &Tensor) -> Tensor {
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
impl<'a> Mul<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
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
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(mut self, rhs: f64) -> Tensor {
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

// S * T
/*
impl<'a> Mul<f64> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        let mut t = rhs.clone();
        if cfg!(noblas) {
            for i in 0..t.size() {
                t.data[i] *= self;
            }
        } else {
            unsafe {
                blas_sys::dscal_(&(t.size() as c_int),
                                 &self,
                                 t.data.as_mut_ptr(),
                                 &1);
            }
        }
        t
    }
}
*/

//////////////
// DIVISION //
//////////////
impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(mut self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        //if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] /= rhs.data[i];
            }
        /*} else {
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
        */
        self
    }
}

impl<'a> Div<&'a Tensor> for Tensor {
    type Output = Tensor;
    fn div(mut self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        //if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] /= rhs.data[i];
            }
        /*} else {
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
        }*/
        self
    }
}

// TODO: Change to separate lifetimes
impl<'a> Div<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        let mut t = self.clone();
        //if cfg!(noblas) {
            for i in 0..self.size() {
                t.data[i] /= rhs.data[i];
            }
        /*} else {
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
        */
        t
    }
}

// T * S
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(mut self, rhs: f64) -> Tensor {
        if cfg!(noblas) {
            for i in 0..self.size() {
                self.data[i] /= rhs;
            }
        } else {
            unsafe {
                blas_sys::dscal_(&(self.size() as c_int),
                                 &(1.0 / rhs),
                                 self.data.as_mut_ptr(),
                                 &1);
            }
        }
        self
    }
}


//////////////////
// OTHER TRAITS //
//////////////////
impl PartialEq<Tensor> for Tensor {
    fn eq(&self, rhs: &Tensor) -> bool {
        self.shape == rhs.shape && self.data == rhs.data
    }
}

impl Eq for Tensor { }

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor{data: self.data.clone(), shape: self.shape.clone()}
    }
}
