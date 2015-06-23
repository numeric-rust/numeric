use std::vec::Vec;
use std::ops::{Add ,Sub, Mul, Div, Index, IndexMut};
use libc::{c_int, c_char};
use std::cmp::{PartialEq, Eq};
use blas_sys;
use std::fmt;

/// An implementation of an N-dimensional matrix.
///
/// A quick example:
///
/// ```
/// let t = Tensor::new(vec![1.0, 3.0, 2.0, 2.0]).reshaped(&[2, 2]);
/// println!("{}", t);
/// ```
///
/// Will output:
///
/// ```
/// [[  1.00   3.00]
///  [  2.00   2.00]]
/// ```
pub struct Tensor {
    /// The underlying data matrix, stored in row-major order.
    data: Vec<f64>,

    /// The shape of the tensor.
    shape: Vec<usize>,
}

/// Used for advanced slicing of a `Tensor`.
///
#[derive(Copy, Clone)]
pub enum AxisIndex {
    /// Indexes from start to end for this axis.
    Full,
    /// Indexes from start to end for all axes in the middle. A maximum of one can be used.
    Ellipsis,
    /// Creates a new axis of length 1 at this location.
    NewAxis,
    /// Picks one elements of an axis. This will remove that axis from the tensor.
    Index(isize),
    /// Specifies a half-open range. Slice(2, 5) will pick out indices 2, 3 and 4.
    Slice(isize, isize),

    /// Specifies the start (inclusive) and to the end.
    SliceFrom(isize),

    /// Specifies the end (exclusive) from the start.
    SliceTo(isize),
}

impl Tensor {
    /// Creates a new tensor with no elements of shape `(0,)`
    pub fn empty() -> Tensor {
        Tensor{data: Vec::new(), shape: vec![0]}
    }

    /// Creates a new tensor from a `Vec` object. It will take ownership of the vector.
    pub fn new(data: Vec<f64>) -> Tensor {
        let len = data.len();
        Tensor{data: data, shape: vec![len]}
    }

    /// Creates a new tensor with integer values starting at 0 and counting up:
    /// 
    /// ```
    /// Tensor::range(5) // [  0.00   1.00   2.00   3.00   4.00]
    /// ```
    pub fn range(size: usize) -> Tensor {
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(i as f64);
        }
        Tensor{data: data, shape: vec![size]}
    }

    /// Creates a new tensor of a given shape filled with the specified value.
    pub fn filled(shape: &[usize], v: f64) -> Tensor {
        let size = shape_product(shape);
        let sh = shape.to_vec();

        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(v);
        }
        Tensor{data: data, shape: sh}
    }

    /// Creates a zero-filled tensor of the specified shape.
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::filled(shape, 0.0)
    }

    /// Creates a one-filled tensor of the specified shape.
    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::filled(shape, 1.0)
    }

    /// Creates an identify 2-D tensor (matrix). That is, all elements are zero except the diagonal
    /// which is filled with ones.
    pub fn eye(size: usize) -> Tensor {
        let mut t = Tensor::zeros(&[size, size]);
        for k in 0..size {
            t.set(k, k, 1.0);
        }
        t
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// Returns a reference of the underlying data vector.
    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }

    /// Flattens the tensor to one-dimensional. Takes ownership and returns a new tensor.
    pub fn flatten(self) -> Tensor {
        let s = self.size();
        Tensor{data: self.data, shape: vec![s]}
    }

    /// Returns the strides of tensor for each axis.
    pub fn strides(&self) -> Vec<usize> {
        let mut ss = vec![1; self.shape.len()];
        for k in 1..ss.len() {
            let i = ss.len() - 1 - k;
            ss[i] = ss[i + 1] * self.shape[i + 1];
        }
        ss
    }

    /// Returns number of elements in the tensor.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of axes. This is the same as the length of the shape array.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn resolve_axis(&self, axis: usize, index: isize) -> usize {
        if index < 0 {
            (self.shape[axis] as isize + index) as usize
        } else {
            index as usize
        }
    }

    fn expand_slices(&self, slices_raw: &[AxisIndex]) -> (Vec<AxisIndex>, Vec<usize>) {
        // The returned axis will not contain any AxisIndex::Ellipsis
        let mut slices: Vec<AxisIndex> = Vec::with_capacity(self.shape.len());
        let mut newaxes: Vec<usize> = Vec::with_capacity(self.shape.len());

        // Count how many non NewAxis and non Ellipsis
        let mut nondotted = 0;
        for s in slices_raw {
            match *s {
                AxisIndex::NewAxis | AxisIndex::Ellipsis => {
                    nondotted += 0;
                },
                _ => {
                    nondotted += 1;
                }
            }
        }

        // Add an extra index to newaxes that represent insertion before the first axis
        newaxes.push(0);
        let mut ellipsis_found = false;
        for s in slices_raw {
            match *s {
                AxisIndex::Ellipsis => {
                    assert!(!ellipsis_found, "At most one AxisIndex::Ellipsis may be used");
                    assert!(self.shape.len() >= nondotted);

                    for _ in 0..(self.shape.len() - nondotted) {
                        slices.push(AxisIndex::Full);
                        newaxes.push(0);
                    }
                    ellipsis_found = true;
                    //newaxes.push(0);
                },
                AxisIndex::NewAxis => {
                    // Ignore these at this point
                    let n = newaxes.len();
                    newaxes[n - 1] += 1;
                },
                _ => {
                    newaxes.push(0);
                    slices.push(*s);
                }
            }
        }
        while slices.len() < self.shape.len() {
            slices.push(AxisIndex::Full);
        }
        while newaxes.len() < self.shape.len() + 1 {
            newaxes.push(0)
        }
        assert!(slices.len() == self.shape.len(), "Too many indices specified");
        debug_assert!(newaxes.len() == self.shape.len() + 1, "newaxis wrong length");

        (slices, newaxes)
    }

    /// Takes slices (subsets) of tensors and returns a tensor as a new object. Uses the
    /// `AxisIndex` enum to specify indexing for each axis.
    ///
    /// ```
    /// let t = Tensor::ones(&[2, 3, 4]);
    ///
    /// t.slice([AxisIndex::Ellipsis, AxisIndex::Slice(1, 3)] // shape [2, 3, 2]
    /// t.slice([AxisIndex::Index(-1)]) // shape [3, 4]
    /// t.slice([AxisIndex::Full, AxisIndex::SliceFrom(1), AxisIndex::Index(1)]) // shape [2, 2]
    /// ```
    pub fn slice(&self, slices_raw: &[AxisIndex]) -> Tensor {
        let (slices, newaxes) = self.expand_slices(slices_raw);

        let n = slices.len();
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(n);
        let mut dims: Vec<usize> = Vec::with_capacity(n);
        let mut indices: Vec<usize> = Vec::with_capacity(n);
        let mut shape: Vec<isize> = Vec::with_capacity(n);
        let mut axis = 0;
        for _ in 0..newaxes[0] {
            shape.push(1);
        }
        for s in slices {
            let (st, en, keepdim) = match s {
                AxisIndex::Index(i) => {
                    (self.resolve_axis(axis, i), self.resolve_axis(axis, i + 1), false)
                },
                AxisIndex::Full => {
                    (0, self.shape[axis], true)
                },
                AxisIndex::Slice(start, end) => {
                    (self.resolve_axis(axis, start), self.resolve_axis(axis, end), true)
                },
                AxisIndex::SliceTo(end) => {
                    (0, self.resolve_axis(axis, end), true)
                },
                AxisIndex::SliceFrom(start) => {
                    (self.resolve_axis(axis, start), self.shape[axis], true)
                },
                AxisIndex::Ellipsis | AxisIndex::NewAxis => {
                    // Should have been removed by expand_slices at this point
                    unreachable!();
                },
            };

            ranges.push((st, en));
            indices.push(st);
            dims.push(en - st);
            if keepdim {
                shape.push((en - st) as isize);
            }
            for _ in 0..newaxes[axis + 1] {
                shape.push(1);
            }
            axis += 1;
        }

        let mut t = Tensor::zeros(&dims);
        let strides = self.strides();

        let mut index = 0;
        for si in 0..strides.len() {
            index += strides[si] * indices[si];
        }

        let mut base_i = 0;
        for _ in 0..t.data.len() {
            let mut c = strides.len() - 1;

            t.data[base_i] = self.data[index];
            index += strides[c];
            indices[c] += strides[c];
            while indices[c] >= ranges[c].1 {
                if c == 0 {
                    break;
                }
                // Reset
                indices[c] = ranges[c].0;
                index -= dims[c] * strides[c];
                index += strides[c - 1];
                indices[c - 1] += 1;
                c -= 1;
            }

            base_i += 1;
        }

        t.reshaped(&shape[..])
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

    /// Takes the product of two tensors. If the tensors are both matrices (2D), then a matrix
    /// multiplication is taken. If the tensors are both vectors (1D), the scalar product is taken.
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
            Tensor::new(vec![v])
        } else {
            panic!("Dot product is not supported for the matrix dimensions provided");
        }
    }
}

fn shape_product(shape: &[usize]) -> usize {
    shape.iter().fold(1, |acc, &v| acc * v)
}

// Flattened indexing (this will index it as one-dimensional)
impl Index<usize> for Tensor {
    type Output = f64;
    fn index<'a>(&'a self, _index: usize) -> &'a f64 {
        &self.data[_index as usize]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut<'a>(&'a mut self, _index: usize) -> &'a mut f64 {
        &mut self.data[_index as usize]
    }
}

// 1-D indexing
impl Index<(usize,)> for Tensor {
    type Output = f64;
    fn index<'a>(&'a self, _index: (usize,)) -> &'a f64 {
        assert!(self.ndim() == 1);
        &self.data[_index.0]
    }
}

impl IndexMut<(usize,)> for Tensor {
    fn index_mut<'a>(&'a mut self, _index: (usize,)) -> &'a mut f64 {
        assert!(self.ndim() == 1);
        &mut self.data[_index.0]
    }
}

// 2-D indexing
impl Index<(usize, usize)> for Tensor {
    type Output = f64;
    fn index<'a>(&'a self, _index: (usize, usize)) -> &'a f64 {
        assert!(self.ndim() == 2);
        &self.data[(_index.0 * self.shape[1] + _index.1)]
    }
}

impl IndexMut<(usize, usize)> for Tensor {
    fn index_mut<'a>(&'a mut self, _index: (usize, usize)) -> &'a mut f64 {
        assert!(self.ndim() == 2);
        &mut self.data[(_index.0 * self.shape[1] + _index.1)]
    }
}

// 3-D indexing
impl Index<(usize, usize, usize)> for Tensor {
    type Output = f64;
    fn index<'a>(&'a self, _index: (usize, usize, usize)) -> &'a f64 {
        assert!(self.ndim() == 3);
        &self.data[(_index.0 * self.shape[1] * self.shape[2] +
                    _index.1 * self.shape[2] +
                    _index.2)]
    }
}

impl IndexMut<(usize, usize, usize)> for Tensor {
    fn index_mut<'a>(&'a mut self, _index: (usize, usize, usize)) -> &'a mut f64 {
        assert!(self.ndim() == 3);
        &mut self.data[(_index.0 * self.shape[1] * self.shape[2] +
                        _index.1 * self.shape[2] +
                        _index.2)]
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mv = &self.data[..];
        let mut s = "".to_string();
        if self.ndim() == 1 {
            s.push_str("[");
            for i in 0..self.shape[0] {
                if i > 0 {
                    s.push_str(" ");
                }
                s = format!("{}{:6.2}", s, mv[i]);
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
        } else {
            s = format!("Tensor({:?})", self.shape);
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
