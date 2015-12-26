//! The tensor module defines an N-dimensional matrix for use in scientific computing.
//!
//! Many of the things in this module are lifted out of the `tensor` namespace, which means you can
//! do things like:
//!
//! ```
//! use numeric::Tensor;
//! ```

use std::vec::Vec;
use {TensorType, Numeric};
use num::traits::cast;
use std::rc::Rc;

/// An implementation of an N-dimensional matrix.
/// A quick example:
///
/// ```
/// use numeric::Tensor;
/// let t = Tensor::new(vec![1.0f64, 3.0, 2.0, 2.0]).reshape(&[2, 2]);
/// println!("t = {}", t);
/// ```
///
/// Will output:
///
/// ```text
/// t =
///  1 3
///  2 2
/// [Tensor<f64> of shape 2x2]
/// ```
pub struct Tensor<T> {
    /// The underlying data matrix, stored in row-major order.
    data: Rc<Vec<T>>,

    /// The shape of the tensor.
    shape: Vec<usize>,

    /// The strides for each axis.
    strides: Vec<isize>,

    /// The offsets for each axis.
    offsets: Vec<isize>
}

// Common type-specific tensors

/// Type alias for `Tensor<f64>`
pub type DoubleTensor = Tensor<f64>;

/// Type alias for `Tensor<f32>`
pub type SingleTensor = Tensor<f32>;

/// Used for advanced slicing of a `Tensor`.
#[derive(Copy, Clone)]
pub enum AxisIndex {
    /// Indexes from start to end for this axis.
    Full,
    /// Indexes from start to end for all axes in the middle. A maximum of one can be used.
    Ellipsis,
    /// Creates a new axis of length 1 at this location.
    NewAxis,
    /// Picks one element of an axis. This will remove that axis from the tensor.
    Index(isize),
    /// Specifies a half-open range. Slice(2, 5) will pick out indices 2, 3 and 4.
    Slice(isize, isize),
    /// Specifies the start (inclusive) and to the end.
    SliceFrom(isize),
    /// Specifies the end (exclusive) from the start.
    SliceTo(isize),
}

mod dot;

mod display;
mod generics;

mod summary;
mod eq;
mod indexing;
mod concat;
mod convert;

mod binary;

use num::traits::{Num, NumCast};

impl<T: TensorType> Tensor<T> {
    pub unsafe fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.slice_mut().as_mut_ptr()
    }

    /// Creates a new tensor from a `Vec` object. It will take ownership of the vector.
    pub fn new(data: Vec<T>) -> Tensor<T> {
        let len = data.len();
        Tensor{data: Rc::new(data), shape: vec![len]}
    }

    /// Creates a zero-filled tensor of the specified shape.
    pub fn empty(shape: &[usize]) -> Tensor<T> {
        //let data = Vector::with_capacity(
        let size = shape_product(shape);
        let sh = shape.to_vec();

        let mut data = Vec::with_capacity(size);
        // Fill with potentially random elements.
        // TODO: Possibly revise this (solution that doesn't need unsafe?)
        unsafe {
            data.set_len(size);
        }
        Tensor{data: Rc::new(data), shape: sh}
    }

    /// Returns a flat slice of the tensor.
    pub fn slice(&self) -> &[T] {
        &self.data[..]
    }

    /// Returns a mutable flat slice of the tensors. This will cause a copy unless the tensor is
    /// unique. This is mostly used internally.
    pub fn slice_mut(&mut self) -> &mut [T] {
        &mut Rc::make_mut(&mut self.data)[..]
    }

    /// Creates a Tensor representing a scalar
    pub fn scalar(value: T) -> Tensor<T> {
        Tensor{data: Rc::new(vec![value]), shape: vec![]}
    }

    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0 && self.size() == 1
    }

    pub fn scalar_value(&self) -> T {
        debug_assert!(self.is_scalar(), "Tensor must be scalar to use scalar_value()");
        self.data[0]
    }

    /// Creates a new tensor of a given shape filled with the specified value.
    pub fn filled(shape: &[usize], v: T) -> Tensor<T> {
        let size = shape_product(shape);
        let sh = shape.to_vec();

        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(v);
        }
        Tensor{data: Rc::new(data), shape: sh}
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// Returns length of single dimension.
    pub fn dim(&self, axis: usize) -> usize {
        self.shape[axis]
    }

    /// Returns a reference of the underlying data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Flattens the tensor to one-dimensional. Takes ownership and returns a new tensor.
    pub fn flatten(self) -> Tensor<T> {
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
    #[inline]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of axes. This is the same as the length of the shape array.
    #[inline]
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

    fn expand_indices(&self, selection: &[AxisIndex]) -> (Vec<AxisIndex>, Vec<usize>) {
        // The returned axis will not contain any AxisIndex::Ellipsis
        let mut sel: Vec<AxisIndex> = Vec::with_capacity(self.shape.len());
        let mut newaxes: Vec<usize> = Vec::with_capacity(self.shape.len());

        // Count how many non NewAxis and non Ellipsis
        let mut nondotted = 0;
        for s in selection {
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
        for s in selection {
            match *s {
                AxisIndex::Ellipsis => {
                    assert!(!ellipsis_found, "At most one AxisIndex::Ellipsis may be used");
                    assert!(self.shape.len() >= nondotted);

                    for _ in 0..(self.shape.len() - nondotted) {
                        sel.push(AxisIndex::Full);
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
                    sel.push(*s);
                }
            }
        }
        while sel.len() < self.shape.len() {
            sel.push(AxisIndex::Full);
        }
        while newaxes.len() < self.shape.len() + 1 {
            newaxes.push(0)
        }
        assert!(sel.len() == self.shape.len(), "Too many indices specified");
        debug_assert!(newaxes.len() == self.shape.len() + 1, "newaxis wrong length");

        (sel, newaxes)
    }

    fn prepare_index(&self, selection: &[AxisIndex]) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>, Vec<usize>) {
        let (sel, newaxes) = self.expand_indices(selection);
        let mut axis = 0;
        let n = sel.len();

        let mut shape: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..newaxes[0] {
            shape.push(1);
        }
        let mut dims: Vec<usize> = Vec::with_capacity(n);
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(n);
        let mut indices: Vec<usize> = Vec::with_capacity(n);
        for s in sel {
            let (st, en, keepdim) = match s {
                AxisIndex::Index(i) => {
                    let idx = self.resolve_axis(axis, i);
                    (idx, idx + 1, false)
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
                    // Should have been removed by expand_indices at this point
                    unreachable!();
                },
            };

            ranges.push((st, en));
            indices.push(st);
            dims.push(en - st);
            if keepdim {
                shape.push((en - st) as usize);
            }
            for _ in 0..newaxes[axis + 1] {
                shape.push(1);
            }
            axis += 1;
        }
        (ranges, indices, dims, shape)
    }

    /// Takes slices (subsets) of tensors and returns a tensor as a new object. Uses the
    /// `AxisIndex` enum to specify indexing for each axis.
    ///
    /// ```
    /// use numeric::{DoubleTensor, AxisIndex};
    ///
    /// let t = DoubleTensor::ones(&[2, 3, 4]);
    ///
    /// t.index(&[AxisIndex::Ellipsis, AxisIndex::Slice(1, 3)]); // shape [2, 3, 2]
    /// t.index(&[AxisIndex::Index(-1)]); // shape [3, 4]
    /// t.index(&[AxisIndex::Full, AxisIndex::SliceFrom(1), AxisIndex::Index(1)]); // shape [2, 2]
    /// ```
    pub fn index(&self, selection: &[AxisIndex]) -> Tensor<T> {
        let (ranges, mut indices, dims, shape) = self.prepare_index(&selection);

        let mut t = Tensor::empty(&dims);
        let strides = self.strides();

        let mut index = 0;
        for si in 0..strides.len() {
            index += strides[si] * indices[si];
        }

        {
            let n = t.size();
            let mut data = t.slice_mut();
            let mut base_i = 0;
            for _ in 0..n {
                let mut c = strides.len() - 1;

                data[base_i] = self.data[index];
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
        }

        t.reshape_proper(&shape[..])
    }

    /// Similar to `index`, except this updates the tensor with `other` instead of returning them.
    pub fn index_set(&mut self, selection: &[AxisIndex], other: &Tensor<T>) {
        let (ranges, mut indices, dims, shape) = self.prepare_index(&selection);
        assert!(shape == other.shape, "Shape not matching");

        let strides = self.strides();

        let mut index = 0;
        for si in 0..strides.len() {
            index += strides[si] * indices[si];
        }

        let n = self.size();
        let mut data = self.slice_mut();
        let mut base_i = 0;
        for _ in 0..other.size() {
            let mut c = strides.len() - 1;

            assert!(index < n);
            data[index] = other[base_i];
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
    }

    pub fn bool_index(&self, indices: &Tensor<bool>) -> Tensor<T> {
        let mut s = 0;
        for i in 0..indices.size() {
            if indices[i] {
                s += 1;
            }
        }
        let mut t = Tensor::empty(&[s]);
        let mut j = 0;
        for i in 0..indices.size() {
            if indices[i] {
                t[j] = self.data[i];
                j += 1;
            }
        }
        t
    }

    pub fn bool_index_set(&mut self, indices: &Tensor<bool>, values: &Tensor<T>) {
        let mut s = 0;
        for i in 0..indices.size() {
            if indices[i] {
                s += 1;
            }
        }
        if values.is_scalar() {
            let v = values.scalar_value();
            let mut data = self.slice_mut();
            for i in 0..indices.size() {
                if indices[i] {
                    data[i] = v;
                }
            }
        } else {
            assert!(values.size() == s);
            let mut j = 0;
            let mut data = self.slice_mut();
            for i in 0..indices.size() {
                if indices[i] {
                    data[i] = values[j];
                    j += 1;
                }
            }
        }
    }


    /// Takes a flatten index (in row-major order) and returns a vector of the per-axis indices.
    pub fn unravel_index(&self, index: usize) -> Vec<usize> {
        let strides = self.strides();
        let mut ii: Vec<usize> = Vec::with_capacity(self.ndim());
        let mut c = index;
        for i in 0..self.ndim() {
            ii.push(c / strides[i]);
            c %= strides[i];
        }
        ii
    }

    /// Takes an array of per-axis indices and returns a flattened index (in row-major order).
    pub fn ravel_index(&self, ii: &[usize]) -> usize {
        assert_eq!(ii.len(), self.ndim());
        let strides = self.strides();
        let mut index = 0;
        for i in 0..self.ndim() {
            index += strides[i] * ii[i];
        }
        index
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

    fn reshape_proper(self, proper_shape: &[usize]) -> Tensor<T> {
        let s = proper_shape.iter().fold(1, |acc, &item| acc * item);
        assert_eq!(self.size(), s);
        Tensor{data: self.data, shape: proper_shape.to_vec()}
    }

    /// Reshapes the data. This moves the data, so no memory is allocate.
    pub fn reshape(self, shape: &[isize]) -> Tensor<T> {
        let proper_shape = self.convert_shape(shape);
        self.reshape_proper(&proper_shape[..])
    }

    #[inline]
    fn get2(&self, i: usize, j: usize) -> T {
        self.data[i * self.shape[1] + j]
    }

    #[inline]
    fn set2(&mut self, i: usize, j: usize, v: T) {
        let sh1 = self.shape[1];
        let mut data = self.slice_mut();
        data[i * sh1 + j] = v;
    }

    /// Sets all the values according to another tensor of the same shape.
    pub fn set(&mut self, other: &Tensor<T>) -> () {
        assert!(self.shape() == other.shape());
        let n = self.size();
        let mut data = self.slice_mut();
        for i in 0..n {
            data[i] = other.data[i];
        }
    }
}

impl<T: TensorType + Num + NumCast> Tensor<T> {
    /// Creates a zero-filled tensor of the specified shape.
    pub fn zeros(shape: &[usize]) -> Tensor<T> {
        Tensor::filled(shape, T::zero())
    }

    /// Creates a one-filled tensor of the specified shape.
    pub fn ones(shape: &[usize]) -> Tensor<T> {
        Tensor::filled(shape, T::one())
    }

    /// Creates an identify 2-D tensor (matrix). That is, all elements are zero except the diagonal
    /// which is filled with ones.
    pub fn eye(size: usize) -> Tensor<T> {
        let mut t = Tensor::zeros(&[size, size]);
        for k in 0..size {
            t.set2(k, k, T::one());
        }
        t
    }

    /// Swaps two axes. This returns a new Tensor, since the memory needs to be re-arranged.
    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> Tensor<T> {
        assert!(axis1 < self.ndim());
        assert!(axis2 < self.ndim());
        assert!(axis1 != axis2);

        let mut shape = self.shape.clone();
        let s = shape[axis1];
        shape[axis1] = shape[axis2];
        shape[axis2] = s;

        // TODO: This is slow and can be improved
        let mut t = Tensor::zeros(&shape);
        for i in 0..self.size() {
            let mut ii = self.unravel_index(i);
            let s = ii[axis1];
            ii[axis1] = ii[axis2];
            ii[axis2] = s;
            t[&ii] = self.data[i];
        }
        t
    }

    /// Transposes a matrix (for now, requires it to be 2D).
    pub fn transpose(&self) -> Tensor<T> {
        assert!(self.ndim() == 2, "Can only transpose a matrix (2D tensor)");
        self.swapaxes(0, 1)
    }

    /// Creates a new vector with integer values starting at 0 and counting up:
    /// 
    /// ```
    /// use numeric::DoubleTensor;
    ///
    /// let t = DoubleTensor::range(5); // [  0.00   1.00   2.00   3.00   4.00]
    /// ```
    pub fn range(size: usize) -> Tensor<T> {
        let mut data = Vec::with_capacity(size);
        let mut v = T::zero();
        for _ in 0..size {
            data.push(v);
            v = v + T::one();
        }
        Tensor{data: Rc::new(data), shape: vec![size]}
    }

    /// Creates a new vector between two values at constant increments. The number of elements is
    /// specified.
    pub fn linspace(start: T, stop: T, num: usize) -> Tensor<T> {
        let mut t = Tensor::empty(&[num]);
        let mut fi: T = T::zero();
        let d: T = (stop - start) / (cast::<usize, T>(num).unwrap() - T::one());
        for i in 0..num {
            t[i] = start + fi * d;
            fi = fi + T::one();
        }
        t
    }
}

impl<T: Numeric> Tensor<T> {
    /// Creates a scalar specified as a `f64` and internally casted to `T`
    pub fn fscalar(value: f64) -> Tensor<T> {
        Tensor{data: Rc::new(vec![cast(value).unwrap()]), shape: vec![]}
    }
}

fn shape_product(shape: &[usize]) -> usize {
    shape.iter().fold(1, |acc, &v| acc * v)
}

impl<T: Copy> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
        Tensor{data: self.data.clone(), shape: self.shape.clone()}
    }
}
