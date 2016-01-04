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

    /// Memory offset.
    mem_offset: usize,

    /// Canonical C contiguous memory layout. This will likely be removed or changed in future
    /// updates.
    canonical: bool,
}

pub struct TensorIterator<T> {
    tensor: Tensor<T>,
    cur_index: Vec<usize>,
    cur_axis: usize,
    cur_pos: isize,
}

impl<T: TensorType> Iterator for TensorIterator<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        let dims = self.tensor.ndim();
        if dims == 0 {
            if self.cur_axis == 0 {
                self.cur_axis = 1;
                return Some(self.tensor.scalar_value());
            }
        } else {
            loop {
                if self.cur_index[self.cur_axis] == self.tensor.shape[self.cur_axis] {
                    if self.cur_axis == 0 {
                        break;
                    }
                    self.cur_pos -= (self.tensor.shape[self.cur_axis] as isize) *
                                     self.tensor.strides[self.cur_axis];
                    self.cur_index[self.cur_axis] = 0;
                    self.cur_axis -= 1;
                    self.cur_index[self.cur_axis] += 1;
                } else {
                    let x = self.tensor.data[self.cur_pos as usize];
                    self.cur_axis = dims - 1;
                    self.cur_index[self.cur_axis] += 1;
                    self.cur_pos += self.tensor.strides[self.cur_axis];
                    return Some(x);
                }
                self.cur_pos += self.tensor.strides[self.cur_axis];
            }
        }
        None
    }
}

// Common type-specific tensors

/// Type alias for `Tensor<f64>`
pub type DoubleTensor = Tensor<f64>;

/// Type alias for `Tensor<f32>`
pub type SingleTensor = Tensor<f32>;

/// Used for advanced slicing of a `Tensor`.
#[derive(Copy, Clone, Debug)]
pub enum AxisIndex {
    /// Indexes from start to end for this axis.
    Full,
    /// Indexes from start to end for all axes in the middle. A maximum of one can be used.
    Ellipsis,
    /// Creates a new axis of length 1 at this location.
    NewAxis,
    /// Picks one element of an axis. This will remove that axis from the tensor.
    Index(isize),
    /// Makes a strided slice `(start, end, step)`, with the same semantics as Python's Numpy. If
    /// `start` is specified as `None`, it will start from the first element if `step` is positive
    /// and last element if `step` is negative. If `end` is `None`, it will imply beyond the last
    /// element if `step` is positive and one before the first element if `step` is negative.
    StridedSlice(Option<isize>, Option<isize>, isize),
}

pub use AxisIndex::{Full, Ellipsis, NewAxis, Index, StridedSlice};

#[macro_use]
pub mod macros;

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

fn default_strides_old(shape: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(shape.len());

    let mut prod = 1;
    for i in (0..shape.len()).rev() {
        strides.insert(0, prod as isize);
        prod *= shape[i];
    }
    strides
}


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
        Tensor { data: Rc::new(data), shape: vec![len], strides: vec![1],
                 mem_offset: 0, canonical: true }
    }

    /// Creates a zero-filled tensor of the specified shape.
    pub fn empty(shape: &[usize]) -> Tensor<T> {
        //let data = Vector::with_capacity(
        let strides = default_strides_old(shape);
        let size = shape_product(shape);
        let sh = shape.to_vec();

        let mut data = Vec::with_capacity(size);
        // Fill with potentially random elements.
        // TODO: Possibly revise this (solution that doesn't need unsafe?)
        unsafe {
            data.set_len(size);
        }
        Tensor { data: Rc::new(data), shape: sh, strides: strides, mem_offset: 0, canonical: true }
    }

    /// Returns a flat slice of the tensor. Only works for canonical tensors.
    pub fn slice(&self) -> &[T] {
        assert!(self.canonical);
        &self.data[..]
    }

    /// Returns a mutable flat slice of the tensor. Only works for canonical tensors.
    /// Will make a copy of the underyling data if the tensor is not unique.
    pub fn slice_mut(&mut self) -> &mut [T] {
        assert!(self.canonical);
        &mut Rc::make_mut(&mut self.data)[..]
    }

    pub fn iter(&self) -> TensorIterator<T> {
        if self.ndim() == 0 {
            TensorIterator{
                tensor: self.clone(),
                cur_index: vec![],
                cur_axis: 0,
                cur_pos: self.mem_offset as isize,
            }
        } else {
            TensorIterator{
                tensor: self.clone(),
                cur_index: vec![0; self.ndim()],
                cur_axis: self.ndim() - 1,
                cur_pos: self.mem_offset as isize,
            }
        }
    }

    /// Creates a Tensor representing a scalar
    pub fn scalar(value: T) -> Tensor<T> {
        Tensor { data: Rc::new(vec![value]), shape: vec![],
                 strides: vec![], mem_offset: 0, canonical: true }
    }

    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0 && self.size() == 1
    }

    pub fn scalar_value(&self) -> T {
        debug_assert!(self.is_scalar(), "Tensor must be scalar to use scalar_value()");
        self.data[self.mem_offset]
    }

    /// Creates a new tensor of a given shape filled with the specified value.
    pub fn filled(shape: &[usize], v: T) -> Tensor<T> {
        let size = shape_product(shape);
        let strides = default_strides_old(shape);
        let sh = shape.to_vec();
        let data = vec![v; size];
        Tensor { data: Rc::new(data), shape: sh, strides: strides, mem_offset: 0, canonical: true }
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

    /// Flattens the tensor to one-dimensional.
    pub fn flatten(&self) -> Tensor<T> {
        let t = self.canonize();
        let s = t.size();
        Tensor { data: t.data, shape: vec![s], strides: vec![1], mem_offset: 0, canonical: true }
    }

    /// Make a dense copy of the tensor. This means it will have default strides and no memory
    /// offset.
    pub fn canonize(&self) -> Tensor<T> {
        if self.canonical {
            Tensor {
                data: self.data.clone(),
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                mem_offset: self.mem_offset,
                canonical: true,
            }
        } else {
            let s = self.shape.iter().fold(1, |acc, &item| acc * item);
            let mut v: Vec<T> = Vec::with_capacity(s);
            let def_strides = default_strides_old(&self.shape);
            let mut i = self.mem_offset as isize;
            let dims = self.shape.len();
            let mut cur_index: Vec<usize> = vec![0; dims];
            let mut cur_axis = dims - 1;
            loop {
                if cur_index[cur_axis] == self.shape[cur_axis] {
                    if cur_axis == 0 {
                        break;
                    }
                    i -= (self.shape[cur_axis] as isize) * self.strides[cur_axis];
                    cur_index[cur_axis] = 0;
                    cur_axis -= 1;
                    cur_index[cur_axis] += 1;
                } else {
                    let x = self.data[i as usize];
                    v.push(x);
                    cur_axis = dims - 1;
                    cur_index[cur_axis] += 1;
                }
                i += self.strides[cur_axis];
            }
            Tensor {
                data: Rc::new(v),
                shape: self.shape.clone(),
                strides: def_strides,
                mem_offset: 0,
                canonical: true,
            }
        }
    }

    pub fn canonize_inplace(&mut self) -> () {
        if !self.canonical {
            let t = self.canonize();
            self.data = t.data;
            self.shape = t.shape;
            self.strides = t.strides;
            self.mem_offset = t.mem_offset;
            self.canonical = true;
        }
    }

    /*
    fn default_strides(&self) -> Vec<isize> {
        let mut ss = vec![1; self.shape.len()];
        for k in 1..ss.len() {
            let i = ss.len() - 1 - k;
            ss[i] = ss[i + 1] * (self.shape[i + 1] as isize);
        }
        ss
    }
    */

    /// Returns the strides of tensor for each axis.
    /*
    pub fn strides(&self) -> Vec<isize> {
        /*
        let mut ss = vec![1; self.shape.len()];
        for k in 1..ss.len() {
            let i = ss.len() - 1 - k;
            ss[i] = ss[i + 1] * self.shape[i + 1];
        }
        ss
        */
        self.strides.clone()
    }
*/

    /// Returns number of elements in the underlying vector.
    #[inline]
    fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Returns number of elements in the tensor.
    #[inline]
    pub fn size(&self) -> usize {
        shape_product(&self.shape)
    }

    /// Returns the number of axes. This is the same as the length of the shape array.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /*
    fn resolve_axis(&self, axis: usize, index: isize) -> usize {
        if index < 0 {
            (self.shape[axis] as isize + index) as usize
        } else {
            index as usize
        }
    }
    */

    fn expand_indices(&self, selection: &[AxisIndex]) -> (Vec<AxisIndex>, Vec<isize>) {
        // The returned axis will not contain any AxisIndex::Ellipsis
        let mut sel: Vec<AxisIndex> = Vec::with_capacity(self.shape.len());
        let mut newaxes: Vec<isize> = Vec::with_capacity(self.shape.len());

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

    /// Takes slices (subsets) of tensors and returns a tensor as a new object. Uses the
    /// `AxisIndex` enum to specify indexing for each axis.
    ///
    /// ```
    /// use numeric::{DoubleTensor, Ellipsis, StridedSlice, Index, Full};
    ///
    /// let t = DoubleTensor::ones(&[2, 3, 4]);
    ///
    /// t.index(&[Ellipsis, StridedSlice(Some(1), Some(3), 1)]); // shape [2, 3, 2]
    /// t.index(&[Index(-1)]); // shape [3, 4]
    /// t.index(&[Full, StridedSlice(Some(1), None, 1), Index(1)]); // shape [2, 2]
    /// ```
    pub fn index(&self, selection: &[AxisIndex]) -> Tensor<T> {
        let (sel, mut newaxes) = self.expand_indices(selection);
        debug_assert!(sel.len() == self.ndim());
        debug_assert!(newaxes.len() == self.ndim() + 1);

        let dims = sel.len();

        let mut shape: Vec<usize> = Vec::with_capacity(dims);
        let mut strides: Vec<isize> = Vec::with_capacity(dims);
        let mut offsets: Vec<usize> = Vec::with_capacity(dims);

        //let mut ss = 1usize;
        let mut i = dims;
        for &s in sel.iter().rev() {
            let axis_size = self.shape[i - 1];
            let (offset, size, step): (usize, isize, isize) = match s {
                AxisIndex::Full => {
                    (0, axis_size as isize, 1)
                },
                AxisIndex::Index(idx) => {
                    newaxes[i - 1] -= 1;
                    if idx >= 0 {
                        (idx as usize, 1isize, 1)
                    } else if -idx as usize <= axis_size {
                        ((axis_size as isize + idx) as usize, 1isize, 1)
                    } else {
                        // Out of index
                        panic!("Out of index");
                    }
                },
                AxisIndex::StridedSlice(maybe_start, maybe_end, step) => {
                    let st = match maybe_start {
                        Some(v) => if v >= 0 { v
                        } else {
                            axis_size as isize + v
                        },
                        None => if step >= 0 {
                            0
                        } else {
                            axis_size as isize - 1
                        },
                    };

                    let en = match maybe_end {
                        Some(v) => if v >= 0 {
                            v
                        } else {
                            axis_size as isize + v
                        },
                        None => if step >= 0 {
                            axis_size as isize
                        } else {
                            -1
                        },
                    };

                    (st as usize, ((en - st).abs() + step.abs() - 1) / step.abs(), step)
                },
                AxisIndex::Ellipsis | AxisIndex::NewAxis => {
                    // Should have been removed by expand_indices at this point
                    unreachable!();
                },
            };

            shape.insert(0, size as usize);
            strides.insert(0, self.strides[i-1] * step);
            offsets.insert(0, offset);
            i -= 1;
        }

        let mut mem_offset = self.mem_offset as isize;
        for j in 0..dims {
            mem_offset += self.strides[j] * offsets[j] as isize;
        }

        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        for i in 0..shape.len() {
            if newaxes[i] >= 0 {
                for _ in 0..newaxes[i] {
                    new_shape.push(1);
                    new_strides.push(strides[i]);
                }
                new_shape.push(shape[i]);
                new_strides.push(strides[i]);
            }
        }
        for _ in 0..newaxes[newaxes.len() - 1] {
            new_shape.push(1);
            new_strides.push(1);
        }

        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            mem_offset: mem_offset as usize,
            canonical: false,
        }
    }

    /// Similar to `index`, except this updates the tensor with `other` instead of returning them.
    pub fn index_set(&mut self, selection: &[AxisIndex], other: &Tensor<T>) {
        // TODO: This is a quick and dirty way and can be made much faster
        let indices: Tensor<usize> = Tensor::range(self.size()).reshape_proper(&self.shape[..]);
        let sub = indices.index(selection);
        assert!(sub.shape() == other.shape());
        assert!(sub.size() == other.size());

        self.canonize_inplace();
        let mut data = self.slice_mut();
        for (i, v) in sub.iter().zip(other.iter()) {
            data[i] = v;
        }
    }

    pub fn bool_index(&self, indices: &Tensor<bool>) -> Tensor<T> {
        let mut s = 0;
        for v in indices.iter() {
            if v {
                s += 1;
            }
        }
        let mut t = Tensor::empty(&[s]);
        let mut j = 0;
        {
            let mut data = t.slice_mut();
            for (idx, v) in indices.iter().zip(self.iter()) {
                if idx {
                    data[j] = v;
                    j += 1;
                }
            }
        }
        t
    }

    pub fn bool_index_set(&mut self, indices: &Tensor<bool>, values: &Tensor<T>) {
        self.canonize_inplace();
        let mut s = 0;
        for v in indices.iter() {
            if v {
                s += 1;
            }
        }
        if values.is_scalar() {
            let v = values.scalar_value();
            let mut data = self.slice_mut();
            let mut j = 0;
            for idx in indices.iter() {
                if idx {
                    data[j] = v;
                    j += 1;
                }
            }
        } else {
            assert!(values.size() == s);
            let mut data = self.slice_mut();
            let mut value_iter = values.iter();
            for (i, idx) in indices.iter().enumerate() {
                if idx {
                    data[i] = value_iter.next().unwrap();
                }
            }
        }
    }

    /// Takes a flatten index (if in row-major order) and returns a vector of the per-axis indices.
    pub fn unravel_index(&self, index: usize) -> Vec<usize> {
        // Can only be used if tensor is canonical
        assert!(self.canonical);

        let mut ii: Vec<usize> = Vec::with_capacity(self.ndim());
        let mut c = index;
        for i in 0..self.ndim() {
            ii.push(c / self.strides[i] as usize);
            c %= self.strides[i] as usize;
        }
        ii
    }

    /// Takes an array of per-axis indices and returns a flattened index (in row-major order).
    pub fn ravel_index(&self, ii: &[usize]) -> usize {
        assert_eq!(ii.len(), self.ndim());
        let mut index = 0;
        for i in 0..self.ndim() {
            index += self.strides[i] * ii[i] as isize;
        }
        index as usize
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
        // TODO: Are there cases where we do not need to canonize?
        //self.canonize_inplace();
        let t = self.canonize();

        let s = proper_shape.iter().fold(1, |acc, &item| acc * item);
        assert_eq!(t.size(), s);
        let strides = default_strides_old(&proper_shape);
        Tensor { data: t.data, shape: proper_shape.to_vec(),
               strides: strides, mem_offset: t.mem_offset, canonical: t.canonical}
    }

    /// Reshapes the data. This moves the data, so no memory is allocate.
    pub fn reshape(self, shape: &[isize]) -> Tensor<T> {
        let proper_shape = self.convert_shape(shape);
        self.reshape_proper(&proper_shape)
    }

    #[inline]
    fn get2(&self, i: usize, j: usize) -> T {
        self.data[self.mem_offset + (i as isize * self.strides[0] +
                                     j as isize * self.strides[1]) as usize]
    }

    #[inline]
    fn set2(&mut self, i: usize, j: usize, v: T) {
        self.canonize_inplace();
        let i = self.mem_offset + (i as isize * self.strides[0] +
                                   j as isize * self.strides[1]) as usize;
        let mut data = self.slice_mut();
        data[i] = v;
    }

    /// Sets all the values according to another tensor.
    pub fn set(&mut self, other: &Tensor<T>) -> () {
        let data_size = self.data_size();
        if data_size != other.size() {
            unsafe {
                Rc::make_mut(&mut self.data).set_len(other.size());
            }
        }

        let mut data = self.slice_mut();
        for (i, v) in other.iter().enumerate() {
            data[i] = v;
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

        let mut t = self.clone();
        let tmp = t.strides[axis1];
        t.strides[axis1] = t.strides[axis2];
        t.strides[axis2] = tmp;

        let tmp2 = t.shape[axis1];
        t.shape[axis1] = t.shape[axis2];
        t.shape[axis2] = tmp2;

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
        Tensor {
            data: Rc::new(data),
            shape: vec![size],
            strides: vec![1],
            mem_offset: 0,
            canonical: true,
        }
    }

    /// Creates a new vector between two values at constant increments. The number of elements is
    /// specified.
    pub fn linspace(start: T, stop: T, num: usize) -> Tensor<T> {
        let mut t = Tensor::empty(&[num]);
        let mut fi: T = T::zero();
        let d: T = (stop - start) / (cast::<usize, T>(num).unwrap() - T::one());
        {
            let mut data = t.slice_mut();
            for i in 0..num {
                data[i] = start + fi * d;
                fi = fi + T::one();
            }
        }
        t
    }
}

impl<T: Numeric> Tensor<T> {
    /// Creates a scalar specified as a `f64` and internally casted to `T`
    pub fn fscalar(value: f64) -> Tensor<T> {
        Tensor {
            data: Rc::new(vec![cast(value).unwrap()]),
            shape: vec![],
            strides: vec![],
            mem_offset: 0,
            canonical: true,
        }
    }
}

fn shape_product(shape: &[usize]) -> usize {
   shape.iter().fold(1, |acc, &v| acc * v)
}

impl<T: Copy> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            mem_offset: self.mem_offset,
            canonical: self.canonical,
        }
    }
}

#[test]
fn test_strided_slice1() {
    let t0 = Tensor::range(40).reshape(&[5, 4, 2]);
    let t1 = t0.index(&[StridedSlice(Some(0), Some(5), 2),
                        StridedSlice(Some(0), Some(4), 2),
                        StridedSlice(Some(0), Some(2), 2)]);
    let ta = Tensor::new(vec![0, 4, 16, 20, 32, 36]).reshape(&[3, 2, 1]);
    assert!(t1 == ta);
}

#[test]
fn test_strided_slice2() {
    let t0 = Tensor::range(5 * 7 * 10).reshape(&[5, 7, 10]);
    let t1 = t0.index(&[StridedSlice(Some(1), Some(5), 2),
                        StridedSlice(Some(2), Some(5), 3),
                        StridedSlice(Some(4), Some(10), 4)]);
    let ta = Tensor::new(vec![94, 98, 234, 238]).reshape(&[2, 1, 2]);
    assert!(t1 == ta);
}

#[test]
fn test_strided_slice3() {
    let t0 = Tensor::range(40).reshape(&[5, 4, 2]);
    let t1 = t0.index(&[StridedSlice(Some(4), Some(0), -2),
                        StridedSlice(Some(3), Some(0), -1),
                        StridedSlice(Some(0), Some(2), 2)]);
    let ta = Tensor::new(vec![38, 36, 34, 22, 20, 18]).reshape(&[2, 3, 1]);
    assert!(t1 == ta);
}

#[test]
fn test_strided_slice4() {
    let t0 = Tensor::range(5 * 7 * 10).reshape(&[5, 7, 10]);
    let t1 = t0.index(&[StridedSlice(Some(0), Some(2), 1),
                        StridedSlice(Some(2), Some(5), 3),
                        StridedSlice(Some(4), Some(10), 4)]);
    let ta = Tensor::new(vec![24, 28, 94, 98]).reshape(&[2, 1, 2]);
    assert!(t1 == ta);
}

#[test]
fn test_strided_slice5() {
    let t0 = Tensor::range(5 * 7 * 10).reshape(&[5, 7, 10]);
    let t1 = t0.index(&[StridedSlice(Some(4), Some(1), -1),
                        StridedSlice(Some(2), Some(5), 3),
                        StridedSlice(Some(4), Some(10), 4)]);
    let ta = Tensor::new(vec![304, 308, 234, 238, 164, 168]).reshape(&[3, 1, 2]);
    assert!(t1 == ta);
}

#[test]
fn test_strided_slice6() {
    let t0 = Tensor::range(5 * 7 * 10).reshape(&[5, 7, 10]);
    let t1 = t0.index(&[StridedSlice(Some(4), Some(1), -1),
                        StridedSlice(Some(5), Some(1), -3),
                        StridedSlice(Some(9), Some(0), -4)]);
    let ta = Tensor::new(vec![339, 335, 331, 309, 305, 301, 269, 265, 261,
                              239, 235, 231, 199, 195, 191, 169, 165, 161]).reshape(&[3, 2, 3]);
    assert!(t1 == ta);
}

#[test]
fn test_strided_slice7() {
    // Twice indexed
    let t0 = Tensor::range(15 * 21 * 23).reshape(&[15, 21, 23]);
    let t1 = t0.index(&[StridedSlice(Some(5), Some(15), 2),
                        StridedSlice(Some(19), Some(5), -1),
                        StridedSlice(Some(20), Some(10), -1)]);
    let t2 = t1.index(&[StridedSlice(Some(4), Some(0), -1),
                        StridedSlice(Some(0), Some(10), 3),
                        StridedSlice(Some(5), Some(0), -3)]);
    let ta = Tensor::new(vec![6731, 6734, 6662, 6665, 6593, 6596,
                              6524, 6527, 5765, 5768, 5696, 5699,
                              5627, 5630, 5558, 5561, 4799, 4802,
                              4730, 4733, 4661, 4664, 4592, 4595,
                              3833, 3836, 3764, 3767, 3695, 3698,
                              3626, 3629]).reshape(&[4, 4, 2]);
    assert!(t2 == ta);
}

#[test]
fn test_strided_slice8() {
    let t0 = Tensor::range(40).reshape(&[5, 4, 2]);
    let t1 = t0.index(&[StridedSlice(Some(3), None, -2),
                        StridedSlice(None, None, -1),
                        StridedSlice(None, None, 2)]);
    let ta = Tensor::new(vec![30, 28, 26, 24, 14, 12, 10,  8]).reshape(&[2, 4, 1]);
    assert!(t1 == ta);
}

#[test]
fn test_strided_slice9() {
    // Twice indexed
    let t0 = Tensor::range(15 * 21 * 23).reshape(&[15, 21, 23]);
    let t1 = t0.index(&[StridedSlice(Some(-10), None, 2),
                        StridedSlice(Some(-2), Some(-16), -1),
                        StridedSlice(Some(20), Some(-13), -1)]);
    let t2 = t1.index(&[StridedSlice(Some(4), Some(0), -1),
                        StridedSlice(Some(0), Some(10), 3),
                        StridedSlice(Some(5), Some(0), -3)]);
    let ta = Tensor::new(vec![6731, 6734, 6662, 6665, 6593, 6596,
                              6524, 6527, 5765, 5768, 5696, 5699,
                              5627, 5630, 5558, 5561, 4799, 4802,
                              4730, 4733, 4661, 4664, 4592, 4595,
                              3833, 3836, 3764, 3767, 3695, 3698,
                              3626, 3629]).reshape(&[4, 4, 2]);
    assert!(t2 == ta);
}


#[test]
fn test_single_index1() {
    let t0 = Tensor::range(40).reshape(&[5, 4, 2]);
    let t1 = t0.index(&[StridedSlice(Some(3), None, -2),
                        Index(1),
                        StridedSlice(None, None, 2)]);
    let ta = Tensor::new(vec![26, 10]).reshape(&[2, 1]);
    assert!(t1 == ta);
}

#[test]
fn test_single_index2() {
    let t0 = Tensor::range(120).reshape(&[5, 4, 2, 3]);
    let t1 = t0.index(&[Index(3),
                        StridedSlice(Some(3), None, -2),
                        Index(1),
                        StridedSlice(None, None, 2)]);
    let ta = Tensor::new(vec![93, 95, 81, 83]).reshape(&[2, 2]);
    assert!(t1 == ta);
}

#[test]
fn test_single_index3() {
    let t0 = Tensor::range(120).reshape(&[5, 4, 2, 3]);
    let t1 = t0.index(&[Index(3), Index(2)]);
    let ta = Tensor::new(vec![84, 85, 86, 87, 88, 89]).reshape(&[2, 3]);
    assert!(t1 == ta);
}

#[test]
fn test_single_index4() {
    let t0 = Tensor::range(120).reshape(&[5, 4, 2, 3]);
    let t1 = t0.index(&[Index(3), Full, Index(0)]);
    let ta = Tensor::new(vec![72, 73, 74, 78, 79, 80, 84, 85, 86, 90, 91, 92]).reshape(&[4, 3]);
    assert!(t1 == ta);
}

#[test]
fn test_single_index5() {
    let t0 = Tensor::range(120).reshape(&[5, 4, 2, 3]);
    let t1 = t0.index(&[Index(1), Ellipsis, Index(-1)]);
    let ta = Tensor::new(vec![26, 29, 32, 35, 38, 41, 44, 47]).reshape(&[4, 2]);
    assert!(t1 == ta);
}


#[test]
fn test_new_axis1() {
    let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
    let t1 = t0.index(&[NewAxis]);
    assert!(t1.flatten() == t0.flatten());
    assert!(t1.shape == [1, 5, 4, 2]);
}

#[test]
fn test_new_axis2() {
    let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
    let t1 = t0.index(&[Full, NewAxis]);
    assert!(t1.flatten() == t0.flatten());
    assert!(t1.shape == [5, 1, 4, 2]);
}

#[test]
fn test_new_axis3() {
    let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
    let t1 = t0.index(&[Full, NewAxis, NewAxis, Full, NewAxis]);
    assert!(t1.flatten() == t0.flatten());
    assert!(t1.shape == [5, 1, 1, 4, 1, 2]);
}

#[test]
fn test_new_axis4() {
    let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
    let t1 = t0.index(&[Ellipsis, NewAxis]);
    assert!(t1.flatten() == t0.flatten());
    assert!(t1.shape == [5, 4, 2, 1]);
}

#[test]
fn test_new_axis5() {
    let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
    let t1 = t0.index(&[NewAxis, Ellipsis, NewAxis, Full]);
    assert!(t1.flatten() == t0.flatten());
    assert!(t1.shape == [1, 5, 4, 1, 2]);
}
