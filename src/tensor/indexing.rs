#[allow(unused_imports)]
use tensor::Tensor;
use std::ops::{Index, IndexMut};
use TensorType;

// Vector indexing
impl<'b, T: TensorType> Index<&'b [usize]> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, ii: &'b [usize]) -> &'a T {
        let index = self.ravel_index(ii);
        &self.data[index]
    }
}

impl<'b, T: TensorType> IndexMut<&'b [usize]> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, ii: &'b [usize]) -> &'a mut T {
        let index = self.ravel_index(ii);
        &mut self.slice_mut()[index]
    }
}

impl<'b, T: TensorType> Index<&'b Vec<usize>> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, ii: &'b Vec<usize>) -> &'a T {
        let index = self.ravel_index(&ii[..]);
        &self.data[index]
    }
}

impl<'b, T: TensorType> IndexMut<&'b Vec<usize>> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, ii: &'b Vec<usize>) -> &'a mut T {
        let index = self.ravel_index(&ii[..]);
        &mut self.slice_mut()[index]
    }
}

// Flattened indexing (this will index it as one-dimensional)
impl<T: TensorType> Index<usize> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: usize) -> &'a T {
        &self.data[_index]
    }
}

impl<T: TensorType> IndexMut<usize> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: usize) -> &'a mut T {
        &mut self.slice_mut()[_index]
    }
}

// 1-D indexing
impl<T: TensorType> Index<(usize,)> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: (usize,)) -> &'a T {
        assert!(self.ndim() == 1);
        &self.data[_index.0]
    }
}

impl<T: TensorType> IndexMut<(usize,)> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: (usize,)) -> &'a mut T {
        assert!(self.ndim() == 1);
        &mut self.slice_mut()[_index.0]
    }
}

// 2-D indexing
impl<T: TensorType> Index<(usize, usize)> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: (usize, usize)) -> &'a T {
        assert!(self.ndim() == 2);
        &self.data[(_index.0 * self.shape[1] + _index.1)]
    }
}

impl<T: TensorType> IndexMut<(usize, usize)> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: (usize, usize)) -> &'a mut T {
        assert!(self.ndim() == 2);
        let i = _index.0 * self.shape[1] + _index.1;
        &mut self.slice_mut()[i]
    }
}

// 3-D indexing
impl<T: TensorType> Index<(usize, usize, usize)> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: (usize, usize, usize)) -> &'a T {
        assert!(self.ndim() == 3);
        &self.data[(_index.0 * self.shape[1] * self.shape[2] +
                    _index.1 * self.shape[2] +
                    _index.2)]
    }
}

impl<T: TensorType> IndexMut<(usize, usize, usize)> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: (usize, usize, usize)) -> &'a mut T {
        assert!(self.ndim() == 3);
        let (sh1, sh2) = (self.shape[1], self.shape[2]);
        &mut self.slice_mut()[(_index.0 * sh1 * sh2 +
                               _index.1 * sh2 + _index.2)]
    }
}
