use tensor::Tensor;
use num::traits::Num;
use std::ops::{Index, IndexMut};

// Flattened indexing (this will index it as one-dimensional)
impl<T> Index<usize> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: usize) -> &'a T {
        &self.data[_index as usize]
    }
}

impl<T> IndexMut<usize> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: usize) -> &'a mut T {
        &mut self.data[_index as usize]
    }
}

// 1-D indexing
impl<T: Copy + Num> Index<(usize,)> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: (usize,)) -> &'a T {
        assert!(self.ndim() == 1);
        &self.data[_index.0]
    }
}

impl<T: Copy + Num> IndexMut<(usize,)> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: (usize,)) -> &'a mut T {
        assert!(self.ndim() == 1);
        &mut self.data[_index.0]
    }
}

// 2-D indexing
impl<T: Copy + Num> Index<(usize, usize)> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: (usize, usize)) -> &'a T {
        assert!(self.ndim() == 2);
        &self.data[(_index.0 * self.shape[1] + _index.1)]
    }
}

impl<T: Copy + Num> IndexMut<(usize, usize)> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: (usize, usize)) -> &'a mut T {
        assert!(self.ndim() == 2);
        &mut self.data[(_index.0 * self.shape[1] + _index.1)]
    }
}

// 3-D indexing
impl<T: Copy + Num> Index<(usize, usize, usize)> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: (usize, usize, usize)) -> &'a T {
        assert!(self.ndim() == 3);
        &self.data[(_index.0 * self.shape[1] * self.shape[2] +
                    _index.1 * self.shape[2] +
                    _index.2)]
    }
}

impl<T: Copy + Num> IndexMut<(usize, usize, usize)> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: (usize, usize, usize)) -> &'a mut T {
        assert!(self.ndim() == 3);
        &mut self.data[(_index.0 * self.shape[1] * self.shape[2] +
                        _index.1 * self.shape[2] +
                        _index.2)]
    }
}
