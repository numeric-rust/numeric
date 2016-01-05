use std::ops::{Index, IndexMut};
use tensor::Tensor;
use traits::TensorTrait;

// Vector indexing
impl<'b, T: TensorTrait> Index<&'b [usize]> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, ii: &'b [usize]) -> &'a T {
        assert!(self.canonical);
        let index = self.ravel_index(ii);
        &self.data[index]
    }
}

impl<'b, T: TensorTrait> IndexMut<&'b [usize]> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, ii: &'b [usize]) -> &'a mut T {
        assert!(self.canonical);
        let index = self.ravel_index(ii);
        &mut self.slice_mut()[index]
    }
}

impl<'b, T: TensorTrait> Index<&'b Vec<usize>> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, ii: &'b Vec<usize>) -> &'a T {
        assert!(self.canonical);
        let index = self.ravel_index(&ii[..]);
        &self.data[index]
    }
}

impl<'b, T: TensorTrait> IndexMut<&'b Vec<usize>> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, ii: &'b Vec<usize>) -> &'a mut T {
        assert!(self.canonical);
        //self.canonize_inplace();
        let index = self.ravel_index(&ii[..]);
        &mut self.slice_mut()[index]
    }
}

/*
// Flattened indexing (this will index it as one-dimensional)
impl<T: TensorTrait> Index<usize> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: usize) -> &'a T {
        &self.data[self.mem_offset + _index]
    }
}

impl<T: TensorTrait> IndexMut<usize> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: usize) -> &'a mut T {
        //self.harden_inplace();
        let offset = self.mem_offset;
        &mut self.slice_mut()[offset + _index]
    }
}
*/

// 1-D indexing
impl<T: TensorTrait> Index<(usize,)> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: (usize,)) -> &'a T {
        assert!(self.ndim() == 1);
        &self.data[(self.mem_offset as isize + _index.0 as isize * self.strides[0]) as usize]
    }
}

impl<T: TensorTrait> IndexMut<(usize,)> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: (usize,)) -> &'a mut T {
        assert!(self.ndim() == 1);
        let offset = self.mem_offset as isize;
        let s0 = self.strides[0];
        &mut self.slice_mut()[(offset + _index.0 as isize * s0) as usize]
    }
}

// 2-D indexing
impl<T: TensorTrait> Index<(usize, usize)> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: (usize, usize)) -> &'a T {
        assert!(self.ndim() == 2);
        //&self.data[(_index.1 * self.shape[1] + _index.1)]
        &self.data[self.mem_offset + (_index.0 as isize * self.strides[0] + _index.1 as isize * self.strides[1]) as usize]
    }
}
impl<T: TensorTrait> IndexMut<(usize, usize)> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: (usize, usize)) -> &'a mut T {
        assert!(self.ndim() == 2);
        //self.harden_inplace();
        let i = self.mem_offset + (_index.0 as isize * self.strides[0] + _index.1 as isize * self.strides[1]) as usize;
        &mut self.slice_mut()[i]
    }
}

// 3-D indexing
impl<T: TensorTrait> Index<(usize, usize, usize)> for Tensor<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: (usize, usize, usize)) -> &'a T {
        assert!(self.ndim() == 3);
        &self.data[(self.mem_offset as isize +
                    _index.0 as isize * self.strides[0] +
                    _index.1 as isize * self.strides[1] +
                    _index.2 as isize * self.strides[2]) as usize]
    }
}

impl<T: TensorTrait> IndexMut<(usize, usize, usize)> for Tensor<T> {
    fn index_mut<'a>(&'a mut self, _index: (usize, usize, usize)) -> &'a mut T {
        assert!(self.ndim() == 3);
        let offset = self.mem_offset as isize;
        let (s0, s1, s2) = (self.strides[1], self.strides[2], self.strides[3]);
        &mut self.slice_mut()[(offset +
                               _index.0 as isize * s0 +
                               _index.1 as isize * s1 +
                               _index.2 as isize * s2) as usize]
    }
}

