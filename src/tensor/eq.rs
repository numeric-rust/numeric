use tensor::Tensor;
use std::cmp::{PartialEq, Eq};
use num::traits::Num;

impl<T: Copy + Num> PartialEq<Tensor<T>> for Tensor<T> {
    fn eq(&self, rhs: &Tensor<T>) -> bool {
        self.shape == rhs.shape && self.data == rhs.data
    }
}

impl<T: Copy + Num> Eq for Tensor<T> { }
