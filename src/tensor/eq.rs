use tensor::Tensor;
use std::cmp::{PartialEq, Eq};
use TensorType;

impl<T: TensorType + PartialOrd> PartialEq<Tensor<T>> for Tensor<T> {
    fn eq(&self, rhs: &Tensor<T>) -> bool {
        self.shape == rhs.shape && self.data == rhs.data
    }
}

impl<T: TensorType + PartialOrd> Eq for Tensor<T> { }
