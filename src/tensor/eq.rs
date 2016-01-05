use std::cmp::{PartialEq, Eq};
use tensor::Tensor;
use traits::TensorTrait;

impl<T: TensorTrait + PartialOrd> PartialEq<Tensor<T>> for Tensor<T> {
    fn eq(&self, rhs: &Tensor<T>) -> bool {
        // Iterators are slow, but it should be faster to iterate
        // without iterators than to canonize, right?

        /*
        if self.shape != rhs.shape {
            return false;
        }

        let mut ok = true;
        for (v1, v2) in self.iter().zip(rhs.iter()) {
            ok = ok && v1 == v2;
        }
        ok
        */
        let t0 = self.canonize();
        let t1 = rhs.canonize();

        t0.shape == t1.shape && t0.data == t1.data
    }
}

impl<T: TensorTrait + PartialOrd> Eq for Tensor<T> { }


