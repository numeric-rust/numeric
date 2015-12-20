use std::ops::{Add, Mul, BitAnd, BitOr, BitXor};
use tensor::Tensor;
use Numeric;

impl<T: Numeric> Tensor<T> {
    pub fn max(&self) -> T {
        debug_assert!(self.size() > 0, "Can't take max of empty tensor");
        let mut m = self.data[0];
        for i in 1..self.size() {
            if self.data[i] > m {
                m = self.data[i];
            }
        }
        m
    }

    pub fn min(&self) -> T {
        debug_assert!(self.size() > 0, "Can't take min of empty tensor");
        let mut m = self.data[0];
        for i in 1..self.size() {
            if self.data[i] < m {
                m = self.data[i];
            }
        }
        m
    }

    pub fn sum(&self) -> T {
        let mut s = T::zero();
        for i in 0..self.size() {
            s = s + self.data[i];
        }
        s
    }

    pub fn mean(&self) -> T {
        let mut s = T::zero();
        let mut t = T::zero();
        for i in 0..self.size() {
            s = s + self.data[i];
            t = t + T::one();
        }
        s / t
    }
}

macro_rules! add_impl {
    ($trait_name:ident, $func_name:ident, $new_func_name:ident) => (
        impl<T: Copy + $trait_name<Output=T>> Tensor<T> {
            pub fn $new_func_name(&self, axis: usize) -> Tensor<T> {
                assert!(axis < self.ndim(), "Reduced axis must exist");

                let mut sh = Vec::with_capacity(self.ndim() - 1);
                for i in 0..self.ndim() {
                    if i != axis {
                        sh.push(self.shape[i]);
                    }
                }

                let mut t = Tensor::empty(&sh);

                let strides = self.strides();
                let stride = strides[axis];
                let axis_size = self.shape[axis];
                let t_size = t.size();
                for start in 0..t_size {
                    let mut index = t.unravel_index(start);
                    index.insert(axis, 0);
                    let orig_index = self.ravel_index(&index[..]);
                    t.data[start] = self.data[orig_index];
                    for k in 1..axis_size {
                        t.data[start] = t.data[start].$func_name(self.data[orig_index + k * stride]);
                    }
                }
                t
            }
        }
    )
}

add_impl!(Add, add, sum_axis);
add_impl!(Mul, mul, prod_axis);

add_impl!(BitAnd, bitand, bitand_axis);
add_impl!(BitOr, bitor, bitor_axis);
add_impl!(BitXor, bitxor, bitxor_axis);
