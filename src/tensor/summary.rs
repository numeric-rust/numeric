use std::ops::{Add, Mul, BitAnd, BitOr, BitXor};
use tensor::{Tensor, Full, Index};
use Numeric;
use TensorType;

impl<T: Numeric> Tensor<T> {
    pub fn max(&self) -> T {
        assert!(self.size() > 0, "Can't take max of empty tensor");
        let mut m = T::zero();
        for (i, v) in self.iter().enumerate() {
            if i == 0 {
                m = v;
            } else if v > m {
                m = v;
            }
        }
        m
    }

    pub fn min(&self) -> T {
        assert!(self.size() > 0, "Can't take min of empty tensor");
        let mut m = T::zero();
        for (i, v) in self.iter().enumerate() {
            if i == 0 {
                m = v;
            } else if v < m {
                m = v;
            }
        }
        m
    }

    pub fn sum(&self) -> T {
        let mut s = T::zero();
        for v in self.iter() {
            s = s + v;
        }
        s
    }

    pub fn mean(&self) -> T {
        let mut s = T::zero();
        let mut t = T::zero();
        for v in self.iter() {
            s = s + v;
            t = t + T::one();
        }
        s / t
    }
}

macro_rules! add_impl {
    ($trait_name:ident, $func_name:ident, $new_func_name:ident) => (
        impl<T: TensorType + $trait_name<Output=T>> Tensor<T> {
            pub fn $new_func_name(&self, axis: usize) -> Tensor<T> {
                assert!(axis < self.ndim(), "Reduced axis must exist");
                let mut sel = vec![Full; axis];
                sel.push(Index(0));

                let mut t = self.index(&sel[..]);
                let d = self.dim(axis);
                for i in 1..d {
                    sel[axis] = Index(i as isize);
                    t = (&t).$func_name(&self.index(&sel[..]));
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
