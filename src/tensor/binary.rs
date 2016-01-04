use tensor::Tensor;
use TensorType;

macro_rules! add_impl {
    ($new_fname:ident, $fname:ident) => (
        /// Element-wise comparison.
        impl<T: TensorType + PartialOrd> Tensor<T> {
            pub fn $new_fname(&self, rhs: &Tensor<T>) -> Tensor<bool> {
                let mut y = Tensor::empty(&self.shape());
                {
                    let mut data = y.slice_mut();
                    if rhs.is_scalar() {
                        let v2 = rhs.scalar_value();
                        for (i, v1) in rhs.iter().enumerate() {
                            data[i] = v1.$fname(&v2);
                        }
                    } else {
                        assert_eq!(self.shape(), rhs.shape());
                        for (i, (v1, v2)) in self.iter().zip(rhs.iter()).enumerate() {
                            data[i] = v1.$fname(&v2);
                        }
                    }
                }
                y
            }
        }
    )
}

add_impl!(elem_gt, gt);
add_impl!(elem_ge, ge);
add_impl!(elem_lt, lt);
add_impl!(elem_le, le);
add_impl!(elem_eq, eq);
add_impl!(elem_ne, ne);

impl Tensor<bool> {
    pub fn all(&self) -> bool {
        for v in self.iter() {
            if !v {
                return false;
            }
        }
        true
    }

    pub fn any(&self) -> bool {
        for v in self.iter() {
            if v {
                return true;
            }
        }
        false
    }
}
