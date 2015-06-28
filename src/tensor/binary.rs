use tensor::Tensor;
use TensorType;

macro_rules! add_impl {
    ($new_fname:ident, $fname:ident) => (
        /// Element-wise comparison.
        impl<T: TensorType + PartialOrd> Tensor<T> {
            pub fn $new_fname(&self, rhs: &Tensor<T>) -> Tensor<bool> {
                let mut y = Tensor::empty(&self.shape());
                if rhs.is_scalar() {
                    let v = rhs.scalar_value();
                    for i in 0..self.size() {
                        y[i] = self.data[i].$fname(&v);
                    }
                } else {
                    assert_eq!(self.shape(), rhs.shape());
                    for i in 0..self.size() {
                        y[i] = self.data[i].$fname(&rhs.data[i]);
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
        for i in 0..self.size() {
            if !self.data[i] {
                return false;
            }
        }
        true
    }

    pub fn any(&self) -> bool {
        for i in 0..self.size() {
            if self.data[i] {
                return true;
            }
        }
        false
    }
}
