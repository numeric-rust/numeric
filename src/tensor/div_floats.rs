use std::ops::Div; use tensor::Tensor;

/*
macro_rules! add_impl {
    ($t:ty) => (
        impl Div<Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn div(mut self, rhs: Tensor<$t>) -> Tensor<$t> {
                assert_eq!(self.shape, rhs.shape);
                for i in 0..self.size() {
                    self.data[i] /= rhs.data[i];
                }
                self
            }
        }

        impl<'a> Div<&'a Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn div(mut self, rhs: &Tensor<$t>) -> Tensor<$t> {
                if rhs.is_scalar() {
                    let v = rhs[0];
                    for i in 0..self.size() {
                        self.data[i] /= v;
                    }
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    for i in 0..self.size() {
                        self.data[i] /= rhs.data[i];
                    }
                }
                self
            }
        }

        // TODO: Change to separate lifetimes?
        impl<'a> Div<&'a Tensor<$t>> for &'a Tensor<$t> {
            type Output = Tensor<$t>;
            fn div(self, rhs: &Tensor<$t>) -> Tensor<$t> {
                if rhs.is_scalar() {
                    let mut t: Tensor<$t> = self.clone();
                    let v = rhs[0];
                    for i in 0..self.size() {
                        t.data[i] /= v;
                    }
                    t
                } else if self.is_scalar() {
                    let mut t: Tensor<$t> = rhs.clone();
                    let v = self[0];
                    for i in 0..self.size() {
                        t.data[i] /= v;
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    let mut t = self.clone();
                    for i in 0..self.size() {
                        t.data[i] /= rhs.data[i];
                    }
                    t
                }
            }
        }

        // T / S
        impl Div<$t> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn div(mut self, rhs: $t) -> Tensor<$t> {
                for i in 0..self.size() {
                    self.data[i] /= rhs;
                }
                self
            }
        }
    )
}

add_impl!(f32);
add_impl!(f64);
*/
