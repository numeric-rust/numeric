use tensor::Tensor;
use std::ops::Div;

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(mut self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.size() {
            self.data[i] /= rhs.data[i];
        }
        self
    }
}

impl<'a> Div<&'a Tensor> for Tensor {
    type Output = Tensor;
    fn div(mut self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.size() {
            self.data[i] /= rhs.data[i];
        }
        self
    }
}

// TODO: Change to separate lifetimes
impl<'a> Div<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        let mut t = self.clone();
        for i in 0..self.size() {
            t.data[i] /= rhs.data[i];
        }
        t
    }
}
