use tensor::Tensor;
use std::ops::Mul;

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(mut self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.size() {
            self.data[i] *= rhs.data[i];
        }
        self
    }
}

impl<'a> Mul<&'a Tensor> for Tensor {
    type Output = Tensor;
    fn mul(mut self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.size() {
            self.data[i] *= rhs.data[i];
        }
        self
    }
}

// TODO: Change to separate lifetimes
impl<'a> Mul<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        let mut t = self.clone();
        for i in 0..self.size() {
            t.data[i] *= rhs.data[i];
        }
        t
    }
}

// T * S
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(mut self, rhs: f64) -> Tensor {
        for i in 0..self.size() {
            self.data[i] *= rhs;
        }
        self
    }
}
