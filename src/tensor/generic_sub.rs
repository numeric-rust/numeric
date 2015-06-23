use tensor::Tensor;
use std::ops::Sub;

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(mut self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.size() {
            self.data[i] -= rhs.data[i];
        }
        self
    }
}

impl<'a> Sub<&'a Tensor> for Tensor {
    type Output = Tensor;
    fn sub(mut self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.size() {
            self.data[i] -= rhs.data[i];
        }
        self
    }
}

impl<'a> Sub for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape);
        let mut t = self.clone();
        for i in 0..self.size() {
            t.data[i] -= rhs.data[i];
        }
        t
    }
}

impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(mut self, rhs: f64) -> Tensor {
        for i in 0..self.size() {
            self.data[i] -= rhs;
        }
        self
    }
}

