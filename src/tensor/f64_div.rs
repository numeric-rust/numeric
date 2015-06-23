use std::ops::Div;
use tensor::Tensor;

impl Div<Tensor<f64>> for Tensor<f64> {
    type Output = Tensor<f64>;
    fn div(mut self, rhs: Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.size() {
            self.data[i] /= rhs.data[i];
        }
        self
    }
}

impl<'a> Div<&'a Tensor<f64>> for Tensor<f64> {
    type Output = Tensor<f64>;
    fn div(mut self, rhs: &Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.size() {
            self.data[i] /= rhs.data[i];
        }
        self
    }
}

// TODO: Change to separate lifetimes?
impl<'a> Div<&'a Tensor<f64>> for &'a Tensor<f64> {
    type Output = Tensor<f64>;
    fn div(self, rhs: &Tensor<f64>) -> Tensor<f64> {
        assert_eq!(self.shape, rhs.shape);
        let mut t = self.clone();
        for i in 0..self.size() {
            t.data[i] /= rhs.data[i];
        }
        t
    }
}

// T / S
impl Div<f64> for Tensor<f64> {
    type Output = Tensor<f64>;
    fn div(mut self, rhs: f64) -> Tensor<f64> {
        for i in 0..self.size() {
            self.data[i] /= rhs;
        }
        self
    }
}
