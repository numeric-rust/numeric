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
}
