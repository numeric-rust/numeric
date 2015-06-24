use tensor::Tensor;
use num::traits::Num;

impl<T: Copy + Num> Tensor<T> {
    pub fn concat(&self, rhs: &Tensor<T>, axis: usize) -> Tensor<T> {
        debug_assert!(axis < self.ndim());
        debug_assert!(self.ndim() == rhs.ndim());

        let mut shape = Vec::with_capacity(self.ndim());
        for i in 0..self.ndim() {
            if i != axis {
                if self.shape[i] != rhs.shape[i] {
                    panic!("When using concat, all axes must be the same except the joining one");
                }
                shape.push(self.shape[i]);
            } else {
                shape.push(self.shape[i] + rhs.shape[i]);
            }
        }

        let mut t = Tensor::zeros(&shape);
        for i in 0..self.size() {
            let ii = self.unravel_index(i);
            t[&ii] = self.data[i];
        }

        let offset = self.shape[axis];
        for i in 0..rhs.size() {
            let mut ii = rhs.unravel_index(i);
            ii[axis] += offset;
            t[&ii] = rhs.data[i];
        }
        t
    }

}
