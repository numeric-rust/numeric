use tensor::Tensor;
use TensorType;

impl<T: TensorType> Tensor<T> {
    pub fn concat(lhs: &Tensor<T>, rhs: &Tensor<T>, axis: usize) -> Tensor<T> {
        debug_assert!(axis < lhs.ndim());
        debug_assert!(lhs.ndim() == rhs.ndim());

        let mut shape = Vec::with_capacity(lhs.ndim());
        for i in 0..lhs.ndim() {
            if i != axis {
                if lhs.shape[i] != rhs.shape[i] {
                    panic!("When using concat, all axes must be the same except the joining one");
                }
                shape.push(lhs.shape[i]);
            } else {
                shape.push(lhs.shape[i] + rhs.shape[i]);
            }
        }

        let mut t = Tensor::empty(&shape);
        for i in 0..lhs.size() {
            let ii = lhs.unravel_index(i);
            t[&ii] = lhs.data[i];
        }

        let offset = lhs.shape[axis];
        for i in 0..rhs.size() {
            let mut ii = rhs.unravel_index(i);
            ii[axis] += offset;
            t[&ii] = rhs.data[i];
        }
        t
    }

}
