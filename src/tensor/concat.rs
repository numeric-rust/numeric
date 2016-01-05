use tensor::Tensor;
use traits::TensorTrait;

impl<T: TensorTrait> Tensor<T> {
    pub fn concat(lhs: &Tensor<T>, rhs: &Tensor<T>, axis: usize) -> Tensor<T> {
        debug_assert!(axis < lhs.ndim());
        debug_assert!(lhs.ndim() == rhs.ndim());

        let t1 = lhs.canonize();
        let t2 = rhs.canonize();

        let mut shape = Vec::with_capacity(t1.ndim());
        for i in 0..t1.ndim() {
            if i != axis {
                if t1.shape[i] != t2.shape[i] {
                    panic!("When using concat, all axes must be the same except the joining one");
                }
                shape.push(t1.shape[i]);
            } else {
                shape.push(t1.shape[i] + t2.shape[i]);
            }
        }

        let mut t = Tensor::empty(&shape);
        for i in 0..t1.size() {
            let ii = t1.unravel_index(i);
            t[&ii] = t1.data[i];
        }

        let offset = t1.shape[axis];
        for i in 0..t2.size() {
            let mut ii = t2.unravel_index(i);
            ii[axis] += offset;
            t[&ii] = t2.data[i];
        }
        t
    }

}
