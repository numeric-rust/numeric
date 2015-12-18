use tensor::Tensor;
use blas;

macro_rules! add_impl {
    ($t:ty, $gemv:ident, $gemm:ident, $dot:ident) => (
        impl Tensor<$t> {
            /// Takes the product of two tensors. If the tensors are both matrices (2D), then a
            /// matrix multiplication is taken. If the tensors are both vectors (1D), the scalar
            /// product is taken.
            pub fn dot(&self, rhs: &Tensor<$t>) -> Tensor<$t> {
                if self.ndim() == 2 && rhs.ndim() == 1 {
                    assert_eq!(self.shape[1], rhs.shape[0]);
                    let mut t3 = Tensor::empty(&[self.shape[0]]);
                    if cfg!(noblas) {
                        // Naive implementation, BLAS will be much faster
                        for i in 0..self.shape[0] {
                            let mut v = 0.0;
                            for k in 0..self.shape[1] {
                                v += self.get(i, k) * rhs.data[k];
                            }
                            t3.data[i] = v;
                        }
                    } else {
                        blas::$gemv(b'T', self.shape[1], self.shape[0], 1.0, &self.data,
                                    self.shape[1], &rhs.data, 1, 0.0, &mut t3.data, 1);
                    }
                    t3
                } else if self.ndim() == 2 && rhs.ndim() == 2 {
                    assert_eq!(self.shape[1], rhs.shape[0]);
                    let mut t3 = Tensor::empty(&[self.shape[0], rhs.shape[1]]);
                    if cfg!(noblas) {
                        // Naive implementation, BLAS will be much faster
                        for i in 0..self.shape[0] {
                            for j in 0..rhs.shape[1] {
                                let mut v = 0.0;
                                for k in 0..self.shape[1] {
                                    v += self.get(i, k) * rhs.get(k, j);
                                }
                                t3.set2(i, j, v);
                            }
                        }
                    } else {
                        // Note: dgemm assumes column-major while we have row-major,
                        //       so we have to re-arrange things a bit
                        blas::$gemm(b'N', b'N', rhs.shape[1], self.shape[0], rhs.shape[0], 1.0,
                                    &rhs.data, rhs.shape[1], &self.data, rhs.shape[0], 0.0,
                                    &mut t3.data, rhs.shape[1]);
                    }
                    t3
                } else if self.ndim() == 1 && rhs.ndim() == 1 { // scalar product
                    assert_eq!(self.size(), rhs.size());
                    let mut v = 0.0;
                    if cfg!(noblas) {
                        // Naive implementation, BLAS will be much faster
                        for k in 0..self.shape[0] {
                            v += self.data[k] * rhs.data[k];
                        }
                    } else {
                        v = blas::$dot(self.size(), &self.data, 1, &rhs.data, 1);
                    }
                    Tensor::new(vec![v])
                } else {
                    panic!("Dot product is not supported for the matrix dimensions provided");
                }
            }
        }
    )
}

add_impl!(f32, sgemv, sgemm, sdot);
add_impl!(f64, dgemv, dgemm, ddot);
