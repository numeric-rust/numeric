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
                    {
                        let mut data = t3.slice_mut();
                        if cfg!(noblas) {
                            // Naive implementation, BLAS will be much faster
                            for i in 0..self.shape[0] {
                                let mut v = 0.0;
                                for k in 0..self.shape[1] {
                                    v += self.get2(i, k) * rhs.data[k];
                                }
                                data[i] = v;
                            }
                        } else {
                            let t1 = self.canonize();
                            let t2 = rhs.canonize();
                            blas::$gemv(b'T', t1.shape[1], t1.shape[0], 1.0, &t1.data,
                                        t1.shape[1], &t2.data, 1, 0.0, data, 1);
                        }
                    }
                    t3
                } else if self.ndim() == 1 && rhs.ndim() == 2 {
                    // TODO dot(vector, matrix) with blas
                    assert_eq!(self.shape[0], rhs.shape[0]);
                    let mut t3 = Tensor::empty(&[rhs.shape[1]]);
                    {
                        let mut data = t3.slice_mut();
                        
                        // Naive implementation, BLAS will be much faster
                        for i in 0..rhs.shape[1] {
                            let mut v = 0.0;
                            for k in 0..rhs.shape[0] {
                                v += self.data[k] * rhs.get2(k, i
                                );
                            }
                            data[i] = v;
                        }
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
                                    v += self.get2(i, k) * rhs.get2(k, j);
                                }
                                t3.set2(i, j, v);
                            }
                        }
                    } else {
                        // Note: dgemm assumes column-major while we have row-major,
                        //       so we have to re-arrange things a bit
                        let t1 = self.canonize();
                        let t2 = rhs.canonize();
                        let mut data = t3.slice_mut();
                        blas::$gemm(b'N', b'N', t2.shape[1], t1.shape[0], t2.shape[0], 1.0,
                                    &t2.data, t2.shape[1], &t1.data, t2.shape[0], 0.0,
                                    data, t2.shape[1]);
                    }
                    t3
                } else if self.ndim() == 1 && rhs.ndim() == 1 { // scalar product
                    assert_eq!(self.size(), rhs.size());
                    let mut v = 0.0;
                    if cfg!(noblas) {
                        // Naive implementation, BLAS will be much faster
                        for (v1, v2) in self.iter().zip(rhs.iter()) {
                            v += v1 * v2;
                        }
                    } else {
                        let t1 = self.canonize();
                        let t2 = rhs.canonize();
                        v = blas::$dot(t1.size(), &t1.data, 1, &t2.data, 1);
                    }
                    Tensor::scalar(v)
                } else {
                    panic!("Dot product is not supported for the matrix dimensions provided");
                }
            }
        }
    )
}

add_impl!(f32, sgemv, sgemm, sdot);
add_impl!(f64, dgemv, dgemm, ddot);
