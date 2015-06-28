use tensor::Tensor;
use libc::{c_int, c_char};
use blas_sys;

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
                        unsafe {
                            blas_sys::$gemv(&('T' as c_char),
                                            &(self.shape[1] as c_int),
                                            &(self.shape[0] as c_int),
                                            &1.0,
                                            self.data.as_ptr(),
                                            &(self.shape[1] as c_int),
                                            rhs.data.as_ptr(),
                                            &1,
                                            &0.0,
                                            t3.data.as_mut_ptr(),
                                            &1
                            );
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
                                    v += self.get(i, k) * rhs.get(k, j);
                                }
                                t3.set(i, j, v);
                            }
                        }
                    } else {
                        unsafe {
                            // Note: dgemm assumes column-major while we have row-major,
                            //       so we have to re-arrange things a bit
                            blas_sys::$gemm(&('N' as c_char),
                                            &('N' as c_char),
                                            &(rhs.shape[1] as c_int),
                                            &(self.shape[0] as c_int),
                                            &(rhs.shape[0] as c_int),
                                            &1.0,
                                            rhs.data.as_ptr(),
                                            &(rhs.shape[1] as c_int),
                                            self.data.as_ptr(),
                                            &(rhs.shape[0] as c_int),
                                            &0.0,
                                            t3.data.as_mut_ptr(),
                                            &(rhs.shape[1] as c_int)
                                            );
                        }
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
                        let n = self.size() as c_int;
                        unsafe {
                            v = blas_sys::$dot(&n,
                                               self.data.as_ptr(),
                                               &1,
                                               rhs.data.as_ptr(),
                                               &1);
                        }
                    }
                    Tensor::new(vec![v])
                } else {
                    panic!("Dot product is not supported for the matrix dimensions provided");
                }
            }
        }
    )
}

add_impl!(f32, sgemv_, sgemm_, sdot_);
add_impl!(f64, dgemv_, dgemm_, ddot_);
