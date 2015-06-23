use tensor::Tensor;
use libc::{c_int, c_char};
use blas_sys;

impl Tensor<f64> {
    /// Takes the product of two tensors. If the tensors are both matrices (2D), then a matrix
    /// multiplication is taken. If the tensors are both vectors (1D), the scalar product is taken.
    pub fn dot(t1: &Tensor<f64>, t2: &Tensor<f64>) -> Tensor<f64> {
        if t1.ndim() == 2 && t2.ndim() == 1 {
            assert_eq!(t1.shape[1], t2.shape[0]);
            let mut t3 = Tensor::zeros(&[t1.shape[0]]);
            if cfg!(noblas) {
                // Naive implementation, BLAS will be much faster
                for i in 0..t1.shape[0] {
                    let mut v = 0.0;
                    for k in 0..t1.shape[1] {
                        v += t1.get(i, k) * t2.data[k];
                    }
                    t3.data[i] = v;
                }
            } else {
                unsafe {
                    blas_sys::dgemv_(
                        &('T' as c_char),
                        &(t1.shape[1] as c_int),
                        &(t1.shape[0] as c_int),
                        &1.0,
                        t1.data.as_ptr(),
                        &(t1.shape[1] as c_int),
                        t2.data.as_ptr(),
                        &1,
                        &0.0,
                        t3.data.as_mut_ptr(),
                        &1
                    );
                }
            }
            t3
        } else if t1.ndim() == 2 && t2.ndim() == 2 {
            assert_eq!(t1.shape[1], t2.shape[0]);
            let mut t3 = Tensor::zeros(&[t1.shape[0], t2.shape[1]]);
            if cfg!(noblas) {
                // Naive implementation, BLAS will be much faster
                for i in 0..t1.shape[0] {
                    for j in 0..t2.shape[1] {
                        let mut v = 0.0;
                        for k in 0..t1.shape[1] {
                            v += t1.get(i, k) * t2.get(k, j);
                        }
                        t3.set(i, j, v);
                    }
                }
            } else {
                unsafe {
                    // Note: dgemm assumes column-major while we have row-major,
                    //       so we have to re-arrange things a bit
                    blas_sys::dgemm_(&('N' as c_char),
                                     &('N' as c_char),
                                     &(t2.shape[1] as c_int),
                                     &(t1.shape[0] as c_int),
                                     &(t2.shape[0] as c_int),
                                     &1.0,
                                     t2.data.as_ptr(),
                                     &(t2.shape[1] as c_int),
                                     t1.data.as_ptr(),
                                     &(t2.shape[0] as c_int),
                                     &0.0,
                                     t3.data.as_mut_ptr(),
                                     &(t2.shape[1] as c_int)
                                     );
                }
            }
            t3
        } else if t1.ndim() == 1 && t2.ndim() == 1 { // scalar product
            assert_eq!(t1.size(), t2.size());
            let mut v = 0.0;
            if cfg!(noblas) {
                // Naive implementation, BLAS will be much faster
                for k in 0..t1.shape[0] {
                    v += t1.data[k] * t2.data[k];
                }
            } else {
                let n = t1.size() as c_int;
                unsafe {
                    v = blas_sys::ddot_(&n,
                                        t1.data.as_ptr(),
                                        &1,
                                        t2.data.as_ptr(),
                                        &1);
                }
            }
            Tensor::new(vec![v])
        } else {
            panic!("Dot product is not supported for the matrix dimensions provided");
        }
    }
}
