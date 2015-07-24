use std::ops::Add;
use tensor::Tensor;
use blas;

macro_rules! add_impl {
    ($t:ty, $bfunc:ident) => (
        // There is no point to this one.
        // mut T, T
        /*
        impl Add<Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn add(mut self, rhs: Tensor<$t>) -> Tensor<$t> {
                //println!("add T + T");
                assert_eq!(self.shape, rhs.shape);
                /*
                assert_eq!(self.shape.len(), rhs.shape.len());

                for i in 0..self.shape.len() {
                    assert!(self.shape[i] == rhs.shape[i] ||
                            rhs.shape[i] == 1);
                }
                */

                //let s = rhs.size();
                //if cfg!(noblas) {
                    for i in 0..rhs.size() {
                        self.data[i] += rhs.data[i];
                    }
                /*} else {
                    unsafe {
                        blas_sys::$bfunc(&(self.data.len() as c_int),
                                         &1.0,
                                         rhs.data.as_ptr(),
                                         &1,
                                         self.data.as_mut_ptr(),
                                         &1);
                    }
                }
                */
                self
            }
        }
*/

        // mut T, &T
        impl<'a> Add<&'a Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn add(mut self, rhs: &Tensor<$t>) -> Tensor<$t> {
                //println!("add T + &T");
                if rhs.is_scalar() {
                    let v = rhs[0];
                    for i in 0..self.size() {
                        self.data[i] += v;
                    }
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    if cfg!(noblas) {
                        for i in 0..self.size() {
                            self.data[i] += rhs.data[i];
                        }
                    } else {
                        blas::$bfunc(self.data.len(), 1.0, &rhs.data, 1, &mut self.data, 1);
                    }
                }
                self
            }
        }

        // &T + &T
        impl<'a> Add for &'a Tensor<$t> {
            type Output = Tensor<$t>;
            fn add(self, rhs: &Tensor<$t>) -> Tensor<$t> {
                //println!("add &T + &T");
                if rhs.is_scalar() {
                    let mut t: Tensor<$t> = self.clone();
                    let v = rhs[0];
                    for i in 0..self.size() {
                        t.data[i] += v;
                    }
                    t
                } else if self.is_scalar() {
                    let mut t: Tensor<$t> = rhs.clone();
                    let v = self[0];
                    for i in 0..self.size() {
                        t.data[i] += v;
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    let mut t: Tensor<$t> = self.clone();
                    if cfg!(noblas) {
                        for i in 0..self.size() {
                            t.data[i] += rhs.data[i];
                        }
                    } else {
                        blas::$bfunc(self.data.len(), 1.0, &rhs.data, 1, &mut t.data, 1);
                    }
                    t
                }
            }
        }

        // T + S
        impl Add<$t> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn add(mut self, rhs: $t) -> Tensor<$t> {
                if cfg!(noblas) {
                    for i in 0..self.size() {
                        self.data[i] += rhs;
                    }
                } else {
                    blas::$bfunc(self.data.len(), 1.0, &rhs, 0, &mut self.data, 1);
                }
                self
            }
        }
    )
}

add_impl!(f32, saxpy);
add_impl!(f64, daxpy);
