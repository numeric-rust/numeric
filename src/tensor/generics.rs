use tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div, Rem, Neg, BitAnd, BitOr, BitXor};

macro_rules! add_impl {
    ($trait_name:ident, $func_name:ident) => (
        // T <op> T
        impl<T: Copy + $trait_name<Output=T>> $trait_name for Tensor<T> {
            type Output = Tensor<T>;
            fn $func_name(mut self, rhs: Self::Output) -> Self::Output {
                if rhs.is_scalar() {
                    let v: T = rhs.scalar_value() as T;
                    for i in 0..self.size() {
                        self.data[i] = self.data[i].$func_name(v);
                    }
                    self
                } else if self.is_scalar() {
                    let mut t = Tensor::empty(&rhs.shape);
                    let v = self.scalar_value();
                    for i in 0..t.size() {
                        t.data[i] = v.$func_name(rhs.data[i]);
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    for i in 0..self.size() {
                        self.data[i] = self.data[i].$func_name(rhs.data[i]);
                    }
                    self
                }
            }
        }
        // T <op> &T
        impl<'a, T: Copy + $trait_name<Output=T>> $trait_name<&'a Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;
            fn $func_name(mut self, rhs: &Self::Output) -> Self::Output {
                //println!("add T + &T");
                if rhs.is_scalar() {
                    let v = rhs[0];
                    for i in 0..self.size() {
                        self.data[i] = self.data[i].$func_name(v);
                    }
                    self
                } else if self.is_scalar() {
                    let mut t: Tensor<T> = Tensor::empty(&rhs.shape);
                    let v = self[0];
                    for i in 0..t.size() {
                        t.data[i] = v.$func_name(rhs.data[i]);
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    for i in 0..self.size() {
                        self.data[i] = self.data[i].$func_name(rhs.data[i]);
                    }
                    self
                }
            }
        }

        // &T <op> &T
        impl<'a, T: Copy + $trait_name<Output=T>> $trait_name<&'a Tensor<T>> for &'a Tensor<T> {
            type Output = Tensor<T>;
            fn $func_name(self, rhs: &Self::Output) -> Self::Output {
                //println!("$fname &T + &T");
                if rhs.is_scalar() {
                    let mut t = self.clone();
                    let v = rhs[0];
                    for i in 0..t.size() {
                        t.data[i] = t.data[i].$func_name(v);
                    }
                    t
                } else if self.is_scalar() {
                    let mut t = Tensor::empty(&rhs.shape);
                    let v = self[0];
                    for i in 0..t.size() {
                        t.data[i] = v.$func_name(rhs.data[i]);
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    let mut t = self.clone();
                    for i in 0..t.size() {
                        t.data[i] = t.data[i].$func_name(rhs.data[i]);
                    }
                    t
                }
            }
        }

        // T <op> S
        impl<T: Copy + $trait_name<Output=T>> $trait_name<T> for Tensor<T> {
            type Output = Tensor<T>;
            fn $func_name(mut self, rhs: T) -> Self::Output {
                for i in 0..self.size() {
                    self.data[i] = self.data[i].$func_name(rhs);
                }
                self
            }
        }
    )
}

// Any operation supported on T should be supported on Tensor<T>, as long as T supports Copy
add_impl!(Add, add);
add_impl!(Sub, sub);
add_impl!(Mul, mul);
add_impl!(Div, div);
add_impl!(Rem, rem);

add_impl!(BitAnd, bitand);
add_impl!(BitOr, bitor);
add_impl!(BitXor, bitxor);


// -T
impl<T: Copy + Neg<Output=T>> Neg for Tensor<T> {
    type Output = Tensor<T>;
    fn neg(mut self) -> Self::Output {
        for i in 0..self.size() {
            self.data[i] = -self.data[i];
        }
        self
    }
}

// -&T
impl<'a, T: Copy + Neg<Output=T>> Neg for &'a Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Self::Output {
        let mut t = Tensor::empty(&self.shape);
        for i in 0..self.size() {
            t.data[i] = -self.data[i];
        }
        t
    }
}
