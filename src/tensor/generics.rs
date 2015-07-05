use tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div, Rem, BitAnd, BitOr, BitXor};

/*
macro_rules! add_impl {
    ($trait_name:ident, $fname:ident, $($t:ty)*) => ($(
        // mut T <op> T
        impl $trait_name<Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn $fname(mut self, rhs: Tensor<$t>) -> Tensor<$t> {
                if rhs.is_scalar() {
                    let v = rhs[0];
                    for i in 0..self.size() {
                        self.data[i] = self.data[i].$fname(v);
                    }
                    self
                } else if self.is_scalar() {
                    let mut t: Tensor<$t> = Tensor::empty(&rhs.shape);
                    let v = self[0];
                    for i in 0..t.size() {
                        t.data[i] = v.$fname(rhs.data[i]);
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    for i in 0..self.size() {
                        self.data[i] = self.data[i].$fname(rhs.data[i]);
                    }
                    self
                }
            }
        }

        // mut T <op> &T
        impl<'a> $trait_name<&'a Tensor<$t>> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn $fname(mut self, rhs: &Tensor<$t>) -> Tensor<$t> {
                //println!("$fname T + &T");
                if rhs.is_scalar() {
                    let v = rhs[0];
                    for i in 0..self.size() {
                        self.data[i] = self.data[i].$fname(v);
                    }
                    self
                } else if self.is_scalar() {
                    let mut t: Tensor<$t> = Tensor::empty(&rhs.shape);
                    let v = self[0];
                    for i in 0..t.size() {
                        t.data[i] = v.$fname(rhs.data[i]);
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    for i in 0..self.size() {
                        self.data[i] = self.data[i].$fname(rhs.data[i]);
                    }
                    self
                }
            }
        }

        // &T <op> &T
        impl<'a> $trait_name for &'a Tensor<$t> {
            type Output = Tensor<$t>;
            fn $fname(self, rhs: &Tensor<$t>) -> Tensor<$t> {
                //println!("$fname &T + &T");
                if rhs.is_scalar() {
                    let mut t = self.clone();
                    let v = rhs[0];
                    for i in 0..t.size() {
                        t.data[i] = t.data[i].$fname(v);
                    }
                    t
                } else if self.is_scalar() {
                    let mut t = Tensor::empty(&rhs.shape);
                    let v = self[0];
                    for i in 0..t.size() {
                        t.data[i] = v.$fname(rhs.data[i]);
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    let mut t = self.clone();
                    for i in 0..t.size() {
                        t.data[i] = t.data[i].$fname(rhs.data[i]);
                    }
                    t
                }
            }
        }

        // T <op> S
        impl $trait_name<$t> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn $fname(mut self, rhs: $t) -> Tensor<$t> {
                for i in 0..self.size() {
                    self.data[i] = self.data[i].$fname(rhs);
                }
                self
            }
        }
    )*)
}

// TODO: It would be better if these relied only on the traits, so that custom-made tensort types
// could be used.
//add_impl! { Add, add, usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }
add_impl! { Sub, sub, usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }
add_impl! { Mul, mul, usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }
add_impl! { Div, div, usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }
add_impl! { Rem, rem, usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

add_impl! { BitAnd, bitand, usize u8 u16 u32 u64 isize i8 i16 i32 i64 bool }
add_impl! { BitOr, bitor, usize u8 u16 u32 u64 isize i8 i16 i32 i64 bool }
add_impl! { BitXor, bitxor, usize u8 u16 u32 u64 isize i8 i16 i32 i64 bool }
*/

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

