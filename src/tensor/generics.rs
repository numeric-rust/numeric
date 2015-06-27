use tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div, Rem};

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
                    let mut t: Tensor<$t> = Tensor::zeros(&rhs.shape);
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
                    let mut t: Tensor<$t> = Tensor::zeros(&rhs.shape);
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
                    let mut t: Tensor<$t> = self.clone();
                    let v = rhs[0];
                    for i in 0..t.size() {
                        t.data[i] = t.data[i].$fname(v);
                    }
                    t
                } else if self.is_scalar() {
                    let mut t: Tensor<$t> = Tensor::zeros(&rhs.shape);
                    let v = self[0];
                    for i in 0..t.size() {
                        t.data[i] = v.$fname(rhs.data[i]);
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    let mut t: Tensor<$t> = self.clone();
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

add_impl! { Add, add, usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }
add_impl! { Sub, sub, usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }
add_impl! { Mul, mul, usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }
add_impl! { Div, div, usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }
add_impl! { Rem, rem, usize u8 u16 u32 u64 isize i8 i16 i32 i64 }
