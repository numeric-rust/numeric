use tensor::Tensor;
use traits::TensorTrait;
use std::ops::{Add, Sub, Mul, Div, Rem, Neg, BitAnd, BitOr, BitXor};

// T <op> &T
macro_rules! add_impl {
    ($trait:ident, $func_name:ident, $func_name_with_mul:ident) => (
        // T <op> T
        impl<T: TensorTrait + $trait<Output=T>> $trait for Tensor<T> {
            type Output = Tensor<T>;
            fn $func_name(mut self, rhs: Self::Output) -> Self::Output {
                if rhs.is_scalar() {
                    self.canonize_inplace();
                    {
                        let n = self.size();
                        let mut data = self.slice_mut();
                        let v: T = rhs.scalar_value() as T;
                        for i in 0..n {
                            data[i] = data[i].$func_name(v);
                        }
                    }
                    self
                } else if self.is_scalar() {
                    let mut t = Tensor::empty(&rhs.shape);
                    {
                        let n = t.size();
                        let mut data = t.slice_mut();
                        let v = self.scalar_value();
                        for i in 0..n {
                            data[i] = v.$func_name(rhs.data[i]);
                        }
                    }
                    t
                } else {
                    self.canonize_inplace();
                    {
                        let n = self.size();
                        assert_eq!(self.shape, rhs.shape);
                        let mut data = self.slice_mut();
                        for i in 0..n {
                            data[i] = data[i].$func_name(rhs.data[i]);
                        }
                    }
                    self
                }
            }
        }

        // T <op> &T
        impl<'a, T: TensorTrait + $trait<Output=T>> $trait<&'a Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;
            fn $func_name(mut self, rhs: &Self::Output) -> Self::Output {
                if rhs.is_scalar() {
                    self.canonize_inplace();
                    {
                        let n = self.size();
                        let mut data = self.slice_mut();
                        let v = rhs.scalar_value();
                        for i in 0..n {
                            data[i] = data[i].$func_name(v);
                        }
                    }
                    self
                } else if self.is_scalar() {
                    let mut t: Tensor<T> = Tensor::empty(&rhs.shape);
                    {
                        let n = t.size();
                        let mut data = t.slice_mut();
                        let v = self.scalar_value();
                        for i in 0..n {
                            data[i] = v.$func_name(rhs.data[i]);
                        }
                    }
                    t
                } else {
                    self.canonize_inplace();
                    {
                        let n = self.size();
                        assert_eq!(self.shape, rhs.shape);
                        let mut data = self.slice_mut();
                        for i in 0..n {
                            data[i] = data[i].$func_name(rhs.data[i]);
                        }
                    }
                    self
                }
            }
        }

        // T <op> &T  (with out)
        impl<T: TensorTrait + $trait<Output=T>> Tensor<T> {
            pub fn $func_name_with_mul(&self, rhs: &Tensor<T>, out: &mut Tensor<T>) -> () {
                out.canonize_inplace();
                if rhs.is_scalar() {
                    assert!(out.shape() == self.shape());
                    let n = out.size();
                    let mut data = out.slice_mut();
                    let v = rhs.scalar_value();
                    for i in 0..n {
                        data[i] = data[i].$func_name(v);
                    }
                } else if self.is_scalar() {
                    assert!(out.shape() == rhs.shape());
                    let mut data = out.slice_mut();
                    let v = self.scalar_value();
                    for i in 0..rhs.size() {
                        data[i] = v.$func_name(rhs.data[i]);
                    }
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    let n = out.size();
                    let mut data = out.slice_mut();
                    for i in 0..n {
                        data[i] = self.data[i].$func_name(rhs.data[i]);
                    }
                }
            }
        }

        // &T <op> &T
        impl<'a, T: TensorTrait + $trait<Output=T>> $trait<&'a Tensor<T>> for &'a Tensor<T> {
            type Output = Tensor<T>;
            fn $func_name(self, rhs: &Self::Output) -> Self::Output {
                //println!("$fname &T + &T");
                if rhs.is_scalar() {
                    //let mut t = self.clone();
                    //let mut t: Tensor<T> = Tensor::empty(&self.shape);
                    let mut t = self.canonize();
                    {
                        let n = t.size();
                        let mut data = t.slice_mut();
                        let v = rhs.scalar_value();
                        for i in 0..n {
                            data[i] = data[i].$func_name(v);
                        }
                    }
                    t
                } else if self.is_scalar() {
                    let mut t = Tensor::empty(&rhs.shape);
                    {
                        let n = t.size();
                        let mut data = t.slice_mut();
                        let v = self.scalar_value();
                        for i in 0..n {
                            data[i] = v.$func_name(rhs.data[i]);
                        }
                    }
                    t
                } else {
                    assert_eq!(self.shape, rhs.shape);
                    //let mut t: Tensor<T> = Tensor::empty(&self.shape);
                    let mut t = self.canonize();
                    {
                        let mut data = t.slice_mut();
                        for (i, v) in rhs.iter().enumerate() {
                            data[i] = data[i].$func_name(v);
                        }
                    }
                    t
                }
            }
        }

        // T <op> S
        impl<T: TensorTrait + $trait<Output=T>> $trait<T> for Tensor<T> {
            type Output = Tensor<T>;
            fn $func_name(mut self, rhs: T) -> Self::Output {
                self.canonize_inplace();
                {
                    let n = self.size();
                    let mut data = self.slice_mut();
                    for i in 0..n {
                        data[i] = data[i].$func_name(rhs);
                    }
                }
                self
            }
        }
    )
}

// Any operation supported on T should be supported on Tensor<T>, as long as T supports TensorTrait
add_impl!(Add, add, add_with_out);
add_impl!(Sub, sub, sub_with_out);
add_impl!(Mul, mul, mul_with_out);
add_impl!(Div, div, div_with_out);
add_impl!(Rem, rem, rem_with_out);

add_impl!(BitAnd, bitand, bitand_with_out);
add_impl!(BitOr, bitor, bitor_with_out);
add_impl!(BitXor, bitxor, bitxor_with_out);

// -T
impl<T: TensorTrait + Neg<Output=T>> Neg for Tensor<T> {
    type Output = Tensor<T>;
    fn neg(mut self) -> Self::Output {
        self.canonize_inplace();
        {
            let n = self.size();
            let mut data = self.slice_mut();
            for i in 0..n {
                data[i] = -data[i];
            }
        }
        self
    }
}

// -&T
impl<'a, T: TensorTrait + Neg<Output=T>> Neg for &'a Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Self::Output {
        let mut t = Tensor::empty(&self.shape);
        {
            let mut data = t.slice_mut();
            for (i, v) in self.iter().enumerate() {
                data[i] = -v;
            }
        }
        t
    }
}

/*
#[test]
fn test_negate1() {
    let t0 = tensor![1.0, 10.0; 23.0, -4.0];
    let t1 = -&t0;

    let ta = tensor![-1.0, -10.0; -23.0, 4.0];
    assert!(t1 == ta);
}

#[test]
fn test_negate2() {
    let t0 = Tensor::range(120).reshape(&[5, 4, 2, 3]);
    let t1 = t0.index(&[AxisIndex::Index(3),
                        AxisIndex::StridedSlice(Some(3), None, -2),
                        AxisIndex::Index(1),
                        AxisIndex::StridedSlice(None, None, 2)]);
    let t2 = -&t1;

    let ta = tensor![-93, -95; -81, -83];
    assert!(t2 == ta);
}
*/
