use tensor::Tensor;
use traits::TensorTrait;
use std::ops::{Add, Sub, Mul, Div, Rem, Neg, BitAnd, BitOr, BitXor};

fn compatible_shapes_for_elementwise_op(sh1: &[usize], sh2: &[usize]) -> bool {
    let mut ok = true;
    if sh1.len() == sh2.len() {
        for i in 0..sh1.len() {
            ok = ok && (sh1[i] == sh2[i] || sh1[i] == 1 || sh2[i] == 1);
        }
    } else if sh1.len() > sh2.len() {
        let diff = sh1.len() - sh2.len();
        for i in 0..sh2.len() {
            ok = ok && (sh1[i+diff] == sh2[i] || sh1[i+diff] == 1 || sh2[i] == 1);
        }
    } else if sh2.len() > sh1.len() {
        let diff = sh2.len() - sh1.len();
        for i in 0..sh1.len() {
            ok = ok && (sh1[i] == sh2[i+diff] || sh1[i] == 1 || sh2[i+diff] == 1);
        }
    }
    ok
}

/// Help function for test_compatible_shapes_for_elementwise_op, since it is a
/// symmetric function.
fn test_compatible_assert_symmetric(sh1: &[usize], sh2: &[usize], expected: bool) -> () {
    assert!(compatible_shapes_for_elementwise_op(sh1, sh2) == expected);
    assert!(compatible_shapes_for_elementwise_op(sh2, sh1) == expected);
}

#[test]
fn test_compatible_shapes_for_elementwise_op() {
    // Compatible test cases

    //assert!(compatible_shapes_for_elementwise_op(&[2, 30], &[2, 30]));
    test_compatible_assert_symmetric(&[2, 30], &[2, 30], true);
    test_compatible_assert_symmetric(&[2, 3, 4], &[2, 3, 4], true);
    test_compatible_assert_symmetric(&[1, 1, 1], &[1, 1, 1], true);
    test_compatible_assert_symmetric(&[2, 3, 4, 5], &[2, 3, 4, 5], true);

    // One broadcasted axis
    test_compatible_assert_symmetric(&[1, 3, 4], &[2, 3, 4], true);
    test_compatible_assert_symmetric(&[10, 1, 5], &[10, 2, 5], true);
    test_compatible_assert_symmetric(&[10, 20, 1], &[10, 20, 100], true);
    test_compatible_assert_symmetric(&[10, 20, 1, 300, 2], &[10, 20, 100, 300, 2], true);

    // Two broadcasted axes
    test_compatible_assert_symmetric(&[1, 1, 4], &[2, 3, 4], true);
    test_compatible_assert_symmetric(&[10, 1, 1], &[10, 2, 5], true);
    test_compatible_assert_symmetric(&[1, 20, 1], &[10, 20, 100], true);
    test_compatible_assert_symmetric(&[1, 20, 1, 300, 2], &[10, 20, 100, 300, 2], true);

    // Mixed broadcasting
    test_compatible_assert_symmetric(&[10, 1, 100, 300, 2], &[10, 20, 1, 300, 2], true);
    test_compatible_assert_symmetric(&[1, 5, 100, 1, 1], &[10, 1, 1, 300, 2], true);
    test_compatible_assert_symmetric(&[1, 1, 100, 1, 1], &[10, 1, 1, 300, 2], true);

    // One is lower dimensional
    test_compatible_assert_symmetric(&[1, 100, 300, 2], &[10, 20, 1, 300, 2], true);
    test_compatible_assert_symmetric(&[100, 300, 2], &[10, 20, 1, 300, 2], true);
    test_compatible_assert_symmetric(&[300, 2], &[10, 20, 1, 300, 2], true);
    test_compatible_assert_symmetric(&[2], &[10, 20, 1, 300, 2], true);
    test_compatible_assert_symmetric(&[1], &[10, 20, 1, 300, 2], true);

    // Scalars
    test_compatible_assert_symmetric(&[], &[10, 20, 1, 300, 2], true);
    test_compatible_assert_symmetric(&[], &[], true);

    // Incompatible test cases
    test_compatible_assert_symmetric(&[2, 3, 4], &[3, 2, 4], false);
    test_compatible_assert_symmetric(&[2, 3], &[3, 3], false);
    test_compatible_assert_symmetric(&[5], &[3], false);
    test_compatible_assert_symmetric(&[1, 5], &[3], false);
    test_compatible_assert_symmetric(&[1, 5], &[1, 3], false);
    test_compatible_assert_symmetric(&[1, 2, 3, 4], &[5, 4], false);
}

// T <op> &T
macro_rules! add_impl {
    ($trait_name:ident, $func_name:ident, $func_name_with_mul:ident) => (
        // T <op> T
        impl<T: TensorTrait + $trait_name<Output=T>> $trait_name for Tensor<T> {
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
        impl<'a, T: TensorTrait + $trait_name<Output=T>> $trait_name<&'a Tensor<T>> for Tensor<T> {
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
        impl<T: TensorTrait + $trait_name<Output=T>> Tensor<T> {
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
        impl<'a, T: TensorTrait + $trait_name<Output=T>> $trait_name<&'a Tensor<T>> for &'a Tensor<T> {
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
        impl<T: TensorTrait + $trait_name<Output=T>> $trait_name<T> for Tensor<T> {
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
