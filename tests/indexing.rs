use numeric::Tensor;

macro_rules! add_impl {
    ($t:ty, $m:ident) => (
        mod $m {
            use numeric::Tensor;
            type T = Tensor<$t>;

            /*
            #[test]
            fn indexing() {
                let t = T::range(6);
                let v = t[2];
                assert_eq!(v, 2.0);
            }

            #[test]
            fn indexing_mut() {
                let mut t = T::zeros(&[3]);
                t[2] = 1.0;
                assert!(t == T::new(vec![0.0, 0.0, 1.0]));
            }
            */

            #[test]
            fn indexing_tuple_1() {
                let t = T::range(6);
                let v = t[(2,)];
                assert_eq!(v, 2.0);
            }

            #[test]
            fn indexing_tuple_2() {
                let t = T::range(6).reshape(&[2, 3]);
                let v = t[(1, 2)];
                assert_eq!(v, 5.0);
            }

            #[test]
            fn indexing_tuple_3() {
                let t = T::range(100).reshape(&[2, 5, 10]);
                let v = t[(1, 2, 3)];
                assert_eq!(v, 73.0);
            }
        }
    )
}

add_impl!(f32, float32);
add_impl!(f64, float64);

// Ravelling and unravelling indices

#[test]
fn ravel_index() {
    let t: Tensor<f64> = Tensor::zeros(&[3, 4, 5, 6]);
    assert_eq!(t.ravel_index(&[0, 1, 0, 3]), 33);
    assert_eq!(t.ravel_index(&[0, 1, 4, 3]), 57);
    assert_eq!(t.ravel_index(&[1, 1, 4, 1]), 175);
    assert_eq!(t.ravel_index(&[2, 2, 0, 0]), 300);
}

#[test]
fn unravel_index() {
    let t: Tensor<f64> = Tensor::zeros(&[3, 4, 5, 6]);
    assert_eq!(t.unravel_index(33), vec![0, 1, 0, 3]);
    assert_eq!(t.unravel_index(57), vec![0, 1, 4, 3]);
    assert_eq!(t.unravel_index(175), vec![1, 1, 4, 1]);
    assert_eq!(t.unravel_index(300), vec![2, 2, 0, 0]);
}
