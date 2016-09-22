macro_rules! add_impl {
    ($t:ty, $m:ident) => (
        mod $m {
            use numeric::{Tensor, AxisIndex};
            type T = Tensor<$t>;

            #[test]
            fn mv_tensor_rf_tensor_2() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  1., -1., 2.]).reshape(&[2, 3]);
                let answer = T::new(vec![-0., 3., 0.25, 10., 3., 0.]).reshape(&[2, 3]);
                assert!(t1.clone() / &t2 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_3() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  1., -1., 2.]).reshape(&[2, 3]);
                let answer = T::new(vec![-0., 3., 0.25, 10., 3., 0.]).reshape(&[2, 3]);
                assert!(&t1 / &t2 == answer);
            }

            #[test]
            #[should_panic(expected = "assertion failed")]
            fn mv_tensor_rf_tensor_mismatched_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  0.]).reshape(&[2, 2]);
                t1 / &t2; // mis-matched shape
            }

            #[test]
            fn rf_tensor_rf_tensor_slices_reverse_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  1., -1., 2.]).reshape(&[2, 3])
                    .index(&[AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![-0., -3., 1., -5., -3., 0.]).reshape(&[2, 3]);
                assert!(&t1 / &t2 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_slices_reverse_2() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  1., -1., 2.]).reshape(&[2, 3])
                    .index(&[AxisIndex::Full, AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![0., 3., -1., 5., 3., 0.]).reshape(&[2, 3]);
                assert!(&t1 / &t2 == answer);
            }

            #[test]
            fn mv_tensor_rf_tensor_slices_reverse_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  1., -1., 2.]).reshape(&[2, 3])
                    .index(&[AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![-0., -3., 1., -5., -3., 0.]).reshape(&[2, 3]);
                assert!(t1 / &t2 == answer);
            }

            #[test]
            fn mv_tensor_rf_tensor_slices_reverse_2() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  1., -1., 2.]).reshape(&[2, 3])
                    .index(&[AxisIndex::Full, AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![0., 3., -1., 5., 3., 0.]).reshape(&[2, 3]);
                assert!(t1 / &t2 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_slices_subset_1() {
                let t1 = T::range(20)
                    .index(&[AxisIndex::StridedSlice(Some(2), Some(7), 2)]);
                let t2 = T::range(10)
                    .index(&[AxisIndex::StridedSlice(Some(1), Some(4), 1)]);
                let answer = T::new(vec![2., 2., 2.]);
                assert!(&t1 / &t2 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_1() {
                let t1 = T::new(vec![10., 20., 40., 5., 10., 20.]).reshape(&[3, 2]);
                let t2 = T::new(vec![2., 4.]);
                let answer = T::new(vec![5., 5., 20., 1.25, 5., 5.]).reshape(&[3, 2]);
                assert!(&t1 / &t2 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_2() {
                let t1 = T::new(vec![10., 20., 40., 5., 10., 20.]).reshape(&[3, 2]);
                let t2 = T::new(vec![2., 4., 1.]).index(&[AxisIndex::Full, AxisIndex::NewAxis]);
                let answer = T::new(vec![5., 10., 10., 1.25, 10., 20.]).reshape(&[3, 2]);
                assert!(&t1 / &t2 == answer);
            }

            #[test]
            fn rf_tensor_rf_tscalar_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::scalar(-2.);
                let answer = T::new(vec![0., -1.5, -1., -5., 1.5, 0.]).reshape(&[2, 3]);
                assert!(&t1 / &t2 == answer);
            }

            #[test]
            fn rf_tensor_scalar_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let answer = T::new(vec![-0., -1.5, -1., -5., 1.5, -0.]).reshape(&[2, 3]);
                assert!(&t1 / -2. == answer);
            }

            #[test]
            fn mv_tensor_scalar_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let answer = T::new(vec![-0., -1.5, -1., -5., 1.5, -0.]).reshape(&[2, 3]);
                assert!(t1 / -2. == answer);
            }
        }
    )
}

add_impl!(f32, float32);
add_impl!(f64, float64);
