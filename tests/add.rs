
macro_rules! add_impl {
    ($t:ty, $m:ident) => (
        mod $m {
            use numeric::{Tensor, AxisIndex};
            type T = Tensor<$t>;

            #[test]
            fn tensor_tensor_2() {
                let t1 = T::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2.0, 1.0, 8.0,  0.0, -1.0, 2.0]).reshape(&[2, 3]);
                let answer = T::new(vec![-2.0, 4.0, 10.0, 10.0, -4.0, 2.0]).reshape(&[2, 3]);
                assert!(t1.clone() + &t2 == answer);
                assert!(t2.clone() + &t1 == answer);
            }

            #[test]
            fn tensor_tensor_3() {
                let t1 = T::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2.0, 1.0, 8.0,  0.0, -1.0, 2.0]).reshape(&[2, 3]);
                let answer = T::new(vec![-2.0, 4.0, 10.0, 10.0, -4.0, 2.0]).reshape(&[2, 3]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            #[should_panic(expected = "assertion failed")]
            fn tensor_tensor_mismatched_1() {
                let t1 = T::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2.0, 1.0, 8.0,  0.0]).reshape(&[2, 2]);
                t1 + &t2; // mis-matched shape
            }

            #[test]
            fn tensor_slices_reverse_ref_1() {
                let t1 = T::range(3 * 7 * 2).reshape(&[3, 7, 2]);
                let t2 = T::range(3 * 7 * 2).reshape(&[3, 7, 2]).index(&[AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50., 52., 54.,
                                         28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50., 52., 54.,
                                         28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50., 52., 54.])
                    .reshape(&[3, 7, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_slices_reverse_move_1() {
                let t1 = T::range(3 * 7 * 2).reshape(&[3, 7, 2]);
                let t2 = T::range(3 * 7 * 2).reshape(&[3, 7, 2]).index(&[AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50., 52., 54.,
                                         28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50., 52., 54.,
                                         28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50., 52., 54.])
                    .reshape(&[3, 7, 2]);
                assert!(t1.clone() + &t2 == answer);
                assert!(t2.clone() + &t1 == answer);
            }

            #[test]
            fn tensor_slices_reverse_ref_2() {
                let t1 = T::range(3 * 5 * 2).reshape(&[3, 5, 2]);
                let t2 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::Full, AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![8., 10., 8., 10., 8., 10., 8., 10., 8., 10.,
                                         28., 30., 28., 30., 28., 30., 28., 30., 28., 30.,
                                         48., 50., 48., 50., 48., 50., 48., 50., 48., 50.])
                    .reshape(&[3, 5, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_slices_reverse_move_2() {
                let t1 = T::range(3 * 5 * 2).reshape(&[3, 5, 2]);
                let t2 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::Full, AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![8., 10., 8., 10., 8., 10., 8., 10., 8., 10.,
                                         28., 30., 28., 30., 28., 30., 28., 30., 28., 30.,
                                         48., 50., 48., 50., 48., 50., 48., 50., 48., 50.])
                    .reshape(&[3, 5, 2]);
                assert!(t1.clone() + &t2 == answer);
                assert!(t2.clone() + &t1 == answer);
            }

            #[test]
            fn tensor_slices_reverse_ref_3() {
                let t1 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::Ellipsis, AxisIndex::StridedSlice(None, None, -1)]);
                let t2 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::StridedSlice(None, None, -1), AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::filled(&[3, 5, 2], 29.);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_slices_reverse_move_3() {
                let t1 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::Ellipsis, AxisIndex::StridedSlice(None, None, -1)]);
                let t2 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::StridedSlice(None, None, -1), AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::filled(&[3, 5, 2], 29.);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_slices_subset_ref_1() {
                let t1 = T::range(3 * 5 * 2).reshape(&[5, 2, 3])
                    .index(&[AxisIndex::StridedSlice(None, Some(3), 1)]);
                let t2 = T::range(3 * 5 * 2).reshape(&[5, 2, 3])
                    .index(&[AxisIndex::StridedSlice(Some(-3), None, 1)]);
                //let answer = T::range(18) * &T::scalar(2) + T::scalar(12);
                let answer = T::new(vec![12., 14., 16., 18., 20., 22., 24., 26., 28.,
                                         30., 32., 34., 36., 38., 40., 42., 44., 46.])
                    .reshape(&[3, 2, 3]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_slices_subset_ref_2() {
                let t1 = T::range(3 * 7 * 2).reshape(&[3, 7, 2])
                    .index(&[AxisIndex::StridedSlice(None, Some(2), 1),
                             AxisIndex::StridedSlice(Some(2), Some(5), 1)]);
                let t2 = T::range(3 * 3 * 2).reshape(&[3, 3, 2])
                    .index(&[AxisIndex::StridedSlice(Some(-2), None, 1)]);
                let answer = T::new(vec![10., 12., 14., 16., 18., 20.,
                                         30., 32., 34., 36., 38., 40.])
                    .reshape(&[2, 3, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_slices_subset_ref_3() {
                let t1 = T::range(5 * 7 * 2).reshape(&[5, 7, 2])
                    .index(&[AxisIndex::StridedSlice(None, None, 2),
                             AxisIndex::StridedSlice(None, Some(4), -2)]);
                let t2 = T::range(5 * 7 * 2).reshape(&[5, 7, 2])
                    .index(&[AxisIndex::StridedSlice(None, Some(3), 1),
                             AxisIndex::StridedSlice(Some(3), Some(2), -1),
                             AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![19., 19., 61., 61., 103., 103.])
                    .reshape(&[3, 1, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_broadcast_ref_1() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(3 * 2).reshape(&[1, 3, 2]);
                let answer = T::new(vec![
                    0., 2., 4., 6., 8., 10., 6., 8., 10., 12., 14., 16., 12.,
                    14., 16., 18., 20., 22., 18., 20., 22., 24., 26., 28., 24.,
                    26., 28., 30., 32., 34.]).reshape(&[5, 3, 2]); assert!(&t1 + &t2 == answer);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_broadcast_ref_2() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(5 * 2).reshape(&[5, 1, 2]);
                let answer = T::new(vec![
                    0., 2., 2., 4., 4., 6., 8., 10., 10., 12., 12.,
                    14., 16., 18., 18., 20., 20., 22., 24., 26., 26., 28.,
                    28., 30., 32., 34., 34., 36., 36., 38.]).reshape(&[5, 3, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_broadcast_ref_3() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(5 * 3).reshape(&[5, 3, 1]);
                let answer = T::new(vec![
                    0., 1., 3., 4., 6., 7., 9., 10., 12., 13., 15.,
                    16., 18., 19., 21., 22., 24., 25., 27., 28., 30., 31.,
                    33., 34., 36., 37., 39., 40., 42., 43.]).reshape(&[5, 3, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_broadcast_ref_4() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(5).reshape(&[5, 1, 1]);
                let answer = T::new(vec![
                    0., 1., 2., 3., 4., 5., 7., 8., 9., 10., 11.,
                    12., 14., 15., 16., 17., 18., 19., 21., 22., 23., 24.,
                    25., 26., 28., 29., 30., 31., 32., 33.]).reshape(&[5, 3, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_broadcast_ref_5() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(3).reshape(&[1, 3, 1]);
                let answer = T::new(vec![
                    0., 1., 3., 4., 6., 7., 6., 7., 9., 10., 12.,
                    13., 12., 13., 15., 16., 18., 19., 18., 19., 21., 22.,
                    24., 25., 24., 25., 27., 28., 30., 31.]).reshape(&[5, 3, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_broadcast_ref_6() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(2).reshape(&[1, 1, 2]);
                let answer = T::new(vec![
                    0., 2., 2., 4., 4., 6., 6., 8., 8., 10., 10.,
                    12., 12., 14., 14., 16., 16., 18., 18., 20., 20., 22.,
                    22., 24., 24., 26., 26., 28., 28., 30.]).reshape(&[5, 3, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_broadcast_ref_7() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::ones(&[1, 1, 1]);
                let answer = T::new(vec![
                    1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
                    12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.,
                    23., 24., 25., 26., 27., 28., 29., 30.]).reshape(&[5, 3, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_broadcast_ref_8() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(3 * 2).reshape(&[3, 2]);
                let answer = T::new(vec![
                    0., 2., 4., 6., 8., 10., 6., 8., 10., 12., 14., 16., 12.,
                    14., 16., 18., 20., 22., 18., 20., 22., 24., 26., 28., 24.,
                    26., 28., 30., 32., 34.]).reshape(&[5, 3, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_broadcast_ref_9() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(2);
                let answer = T::new(vec![
                    0., 2., 2., 4., 4., 6., 6., 8., 8., 10., 10.,
                    12., 12., 14., 14., 16., 16., 18., 18., 20., 20., 22.,
                    22., 24., 24., 26., 26., 28., 28., 30.]).reshape(&[5, 3, 2]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_tscalar_ref_1() {
                let t1 = T::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshape(&[2, 3]);
                let t2 = T::scalar(-2.0);
                let answer = T::new(vec![-2.0, 1.0, 0.0, 8.0, -5.0, -2.0]).reshape(&[2, 3]);
                assert!(&t1 + &t2 == answer);
                assert!(&t2 + &t1 == answer);
            }

            #[test]
            fn tensor_scalar_move_1() {
                let t1 = T::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshape(&[2, 3]);
                let t = t1 + 3.0;
                let answer = T::new(vec![3.0, 6.0, 5.0, 13.0, 0.0, 3.0]).reshape(&[2, 3]);
                assert!(t == answer);
            }
        }
    )
}

add_impl!(f32, float32);
add_impl!(f64, float64);
