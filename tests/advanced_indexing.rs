macro_rules! add_impl {
    ($t:ty, $m:ident) => (
        mod $m {
            use numeric::{Tensor, Full, NewAxis, Ellipsis, Index, StridedSlice};
            type T = Tensor<$t>;

            /*
            #[test]
            fn index_1() {
                let t = T::range(200).reshape(&[2, 20, 5]);
                let t2 = t.index(&[AxisIndex::StridedSlice(Some(1), None, 1),
                                   AxisIndex::StridedSlice(None, Some(5), 1),
                                   AxisIndex::StridedSlice(Some(2), Some(4), 1)]);
                let answer = T::new(vec![102.0, 103.0, 107.0, 108.0, 112.0,
                                         113.0, 117.0, 118.0, 122.0, 123.0]).reshape(&[1, 5, 2]);
                assert!(t2 == answer);
            }

            #[test]
            fn index_2() {
                let t = T::range(210).reshape(&[7, 5, 2, 3]);
                let t2 = t.index(&[AxisIndex::StridedSlice(Some(1), Some(3), 1),
                                   AxisIndex::SliceTo(2),
                                   AxisIndex::Full,
                                   AxisIndex::Slice(1, 2)]);
                let answer = T::new(vec![31.0, 34.0, 37.0, 40.0,
                                         61.0, 64.0, 67.0, 70.0]).reshape(&[2, 2, 2, 1]);
                assert!(t2 == answer);
            }

            #[test]
            fn index_index_1() {
                let t = T::range(210).reshape(&[7, 5, 2, 3]);
                let t2 = t.index(&[AxisIndex::Index(2),
                                   AxisIndex::SliceTo(2),
                                   AxisIndex::Full,
                                   AxisIndex::Index(2)]);
                let answer = T::new(vec![62., 65., 68., 71.]).reshape(&[2, 2]);
                assert!(t2 == answer);
            }

            #[test]
            fn index_ellipsis() {
                let t = T::range(3*2*3*7).reshape(&[3, 2, 3, 7]);
                let t2 = t.index(&[AxisIndex::Index(1),
                                   AxisIndex::Ellipsis,
                                   AxisIndex::SliceFrom(5)]);
                let answer = T::new(vec![47., 48., 54., 55.,  61., 62.,
                                         68., 69., 75., 76.,  82., 83.]).reshape(&[2, 3, 2]);
                assert!(t2 == answer);
            }

            #[test]
            fn index_implied_full() {
                let t = T::range(3*2*7*3).reshape(&[3, 2, 7, 3]);
                let t2 = t.index(&[AxisIndex::Index(1),
                                   AxisIndex::Index(1),
                                   AxisIndex::Slice(1, 3)]);
                let answer = T::new(vec![66., 67., 68.,
                                         69., 70., 71.]).reshape(&[2, 3]);
                assert!(t2 == answer);
            }

            #[test]
            #[should_panic]
            fn index_ellipsis_multiple() {
                let t = T::range(3*2*3*7).reshape(&[3, 2, 3, 7]);
                // Can't use more than one Ellipsis
                t.index(&[AxisIndex::Ellipsis,
                          AxisIndex::Index(2),
                          AxisIndex::Ellipsis]);
            }
            */


            #[test]
            fn test_strided_slice1() {
                let t0 = Tensor::range(40).reshape(&[5, 4, 2]);
                let t1 = t0.index(&[StridedSlice(Some(0), Some(5), 2),
                                    StridedSlice(Some(0), Some(4), 2),
                                    StridedSlice(Some(0), Some(2), 2)]);
                let ta = Tensor::new(vec![0, 4, 16, 20, 32, 36]).reshape(&[3, 2, 1]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_strided_slice2() {
                let t0 = Tensor::range(5 * 7 * 10).reshape(&[5, 7, 10]);
                let t1 = t0.index(&[StridedSlice(Some(1), Some(5), 2),
                                    StridedSlice(Some(2), Some(5), 3),
                                    StridedSlice(Some(4), Some(10), 4)]);
                let ta = Tensor::new(vec![94, 98, 234, 238]).reshape(&[2, 1, 2]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_strided_slice3() {
                let t0 = Tensor::range(40).reshape(&[5, 4, 2]);
                let t1 = t0.index(&[StridedSlice(Some(4), Some(0), -2),
                                    StridedSlice(Some(3), Some(0), -1),
                                    StridedSlice(Some(0), Some(2), 2)]);
                let ta = Tensor::new(vec![38, 36, 34, 22, 20, 18]).reshape(&[2, 3, 1]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_strided_slice4() {
                let t0 = Tensor::range(5 * 7 * 10).reshape(&[5, 7, 10]);
                let t1 = t0.index(&[StridedSlice(Some(0), Some(2), 1),
                                    StridedSlice(Some(2), Some(5), 3),
                                    StridedSlice(Some(4), Some(10), 4)]);
                let ta = Tensor::new(vec![24, 28, 94, 98]).reshape(&[2, 1, 2]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_strided_slice5() {
                let t0 = Tensor::range(5 * 7 * 10).reshape(&[5, 7, 10]);
                let t1 = t0.index(&[StridedSlice(Some(4), Some(1), -1),
                                    StridedSlice(Some(2), Some(5), 3),
                                    StridedSlice(Some(4), Some(10), 4)]);
                let ta = Tensor::new(vec![304, 308, 234, 238, 164, 168]).reshape(&[3, 1, 2]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_strided_slice6() {
                let t0 = Tensor::range(5 * 7 * 10).reshape(&[5, 7, 10]);
                let t1 = t0.index(&[StridedSlice(Some(4), Some(1), -1),
                                    StridedSlice(Some(5), Some(1), -3),
                                    StridedSlice(Some(9), Some(0), -4)]);
                let ta = Tensor::new(vec![339, 335, 331, 309, 305, 301, 269, 265, 261,
                                          239, 235, 231, 199, 195, 191, 169, 165, 161]).reshape(&[3, 2, 3]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_strided_slice7() {
                // Twice indexed
                let t0 = Tensor::range(15 * 21 * 23).reshape(&[15, 21, 23]);
                let t1 = t0.index(&[StridedSlice(Some(5), Some(15), 2),
                                    StridedSlice(Some(19), Some(5), -1),
                                    StridedSlice(Some(20), Some(10), -1)]);
                let t2 = t1.index(&[StridedSlice(Some(4), Some(0), -1),
                                    StridedSlice(Some(0), Some(10), 3),
                                    StridedSlice(Some(5), Some(0), -3)]);
                let ta = Tensor::new(vec![6731, 6734, 6662, 6665, 6593, 6596,
                                          6524, 6527, 5765, 5768, 5696, 5699,
                                          5627, 5630, 5558, 5561, 4799, 4802,
                                          4730, 4733, 4661, 4664, 4592, 4595,
                                          3833, 3836, 3764, 3767, 3695, 3698,
                                          3626, 3629]).reshape(&[4, 4, 2]);
                assert!(t2 == ta);
            }

            #[test]
            fn test_strided_slice8() {
                let t0 = Tensor::range(40).reshape(&[5, 4, 2]);
                let t1 = t0.index(&[StridedSlice(Some(3), None, -2),
                                    StridedSlice(None, None, -1),
                                    StridedSlice(None, None, 2)]);
                let ta = Tensor::new(vec![30, 28, 26, 24, 14, 12, 10,  8]).reshape(&[2, 4, 1]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_strided_slice9() {
                // Twice indexed
                let t0 = Tensor::range(15 * 21 * 23).reshape(&[15, 21, 23]);
                let t1 = t0.index(&[StridedSlice(Some(-10), None, 2),
                                    StridedSlice(Some(-2), Some(-16), -1),
                                    StridedSlice(Some(20), Some(-13), -1)]);
                let t2 = t1.index(&[StridedSlice(Some(4), Some(0), -1),
                                    StridedSlice(Some(0), Some(10), 3),
                                    StridedSlice(Some(5), Some(0), -3)]);
                let ta = Tensor::new(vec![6731, 6734, 6662, 6665, 6593, 6596,
                                          6524, 6527, 5765, 5768, 5696, 5699,
                                          5627, 5630, 5558, 5561, 4799, 4802,
                                          4730, 4733, 4661, 4664, 4592, 4595,
                                          3833, 3836, 3764, 3767, 3695, 3698,
                                          3626, 3629]).reshape(&[4, 4, 2]);
                assert!(t2 == ta);
            }


            #[test]
            fn test_single_index1() {
                let t0 = Tensor::range(40).reshape(&[5, 4, 2]);
                let t1 = t0.index(&[StridedSlice(Some(3), None, -2),
                                    Index(1),
                                    StridedSlice(None, None, 2)]);
                let ta = Tensor::new(vec![26, 10]).reshape(&[2, 1]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_single_index2() {
                let t0 = Tensor::range(120).reshape(&[5, 4, 2, 3]);
                let t1 = t0.index(&[Index(3),
                                    StridedSlice(Some(3), None, -2),
                                    Index(1),
                                    StridedSlice(None, None, 2)]);
                let ta = Tensor::new(vec![93, 95, 81, 83]).reshape(&[2, 2]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_single_index3() {
                let t0 = Tensor::range(120).reshape(&[5, 4, 2, 3]);
                let t1 = t0.index(&[Index(3), Index(2)]);
                let ta = Tensor::new(vec![84, 85, 86, 87, 88, 89]).reshape(&[2, 3]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_single_index4() {
                let t0 = Tensor::range(120).reshape(&[5, 4, 2, 3]);
                let t1 = t0.index(&[Index(3), Full, Index(0)]);
                let ta = Tensor::new(vec![72, 73, 74, 78, 79, 80, 84, 85, 86, 90, 91, 92]).reshape(&[4, 3]);
                assert!(t1 == ta);
            }

            #[test]
            fn test_single_index5() {
                let t0 = Tensor::range(120).reshape(&[5, 4, 2, 3]);
                let t1 = t0.index(&[Index(1), Ellipsis, Index(-1)]);
                let ta = Tensor::new(vec![26, 29, 32, 35, 38, 41, 44, 47]).reshape(&[4, 2]);
                assert!(t1 == ta);
            }


            #[test]
            fn test_new_axis1() {
                let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
                let t1 = t0.index(&[NewAxis]);
                assert!(t1.flatten() == t0.flatten());
                assert!(t1.shape() == &[1, 5, 4, 2]);
            }

            #[test]
            fn test_new_axis2() {
                let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
                let t1 = t0.index(&[Full, NewAxis]);
                assert!(t1.flatten() == t0.flatten());
                assert!(t1.shape() == &[5, 1, 4, 2]);
            }

            #[test]
            fn test_new_axis3() {
                let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
                let t1 = t0.index(&[Full, NewAxis, NewAxis, Full, NewAxis]);
                assert!(t1.flatten() == t0.flatten());
                assert!(t1.shape() == &[5, 1, 1, 4, 1, 2]);
            }

            #[test]
            fn test_new_axis4() {
                let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
                let t1 = t0.index(&[Ellipsis, NewAxis]);
                assert!(t1.flatten() == t0.flatten());
                assert!(t1.shape() == &[5, 4, 2, 1]);
            }

            #[test]
            fn test_new_axis5() {
                let t0: Tensor<f64> = Tensor::range(40).reshape(&[5, 4, 2]);
                let t1 = t0.index(&[NewAxis, Ellipsis, NewAxis, Full]);
                assert!(t1.flatten() == t0.flatten());
                assert!(t1.shape() == &[1, 5, 4, 1, 2]);
            }

            #[test]
            fn index_new_axis_empty() {
                let t = T::new(vec![]);
                assert!(t.index(&[NewAxis]).shape() == &[1, 0]);
                assert!(t.index(&[NewAxis, NewAxis]).shape() == &[1, 1, 0]);
                assert!(t.index(&[Ellipsis, NewAxis]).shape() == &[0, 1]);
                assert!(t.index(&[Full, NewAxis]).shape() == &[0, 1]);
                assert!(t.index(&[NewAxis, Ellipsis, NewAxis]).shape() == &[1, 0, 1]);
            }

            #[test]
            fn index_new_axis_multidimensional() {
                let t = T::range(210).reshape(&[7, 5, 2, 3]);

                assert!(t.index(&[NewAxis]).shape() == &[1, 7, 5, 2, 3]);
                assert!(t.index(&[Full, NewAxis]).shape() == &[7, 1, 5, 2, 3]);
                assert!(t.index(&[Full, Full, NewAxis]).shape() == &[7, 5, 1, 2, 3]);
                assert!(t.index(&[Full, NewAxis, Full]).shape() == &[7, 1, 5, 2, 3]);
                assert!(t.index(&[Full, NewAxis, Full, Full, NewAxis]).shape() ==
                        &[7, 1, 5, 2, 1, 3]);
                assert!(t.index(&[Ellipsis, NewAxis, Full, Full, NewAxis]).shape() ==
                        &[7, 5, 1, 2, 3, 1]);
                assert!(t.index(&[NewAxis, Ellipsis, NewAxis, NewAxis]).shape() ==
                        &[1, 7, 5, 2, 3, 1, 1]);
            }
        }
    )
}

add_impl!(f32, float32);
add_impl!(f64, float64);
