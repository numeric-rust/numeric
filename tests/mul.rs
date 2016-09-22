macro_rules! add_impl {
    ($t:ty, $m:ident) => (
        mod $m {
            use numeric::{Tensor, AxisIndex};
            type T = Tensor<$t>;

            #[test]
            fn mv_tensor_rf_tensor_2() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  0., -1., 2.]).reshape(&[2, 3]);
                let answer = T::new(vec![0., 3., 16., 0., 3., 0.]).reshape(&[2, 3]);
                assert!(t1.clone() * &t2 == answer);
                assert!(t2.clone() * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_3() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  0., -1., 2.]).reshape(&[2, 3]);
                let answer = T::new(vec![0., 3., 16., 0., 3., 0.]).reshape(&[2, 3]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            #[should_panic(expected = "assertion failed")]
            fn mv_tensor_rf_tensor_mismatched_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::new(vec![-2., 1., 8.,  0.]).reshape(&[2, 2]);
                t1 * &t2; // mis-matched shape
            }

            #[test]
            fn rf_tensor_rf_tensor_slices_reverse_1() {
                let t1 = T::range(3 * 7 * 2).reshape(&[3, 7, 2]);
                let t2 = T::range(3 * 7 * 2).reshape(&[3, 7, 2]).index(&[AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![   0.,   29.,   60.,   93.,  128.,  165.,  204.,  245.,  288.,
                                          333.,  380.,  429.,  480.,  533.,  196.,  225.,  256.,  289.,
                                          324.,  361.,  400.,  441.,  484.,  529.,  576.,  625.,  676.,
                                          729.,    0.,   29.,   60.,   93.,  128.,  165.,  204.,  245.,
                                          288.,  333.,  380.,  429.,  480.,  533.])
                    .reshape(&[3, 7, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn mv_tensor_rf_tensor_slices_reverse_1() {
                let t1 = T::range(3 * 7 * 2).reshape(&[3, 7, 2]);
                let t2 = T::range(3 * 7 * 2).reshape(&[3, 7, 2]).index(&[AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![   0.,   29.,   60.,   93.,  128.,  165.,  204.,  245.,  288.,
                                          333.,  380.,  429.,  480.,  533.,  196.,  225.,  256.,  289.,
                                          324.,  361.,  400.,  441.,  484.,  529.,  576.,  625.,  676.,
                                          729.,    0.,   29.,   60.,   93.,  128.,  165.,  204.,  245.,
                                          288.,  333.,  380.,  429.,  480.,  533.])
                    .reshape(&[3, 7, 2]);
                assert!(t1.clone() * &t2 == answer);
                assert!(t2.clone() * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_slices_reverse_2() {
                let t1 = T::range(3 * 5 * 2).reshape(&[3, 5, 2]);
                let t2 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::Full, AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![   0.,    9.,   12.,   21.,   16.,   25.,   12.,   21.,    0.,
                                            9.,  180.,  209.,  192.,  221.,  196.,  225.,  192.,  221.,
                                          180.,  209.,  560.,  609.,  572.,  621.,  576.,  625.,  572.,
                                          621.,  560.,  609.])
                    .reshape(&[3, 5, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn mv_tensor_rf_tensor_slices_reverse_2() {
                let t1 = T::range(3 * 5 * 2).reshape(&[3, 5, 2]);
                let t2 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::Full, AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![   0.,    9.,   12.,   21.,   16.,   25.,   12.,   21.,    0.,
                                            9.,  180.,  209.,  192.,  221.,  196.,  225.,  192.,  221.,
                                          180.,  209.,  560.,  609.,  572.,  621.,  576.,  625.,  572.,
                                          621.,  560.,  609.])
                    .reshape(&[3, 5, 2]);
                assert!(t1.clone() * &t2 == answer);
                assert!(t2.clone() * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_slices_reverse_3() {
                let t1 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::Ellipsis, AxisIndex::StridedSlice(None, None, -1)]);
                let t2 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::StridedSlice(None, None, -1), AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![  28.,    0.,   78.,   54.,  120.,  100.,  154.,  138.,  180.,
                                          168.,  198.,  190.,  208.,  204.,  210.,  210.,  204.,  208.,
                                          190.,  198.,  168.,  180.,  138.,  154.,  100.,  120.,   54.,
                                           78.,    0.,   28.])
                    .reshape(&[3, 5, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn mv_tensor_rf_tensor_slices_reverse_3() {
                let t1 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::Ellipsis, AxisIndex::StridedSlice(None, None, -1)]);
                let t2 = T::range(3 * 5 * 2).reshape(&[3, 5, 2])
                    .index(&[AxisIndex::StridedSlice(None, None, -1), AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![  28.,    0.,   78.,   54.,  120.,  100.,  154.,  138.,  180.,
                                          168.,  198.,  190.,  208.,  204.,  210.,  210.,  204.,  208.,
                                          190.,  198.,  168.,  180.,  138.,  154.,  100.,  120.,   54.,
                                           78.,    0.,   28.])
                    .reshape(&[3, 5, 2]);
                assert!(t1.clone() * &t2 == answer);
                assert!(t2.clone() * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_slices_subset_1() {
                let t1 = T::range(3 * 5 * 2).reshape(&[5, 2, 3])
                    .index(&[AxisIndex::StridedSlice(None, Some(3), 1)]);
                let t2 = T::range(3 * 5 * 2).reshape(&[5, 2, 3])
                    .index(&[AxisIndex::StridedSlice(Some(-3), None, 1)]);
                let answer = T::new(vec![   0.,   13.,   28.,   45.,   64.,   85.,  108.,  133.,  160.,
                                          189.,  220.,  253.,  288.,  325.,  364.,  405.,  448.,  493.])
                    .reshape(&[3, 2, 3]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_slices_subset_2() {
                let t1 = T::range(3 * 7 * 2).reshape(&[3, 7, 2])
                    .index(&[AxisIndex::StridedSlice(None, Some(2), 1),
                             AxisIndex::StridedSlice(Some(2), Some(5), 1)]);
                let t2 = T::range(3 * 3 * 2).reshape(&[3, 3, 2])
                    .index(&[AxisIndex::StridedSlice(Some(-2), None, 1)]);
                let answer = T::new(vec![  24.,   35.,   48.,   63.,   80.,   99.,
                                          216.,  247.,  280.,  315.,  352.,  391.])
                    .reshape(&[2, 3, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_slices_subset_3() {
                let t1 = T::range(5 * 7 * 2).reshape(&[5, 7, 2])
                    .index(&[AxisIndex::StridedSlice(None, None, 2),
                             AxisIndex::StridedSlice(None, Some(4), -2)]);
                let t2 = T::range(5 * 7 * 2).reshape(&[5, 7, 2])
                    .index(&[AxisIndex::StridedSlice(None, Some(3), 1),
                             AxisIndex::StridedSlice(Some(3), Some(2), -1),
                             AxisIndex::StridedSlice(None, None, -1)]);
                let answer = T::new(vec![   84.,    78.,   840.,   820.,  2380.,  2346.])
                    .reshape(&[3, 1, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_1() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(3 * 2).reshape(&[1, 3, 2]);
                let answer = T::new(vec![
                          0.,    1.,    4.,    9.,   16.,   25.,    0.,    7.,   16.,
                         27.,   40.,   55.,    0.,   13.,   28.,   45.,   64.,   85.,
                          0.,   19.,   40.,   63.,   88.,  115.,    0.,   25.,   52.,
                         81.,  112.,  145.]).reshape(&[5, 3, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_2() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(5 * 2).reshape(&[5, 1, 2]);
                let answer = T::new(vec![
                          0.,    1.,    0.,    3.,    0.,    5.,   12.,   21.,   16.,
                         27.,   20.,   33.,   48.,   65.,   56.,   75.,   64.,   85.,
                        108.,  133.,  120.,  147.,  132.,  161.,  192.,  225.,  208.,
                        243.,  224.,  261.]).reshape(&[5, 3, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_3() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(5 * 3).reshape(&[5, 3, 1]);
                let answer = T::new(vec![
                          0.,    0.,    2.,    3.,    8.,   10.,   18.,   21.,   32.,
                         36.,   50.,   55.,   72.,   78.,   98.,  105.,  128.,  136.,
                        162.,  171.,  200.,  210.,  242.,  253.,  288.,  300.,  338.,
                        351.,  392.,  406.]).reshape(&[5, 3, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_4() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(5).reshape(&[5, 1, 1]);
                let answer = T::new(vec![
                          0.,    0.,    0.,    0.,    0.,    0.,    6.,    7.,    8.,
                          9.,   10.,   11.,   24.,   26.,   28.,   30.,   32.,   34.,
                         54.,   57.,   60.,   63.,   66.,   69.,   96.,  100.,  104.,
                        108.,  112.,  116.]).reshape(&[5, 3, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_5() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(3).reshape(&[1, 3, 1]);
                let answer = T::new(vec![
                         0.,   0.,   2.,   3.,   8.,  10.,   0.,   0.,   8.,   9.,  20.,
                        22.,   0.,   0.,  14.,  15.,  32.,  34.,   0.,   0.,  20.,  21.,
                        44.,  46.,   0.,   0.,  26.,  27.,  56.,  58.]).reshape(&[5, 3, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_6() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(2).reshape(&[1, 1, 2]);
                let answer = T::new(vec![
                         0.,   1.,   0.,   3.,   0.,   5.,   0.,   7.,   0.,   9.,   0.,
                        11.,   0.,  13.,   0.,  15.,   0.,  17.,   0.,  19.,   0.,  21.,
                         0.,  23.,   0.,  25.,   0.,  27.,   0.,  29.]).reshape(&[5, 3, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_7() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::ones(&[1, 1, 1]);
                let answer = t1.clone();
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_8() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(3 * 2).reshape(&[3, 2]);
                let answer = T::new(vec![
                          0.,    1.,    4.,    9.,   16.,   25.,    0.,    7.,   16.,
                         27.,   40.,   55.,    0.,   13.,   28.,   45.,   64.,   85.,
                          0.,   19.,   40.,   63.,   88.,  115.,    0.,   25.,   52.,
                         81.,  112.,  145.]).reshape(&[5, 3, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tensor_broadcast_9() {
                let t1 = T::range(5 * 3 * 2).reshape(&[5, 3, 2]);
                let t2 = T::range(2);
                let answer = T::new(vec![
                         0.,   1.,   0.,   3.,   0.,   5.,   0.,   7.,   0.,   9.,   0.,
                        11.,   0.,  13.,   0.,  15.,   0.,  17.,   0.,  19.,   0.,  21.,
                         0.,  23.,   0.,  25.,   0.,  27.,   0.,  29.]).reshape(&[5, 3, 2]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_rf_tscalar_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let t2 = T::scalar(-2.);
                let answer = T::new(vec![0., -6., -4., -20., 6., 0.]).reshape(&[2, 3]);
                assert!(&t1 * &t2 == answer);
                assert!(&t2 * &t1 == answer);
            }

            #[test]
            fn rf_tensor_scalar_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let answer = T::new(vec![0., 9., 6., 30., -9., 0.]).reshape(&[2, 3]);
                assert!(&t1 * 3. == answer);
            }

            #[test]
            fn mv_tensor_scalar_1() {
                let t1 = T::new(vec![ 0., 3., 2., 10., -3., 0.]).reshape(&[2, 3]);
                let answer = T::new(vec![0., 9., 6., 30., -9., 0.]).reshape(&[2, 3]);
                assert!(t1 * 3. == answer);
            }
        }
    )
}

add_impl!(f32, float32);
add_impl!(f64, float64);
