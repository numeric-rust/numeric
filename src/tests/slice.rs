macro_rules! add_impl {
    ($t:ty, $m:ident) => (
        mod $m {
            #[allow(unused_imports)]
            use tensor::{Tensor, AxisIndex};
            type T = Tensor<$t>;

            #[test]
            fn slice_1() {
                let t = T::range(200).reshape(&[2, 20, 5]);
                let t2 = t.slice(&[AxisIndex::SliceFrom(1),
                                   AxisIndex::SliceTo(5),
                                   AxisIndex::Slice(2, 4)]);
                let answer = T::new(vec![102.0, 103.0, 107.0, 108.0, 112.0,
                                              113.0, 117.0, 118.0, 122.0, 123.0]).reshape(&[1, 5, 2]);
                assert!(t2 == answer);
            }

            #[test]
            fn slice_2() {
                let t = T::range(210).reshape(&[7, 5, 2, 3]);
                let t2 = t.slice(&[AxisIndex::Slice(1, 3),
                                   AxisIndex::SliceTo(2),
                                   AxisIndex::Full,
                                   AxisIndex::Slice(1, 2)]);
                let answer = T::new(vec![31.0, 34.0, 37.0, 40.0,
                                              61.0, 64.0, 67.0, 70.0]).reshape(&[2, 2, 2, 1]);
                assert!(t2 == answer);
            }

            #[test]
            fn slice_index_1() {
                let t = T::range(210).reshape(&[7, 5, 2, 3]);
                let t2 = t.slice(&[AxisIndex::Index(2),
                                   AxisIndex::SliceTo(2),
                                   AxisIndex::Full,
                                   AxisIndex::Index(2)]);
                let answer = T::new(vec![62., 65., 68., 71.]).reshape(&[2, 2]);
                assert!(t2 == answer);
            }

            #[test]
            fn slice_ellipsis() {
                let t = T::range(3*2*3*7).reshape(&[3, 2, 3, 7]);
                let t2 = t.slice(&[AxisIndex::Index(1),
                                   AxisIndex::Ellipsis,
                                   AxisIndex::SliceFrom(5)]);
                let answer = T::new(vec![47., 48., 54., 55.,  61., 62.,
                                              68., 69., 75., 76.,  82., 83.]).reshape(&[2, 3, 2]);
                assert!(t2 == answer);
            }

            #[test]
            fn slice_implied_full() {
                let t = T::range(3*2*7*3).reshape(&[3, 2, 7, 3]);
                let t2 = t.slice(&[AxisIndex::Index(1),
                                   AxisIndex::Index(1),
                                   AxisIndex::Slice(1, 3)]);
                let answer = T::new(vec![66., 67., 68.,
                                              69., 70., 71.]).reshape(&[2, 3]);
                assert!(t2 == answer);
            }

            #[test]
            #[should_panic]
            fn slice_ellipsis_multiple() {
                let t = T::range(3*2*3*7).reshape(&[3, 2, 3, 7]);
                // Can't use more than one Ellipsis
                t.slice(&[AxisIndex::Ellipsis,
                          AxisIndex::Index(2),
                          AxisIndex::Ellipsis]);
            }

            #[test]
            fn slice_new_axis_empty() {
                let t = T::new(vec![]);
                assert!(t.slice(&[AxisIndex::NewAxis]).shape() == &[1, 0]);
                assert!(t.slice(&[AxisIndex::NewAxis, AxisIndex::NewAxis]).shape() == &[1, 1, 0]);
                assert!(t.slice(&[AxisIndex::Ellipsis, AxisIndex::NewAxis]).shape() == &[0, 1]);
                assert!(t.slice(&[AxisIndex::Full, AxisIndex::NewAxis]).shape() == &[0, 1]);
                assert!(t.slice(&[AxisIndex::NewAxis, AxisIndex::Ellipsis, AxisIndex::NewAxis]).shape() == &[1, 0, 1]);
            }

            #[test]
            fn slice_new_axis_multidimensional() {
                let t = T::range(210).reshape(&[7, 5, 2, 3]);

                assert!(t.slice(&[AxisIndex::NewAxis]).shape() == &[1, 7, 5, 2, 3]);
                assert!(t.slice(&[AxisIndex::Full,AxisIndex::NewAxis]).shape() == &[7, 1, 5, 2, 3]);
                assert!(t.slice(&[AxisIndex::Full, AxisIndex::Full, AxisIndex::NewAxis]).shape() == &[7, 5, 1, 2, 3]);
                assert!(t.slice(&[AxisIndex::Full, AxisIndex::NewAxis, AxisIndex::Full]).shape() == &[7, 1, 5, 2, 3]);
                assert!(t.slice(&[AxisIndex::Full, AxisIndex::NewAxis, AxisIndex::Full, AxisIndex::Full, AxisIndex::NewAxis]).shape() == &[7, 1, 5, 2, 1, 3]);
                assert!(t.slice(&[AxisIndex::Ellipsis, AxisIndex::NewAxis, AxisIndex::Full, AxisIndex::Full, AxisIndex::NewAxis]).shape() == &[7, 5, 1, 2, 3, 1]);
                assert!(t.slice(&[AxisIndex::NewAxis, AxisIndex::Ellipsis, AxisIndex::NewAxis, AxisIndex::NewAxis]).shape() == &[1, 7, 5, 2, 3, 1, 1]);
            }
        }
    )
}

add_impl!(f32, float32);
add_impl!(f64, float64);
