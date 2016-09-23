macro_rules! add_tests {
    ($t:ty, $m:ident) => (
        mod $m {
            use numeric::{Tensor, AxisIndex};
            type T = Tensor<$t>;

            #[test]
            fn svd_eye() {
                let eye = T::eye(3);
                let (u, s, vt) = eye.svd(true);
                assert!(u == eye);
                assert!(vt == eye);
                assert!(s == T::ones(&[3]));
            }

            // TODO: Needs more tests
        }
    )
}

add_tests!(f32, float32);
add_tests!(f64, float64);
