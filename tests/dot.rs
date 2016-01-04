macro_rules! add_impl {
    ($t:ty, $m:ident) => (
        mod $m {
            use numeric::Tensor;
            type T = Tensor<$t>;

            #[test]
            fn matrix_matrix_1() {
                // Test square matrices
                let t1 = T::new(vec![ 0.0, 3.0, 2.0, 10.0]).reshape(&[2, 2]);
                let t2 = T::new(vec![-2.0, 1.0, 8.0,  0.0]).reshape(&[2, 2]);
                let answer = T::new(vec![24.0, 0.0, 76.0, 2.0]).reshape(&[2, 2]);
                assert!(t1.dot(&t2) == answer);
            }

            #[test]
            fn matrix_matrix_2() {
                let t1 = T::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshape(&[3, 2]);
                let t2 = T::new(vec![-2.0, 1.0, 8.0,  0.0]).reshape(&[2, 2]);
                let answer = T::new(vec![24.0, 0.0, 76.0, 2.0, 6.0, -3.0]).reshape(&[3, 2]);
                assert!(t1.dot(&t2) == answer);
            }

            #[test]
            fn matrix_vector_1() {
                let t1 = T::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshape(&[3, 2]);
                //let t1 = T::new(vec![ 0.0, 2.0, -3.0, 3.0, 10.0, 0.0]).reshape(&[3, 2]);
                let t2 = T::new(vec![-2.0, 1.0]);
                let answer = T::new(vec![3.0, 6.0, 6.0]);
                assert!(t1.dot(&t2) == answer);
            }

            #[test]
            fn vector_vector_1() {
                let t1 = T::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]);
                let t2 = T::new(vec![-2.0, 1.0, 3.0, -2.0, -3.0, 1.0]);
                let answer = T::fscalar(-2.0);
                assert!(t1.dot(&t2) == answer);
            }
        }
    )
}

add_impl!(f32, float32);
add_impl!(f64, float64);
