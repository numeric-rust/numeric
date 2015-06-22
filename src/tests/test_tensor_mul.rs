#[allow(unused_imports)]
use tensor::Tensor;

#[test]
fn test_mul_tensor_tensor_1() {
    let t1 = Tensor::new_data(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]);
    let t2 = Tensor::new_data(vec![-2.0, 1.0, 8.0,  0.0, -1.0, 2.0]);
    let t = t1 * t2;
    assert!(t == Tensor::new_data(vec![0.0, 3.0, 16.0, 0.0, 3.0, 0.0]));
}

#[test]
fn test_mul_tensor_tensor_2() {
    let t1 = Tensor::new_data(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]);
    let t2 = Tensor::new_data(vec![-2.0, 1.0, 8.0,  0.0, -1.0, 2.0]);
    let t = t1 * &t2;
    assert!(t == Tensor::new_data(vec![0.0, 3.0, 16.0, 0.0, 3.0, 0.0]));
}

#[test]
fn test_mul_tensor_tensor_3() {
    let t1 = Tensor::new_data(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]);
    let t2 = Tensor::new_data(vec![-2.0, 1.0, 8.0,  0.0, -1.0, 2.0]);
    let t = &t1 * &t2;
    assert!(t == Tensor::new_data(vec![0.0, 3.0, 16.0, 0.0, 3.0, 0.0]));
}

#[test]
#[should_panic]
fn test_mul_tensor_tensor_mismatched_1() {
    let t1 = Tensor::new_data(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshaped(&[2, 3]);
    let t2 = Tensor::new_data(vec![-2.0, 1.0, 8.0,  0.0]).reshaped(&[2, 2]);
    t1 * t2; // mis-matched shape
}

#[test]
fn test_mul_tensor_scalar_1() {
    let t1 = Tensor::new_data(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]);
    let t = t1 * 3.0;
    assert!(t == Tensor::new_data(vec![0.0, 9.0, 6.0, 30.0, -9.0, 0.0]));
}

/*
#[test]
fn test_mul_tensor_scalar_2() {
    let t1 = Tensor::new_data(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshaped(&[2, 3]);
    let t = &t1 + 3.0;
    println!("{:?}", t1.shape());
    assert!(t == Tensor::new_data(vec![3.0, 6.0, 5.0, 13.0, 0.0, 3.0]).reshaped(&[2, 3]));
}
*/
