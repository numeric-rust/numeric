#[allow(unused_imports)]
use tensor::DoubleTensor;

#[test]
fn tensor_tensor_1() {
    let t1 = DoubleTensor::new(vec![5.0, 3.0, 0.0, 10.0, -3.0, 0.0]);
    let t2 = DoubleTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = t1 / t2;
    assert!(t == DoubleTensor::new(vec![5.0,  1.5,  0.0,  2.5, -0.6,  0.0]));
}

#[test]
fn tensor_tensor_2() {
    let t1 = DoubleTensor::new(vec![5.0, 3.0, 0.0, 10.0, -3.0, 0.0]);
    let t2 = DoubleTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = t1 / &t2;
    assert!(t == DoubleTensor::new(vec![5.0,  1.5,  0.0,  2.5, -0.6,  0.0]));
}

#[test]
fn tensor_tensor_3() {
    let t1 = DoubleTensor::new(vec![5.0, 3.0, 0.0, 10.0, -3.0, 0.0]);
    let t2 = DoubleTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = &t1 / &t2;
    assert!(t == DoubleTensor::new(vec![5.0,  1.5,  0.0,  2.5, -0.6,  0.0]));
}

#[test]
#[should_panic]
fn tensor_tensor_mismatched_1() {
    let t1 = DoubleTensor::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshaped(&[2, 3]);
    let t2 = DoubleTensor::new(vec![-2.0, 1.0, 8.0,  0.0]).reshaped(&[2, 2]);
    t1 / t2; // mis-matched shape
}

#[test]
fn tensor_scalar_1() {
    let t1 = DoubleTensor::new(vec![ 0.0, 3.0, 9.0, -3.0, -3.0, 0.0]);
    let t = t1 / 3.0;
    assert!(t == DoubleTensor::new(vec![0.0, 1.0, 3.0, -1.0, -1.0, 0.0]));
}

/*
#[test]
fn tensor_scalar_2() {
    let t1 = DoubleTensor::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshaped(&[2, 3]);
    let t = &t1 / 3.0;
    println!("{:?}", t1.shape());
    assert!(t == DoubleTensor::new(vec![3.0, 6.0, 5.0, 13.0, 0.0, 3.0]).reshaped(&[2, 3]));
}
*/
