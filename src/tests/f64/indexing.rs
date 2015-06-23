#[allow(unused_imports)]
use tensor::DoubleTensor;

#[test]
fn indexing() {
    let t = DoubleTensor::range(6);
    let v = t[2];
    assert_eq!(v, 2.0);
}

#[test]
fn indexing_mut() {
    let mut t = DoubleTensor::zeros(&[3]);
    t[2] = 1.0;
    assert!(t == DoubleTensor::new(vec![0.0, 0.0, 1.0]));
}

#[test]
fn indexing_tuple_1() {
    let t = DoubleTensor::range(6);
    let v = t[(2,)];
    assert_eq!(v, 2.0);
}

#[test]
fn indexing_tuple_2() {
    let t = DoubleTensor::range(6).reshaped(&[2, 3]);
    let v = t[(1, 2)];
    assert_eq!(v, 5.0);
}

#[test]
fn indexing_tuple_3() {
    let t = DoubleTensor::range(100).reshaped(&[2, 5, 10]);
    let v = t[(1, 2, 3)];
    assert_eq!(v, 73.0);
}
