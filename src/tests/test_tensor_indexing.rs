#[allow(unused_imports)]
use tensor::Tensor;

#[test]
fn test_tensor_indexing() {
    let t = Tensor::range(6);
    let v = t[2];
    assert_eq!(v, 2.0);
}

#[test]
fn test_tensor_indexing_mut() {
    let mut t = Tensor::zeros(&[3]);
    t[2] = 1.0;
    assert!(t == Tensor::new(vec![0.0, 0.0, 1.0]));
}

#[test]
fn test_tensor_indexing_tuple_1() {
    let t = Tensor::range(6);
    let v = t[(2,)];
    assert_eq!(v, 2.0);
}

#[test]
fn test_tensor_indexing_tuple_2() {
    let t = Tensor::range(6).reshaped(&[2, 3]);
    let v = t[(1, 2)];
    assert_eq!(v, 5.0);
}

#[test]
fn test_tensor_indexing_tuple_3() {
    let t = Tensor::range(100).reshaped(&[2, 5, 10]);
    let v = t[(1, 2, 3)];
    assert_eq!(v, 73.0);
}
