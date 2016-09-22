use numeric::Tensor;

#[test]
fn tensor_1d() {
    let x = Tensor::new(vec![1, 2, 3, 4, 5, 6]);
    assert!(x == tensor![1, 2, 3, 4, 5, 6]);
    assert!(x == tensor![1, 2, 3, 4, 5, 6,]);
}

#[test]
fn tensor_2d() {
    let x = Tensor::new(vec![1, 2, 3, 4, 5, 6]).reshape(&[3, 2]);
    assert!(x == tensor![1, 2; 3, 4; 5, 6]);
    assert!(x == tensor![1, 2; 3, 4; 5, 6;]);
}

#[test]
fn tensor_filled_1d() {
    let x = Tensor::new(vec![3, 3, 3, 3, 3]);
    assert!(x == tensor![3; 5]);
}
