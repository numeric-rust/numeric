#[allow(unused_imports)]
use tensor::Tensor;

#[test]
fn test_creation_ones_1() {
    let t1 = Tensor::ones(&[4]);
    assert_eq!(t1.shape(), &[4]);
    assert_eq!(t1.data(), &vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_creation_ones_2() {
    let t2 = Tensor::ones(&[2, 1]);
    assert_eq!(t2.shape(), &[2, 1]);
    assert_eq!(t2.data(), &vec![1.0, 1.0]);
}

#[test]
fn test_creation_ones_3() {
    let t3 = Tensor::ones(&[2, 1, 3]);
    assert_eq!(t3.shape(), &[2, 1, 3]);
    assert_eq!(t3.data(), &vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_creation_zeros_3() {
    let t3 = Tensor::zeros(&[2, 1, 3]);
    assert_eq!(t3.shape(), &[2, 1, 3]);
    assert_eq!(t3.data(), &vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_creation_range() {
    let t1 = Tensor::range(4);
    assert_eq!(t1.shape(), &[4]);
    assert_eq!(t1.data(), &vec![0.0, 1.0, 2.0, 3.0]);

    let t2 = Tensor::range(0);
    assert_eq!(t2.shape(), &[0]);
    assert_eq!(t2.data().len(), 0);
}

#[test]
fn test_partial_eq_1() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 2.0, -3.0]).reshaped(&[3, 2]);
    let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 2.0, -3.0]).reshaped(&[3, 2]);
    assert!(t1 == t2);
}

#[test]
fn test_partial_eq_2() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 2.0, -3.0]).reshaped(&[3, 2]);
    let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 2.0, -3.0]).reshaped(&[2, 3]);
    let p = t1 != t2;
    assert!(p);
}

#[test]
fn test_partial_eq_3() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 2.0, -3.0]).reshaped(&[3, 2]);
    let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 3.0, -3.0]).reshaped(&[3, 2]);
    assert!(t1 != t2);
}
