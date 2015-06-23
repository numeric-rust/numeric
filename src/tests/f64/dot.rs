#[allow(unused_imports)]
use tensor::Tensor;

#[test]
fn matrix_matrix_1() {
    // Test square matrices
    let t1 = Tensor::new(vec![ 0.0, 3.0, 2.0, 10.0]).reshaped(&[2, 2]);
    let t2 = Tensor::new(vec![-2.0, 1.0, 8.0,  0.0]).reshaped(&[2, 2]);
    let answer = Tensor::new(vec![24.0, 0.0, 76.0, 2.0]).reshaped(&[2, 2]);

    println!("t1 = \n{}", t1);
    println!("t2 = \n{}", t2);
    println!("answer = \n{}", Tensor::dot(&t1, &t2));
    assert!(Tensor::dot(&t1, &t2) == answer);
}

#[test]
fn matrix_matrix_2() {
    let t1 = Tensor::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshaped(&[3, 2]);
    let t2 = Tensor::new(vec![-2.0, 1.0, 8.0,  0.0]).reshaped(&[2, 2]);
    let answer = Tensor::new(vec![24.0, 0.0, 76.0, 2.0, 6.0, -3.0]).reshaped(&[3, 2]);

    println!("t1 = \n{}", t1);
    println!("t2 = \n{}", t2);
    println!("answer = \n{}", Tensor::dot(&t1, &t2));
    assert!(Tensor::dot(&t1, &t2) == answer);
}

#[test]
fn matrix_vector_1() {
    let t1 = Tensor::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]).reshaped(&[3, 2]);
    //let t1 = Tensor::new(vec![ 0.0, 2.0, -3.0, 3.0, 10.0, 0.0]).reshaped(&[3, 2]);
    let t2 = Tensor::new(vec![-2.0, 1.0]);
    let answer = Tensor::new(vec![3.0, 6.0, 6.0]);

    println!("t1 = \n{}", t1);
    println!("t2 = \n{}", t2);
    println!("answer = \n{}", Tensor::dot(&t1, &t2));
    assert!(Tensor::dot(&t1, &t2) == answer);
}

#[test]
fn vector_vector_1() {
    let t1 = Tensor::new(vec![ 0.0, 3.0, 2.0, 10.0, -3.0, 0.0]);
    let t2 = Tensor::new(vec![-2.0, 1.0, 3.0, -2.0, -3.0, 1.0]);
    let answer = Tensor::new(vec![-2.0]);

    println!("t1 = \n{}", t1);
    println!("t2 = \n{}", t2);
    println!("answer = \n{}", Tensor::dot(&t1, &t2));
    assert!(Tensor::dot(&t1, &t2) == answer);
}
