#[allow(unused_imports)]
use tensor::Tensor;

#[test]
fn add_axis() {
    let a: Tensor<f64> = Tensor::range(2*3*4).reshape(&[2, 3, 4]);

    let answer0 = Tensor::new(vec![12., 14., 16., 18.,
                                   20., 22., 24., 26.,
                                   28., 30., 32., 34.]).reshape(&[3, 4]);

    let answer1 = Tensor::new(vec![12., 15., 18., 21.,
                                   48., 51., 54., 57.]).reshape(&[2, 4]);

    let answer2 = Tensor::new(vec![6., 22., 38.,
                                   54., 70., 86.]).reshape(&[2, 3]);

    let c0 = a.sum_axis(0);
    assert!(c0 == answer0);

    let c1 = a.sum_axis(1);
    assert!(c1 == answer1);

    let c2 = a.sum_axis(2);
    assert!(c2 == answer2);
}

#[test]
fn bitand_axis() {
    let a = Tensor::new(vec![true, true, true,
                             false, true, false]).reshape(&[2, 3]);

    let answer0 = Tensor::new(vec![false, true, false]);
    let answer1 = Tensor::new(vec![true, false]);

    let c0 = a.bitand_axis(0);
    assert!(c0 == answer0);

    let c1 = a.bitand_axis(1);
    assert!(c1 == answer1);
}
