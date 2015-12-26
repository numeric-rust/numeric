#![feature(test)]

#[macro_use]
extern crate numeric;
extern crate test;

#[bench]
fn tensor(bencher: &mut test::Bencher) {
    const T: f64 = 42.0;
    bencher.iter(|| {
        let tensor = tensor![
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
            T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T;
        ];
        test::black_box(tensor);
    });
}
