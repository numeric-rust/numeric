[![Crates.io](https://img.shields.io/crates/v/numeric.svg)](https://crates.io/crates/numeric)

# Numeric Rust

General-purpose N-dimensional matrix class for Rust. It links to OpenBLAS and
LAPACK to make tensor operations fast (such as matrix multiplications and
linear solvers). It utilizes Rust's move semantics as much as possible to avoid
unnecessary copies.

## Documentation

* http://numeric.rs/doc/numeric/index.html

## Features

Some of the completed and planned features:

* [x] Element-wise addition, subtraction, multiplication, division
* [x] Matrix multiplication and scalar product
* [x] Indexing
* [x] Slicing
* [x] Generic (anything from `Tensor<bool>` to `Tensor<f64>`)
* [x] Mathematical functions
* [x] Linear solver
* [x] Basic random number generation
* [x] Creation macro
* [x] Updating slices
* [x] Saving/loading HDF5
* [x] Strided slices
* [x] Broadcasted axes
* [x] Basic support for complex numbers
* [x] Singular Value Decomposition
* [ ] Matrix inverse

Recent progress is summarized in [CHANGELOG.md](CHANGELOG.md). For planned
features, take a look at [TODO.md](TODO.md).

## Example

```rust
#[macro_use(tensor)]
extern crate numeric;

use numeric::Tensor;

fn main() {
    let a: Tensor<f64> = Tensor::range(6).reshape(&[2, 3]);
    let b = tensor![7.0, 3.0, 2.0; -3.0, 2.0, -5.0];
    let c = tensor![7.0, 3.0, 2.0];

    let d = &a + &b;         // a new tensor is returned
    println!("d = {}", d);

    let e = a.dot(&c);       // matrix multiplication (returns a new tensor)
    println!("e = {}", e);

    let f = a + &b;          // a is moved (no new memory is allocated)
    println!("f = {}", f);

    // Higher-dimensional
    let g: Tensor<f64> = Tensor::ones(&[2, 3, 4, 5]);
    println!("g = {}", g);
}
```

Output:

```
d =
 7 4 4
 0 6 0
[Tensor<f64> of shape 2x3]
e =
  7 43
[Tensor<f64> of shape 2]
f =
 7 4 4
 0 6 0
[Tensor<f64> of shape 2x3]
g =
...
[Tensor<f64> of shape 2x3x4x5]
```

## Contribute

We love pull requests and there are tons of things to work on for this project.
If you want suggestions for contributions, check out [TODO.md](TODO.md) (a
non-exhaustive list of what would be useful additions). Add your name to the
[CONTRIBUTORS.md](CONTRIBUTORS.md) file as part of your PR, no matter how small
it may seem.

## Acknowledgements

Numeric Rust is primarily inspired by Numpy and borrows heavily from that
project. Even the name is a play on **Num**eric **Py**thon. Another source of
some inspiration is Torch7.
