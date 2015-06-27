
# Numeric Rust

N-dimensional matrix class for Rust 1.0. It links to OpenBLAS to make tensor
operations fast (like matrix multipliations). It utilizes Rust's move semantics
as much as possible to avoid unnecessary copies.

## Features

Some of the completed and planned features:

* [x] Element-wise addition, subtraction, multiplication, division
* [x] Matrix multiplication and scalar product
* [x] Indexing
* [x] Slicing
* [x] Generic (anything from `Tensor<bool>` to `Tensor<f64>`)
* [x] Mathematical functions
* [ ] Strided slices
* [ ] Updating slices
* [ ] Broadcasted axes
* [ ] Matrix solver / inverse

## Example

```rust
use numeric::DoubleTensor;

fn main() {
    type T = Tensor<f64>;

    let a = T::range(6).reshaped(&[2, 3]);
    let b = T::new(vec![7.0, 3.0, 2.0, -3.0, 2.0, -5.0]).reshaped(&[2, 3]);
    let c = T::new(vec![7.0, 3.0, 2.0]);

    let d = &a + &b;         // a copy is made
    println!("d = {}", d);

    let e = T::dot(&a, &c);  // matrix multiplication (returns a new tensor)
    println!("e = {}", e);

    let f = a + &b;          // a is moved (no memory is allocated)
    println!("f = {}", f);

    // Higher-dimensional
    println!("g = {}", T::ones(&[2, 3, 4, 5]));
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

## Acknowledgement

Borrowing shamelessly from the great projects Numpy and Torch7.
