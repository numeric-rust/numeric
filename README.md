
# Numeric Rust

N-dimensional matrix class for Rust 1.0. It links to OpenBLAS to give fast 
operations like the matrix multiplication. It utilizes Rust's move semantics 
as much as possible to avoid unnecessary copies.

## Features

Some of the completed and planned features:

* [x] Element-wise addition, subtraction, multiplication, division
* [x] Matrix multiplication and scalar product
* [x] Indexing
* [x] Slicing
* [ ] Strided slices
* [ ] Updating slices
* [ ] Different types (currently `f64` only)
* [ ] Broadcasted axes
* [ ] Matrix solver / inverse

## Example

```rust
use numeric::tensor::Tensor;

fn main() {
    let a = Tensor::range(6).reshaped(&[2, 3]);
    let b = Tensor::new(vec![7.0, 3.0, 2.0, -3.0, 2.0, -5.0]).reshaped(&[2, 3]);
    let c = Tensor::new(vec![7.0, 3.0, 2.0]);

    let d = &a + &b;                // a copy is made
    println!("d = \n{}", d);

    let e = Tensor::dot(&a, &c);    // matrix multiplication (returns a new tensor)
    println!("e = \n{}", e);

    let f = a + &b;                 // a is moved (no memory is allocated)
    println!("f = \n{}", f);

    // Higher-dimensional
    println!("g = \n{}", Tensor::ones(&[2, 3, 4, 5]));
}
```

Output:

```
d =
[[  7.00   4.00   4.00]
 [  0.00   6.00   0.00]]
e =
[  7.00  43.00]
f =
[[  7.00   4.00   4.00]
 [  0.00   6.00   0.00]]
g =
Tensor([2, 3, 4, 5])    
```
