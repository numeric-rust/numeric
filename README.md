
# Numeric Rust

N-dimensional matrix class for Rust 1.0. It links to OpenBLAS to give fast 
operations like the matrix multiplication. It utilizes Rust's move semantics 
as much as possible to avoid unnecessary copies.

## Features

Some of the completed and planned features:

* [x] Element-wise addition, subtraction, multiplication, division
* [x] Matrix multiplication and scalar product
* [ ] Indexing
* [ ] Slicing
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

    let d = &a + &b;                // copy is created
    println!("d = \n{}", d);

    let e = Tensor::dot(&a, &c);    // matrix multiplication
    println!("e = \n{}", e);

    let f = a + &b;                 // a is moved (no memory allocated)
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

