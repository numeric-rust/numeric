//! Linear algebra functions.
//!
//! The functions are often implemented as member functions of `Tensor`, since it offers better
//! handling of generics.
//!
//! # Solving a linear equation
//!
//! ```
//! use numeric::Tensor;
//!
//! let a = Tensor::new(vec![1.0_f64, 0.5, 1.5, -1.0]).reshape(&[2, 2]);
//! let b = Tensor::ones(&[2]);
//!
//! let x = a.solve(&b);
//! ```

mod solve;
mod svd;
