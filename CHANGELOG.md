# Changelog
Numeric does not use semantic versioning, at least not currently for 0.x.y.
Numeric is still experimental and breaking changes occur often. If I break
more things than usual, I will try to bump the minor (x).

The goal of this changelog is to facilitate upgrading the version of Numeric,
despite these frequent breaking changes.

## 0.1.0
Released: 2015-12-25
* Changed internal storage to using `Rc` with copy-on-write semantics
* Added HDF5 support (adds `hdf5-sys` as a dependency)
* Dot product between two vectors results in a proper scalar
* Added `as_ptr` and `as_mut_ptr`
* Added in-place updates (e.g. `mul_with_out`)
* Moved `set` to `set2` and `get` to `get2`
* Added a new `set` that unrelated to the old `set`
* Added `mean`
* Renamed `slice` to `index` (as well as `_set`)
* Renamed `bool_slice` to `bool_index` (as well as (`_set`)
* Added `slice` and `slice_mut` (that returns actual Rust slices)
* Scalars are displayed without decoration

## 0.0.7
Released: 2015-07-28
* Added `tensor!` macro
* Renamed `reshaped` to `reshape`
* Added `abs`
* Switch `blas-sys`/`lapack-sys` dependencies to `blas`/`lapack`
* Added `fscalar`
* Added standard normal random generation
* Added more math functions (e.g. `floor`, `atan2`)
* Math functions now take ownership
* Added `powf` and `powi`
* Fixed `to_f64` (accidentally named `to_f65` in 0.0.6)

## 0.0.6
Released: 2015-07-05
* Made dot a member function (better generic handling)
* Added element-wise operations
* Added bitwise operations
* Added comparative operations (element-wise)
* Added reduction operations (e.g. `sum`, `max`)
* Added `slice_set`
* Made scalars zero-dimensional
* Added `Neg` trait

## 0.0.5
Released: 2015-06-27
* Better display code
* Added matrix solver (LAPACK)

## 0.0.4
Released: 2015-06-27
* Added math functions
* Added random number generation
* Added min/max
* Added concatenation
* Improved indexing functions
* Improved support for various types

## 0.0.3
Released: 2015-06-23
* Made tensors generic
* Improved dot product

## 0.0.2
Released: 2015-06-23
* First Cargo release
* Improved documentation

## 0.0.1
* First release
