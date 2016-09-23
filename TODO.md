# TODO

There are many things that can be done to extend Numeric. These are some of the
more immediately useful things. Feel free to pick one and start working on a PR.

## Core

* Make `index_set` faster
* Make `index_set` broadcastable
* Make elementwise ops that move tensors broadcastable
* Make elementwise binary functions broadcastable
* Extensive testing and units tests for complex numbers
* Improve display function
  * Display higher-dimensional
  * Align at periods
  * Make it prettier

## Tensor functions

* Sort
* Arg-sort
* Make math functions broadcastable

## Sorting and searching

* Sort
* Argsort
* Find indices based on condition

## Linear algebra

* Unit tests for `solve`, `dot` and `diag`
* Matrix inverse and pseudo-inverse
* Determinant
* Condition number
* Trace

## Random number generation

* Multivariate normal
* Shuffle

## Interop

* Numpy
* C

## Documentation

* Create chapters
  * Basic creation
  * Copy-on-write and move semantics
  * Indexing
* Create logo
* Improve landing page at numeric.rs
