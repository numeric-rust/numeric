var searchIndex = {};
searchIndex["numeric"] = {"doc":"Numeric Rust provides a foundation for doing scientific computing with Rust. It aims to be for\nRust what Numpy is for Python.","items":[[0,"traits","numeric","Traits used by Tensor.",null,null],[8,"TensorTrait","numeric::traits","This is the basic trait that must be satisfied for basic elements used in `Tensor`.",null,null],[8,"NumericTrait","","`NumericTrait` extends `TensorTrait` to all the numeric types supported by `Tensor`\n(e.g. `u8` and `f32`).",null,null],[0,"tensor","numeric","The tensor module defines an N-dimensional matrix for use in scientific computing.",null,null],[3,"Tensor","numeric::tensor","An implementation of an N-dimensional matrix.\nA quick example:",null,null],[3,"TensorIterator","","",null,null],[4,"AxisIndex","","Used for advanced slicing of a `Tensor`.",null,null],[13,"Full","","Indexes from start to end for this axis.",0,null],[13,"Ellipsis","","Indexes from start to end for all axes in the middle. A maximum of one can be used.",0,null],[13,"NewAxis","","Creates a new axis of length 1 at this location.",0,null],[13,"Index","","Picks one element of an axis. This will remove that axis from the tensor.",0,null],[13,"StridedSlice","","Makes a strided slice `(start, end, step)`, with the same semantics as Python&#39;s Numpy. If\n`start` is specified as `None`, it will start from the first element if `step` is positive\nand last element if `step` is negative. If `end` is `None`, it will imply beyond the last\nelement if `step` is positive and one before the first element if `step` is negative.",0,null],[11,"dot","","Takes the product of two tensors. If the tensors are both matrices (2D), then a\nmatrix multiplication is taken. If the tensors are both vectors (1D), the scalar\nproduct is taken.",1,null],[11,"dot","","Takes the product of two tensors. If the tensors are both matrices (2D), then a\nmatrix multiplication is taken. If the tensors are both vectors (1D), the scalar\nproduct is taken.",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"fmt","","",1,null],[11,"add","","",1,null],[11,"add","","",1,null],[11,"add_with_out","","",1,null],[11,"add","","",1,null],[11,"sub","","",1,null],[11,"sub","","",1,null],[11,"sub_with_out","","",1,null],[11,"sub","","",1,null],[11,"mul","","",1,null],[11,"mul","","",1,null],[11,"mul_with_out","","",1,null],[11,"mul","","",1,null],[11,"div","","",1,null],[11,"div","","",1,null],[11,"div_with_out","","",1,null],[11,"div","","",1,null],[11,"rem","","",1,null],[11,"rem","","",1,null],[11,"rem_with_out","","",1,null],[11,"rem","","",1,null],[11,"bitand","","",1,null],[11,"bitand","","",1,null],[11,"bitand_with_out","","",1,null],[11,"bitand","","",1,null],[11,"bitor","","",1,null],[11,"bitor","","",1,null],[11,"bitor_with_out","","",1,null],[11,"bitor","","",1,null],[11,"bitxor","","",1,null],[11,"bitxor","","",1,null],[11,"bitxor_with_out","","",1,null],[11,"bitxor","","",1,null],[11,"neg","","",1,null],[11,"max","","",1,null],[11,"min","","",1,null],[11,"sum","","",1,null],[11,"mean","","",1,null],[11,"sum_axis","","",1,null],[11,"prod_axis","","",1,null],[11,"bitand_axis","","",1,null],[11,"bitor_axis","","",1,null],[11,"bitxor_axis","","",1,null],[11,"eq","","",1,null],[11,"index","","",1,null],[11,"index_mut","","",1,null],[11,"index","","",1,null],[11,"index_mut","","",1,null],[11,"index","","",1,null],[11,"index_mut","","",1,null],[11,"index","","",1,null],[11,"index_mut","","",1,null],[11,"index","","",1,null],[11,"index_mut","","",1,null],[11,"concat","","",1,{"inputs":[{"name":"tensor"},{"name":"tensor"},{"name":"usize"}],"output":{"name":"tensor"}}],[11,"convert","","Returns a new tensor with the elements converted to the selected type.",1,null],[11,"to_f32","","Short-hand for `convert::&lt;f32&gt;()`.",1,null],[11,"to_f64","","Short-hand for `convert::&lt;f64&gt;()`.",1,null],[11,"elem_gt","","",1,null],[11,"elem_ge","","",1,null],[11,"elem_lt","","",1,null],[11,"elem_le","","",1,null],[11,"elem_eq","","",1,null],[11,"elem_ne","","",1,null],[11,"all","","",1,null],[11,"any","","",1,null],[6,"DoubleTensor","","Type alias for `Tensor&lt;f64&gt;`",null,null],[6,"SingleTensor","","Type alias for `Tensor&lt;f32&gt;`",null,null],[11,"next","","",2,null],[11,"fmt","","",0,null],[11,"clone","","",0,null],[11,"as_ptr","","",1,null],[11,"as_mut_ptr","","",1,null],[11,"new","","Creates a new tensor from a `Vec` object. It will take ownership of the vector.",1,{"inputs":[{"name":"vec"}],"output":{"name":"tensor"}}],[11,"empty","","Creates a zero-filled tensor of the specified shape.",1,null],[11,"mem_slice","","",1,null],[11,"mem_slice_mut","","",1,null],[11,"slice","","Returns a flat slice of the tensor. Only works for canonical tensors.",1,null],[11,"slice_mut","","Returns a mutable flat slice of the tensor. Only works for canonical tensors.\nWill make a copy of the underyling data if the tensor is not unique.",1,null],[11,"iter","","",1,null],[11,"scalar","","Creates a Tensor representing a scalar",1,{"inputs":[{"name":"t"}],"output":{"name":"tensor"}}],[11,"is_scalar","","",1,null],[11,"scalar_value","","",1,null],[11,"filled","","Creates a new tensor of a given shape filled with the specified value.",1,null],[11,"shape","","Returns the shape of the tensor.",1,null],[11,"dim","","Returns length of single dimension.",1,null],[11,"data","","Returns a reference of the underlying data vector.",1,null],[11,"flatten","","Flattens the tensor to one-dimensional.",1,null],[11,"canonize","","Make a dense copy of the tensor. This means it will have default strides and no memory\noffset.",1,null],[11,"canonize_inplace","","",1,null],[11,"size","","Returns number of elements in the tensor.",1,null],[11,"ndim","","Returns the number of axes. This is the same as the length of the shape array.",1,null],[11,"index","","Takes slices (subsets) of tensors and returns a tensor as a new object. Uses the\n`AxisIndex` enum to specify indexing for each axis.",1,null],[11,"base","","Returns the underlying memory as a vector.",1,null],[11,"index_set","","Similar to `index`, except this updates the tensor with `other` instead of returning them.",1,null],[11,"bool_index","","",1,null],[11,"bool_index_set","","",1,null],[11,"unravel_index","","Takes a flatten index (if in row-major order) and returns a vector of the per-axis indices.",1,null],[11,"ravel_index","","Takes an array of per-axis indices and returns a flattened index (in row-major order).",1,null],[11,"reshape","","Reshapes the data. This moves the data, so no memory is allocated.",1,null],[11,"set","","Sets all the values according to another tensor.",1,null],[11,"swapaxes","","Swaps two axes.",1,null],[11,"transpose","","Transposes a matrix (for now, requires it to be 2D).",1,null],[11,"zeros","","Creates a zero-filled tensor of the specified shape.",1,null],[11,"ones","","Creates a one-filled tensor of the specified shape.",1,null],[11,"eye","","Creates an identity 2-D tensor (matrix). That is, all elements are zero except the diagonal\nwhich is filled with ones.",1,{"inputs":[{"name":"usize"}],"output":{"name":"tensor"}}],[11,"range","","Creates a new vector with integer values starting at 0 and counting up:",1,{"inputs":[{"name":"usize"}],"output":{"name":"tensor"}}],[11,"linspace","","Creates a new vector between two values at constant increments. The number of elements is\nspecified.",1,{"inputs":[{"name":"t"},{"name":"t"},{"name":"usize"}],"output":{"name":"tensor"}}],[11,"fscalar","","Creates a scalar specified as a `f64` and internally casted to `T`",1,{"inputs":[{"name":"f64"}],"output":{"name":"tensor"}}],[11,"clone","","",1,null],[0,"math","numeric","Contains mathematical functions that operate on tensors. These functions are largely modelled\nafter what is available natively in Rust.",null,null],[5,"ln","numeric::math","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"log10","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"log2","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"sin","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"cos","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"tan","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"asin","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"acos","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"atan","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"exp_m1","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"exp","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"exp2","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"ln_1p","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"sinh","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"cosh","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"tanh","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"asinh","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"acosh","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"atanh","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"sqrt","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"floor","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"ceil","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"round","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"trunc","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"fract","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"abs","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"signum","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"is_nan","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"is_finite","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"is_infinite","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"is_normal","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"is_sign_positive","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"is_sign_negative","","",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"log","","",null,{"inputs":[{"name":"tensor"},{"name":"t"}],"output":{"name":"tensor"}}],[5,"atan2","","Calculates atan(y/x).",null,{"inputs":[{"name":"tensor"},{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"powf","","",null,{"inputs":[{"name":"tensor"},{"name":"tensor"}],"output":{"name":"tensor"}}],[5,"powi","","",null,{"inputs":[{"name":"tensor"},{"name":"tensor"}],"output":{"name":"tensor"}}],[0,"random","numeric","The random module provides methods of randomizing tensors.",null,null],[3,"RandomState","numeric::random","",null,null],[11,"new","","Creates a new `RandomState` object with the given seed. The object needs to be captured\nas mutable in order to draw samples from it (since its internal state changes).",3,{"inputs":[{"name":"usize"}],"output":{"name":"randomstate"}}],[11,"uniform","","Generates a tensor by independently drawing samples from a uniform distribution in the\nrange [`low`, `high`). This is appropriate for integer types as well.",3,null],[11,"normal","","Generates a tensor by independently drawing samples from a standard normal.",3,null],[0,"linalg","numeric","Linear algebra functions.",null,null],[5,"diag","numeric::linalg","If passed a vector, creates a diagonal matrix with the vector as its diagonal.\nIf passed a matrix, the diagonal is extracted and returned.",null,{"inputs":[{"name":"tensor"}],"output":{"name":"tensor"}}],[11,"solve","numeric::tensor","Solves the linear equation `Ax = b` and returns `x`. The matrix `A` is `self` and\nmust be a square matrix. The input `b` must be a vector.",1,null],[11,"solve","","Solves the linear equation `Ax = b` and returns `x`. The matrix `A` is `self` and\nmust be a square matrix. The input `b` must be a vector.",1,null],[11,"svd","","Performs a singular value decomposition on the matrix.",1,null],[11,"svd","","Performs a singular value decomposition on the matrix.",1,null],[0,"io","numeric","Saving and loading data to and from disk.",null,null],[5,"load_hdf5_as_u8","numeric::io","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_u16","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_u32","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_u64","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_i8","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_i16","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_i32","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_i64","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_f32","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_f64","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_isize","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[5,"load_hdf5_as_usize","","Load HDF5 file and convert to specified type.",null,{"inputs":[{"name":"path"},{"name":"str"}],"output":{"name":"result"}}],[11,"save_hdf5","numeric::tensor","Saves tensor to an HDF5 file.",1,null],[11,"save_hdf5","","Saves tensor to an HDF5 file.",1,null],[11,"save_hdf5","","Saves tensor to an HDF5 file.",1,null],[11,"save_hdf5","","Saves tensor to an HDF5 file.",1,null],[11,"save_hdf5","","Saves tensor to an HDF5 file.",1,null],[11,"save_hdf5","","Saves tensor to an HDF5 file.",1,null],[11,"save_hdf5","","Saves tensor to an HDF5 file.",1,null],[11,"save_hdf5","","Saves tensor to an HDF5 file.",1,null],[11,"save_hdf5","","Saves tensor to an HDF5 file.",1,null],[11,"save_hdf5","","Saves tensor to an HDF5 file.",1,null],[14,"tensor!","numeric","Macro for creating vectors and matrices.",null,null]],"paths":[[4,"AxisIndex"],[3,"Tensor"],[3,"TensorIterator"],[3,"RandomState"]]};
initSearch(searchIndex);