//! IO module
//!
//! ```
//! // Load HDF5 example here
//! ```

extern crate std;

use libc::{c_char, c_void, c_ulonglong, c_int};
use std::path::Path;
use hdf5_sys as ffi;

use tensor::Tensor;

#[allow(non_camel_case_types)]
type hsize_t = c_ulonglong;
#[allow(non_camel_case_types)]
type hid_t = c_int;

extern fn error_handler(_: hid_t, _: *const c_void) {
    // Suppress errors. We will rely on return statuses alone.
}

macro_rules! add_save {
    ($t:ty, $h5type:expr) => (
        impl Tensor<$t> {
            pub fn save_hdf5(&self, path: &Path) -> std::io::Result<()> {
                let filename = match path.to_str() {
                    Some(v) => v,
                    None => {
                        let err = std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("Path could not be converted to string: {:?}", path));
                        return Err(err);
                    },
                };
                // This could be made an option
                let group = "data";

                unsafe {
                    let filename_cstr = try!(::std::ffi::CString::new(filename));
                    let group_cstr = try!(::std::ffi::CString::new(group));

                    ffi::H5Eset_auto2(0, error_handler, 0 as *const c_void);

                    let file = ffi::H5Fcreate(filename_cstr.as_ptr() as *const c_char,
                                   ffi::H5F_ACC_TRUNC, ffi::H5P_DEFAULT, ffi::H5P_DEFAULT);

                    let mut shape: Vec<u64> = Vec::new();
                    for s in self.shape().iter() {
                        shape.push(*s as u64);
                    }

                    let space = ffi::H5Screate_simple(shape.len() as i32, shape.as_ptr(), std::ptr::null());

                    let dset = ffi::H5Dcreate2(file, group_cstr.as_ptr() as *const c_char,
                                               $h5type, space,
                                               ffi::H5P_DEFAULT,
                                               ffi::H5P_DEFAULT,
                                               ffi::H5P_DEFAULT);

                    let status = ffi::H5Dwrite(dset, $h5type, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                               self.as_ptr() as * const c_void);

                    if status < 0 {
                        let err = std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to write '{}': {:?}", group, path));
                        return Err(err);
                    }


                    ffi::H5Dclose(dset);
                    ffi::H5Fclose(file);
                }
                Ok(())
            }
        }
    )
}

add_save!(u8, ffi::H5T_NATIVE_UINT8);
add_save!(u16, ffi::H5T_NATIVE_UINT16);
add_save!(u32, ffi::H5T_NATIVE_UINT32);
add_save!(u64, ffi::H5T_NATIVE_UINT64);
add_save!(i8, ffi::H5T_NATIVE_INT8);
add_save!(i16, ffi::H5T_NATIVE_INT16);
add_save!(i32, ffi::H5T_NATIVE_INT32);
add_save!(i64, ffi::H5T_NATIVE_INT64);
add_save!(f32, ffi::H5T_NATIVE_FLOAT);
add_save!(f64, ffi::H5T_NATIVE_DOUBLE);


macro_rules! add_load {
    ($name:ident, $t:ty) => (
        pub fn $name(path: &Path, group: &str) -> std::io::Result<Tensor<$t>> {
            let filename = match path.to_str() {
                Some(v) => v,
                None => {
                    let err = std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("Path could not be converted to string: {:?}", path));
                    return Err(err);
                },
            };
            unsafe {
                let filename_cstr = try!(::std::ffi::CString::new(filename));
                let group_cstr = try!(::std::ffi::CString::new(group));


                ffi::H5Eset_auto2(0, error_handler, 0 as *const c_void);

                let file = ffi::H5Fopen(filename_cstr.as_ptr() as *const c_char,
                               ffi::H5F_ACC_RDONLY, ffi::H5P_DEFAULT);

                if file < 0 {
                    let err = std::io::Error::new(std::io::ErrorKind::NotFound, format!("File not found: {:?}", path));
                    return Err(err);
                }

                let dset = ffi::H5Dopen2(file, group_cstr.as_ptr() as *const c_char,
                                        ffi::H5P_DEFAULT);

                if dset < 0 {
                    let err = std::io::Error::new(std::io::ErrorKind::NotFound, format!("Group '{}' not found: {}", group, filename));
                    return Err(err);
                }

                let datatype = ffi::H5Dget_type(dset);

                let space = ffi::H5Dget_space(dset);
                let ndims = ffi::H5Sget_simple_extent_ndims(space);

                let mut shape: Tensor<hsize_t> = Tensor::zeros(&[ndims as usize]);

                if ffi::H5Sget_simple_extent_dims(space, shape.as_mut_ptr(), 0 as *mut hsize_t) != ndims {
                    let err = std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Could not read shape of tesor: {}", filename));
                    return Err(err);
                }

                //let unsigned_shape: Vec<usize> = shape.iter().map(|x| x as usize).collect();
                let unsigned_tensor = shape.convert::<usize>();
                let unsigned_shape = &unsigned_tensor.data();

                let data: Tensor<$t> = {
                    if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_UINT8) == 1 {
                        let mut native_data: Tensor<u8> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_UINT8, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_INT8) == 1 {
                        let mut native_data: Tensor<i8> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_INT8, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_UINT16) == 1 {
                        let mut native_data: Tensor<u16> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_UINT16, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_INT16) == 1 {
                        let mut native_data: Tensor<i16> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_INT16, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_UINT32) == 1 {
                        let mut native_data: Tensor<u32> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_UINT32, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_INT32) == 1 {
                        let mut native_data: Tensor<i32> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_INT32, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_UINT64) == 1 {
                        let mut native_data: Tensor<u64> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_UINT64, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_INT64) == 1 {
                        let mut native_data: Tensor<i64> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_INT64, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_FLOAT) == 1 {
                        let mut native_data: Tensor<f32> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_FLOAT, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else if ffi::H5Tequal(datatype, ffi::H5T_NATIVE_DOUBLE) == 1 {
                        let mut native_data: Tensor<f64> = Tensor::empty(&unsigned_shape[..]);
                        // Finally load the actual data
                        ffi::H5Dread(dset, ffi::H5T_NATIVE_DOUBLE, ffi::H5S_ALL, ffi::H5S_ALL, ffi::H5P_DEFAULT,
                                     native_data.as_mut_ptr() as *mut c_void);
                        native_data.convert::<$t>()
                    } else {
                        let err = std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unable to convert '{}' to {}: {}", group, "f64", filename));
                        return Err(err);
                    }
                };

                ffi::H5Tclose(datatype);
                ffi::H5Dclose(dset);
                ffi::H5Fclose(file);

                Ok(data)
            }
        }
    )
}

add_load!(load_hdf5_as_u8, u8);
add_load!(load_hdf5_as_u16, u16);
add_load!(load_hdf5_as_u32, u32);
add_load!(load_hdf5_as_u64, u64);
add_load!(load_hdf5_as_i8, i8);
add_load!(load_hdf5_as_i16, i16);
add_load!(load_hdf5_as_i32, i32);
add_load!(load_hdf5_as_i64, i64);
add_load!(load_hdf5_as_f32, f32);
add_load!(load_hdf5_as_f64, f64);
add_load!(load_hdf5_as_isize, isize);
add_load!(load_hdf5_as_usize, usize);
