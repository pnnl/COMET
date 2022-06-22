

//! A tensor object (e.g., A) refers to a multi-dimensional array of arithmetic values that can be accessed by using indexing values.

use crate::index::*;

/// Intended to be generic over various arthimetic types, currently only support f64. 
/// Tensor are contructed using either dense or sparse representations.
/// In addition to the represenetation, a tensor also has a shape, which is a list of indices that define the dimensions of the tensor.
/// Tensor data can be loaded from a file, can be filled with a constant value, or can be randomly initialized.
/// Note that the eDSL is not pure rust, so this documentation serves to capture the functionality as best as possible with in the bounds of the Rust language
pub struct Tensor<T> {
    _temp: std::marker::PhantomData<T>
}
impl <T: Default> Tensor<T> {
    /// creates a dense representation of a tensor with shape defined by the array of indicies
    pub fn dense(_index_list: &[Index]) -> Tensor<T> {
        Tensor {_temp: std::marker::PhantomData}
    }
    /// creates a csr representation of a tensor with shape defined by the array of indicies
    pub fn csr(_index_list: &[Index]) -> Tensor<T> {
        Tensor {_temp: std::marker::PhantomData}
    }
    /// creates a dcsr representation of a tensor with shape defined by the array of indicies
    pub fn dcsr(_index_list: &[Index]) -> Tensor<T> {
        Tensor {_temp: std::marker::PhantomData}
    }
    /// creates a coo representation of a tensor with shape defined by the array of indicies
    pub fn coo(_index_list: &[Index]) -> Tensor<T> {
        Tensor {_temp: std::marker::PhantomData}
    }
    /// creates a csf representation of a tensor with shape defined by the array of indicies
    pub fn csf(_index_list: &[Index]) -> Tensor<T> {
        Tensor {_temp: std::marker::PhantomData}
    }    

    /// populate a tensor with a constant value. Currently only valid for dense tensors
    pub fn fill(&mut self, _val: T){}

    /// load a tensor from a file. Currently only valid for sparse tensors
    pub fn fill_from_file(&mut self, _filename: &str){}

    /// randomly initialize a tensor. Currently only valid for dense tensors
    pub fn random(&mut self){}

    /// return the transpose of the tensor
    pub fn transpose(&self) -> Tensor<T> {
        Tensor {_temp: std::marker::PhantomData}
    }

    /// return the sum of the tensor
    pub fn sum(&self) -> T {
        T::default()
    }
}
