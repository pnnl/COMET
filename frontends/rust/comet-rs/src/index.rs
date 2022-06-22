

//! Range-based index label constructs (e.g., [i, j]) represent the range of indices expressed through a scalar, a range, or a range with increment.
//! Index labels can be used both for constructing a tensor or for representing a tensor operation.
//! In a tensor construction, index labels are used to represent each dimension size.
//! In the context of a tensor operation, they represent slicing information of the tensor object where the operation will be applied.

pub struct Index {}
impl Index {
    /// Creates a new dynamic index label, where the size of the index is not known at compile time, and is instead determined at runtime, typically from reading a file containing a representation of a tensor
    pub fn new() -> Index {
        Index {}
    }
    /// Creates a new static index label, where the size of the index is known at compile time
    pub fn with_value(_value: usize) -> Index {
        Index {}
    }
}
