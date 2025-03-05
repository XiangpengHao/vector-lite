use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct Vector<const N: usize>(Vec<f32>);

impl<const N: usize> From<[f32; N]> for Vector<N> {
    fn from(values: [f32; N]) -> Self {
        Self(values.to_vec())
    }
}

impl<const N: usize> Hash for Vector<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, N * 4) };
        bytes.hash(state);
    }
}

impl<const N: usize> PartialEq for Vector<N> {
    // Eq when the underlying memory is the same. Very strict.
    fn eq(&self, other: &Self) -> bool {
        let bytes_left: &[u8] =
            unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, N * 4) };
        let bytes_right: &[u8] =
            unsafe { std::slice::from_raw_parts(other.0.as_ptr() as *const u8, N * 4) };
        bytes_left == bytes_right
    }
}

impl<const N: usize> Eq for Vector<N> {}

impl<const N: usize> Vector<N> {
    /// Try to create a Vector from a `Vec<f32>`.
    ///
    /// # Arguments
    ///
    /// * `values` - The vector to create the Vector from, must be of size N.
    ///
    /// # Returns
    ///
    /// * `Ok(Vector<N>)` - The Vector created from the `Vec<f32>`.
    /// * `Err(Vec<f32>)` - The `Vec<f32>` is not the correct size.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vector_search::Vector;
    /// let vec = vec![1.0, 2.0, 3.0];
    /// let vector: Vector<3> = Vector::try_from(vec).unwrap();
    /// assert_eq!(vector.into_inner(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn try_from(values: Vec<f32>) -> Result<Self, Vec<f32>> {
        if values.len() != N {
            Err(values)
        } else {
            Ok(Self(values))
        }
    }

    /// Get the inner vector.
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }

    fn slice(&self) -> &[f32; N] {
        // Convert the Vec<f32> slice to a fixed-size array reference
        // This is safe because we know the Vector always has exactly N elements
        unsafe { &*(self.0.as_ptr() as *const [f32; N]) }
    }

    pub(crate) fn subtract_from(&self, other: &Vector<N>) -> Vector<N> {
        let vals = self
            .slice()
            .iter()
            .zip(other.slice().iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        debug_assert_eq!(vals.capacity(), N);
        Vector(vals)
    }

    pub(crate) fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.0.len() * 4
    }

    #[cfg(test)]
    pub(crate) fn add(&self, vector: &Vector<N>) -> Vector<N> {
        let vals = self
            .slice()
            .iter()
            .zip(vector.slice().iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        Vector(vals)
    }

    pub(crate) fn avg(&self, vector: &Vector<N>) -> Vector<N> {
        let vals = self
            .slice()
            .iter()
            .zip(vector.slice().iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect::<Vec<_>>();
        Vector(vals)
    }

    pub(crate) fn dot_product(&self, vector: &Vector<N>) -> f32 {
        self.slice()
            .iter()
            .zip(vector.slice().iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub(crate) fn sq_euc_dist(&self, vector: &Vector<N>) -> f32 {
        self.slice()
            .iter()
            .zip(vector.slice().iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }
}
