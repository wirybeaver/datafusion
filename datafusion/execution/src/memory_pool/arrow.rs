// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Adapter for integrating DataFusion's [`MemoryPool`] with Arrow's memory tracking APIs.

use crate::memory_pool::{MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use std::fmt::Debug;
use std::sync::Arc;

/// An adapter that implements Arrow's [`arrow_buffer::MemoryPool`] trait
/// by wrapping a DataFusion [`MemoryPool`].
///
/// All reservations made through this pool grow a single shared
/// [`MemoryReservation`]. This keeps `FairSpillPool`'s bookkeeping correct:
///
/// - [`Self::new`] creates its own consumer registration — use this when the
///   pool stands alone (e.g. tests, standalone Arrow computation).
/// - [`Self::from_reservation`] re-uses an existing reservation — use this
///   inside an operator so that claimed Arrow-buffer bytes are accounted under
///   the *same* pool consumer as the operator's main reservation. That way
///   neither `num_spill` nor `unspillable` changes, and the fair-share
///   formula is unaffected.
#[derive(Debug)]
pub struct ArrowMemoryPool {
    inner: Arc<dyn MemoryPool>,
    shared: Arc<MemoryReservation>,
}

impl ArrowMemoryPool {
    /// Creates a new [`ArrowMemoryPool`] with its own consumer registration.
    pub fn new(inner: Arc<dyn MemoryPool>, consumer: MemoryConsumer) -> Self {
        let shared = Arc::new(consumer.register(&inner));
        Self { inner, shared }
    }

    /// Creates a pool backed by a sibling of `reservation`.
    ///
    /// Calls [`MemoryReservation::new_empty`] to create a zero-size reservation
    /// that shares the same [`Arc`] registration (and therefore the same pool
    /// consumer) as `reservation`. This means:
    ///
    /// - No new consumer is registered: `FairSpillPool::num_spill` and
    ///   `unspillable` are both unaffected.
    /// - Claimed bytes are counted toward the same `spillable`/`unspillable`
    ///   bucket as the operator's main reservation.
    /// - The per-reservation `size` field checked by `FairSpillPool`'s
    ///   fair-share formula is **independent** of the main reservation's size,
    ///   so `update_memory_reservation()` cannot undercut live claim handles.
    pub fn from_reservation(
        inner: Arc<dyn MemoryPool>,
        reservation: &MemoryReservation,
    ) -> Self {
        let shared = Arc::new(reservation.new_empty());
        Self { inner, shared }
    }
}

/// Tracks one buffer's share of the [`ArrowMemoryPool`] shared reservation.
///
/// On resize it adjusts the shared [`MemoryReservation`] by the delta.
/// On drop it releases the buffer's bytes from the shared reservation.
///
/// `MemoryReservation` uses atomic interior mutability, so no external lock is
/// needed: `grow` / `shrink` are `&self` methods.
#[derive(Debug)]
struct SharedClaimHandle {
    shared: Arc<MemoryReservation>,
    size: usize,
}

impl arrow_buffer::MemoryReservation for SharedClaimHandle {
    fn size(&self) -> usize {
        self.size
    }

    fn resize(&mut self, new_size: usize) {
        match new_size.cmp(&self.size) {
            std::cmp::Ordering::Greater => self.shared.grow(new_size - self.size),
            std::cmp::Ordering::Less => self.shared.shrink(self.size - new_size),
            std::cmp::Ordering::Equal => {}
        }
        self.size = new_size;
    }
}

impl Drop for SharedClaimHandle {
    fn drop(&mut self) {
        if self.size > 0 {
            self.shared.shrink(self.size);
        }
    }
}

impl arrow_buffer::MemoryReservation for MemoryReservation {
    fn size(&self) -> usize {
        MemoryReservation::size(self)
    }

    fn resize(&mut self, new_size: usize) {
        MemoryReservation::resize(self, new_size)
    }
}

impl arrow_buffer::MemoryPool for ArrowMemoryPool {
    fn reserve(&self, size: usize) -> Box<dyn arrow_buffer::MemoryReservation> {
        self.shared.grow(size);
        Box::new(SharedClaimHandle {
            shared: Arc::clone(&self.shared),
            size,
        })
    }

    fn available(&self) -> isize {
        // The pool may be overfilled, so this method might return a negative value.
        (self.capacity() as i128 - self.used() as i128)
            .try_into()
            .unwrap_or(isize::MIN)
    }

    fn used(&self) -> usize {
        self.inner.reserved()
    }

    fn capacity(&self) -> usize {
        match self.inner.memory_limit() {
            MemoryLimit::Infinite | MemoryLimit::Unknown => usize::MAX,
            MemoryLimit::Finite(capacity) => capacity,
        }
    }
}

/// Claims all Arrow buffers in `array` against `pool` (idempotent, recursive).
///
/// Uses [`arrow_data::ArrayData::claim`], which covers data buffers, null buffers,
/// and all child arrays. Claiming the same physical buffer twice is a no-op.
pub fn claim_array(array: &dyn Array, pool: &dyn arrow_buffer::MemoryPool) {
    array.to_data().claim(pool);
}

/// Claims all Arrow buffers in every column of `batch` against `pool`.
///
/// See [`claim_array`] for semantics. This is the primary entry point for
/// registering output [`RecordBatch`] memory with a [`MemoryPool`] at output
/// boundaries, replacing manual `get_array_memory_size()` bookkeeping.
pub fn claim_batch(batch: &RecordBatch, pool: &dyn arrow_buffer::MemoryPool) {
    for col in batch.columns() {
        claim_array(col.as_ref(), pool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_pool::{GreedyMemoryPool, UnboundedMemoryPool};
    use arrow::array::Int32Array;
    use arrow_buffer::MemoryPool;

    #[test]
    pub fn can_claim_array() {
        let pool = Arc::new(UnboundedMemoryPool::default());

        let consumer = MemoryConsumer::new("arrow");
        let arrow_pool = ArrowMemoryPool::new(pool, consumer);

        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        claim_array(&array, &arrow_pool);

        assert_eq!(arrow_pool.used(), array.get_buffer_memory_size());

        let slice = array.slice(0, 2);

        // This should be a no-op
        claim_array(&slice, &arrow_pool);

        assert_eq!(arrow_pool.used(), array.get_buffer_memory_size());
    }

    #[test]
    pub fn can_claim_array_with_finite_limit() {
        let pool_capacity = 1024;
        let pool = Arc::new(GreedyMemoryPool::new(pool_capacity));

        let consumer = MemoryConsumer::new("arrow");
        let arrow_pool = ArrowMemoryPool::new(pool, consumer);

        assert_eq!(arrow_pool.capacity(), pool_capacity);
        assert_eq!(arrow_pool.available(), pool_capacity as isize);

        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        claim_array(&array, &arrow_pool);

        assert_eq!(arrow_pool.used(), array.get_buffer_memory_size());
        assert_eq!(
            arrow_pool.available(),
            (pool_capacity - array.get_buffer_memory_size()) as isize
        );
    }
}
