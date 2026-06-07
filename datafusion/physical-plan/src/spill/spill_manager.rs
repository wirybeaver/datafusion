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

//! Define the `SpillManager` struct, which is responsible for reading and writing `RecordBatch`es to raw files based on the provided configurations.

use super::{
    ReadStreamWithReservation, SpillReaderStream,
    in_progress_spill_file::InProgressSpillFile,
};
use crate::coop::cooperative;
use crate::{common::spawn_buffered, metrics::SpillMetrics};
use arrow::array::{BinaryViewArray, GenericByteViewArray, StringViewArray};
use arrow::datatypes::{ByteViewType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result, config::SpillCompression};
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_execution::runtime_env::RuntimeEnv;
use std::borrow::Borrow;
use std::sync::Arc;

/// The `SpillManager` is responsible for the following tasks:
/// - Reading and writing `RecordBatch`es to raw files based on the provided configurations.
/// - Updating the associated metrics.
///
/// Note: The caller (external operators such as `SortExec`) is responsible for interpreting the spilled files.
/// For example, all records within the same spill file are ordered according to a specific order.
#[derive(Debug)]
pub struct SpillManager {
    env: Arc<RuntimeEnv>,
    pub(crate) metrics: SpillMetrics,
    schema: SchemaRef,
    /// Number of batches to buffer in memory during disk reads
    batch_read_buffer_capacity: usize,
    /// general-purpose compression options
    pub(crate) compression: SpillCompression,
    /// Owned reservation split from the operator's reservation.
    /// Per-file write reservations are split from this.
    reservation: MemoryReservation,
}

impl Clone for SpillManager {
    fn clone(&self) -> Self {
        Self {
            env: Arc::clone(&self.env),
            metrics: self.metrics.clone(),
            schema: Arc::clone(&self.schema),
            batch_read_buffer_capacity: self.batch_read_buffer_capacity,
            compression: self.compression,
            reservation: self.reservation.new_empty(),
        }
    }
}

impl SpillManager {
    pub fn new(
        env: Arc<RuntimeEnv>,
        metrics: SpillMetrics,
        schema: SchemaRef,
        reservation: MemoryReservation,
    ) -> Self {
        Self {
            env,
            metrics,
            schema,
            batch_read_buffer_capacity: 2,
            compression: SpillCompression::default(),
            reservation,
        }
    }

    /// Convenience constructor for tests that creates an independent
    /// reservation. Production code must use `new()` with an operator-split
    /// reservation.
    #[cfg(test)]
    pub(crate) fn new_default(
        env: Arc<RuntimeEnv>,
        metrics: SpillMetrics,
        schema: SchemaRef,
    ) -> Self {
        use datafusion_execution::memory_pool::MemoryConsumer;
        let reservation = MemoryConsumer::new("SpillManager")
            .with_can_spill(true)
            .register(&env.memory_pool);
        Self::new(env, metrics, schema, reservation)
    }

    pub fn with_batch_read_buffer_capacity(
        mut self,
        batch_read_buffer_capacity: usize,
    ) -> Self {
        self.batch_read_buffer_capacity = batch_read_buffer_capacity;
        self
    }

    pub fn with_compression_type(mut self, spill_compression: SpillCompression) -> Self {
        self.compression = spill_compression;
        self
    }

    /// Returns the schema for batches managed by this SpillManager
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Creates a temporary file for in-progress operations with automatic
    /// IPC write buffer accounting via the memory pool.
    pub fn create_in_progress_file(
        &self,
        request_msg: &str,
    ) -> Result<InProgressSpillFile> {
        let temp_file = self.env.disk_manager.create_tmp_file(request_msg)?;
        let reservation = self.reservation.new_empty();
        Ok(InProgressSpillFile::new(
            Arc::new(self.clone()),
            temp_file,
            reservation,
        ))
    }

    /// Spill input `batches` into a single file in a atomic operation. If it is
    /// intended to incrementally write in-memory batches into the same spill file,
    /// use [`Self::create_in_progress_file`] instead.
    /// None is returned if no batches are spilled.
    ///
    /// # Errors
    /// - Returns an error if spilling would exceed the disk usage limit configured
    ///   by `max_temp_directory_size` in `DiskManager`
    pub fn spill_record_batch_and_finish(
        &self,
        batches: &[RecordBatch],
        request_msg: &str,
    ) -> Result<Option<RefCountedTempFile>> {
        let mut in_progress_file = self.create_in_progress_file(request_msg)?;

        for batch in batches {
            in_progress_file.append_batch(batch)?;
        }

        in_progress_file.finish()
    }

    /// Spill an iterator of `RecordBatch`es to disk and return the spill file and the size of the largest batch in memory
    /// Note that this expects the caller to provide *non-sliced* batches, so the memory calculation of each batch is accurate.
    pub(crate) fn spill_record_batch_iter_and_return_max_batch_memory(
        &self,
        mut iter: impl Iterator<Item = Result<impl Borrow<RecordBatch>>>,
        request_description: &str,
    ) -> Result<Option<(RefCountedTempFile, usize)>> {
        let mut in_progress_file = self.create_in_progress_file(request_description)?;

        let mut max_record_batch_size = 0;

        iter.try_for_each(|batch| {
            let batch = batch?;
            let borrowed = batch.borrow();
            if borrowed.num_rows() == 0 {
                return Ok(());
            }
            let gc_sliced_size = in_progress_file.append_batch(borrowed)?;
            max_record_batch_size = max_record_batch_size.max(gc_sliced_size);
            Result::<_, DataFusionError>::Ok(())
        })?;

        let file = in_progress_file.finish()?;

        Ok(file.map(|f| (f, max_record_batch_size)))
    }

    /// Spill a stream of `RecordBatch`es to disk and return the spill file and the size of the largest batch in memory
    pub(crate) async fn spill_record_batch_stream_and_return_max_batch_memory(
        &self,
        stream: &mut SendableRecordBatchStream,
        request_description: &str,
    ) -> Result<Option<(RefCountedTempFile, usize)>> {
        use futures::StreamExt;

        let mut in_progress_file = self.create_in_progress_file(request_description)?;

        let mut max_record_batch_size = 0;

        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let gc_sliced_size = in_progress_file.append_batch(&batch)?;

            max_record_batch_size = max_record_batch_size.max(gc_sliced_size);
        }

        let file = in_progress_file.finish()?;

        Ok(file.map(|f| (f, max_record_batch_size)))
    }

    /// Reads a spill file as a stream. The file must be created by the current
    /// `SpillManager`; otherwise an error will be returned.
    ///
    /// Output is produced in FIFO order: the batch appended first is read first.
    ///
    /// # Arg `max_record_batch_memory`
    ///
    /// Most callers should pass `None`. This is mainly useful for the
    /// memory-limited sort-preserving merge path.
    ///
    /// When provided, this value is used only as a validation hint. If a
    /// decoded batch exceeds this threshold, a debug-level log message is
    /// emitted.
    ///
    /// That path uses the maximum spilled batch size to conservatively estimate
    /// the merge degree when merging multiple sorted runs.
    ///
    /// # Arg `reservation`
    ///
    /// Optional caller-owned reservation for tracking decoded batch memory.
    /// When provided along with `max_record_batch_memory`, pre-reserves
    /// `max_record_batch_memory * buffer_capacity` to account for all
    /// batches that can be simultaneously live in the `spawn_buffered`
    /// channel. The reservation is freed when the stream ends or is
    /// dropped.
    ///
    /// Callers that already track decoded batches via their own reservation
    /// (e.g., NLJ build-side, sort merge multi-file path) should pass
    /// `None` to avoid double-counting.
    pub fn read_spill_as_stream(
        &self,
        spill_file_path: RefCountedTempFile,
        max_record_batch_memory: Option<usize>,
        reservation: Option<MemoryReservation>,
    ) -> Result<SendableRecordBatchStream> {
        // Reserve capacity BEFORE spawning the producer task.
        // spawn_buffered immediately starts a producer that decodes batches,
        // so the reservation must be in place before any allocation happens.
        if let (Some(res), Some(max_mem)) = (&reservation, max_record_batch_memory) {
            let capacity_bytes = max_mem.saturating_mul(self.batch_read_buffer_capacity);
            let deficit = capacity_bytes.saturating_sub(res.size());
            if deficit > 0 {
                res.try_grow(deficit)?;
            }
        }

        let stream = Box::pin(cooperative(SpillReaderStream::new(
            Arc::clone(&self.schema),
            spill_file_path,
            max_record_batch_memory,
            None, // per-batch tracking not used for buffered reads
        )));

        let buffered = spawn_buffered(stream, self.batch_read_buffer_capacity);

        match reservation {
            Some(res) => Ok(Box::pin(ReadStreamWithReservation::new(buffered, res))),
            None => Ok(buffered),
        }
    }

    /// Same as `read_spill_as_stream`, but without buffering.
    ///
    /// When `reservation` is provided, per-batch tracking is used:
    /// each decoded batch grows the reservation and the previous batch's
    /// reservation is shrunk. This is correct for unbuffered reads where
    /// only one batch is live at a time.
    pub fn read_spill_as_stream_unbuffered(
        &self,
        spill_file_path: RefCountedTempFile,
        max_record_batch_memory: Option<usize>,
        reservation: Option<MemoryReservation>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(cooperative(SpillReaderStream::new(
            Arc::clone(&self.schema),
            spill_file_path,
            max_record_batch_memory,
            reservation,
        ))))
    }
}

pub(crate) trait GetSlicedSize {
    /// Returns the size of the `RecordBatch` when sliced.
    /// Note: if multiple arrays or even a single array share the same data buffers, we may double count each buffer.
    /// Therefore, make sure we call gc() or gc_view_arrays() before using this method.
    fn get_sliced_size(&self) -> Result<usize>;
}

impl GetSlicedSize for RecordBatch {
    fn get_sliced_size(&self) -> Result<usize> {
        let mut total = 0;
        for array in self.columns() {
            let data = array.to_data();
            total += data.get_slice_memory_size()?;

            // While StringViewArray holds large data buffer for non inlined string, the Arrow layout (BufferSpec)
            // does not include any data buffers. Currently, ArrayData::get_slice_memory_size()
            // under-counts memory size by accounting only views buffer although data buffer is cloned during slice()
            //
            // Therefore, we manually add the sum of the lengths used by all non inlined views
            // on top of the sliced size for views buffer. This matches the intended semantics of
            // "bytes needed if we materialized exactly this slice into fresh buffers".
            // This is a workaround until https://github.com/apache/arrow-rs/issues/8230
            if let Some(sv) = array.as_any().downcast_ref::<StringViewArray>() {
                total += byte_view_data_buffer_size(sv);
            }
            if let Some(bv) = array.as_any().downcast_ref::<BinaryViewArray>() {
                total += byte_view_data_buffer_size(bv);
            }
        }
        Ok(total)
    }
}

fn byte_view_data_buffer_size<T: ByteViewType>(array: &GenericByteViewArray<T>) -> usize {
    array
        .data_buffers()
        .iter()
        .map(|buffer| buffer.capacity())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::SpillManager;
    use crate::common::collect;
    use crate::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
    use crate::spill::{get_record_batch_memory_size, spill_manager::GetSlicedSize};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::{
        array::{ArrayRef, Int32Array, StringArray, StringViewArray},
        record_batch::RecordBatch,
    };
    use datafusion_common::Result;
    use datafusion_execution::runtime_env::RuntimeEnv;
    use std::sync::Arc;

    fn build_test_spill_manager(
        env: Arc<RuntimeEnv>,
        schema: Arc<Schema>,
    ) -> SpillManager {
        let metrics = SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        SpillManager::new_default(env, metrics, schema)
    }

    fn build_writer_batch(schema: Arc<Schema>) -> Result<RecordBatch> {
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .map_err(Into::into)
    }

    #[tokio::test]
    async fn test_read_spill_as_stream_from_another_spill_manager_same_schema()
    -> Result<()> {
        let env = Arc::new(RuntimeEnv::default());
        let writer_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let reader_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));

        let writer =
            build_test_spill_manager(Arc::clone(&env), Arc::clone(&writer_schema));
        let reader = build_test_spill_manager(env, Arc::clone(&reader_schema));
        let written_batch = build_writer_batch(Arc::clone(&writer_schema))?;

        let spill_file = writer
            .spill_record_batch_and_finish(
                std::slice::from_ref(&written_batch),
                "writer",
            )?
            .unwrap();

        // Same-schema reads through a different SpillManager currently pass
        // because only schema compatibility is validated. This is not a
        // supported usage pattern.
        let stream = reader.read_spill_as_stream(spill_file, None, None)?;
        assert_eq!(stream.schema(), reader_schema);

        let batches = collect(stream).await?;
        assert_eq!(batches, vec![written_batch]);

        Ok(())
    }

    #[tokio::test]
    async fn test_read_spill_as_stream_from_another_spill_manager_different_schema()
    -> Result<()> {
        let env = Arc::new(RuntimeEnv::default());
        let writer_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let reader_schema = Arc::new(Schema::new(vec![
            Field::new("other_id", DataType::Int32, true),
            Field::new("other_value", DataType::Utf8, true),
        ]));

        let writer =
            build_test_spill_manager(Arc::clone(&env), Arc::clone(&writer_schema));
        let reader = build_test_spill_manager(env, Arc::clone(&reader_schema));
        let written_batch = build_writer_batch(Arc::clone(&writer_schema))?;

        let spill_file = writer
            .spill_record_batch_and_finish(
                std::slice::from_ref(&written_batch),
                "writer",
            )?
            .unwrap();

        let stream = reader.read_spill_as_stream(spill_file, None, None)?;
        let err = collect(stream)
            .await
            .expect_err("schema mismatch should fail fast");
        let err = err.to_string();
        assert!(err.contains("Spill file schema mismatch"));
        assert!(err.contains("expected"));
        assert!(err.contains("got"));

        Ok(())
    }

    #[test]
    fn check_sliced_size_for_string_view_array() -> Result<()> {
        let array_length = 50;
        let short_len = 8;
        let long_len = 25;

        // Build StringViewArray that includes both inline strings and non inlined strings
        let strings: Vec<String> = (0..array_length)
            .map(|i| {
                if i % 2 == 0 {
                    "a".repeat(short_len)
                } else {
                    "b".repeat(long_len)
                }
            })
            .collect();

        let string_array = StringViewArray::from(strings);
        let array_ref: ArrayRef = Arc::new(string_array);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "strings",
                DataType::Utf8View,
                false,
            )])),
            vec![array_ref],
        )
        .unwrap();

        // We did not slice the batch, so these two memory size should be equal
        assert_eq!(
            batch.get_sliced_size().unwrap(),
            get_record_batch_memory_size(&batch)
        );

        // Slice the batch into half
        let half_batch = batch.slice(0, array_length / 2);
        // Now sliced_size is smaller because the views buffer is sliced
        assert!(
            half_batch.get_sliced_size().unwrap()
                < get_record_batch_memory_size(&half_batch)
        );
        let data = arrow::array::Array::to_data(&half_batch.column(0));
        let views_sliced_size = data.get_slice_memory_size()?;
        // The sliced size should be larger than sliced views buffer size
        assert!(views_sliced_size < half_batch.get_sliced_size().unwrap());

        Ok(())
    }

    #[tokio::test]
    async fn test_spill_write_reservation_balanced() -> Result<()> {
        let env = Arc::new(RuntimeEnv::default());
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let spill_manager =
            build_test_spill_manager(Arc::clone(&env), Arc::clone(&schema));
        let batch = build_writer_batch(schema)?;

        let mut in_progress = spill_manager.create_in_progress_file("test_balanced")?;
        in_progress.append_batch(&batch)?;
        in_progress.append_batch(&batch)?;

        // After each append the grow/shrink is balanced within the call
        let reserved_during = env.memory_pool.reserved();

        let _file = in_progress.finish()?;
        drop(in_progress);

        // After drop, reservation should be fully released
        assert_eq!(
            env.memory_pool.reserved(),
            0,
            "Pool should have zero reserved after InProgressSpillFile is dropped, got {reserved_during} during"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_spill_read_reservation_tracked() -> Result<()> {
        let env = Arc::new(RuntimeEnv::default());
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let spill_manager =
            build_test_spill_manager(Arc::clone(&env), Arc::clone(&schema));
        let batch = build_writer_batch(schema)?;

        // Write a spill file
        let spill_file = spill_manager
            .spill_record_batch_and_finish(&[batch], "test_read")?
            .expect("should have spill file");

        // Read back — reservation should track decoded batches
        let stream = spill_manager.read_spill_as_stream(spill_file, None, None)?;
        let batches = collect(stream).await?;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);

        // After stream is consumed and dropped, pool reserved should be zero
        assert_eq!(
            env.memory_pool.reserved(),
            0,
            "Pool should have zero reserved after read stream is consumed"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_spill_write_balanced_under_exhausted_pool() -> Result<()> {
        use datafusion_execution::memory_pool::GreedyMemoryPool;

        let pool: Arc<dyn datafusion_execution::memory_pool::MemoryPool> =
            Arc::new(GreedyMemoryPool::new(64));
        let env = Arc::new(
            datafusion_execution::runtime_env::RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::clone(&pool))
                .build()?,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));

        // Exhaust most of the pool
        let blocker = datafusion_execution::memory_pool::MemoryConsumer::new("blocker")
            .register(&pool);
        blocker.grow(60);

        let spill_manager =
            build_test_spill_manager(Arc::clone(&env), Arc::clone(&schema));
        let batch = build_writer_batch(schema)?;

        // append_batch uses infallible grow — it must succeed even when
        // the pool is nearly exhausted (spill-write make-progress exception).
        let mut in_progress = spill_manager.create_in_progress_file("exhausted")?;
        in_progress.append_batch(&batch)?;

        // Reservation should be balanced within the call
        let reserved_after_append = in_progress.reservation_size();
        assert_eq!(
            reserved_after_append, 0,
            "Write reservation should be zero after append (grow/shrink balanced)"
        );

        let _file = in_progress.finish()?;
        drop(in_progress);

        // Release the blocker
        blocker.free();

        assert_eq!(
            pool.reserved(),
            0,
            "Pool should be zero after all reservations freed"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_unbuffered_read_reservation_tracks_batches() -> Result<()> {
        use datafusion_execution::memory_pool::MemoryConsumer;

        let env = Arc::new(RuntimeEnv::default());
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let spill_manager =
            build_test_spill_manager(Arc::clone(&env), Arc::clone(&schema));
        let batch = build_writer_batch(Arc::clone(&schema))?;

        let spill_file = spill_manager
            .spill_record_batch_and_finish(&[batch.clone(), batch], "test_read")?
            .expect("should have spill file");

        let read_reservation =
            MemoryConsumer::new("read_test").register(&env.memory_pool);

        let mut stream = spill_manager.read_spill_as_stream_unbuffered(
            spill_file,
            None,
            Some(read_reservation),
        )?;

        use futures::StreamExt;
        // Read first batch — reservation should grow
        let b1 = stream.next().await.unwrap()?;
        assert_eq!(b1.num_rows(), 3);
        let reserved_after_b1 = env.memory_pool.reserved();
        assert!(
            reserved_after_b1 > 0,
            "Reservation should be non-zero after reading first batch"
        );

        // Read second batch — previous shrinks, current grows
        let b2 = stream.next().await.unwrap()?;
        assert_eq!(b2.num_rows(), 3);
        let reserved_after_b2 = env.memory_pool.reserved();
        assert!(
            reserved_after_b2 > 0,
            "Reservation should be non-zero after reading second batch"
        );

        // Stream ends — remaining reservation freed
        assert!(stream.next().await.is_none());
        assert_eq!(
            env.memory_pool.reserved(),
            0,
            "Pool should be zero after unbuffered read stream consumed"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_buffered_read_reservation_prereserves_capacity() -> Result<()> {
        use datafusion_execution::memory_pool::MemoryConsumer;

        let env = Arc::new(RuntimeEnv::default());
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let spill_manager =
            build_test_spill_manager(Arc::clone(&env), Arc::clone(&schema));
        let batch = build_writer_batch(Arc::clone(&schema))?;
        let batch_mem = get_record_batch_memory_size(&batch);

        let spill_file = spill_manager
            .spill_record_batch_and_finish(&[batch], "test_read")?
            .expect("should have spill file");

        let read_reservation =
            MemoryConsumer::new("read_test").register(&env.memory_pool);

        let buffer_capacity = spill_manager.batch_read_buffer_capacity;
        let expected_capacity = batch_mem * buffer_capacity;

        let stream = spill_manager.read_spill_as_stream(
            spill_file,
            Some(batch_mem),
            Some(read_reservation),
        )?;

        // Capacity should be pre-reserved immediately
        assert_eq!(
            env.memory_pool.reserved(),
            expected_capacity,
            "Pool should have capacity pre-reserved for buffered read"
        );

        // Consume the stream
        let batches = collect(stream).await?;
        assert_eq!(batches.len(), 1);

        // After stream consumed and dropped, reservation freed
        assert_eq!(
            env.memory_pool.reserved(),
            0,
            "Pool should be zero after buffered read stream consumed"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_buffered_read_reuses_transferred_reservation() -> Result<()> {
        use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryConsumer};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let batch = build_writer_batch(Arc::clone(&schema))?;
        let batch_mem = get_record_batch_memory_size(&batch);

        // Pool just large enough for the read buffer capacity
        let buffer_capacity = 2usize;
        let capacity_bytes = batch_mem * buffer_capacity;
        let pool_size = capacity_bytes + batch_mem; // extra for write overhead
        let pool: Arc<dyn datafusion_execution::memory_pool::MemoryPool> =
            Arc::new(GreedyMemoryPool::new(pool_size));
        let env = Arc::new(
            datafusion_execution::runtime_env::RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::clone(&pool))
                .build()?,
        );

        let spill_manager =
            build_test_spill_manager(Arc::clone(&env), Arc::clone(&schema));

        let spill_file = spill_manager
            .spill_record_batch_and_finish(&[batch], "test")?
            .expect("should have spill file");

        // Pre-grow read reservation to full capacity (simulating take()
        // from a merge reservation that already holds these bytes)
        let read_reservation = MemoryConsumer::new("read").register(&env.memory_pool);
        read_reservation.grow(capacity_bytes);

        // Consume ALL remaining pool capacity with a competing consumer
        let remaining = pool_size - pool.reserved();
        let blocker = MemoryConsumer::new("blocker").register(&env.memory_pool);
        blocker.grow(remaining);
        assert_eq!(pool.reserved(), pool_size);

        // With grow-to-at-least semantics, read_spill_as_stream should
        // succeed because the pre-grown reservation already covers the
        // required capacity — no additional pool allocation needed.
        let stream = spill_manager.read_spill_as_stream(
            spill_file,
            Some(batch_mem),
            Some(read_reservation),
        )?;

        let batches = collect(stream).await?;
        assert_eq!(batches.len(), 1);

        blocker.free();
        assert_eq!(
            pool.reserved(),
            0,
            "Pool should be zero after all reservations freed"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_unbuffered_read_exhausted_pool_returns_error() -> Result<()> {
        use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryConsumer};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let batch = build_writer_batch(Arc::clone(&schema))?;
        let batch_mem = get_record_batch_memory_size(&batch);

        // Pool large enough for writing but not for a subsequent read
        let pool_size = batch_mem + 64;
        let pool: Arc<dyn datafusion_execution::memory_pool::MemoryPool> =
            Arc::new(GreedyMemoryPool::new(pool_size));
        let env = Arc::new(
            datafusion_execution::runtime_env::RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::clone(&pool))
                .build()?,
        );

        let spill_manager =
            build_test_spill_manager(Arc::clone(&env), Arc::clone(&schema));

        let spill_file = spill_manager
            .spill_record_batch_and_finish(&[batch], "test")?
            .expect("should have spill file");

        // Exhaust the pool completely
        let blocker = MemoryConsumer::new("blocker").register(&pool);
        blocker.grow(pool_size - pool.reserved());

        let read_reservation = MemoryConsumer::new("read").register(&pool);

        // Unbuffered read with pre-reservation: try_grow(batch_mem)
        // should fail with controlled ResourcesExhausted
        let mut stream = spill_manager.read_spill_as_stream_unbuffered(
            spill_file,
            Some(batch_mem),
            Some(read_reservation),
        )?;

        use futures::StreamExt;
        let result = stream.next().await;
        assert!(result.is_some());
        let err = result.unwrap().unwrap_err();
        assert!(
            err.to_string().contains("Resources exhausted"),
            "Expected ResourcesExhausted error, got: {err}"
        );

        blocker.free();
        assert_eq!(pool.reserved(), 0);

        Ok(())
    }
}
