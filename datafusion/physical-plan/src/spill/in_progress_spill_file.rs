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

//! Define the `InProgressSpillFile` struct, which represents an in-progress spill file used for writing `RecordBatch`es to disk, created by `SpillManager`.

use datafusion_common::Result;
use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion_common::exec_datafusion_err;
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::memory_pool::MemoryReservation;

use super::{
    IPCStreamWriter, gc_view_arrays,
    spill_manager::{GetSlicedSize, SpillManager},
};

/// Represents an in-progress spill file used for writing `RecordBatch`es to disk, created by `SpillManager`.
/// Caller is able to use this struct to incrementally append in-memory batches to
/// the file, and then finalize the file by calling the `finish` method.
pub struct InProgressSpillFile {
    pub(crate) spill_writer: Arc<SpillManager>,
    /// Lazily initialized writer
    writer: Option<IPCStreamWriter>,
    /// Lazily initialized in-progress file, it will be moved out when the `finish` method is invoked
    in_progress_file: Option<RefCountedTempFile>,
    /// Memory reservation for tracking IPC write buffer overhead.
    /// `append_batch` reserves memory before writing and releases it
    /// after the write completes. Freed automatically on Drop.
    reservation: MemoryReservation,
}

impl InProgressSpillFile {
    pub fn new(
        spill_writer: Arc<SpillManager>,
        in_progress_file: RefCountedTempFile,
        reservation: MemoryReservation,
    ) -> Self {
        Self {
            spill_writer,
            in_progress_file: Some(in_progress_file),
            writer: None,
            reservation,
        }
    }

    #[cfg(test)]
    pub(crate) fn reservation_size(&self) -> usize {
        self.reservation.size()
    }

    /// Appends a `RecordBatch` to the spill file, initializing the writer if necessary.
    ///
    /// Before writing, performs GC on StringView/BinaryView arrays to compact backing
    /// buffers. When a view array is sliced, it still references the original full buffers,
    /// causing massive spill files without GC (see issue #19414: 820MB → 33MB after GC).
    ///
    /// Returns the post-GC sliced memory size of the batch for memory accounting.
    ///
    /// # Errors
    /// - Returns an error if the file is not active (has been finalized)
    /// - Returns an error if appending would exceed the disk usage limit configured
    ///   by `max_temp_directory_size` in `DiskManager`
    pub fn append_batch(&mut self, batch: &RecordBatch) -> Result<usize> {
        if self.in_progress_file.is_none() {
            return Err(exec_datafusion_err!(
                "Append operation failed: No active in-progress file. The file may have already been finalized."
            ));
        }

        // Named exception (infallible grow): spill writes MUST complete
        // even under memory pressure. Operators free their main reservation
        // before spilling; failing here would prevent memory recovery.
        // The grow/shrink is balanced within each call — the reservation
        // returns to its pre-call size after the write completes or errors.
        let write_overhead = batch.get_array_memory_size();
        self.reservation.grow(write_overhead);

        let result = self.append_batch_inner(batch);

        self.reservation.shrink(write_overhead);
        result
    }

    fn append_batch_inner(&mut self, batch: &RecordBatch) -> Result<usize> {
        let gc_batch = gc_view_arrays(batch)?;

        if self.writer.is_none() {
            let schema = self.spill_writer.schema();
            if let Some(in_progress_file) = &mut self.in_progress_file {
                self.writer = Some(IPCStreamWriter::new(
                    in_progress_file.path(),
                    schema.as_ref(),
                    self.spill_writer.compression,
                )?);

                self.spill_writer.metrics.spill_file_count.add(1);

                in_progress_file.update_disk_usage()?;
                let initial_size = in_progress_file.current_disk_usage();
                self.spill_writer
                    .metrics
                    .spilled_bytes
                    .add(initial_size as usize);
            }
        }
        if let Some(writer) = &mut self.writer {
            let (spilled_rows, _) = writer.write(&gc_batch)?;
            if let Some(in_progress_file) = &mut self.in_progress_file {
                let pre_size = in_progress_file.current_disk_usage();
                in_progress_file.update_disk_usage()?;
                let post_size = in_progress_file.current_disk_usage();

                self.spill_writer.metrics.spilled_rows.add(spilled_rows);
                self.spill_writer
                    .metrics
                    .spilled_bytes
                    .add((post_size - pre_size) as usize);
            } else {
                unreachable!()
            }
        }

        gc_batch.get_sliced_size()
    }

    pub fn flush(&mut self) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
        }
        Ok(())
    }

    /// Returns a reference to the in-progress file, if it exists.
    /// This can be used to get the file path for creating readers before the file is finished.
    pub fn file(&self) -> Option<&RefCountedTempFile> {
        self.in_progress_file.as_ref()
    }

    /// Finalizes the file, returning the completed file reference.
    /// If there are no batches spilled before, it returns `None`.
    pub fn finish(&mut self) -> Result<Option<RefCountedTempFile>> {
        if let Some(writer) = &mut self.writer {
            writer.finish()?;
        } else {
            return Ok(None);
        }

        // Since spill files are append-only, add the file size to spilled_bytes
        if let Some(in_progress_file) = &mut self.in_progress_file {
            // Since writer.finish() writes continuation marker and message length at the end
            let pre_size = in_progress_file.current_disk_usage();
            in_progress_file.update_disk_usage()?;
            let post_size = in_progress_file.current_disk_usage();
            self.spill_writer
                .metrics
                .spilled_bytes
                .add((post_size - pre_size) as usize);
        }

        Ok(self.in_progress_file.take())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_physical_expr_common::metrics::{
        ExecutionPlanMetricsSet, SpillMetrics,
    };
    use futures::TryStreamExt;

    #[tokio::test]
    async fn test_spill_file_uses_spill_manager_schema() -> Result<()> {
        let nullable_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("val", DataType::Int64, true),
        ]));
        let non_nullable_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("val", DataType::Int64, false),
        ]));

        let runtime = Arc::new(RuntimeEnvBuilder::new().build()?);
        let metrics_set = ExecutionPlanMetricsSet::new();
        let spill_metrics = SpillMetrics::new(&metrics_set, 0);
        let spill_manager = Arc::new(SpillManager::new_default(
            runtime,
            spill_metrics,
            Arc::clone(&nullable_schema),
        ));

        let mut in_progress = spill_manager.create_in_progress_file("test")?;

        // First batch: non-nullable val (simulates literal-0 UNION branch)
        let non_nullable_batch = RecordBatch::try_new(
            Arc::clone(&non_nullable_schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![0, 0, 0])),
            ],
        )?;
        in_progress.append_batch(&non_nullable_batch)?;

        // Second batch: nullable val with NULLs (simulates table UNION branch)
        let nullable_batch = RecordBatch::try_new(
            Arc::clone(&nullable_schema),
            vec![
                Arc::new(Int64Array::from(vec![4, 5, 6])),
                Arc::new(Int64Array::from(vec![Some(10), None, Some(30)])),
            ],
        )?;
        in_progress.append_batch(&nullable_batch)?;

        let spill_file = in_progress.finish()?.unwrap();

        let stream = spill_manager.read_spill_as_stream(spill_file, None, None)?;

        // Stream schema should be nullable
        assert_eq!(stream.schema(), nullable_schema);

        let batches = stream.try_collect::<Vec<_>>().await?;
        assert_eq!(batches.len(), 2);

        // Both batches must have the SpillManager's nullable schema
        assert_eq!(
            batches[0],
            non_nullable_batch.with_schema(Arc::clone(&nullable_schema))?
        );
        assert_eq!(batches[1], nullable_batch);

        Ok(())
    }
}
