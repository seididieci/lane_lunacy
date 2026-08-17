// SPDX-License-Identifier: MIT

//! Background chunk builder shared by every presenter.
//!
//! World chunks are a pure, deterministic function of world coordinates, so
//! they can be generated *and* uploaded to GPU buffers before the player ever
//! reaches them. A small worker pool builds and uploads the chunks the car is
//! heading toward (`FrameBuilder` prefetches a few past the leading edge); the
//! render thread only swaps in committed results, so a chunk crossing no longer
//! blocks rendering. The headless snapshot path uses the same cache and simply
//! waits for its first window, so the output is unchanged.
//!
//! The jobs flow over a shared `Mutex<VecDeque>` + `Condvar` (the render thread
//! enqueues, every worker pops), completed meshes flow back over a one-way
//! mpsc channel. On drop the cache flips a shutdown flag and joins the workers,
//! so the allocator Arcs they borrow are released before the scene is torn down.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

use crate::mesh::{build_world_chunk, TerrainDetail};
use crate::render::WORLD_CHUNK_LEN;
use crate::vertex::Vertex3d;

/// A single world-chunk mesh: vertex buffer + index buffer.
pub type WorldChunk = (Subbuffer<[Vertex3d]>, Subbuffer<[u32]>);

/// Worker threads in the chunk pool, capped at 4: a chunk build is CPU-bound
/// pure math, and beyond a few threads the allocator mutex and memory bandwidth
/// stop scaling.
const MAX_CHUNK_WORKERS: usize = 4;

/// How long `wait_for` waits for the workers before building stragglers inline
/// on the render thread. A pure safety net: normal crossings return instantly
/// (the chunk was prefetched) and even a full-window teleport fits in a small
/// fraction of this.
const WAIT_DEADLINE: Duration = Duration::from_secs(2);

struct ChunkJob {
    index: i32,
    detail: TerrainDetail,
    allocator: Arc<StandardMemoryAllocator>,
    generation: u64,
}

struct ChunkDone {
    index: i32,
    chunk: WorldChunk,
    generation: u64,
}

/// Job queue plus the flags the workers watch.
struct Shared {
    jobs: Mutex<VecDeque<ChunkJob>>,
    wake: Condvar,
    shutdown: AtomicBool,
}

impl Shared {
    fn new() -> Self {
        Shared {
            jobs: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            shutdown: AtomicBool::new(false),
        }
    }
}

pub(crate) struct ChunkCache {
    /// Committed chunk meshes by index, including prefetched chunks that are
    /// not yet inside the render window (they hit on a later crossing).
    ready: Mutex<HashMap<i32, WorldChunk>>,
    /// Indices requested but not yet committed (dedupes in-flight requests).
    pending: Mutex<HashSet<i32>>,
    /// Invalidates results produced before a terrain-detail change / reset.
    generation: u64,
    shared: Arc<Shared>,
    done_tx: Sender<ChunkDone>,
    done_rx: Receiver<ChunkDone>,
    workers_spawned: bool,
    workers: Vec<JoinHandle<()>>,
}

impl ChunkCache {
    pub(crate) fn new() -> Self {
        let (done_tx, done_rx) = channel();
        ChunkCache {
            ready: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashSet::new()),
            generation: 0,
            shared: Arc::new(Shared::new()),
            done_tx,
            done_rx,
            workers_spawned: false,
            workers: Vec::new(),
        }
    }

    fn ensure_workers(&mut self) {
        if self.workers_spawned {
            return;
        }
        self.workers_spawned = true;
        let count = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(2)
            .clamp(1, MAX_CHUNK_WORKERS);
        let done_tx = self.done_tx.clone();
        self.workers = (0..count)
            .map(|_| {
                let shared = self.shared.clone();
                let done_tx = done_tx.clone();
                std::thread::Builder::new()
                    .name("chunk-builder".into())
                    .spawn(move || worker_loop(shared, done_tx))
                    .expect("failed to spawn chunk builder thread")
            })
            .collect();
    }

    /// Queues a build for every `index` that is neither committed nor in flight.
    pub(crate) fn request(
        &mut self,
        indices: &[i32],
        detail: TerrainDetail,
        allocator: Arc<StandardMemoryAllocator>,
    ) {
        self.ensure_workers();
        let mut jobs = Vec::new();
        {
            let mut pending = self.pending.lock().unwrap();
            let ready = self.ready.lock().unwrap();
            for &index in indices {
                if ready.contains_key(&index) || pending.contains(&index) {
                    continue;
                }
                pending.insert(index);
                jobs.push(ChunkJob {
                    index,
                    detail,
                    allocator: allocator.clone(),
                    generation: self.generation,
                });
            }
        } // guards dropped: the jobs queue is pushed without holding them
        if !jobs.is_empty() {
            let mut queue = self.shared.jobs.lock().unwrap();
            queue.extend(jobs);
            self.shared.wake.notify_all();
        }
    }

    /// Commits finished builds into the ready map, dropping stale-generation
    /// results (e.g. from before a terrain-detail change). Called every frame,
    /// not only on a crossing, so prefetched chunks are ready well before the
    /// crossing that needs them.
    pub(crate) fn poll(&self) {
        let mut pending = self.pending.lock().unwrap();
        let mut ready = self.ready.lock().unwrap();
        while let Ok(done) = self.done_rx.try_recv() {
            pending.remove(&done.index);
            if done.generation == self.generation {
                ready.insert(done.index, done.chunk);
            }
        }
    }

    /// Blocks until every `want` index is committed, building stragglers inline
    /// after a generous deadline (never a permanent block). A normal crossing
    /// returns immediately because the chunk was prefetched in an earlier frame.
    pub(crate) fn wait_for(
        &self,
        want: &[i32],
        detail: TerrainDetail,
        allocator: Arc<StandardMemoryAllocator>,
    ) {
        let started = Instant::now();
        loop {
            self.poll();
            let ready = self.ready.lock().unwrap();
            let missing: Vec<i32> = want
                .iter()
                .copied()
                .filter(|i| !ready.contains_key(i))
                .collect();
            drop(ready);
            if missing.is_empty() {
                return;
            }
            if started.elapsed() > WAIT_DEADLINE {
                for &index in &missing {
                    self.build_inline(index, detail, allocator.clone());
                }
                return;
            }
            match self.done_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(done) => {
                    self.pending.lock().unwrap().remove(&done.index);
                    if done.generation == self.generation {
                        self.ready.lock().unwrap().insert(done.index, done.chunk);
                    }
                }
                Err(_) => continue,
            }
        }
    }

    fn build_inline(
        &self,
        index: i32,
        detail: TerrainDetail,
        allocator: Arc<StandardMemoryAllocator>,
    ) {
        self.pending.lock().unwrap().remove(&index);
        let chunk = build_chunk_buffers(index, detail, allocator);
        self.ready.lock().unwrap().insert(index, chunk);
    }

    /// Removes a committed chunk and returns it (the render window consumes it).
    pub(crate) fn take(&self, index: i32) -> Option<WorldChunk> {
        self.ready.lock().unwrap().remove(&index)
    }

    /// Drops cached chunks outside `[min, max]` (the window plus a prefetch
    /// margin) so the cache stays bounded while the car drives on.
    pub(crate) fn evict_outside(&self, min: i32, max: i32) {
        let mut ready = self.ready.lock().unwrap();
        ready.retain(|&index, _| index >= min && index <= max);
    }

    /// Drops everything and invalidates in-flight builds (terrain-detail change).
    pub(crate) fn reset(&mut self) {
        self.generation += 1;
        self.ready.lock().unwrap().clear();
        self.pending.lock().unwrap().clear();
        // Drain the not-yet-started jobs: they belong to the old detail.
        self.shared.jobs.lock().unwrap().clear();
    }

    /// Number of chunks committed and ready (including prefetched ones).
    pub(crate) fn cached_count(&self) -> usize {
        self.ready.lock().unwrap().len()
    }

    /// Number of chunk builds requested but not yet committed.
    pub(crate) fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }
}

impl Drop for ChunkCache {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Builds a chunk mesh and uploads it to GPU buffers. Pure and deterministic,
/// so it runs equally well on a worker thread or inline.
fn build_chunk_buffers(
    index: i32,
    detail: TerrainDetail,
    allocator: Arc<StandardMemoryAllocator>,
) -> WorldChunk {
    let (wv, wi) = build_world_chunk(index as f32 * WORLD_CHUNK_LEN, WORLD_CHUNK_LEN, detail);
    let world_vertices = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        wv,
    )
    .expect("world chunk vertices");
    let world_indices = Buffer::from_iter(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        wi,
    )
    .expect("world chunk indices");
    (world_vertices, world_indices)
}

fn worker_loop(shared: Arc<Shared>, done_tx: Sender<ChunkDone>) {
    loop {
        let job = {
            let mut jobs = shared.jobs.lock().unwrap();
            if shared.shutdown.load(Ordering::SeqCst) && jobs.is_empty() {
                return;
            }
            while jobs.is_empty() && !shared.shutdown.load(Ordering::SeqCst) {
                jobs = shared.wake.wait(jobs).unwrap();
            }
            jobs.pop_front()
        };
        let Some(job) = job else {
            continue;
        };
        let chunk = build_chunk_buffers(job.index, job.detail, job.allocator);
        if done_tx
            .send(ChunkDone {
                index: job.index,
                chunk,
                generation: job.generation,
            })
            .is_err()
        {
            return;
        }
    }
}
