use anyhow::{Context as _, Result, ensure};
use collections::FxHashMap;
use etagere::{BucketedAtlasAllocator, size2};
use gpui::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTextureList, AtlasTile, AtlasTileUpdate,
    Bounds, DevicePixels, PlatformAtlas, Point, Size,
};
#[cfg(any(target_os = "linux", target_os = "freebsd"))]
use gpui::{Dmabuf, DmabufId, DmabufImport, DmabufTarget};
use parking_lot::Mutex;
use std::{borrow::Cow, ops, sync::Arc};

use crate::WgpuContext;

/// DRM fourcc `AR24`: B, G, R, A bytes, the order of the color atlas.
#[cfg(any(target_os = "linux", target_os = "freebsd"))]
const DRM_FORMAT_ARGB8888: u32 = u32::from_le_bytes(*b"AR24");

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    size2(size.width.0, size.height.0)
}

fn etagere_point_to_device(point: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels(point.x),
        y: DevicePixels(point.y),
    }
}

/// Shared so dma-buf imports can release themselves from any thread.
pub struct WgpuAtlas(Arc<Mutex<WgpuAtlasState>>);

struct PendingUpload {
    id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
    data: Vec<u8>,
}

/// A region copied on the GPU from an imported dma-buf into a texture.
#[cfg(any(target_os = "linux", target_os = "freebsd"))]
struct PendingCopy {
    source: DmabufId,
    origin: Point<DevicePixels>,
    id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
}

struct WgpuAtlasState {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_texture_size: u32,
    color_texture_format: wgpu::TextureFormat,
    storage: WgpuAtlasStorage,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    pending_uploads: Vec<PendingUpload>,
    /// Run once the GPU finished the copies queued before them.
    completions: Vec<Box<dyn FnOnce() + Send>>,
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    dmabufs: FxHashMap<DmabufId, wgpu::Texture>,
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    next_dmabuf: u64,
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    pending_copies: Vec<PendingCopy>,
    /// Released while a copy from them was still queued.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    released_dmabufs: Vec<DmabufId>,
}

impl Drop for WgpuAtlasState {
    fn drop(&mut self) {
        // Nothing reads for this atlas anymore: producers may reuse what its
        // last updates read.
        for complete in self.completions.drain(..) {
            complete();
        }
    }
}

pub struct WgpuTextureInfo {
    pub view: wgpu::TextureView,
}

impl WgpuAtlas {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_texture_format: wgpu::TextureFormat,
    ) -> Self {
        let max_texture_size = device.limits().max_texture_dimension_2d;
        WgpuAtlas(Arc::new(Mutex::new(WgpuAtlasState {
            device,
            queue,
            max_texture_size,
            color_texture_format,
            storage: WgpuAtlasStorage::default(),
            tiles_by_key: Default::default(),
            pending_uploads: Vec::new(),
            completions: Vec::new(),
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            dmabufs: Default::default(),
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            next_dmabuf: 1,
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            pending_copies: Vec::new(),
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            released_dmabufs: Vec::new(),
        })))
    }

    pub fn from_context(context: &WgpuContext) -> Self {
        Self::new(
            context.device.clone(),
            context.queue.clone(),
            context.color_texture_format(),
        )
    }

    pub fn before_frame(&self) {
        let (device, queue, copies, completions) = {
            let mut lock = self.0.lock();
            lock.flush_uploads();
            (
                lock.device.clone(),
                lock.queue.clone(),
                lock.take_copies(),
                std::mem::take(&mut lock.completions),
            )
        };
        // Submitting and polling may run completion callbacks, which may call
        // back into the atlas: never hold the lock across them.
        if let Some(copies) = copies {
            queue.submit([copies]);
        }
        if !completions.is_empty() {
            queue.on_submitted_work_done(move || {
                for complete in completions {
                    complete();
                }
            });
        }
        let _ = device.poll(wgpu::PollType::Poll);
    }

    pub fn get_texture_info(&self, id: AtlasTextureId) -> WgpuTextureInfo {
        let lock = self.0.lock();
        let texture = &lock.storage[id];
        WgpuTextureInfo {
            view: texture.view.clone(),
        }
    }

    /// Clears all cached textures and tiles, forcing them to be recreated.
    /// Use this for incremental recovery when the device is still valid.
    pub fn clear(&self) {
        let mut lock = self.0.lock();
        lock.storage = WgpuAtlasStorage::default();
        lock.tiles_by_key.clear();
        lock.pending_uploads.clear();
        // Imports stay valid on the same device; their queued copies have no
        // destination left. Completions still run on the next flush.
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        lock.pending_copies.clear();
    }

    /// Handles device lost by clearing all textures and cached tiles.
    /// The atlas will lazily recreate textures as needed on subsequent frames.
    pub fn handle_device_lost(&self, context: &WgpuContext) {
        let (completions, lost) = {
            let mut lock = self.0.lock();
            let lost = (
                std::mem::replace(&mut lock.device, context.device.clone()),
                std::mem::replace(&mut lock.queue, context.queue.clone()),
            );
            lock.color_texture_format = context.color_texture_format();
            lock.storage = WgpuAtlasStorage::default();
            lock.tiles_by_key.clear();
            lock.pending_uploads.clear();
            // Imports belonged to the lost device: producers must import again.
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            {
                lock.pending_copies.clear();
                lock.released_dmabufs.clear();
                lock.dmabufs.clear();
            }
            (std::mem::take(&mut lock.completions), lost)
        };
        // The lost device reads nothing anymore.
        for complete in completions {
            complete();
        }
        // Dropping its queue runs the callbacks still pending on it.
        drop(lost);
    }
}

impl PlatformAtlas for WgpuAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        if let Some(tile) = lock.tiles_by_key.get(key) {
            Ok(Some(*tile))
        } else {
            profiling::scope!("new tile");
            let Some((size, bytes)) = build()? else {
                return Ok(None);
            };
            let tile = lock
                .allocate(size, key.texture_kind())
                .context("failed to allocate")?;
            lock.upload_texture(tile.texture_id, tile.bounds, &bytes);
            lock.tiles_by_key.insert(key.clone(), tile);
            Ok(Some(tile))
        }
    }

    fn insert_blank(&self, key: &AtlasKey, size: Size<DevicePixels>) -> Result<Option<bool>> {
        let mut lock = self.0.lock();
        if lock.tiles_by_key.contains_key(key) {
            return Ok(Some(false));
        }
        let tile = lock
            .allocate(size, key.texture_kind())
            .context("failed to allocate")?;
        lock.tiles_by_key.insert(key.clone(), tile);
        Ok(Some(true))
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.0.lock();

        let Some(tile) = lock.tiles_by_key.remove(key) else {
            return;
        };
        let id = tile.texture_id;
        // Its queued copies would land on whatever takes its place.
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        lock.pending_copies
            .retain(|copy| copy.id != id || !copy.bounds.intersects(&tile.bounds));

        let Some(texture_slot) = lock.storage[id.kind].textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                lock.pending_uploads
                    .retain(|upload| upload.id != texture.id);
                #[cfg(any(target_os = "linux", target_os = "freebsd"))]
                lock.pending_copies.retain(|copy| copy.id != texture.id);
                lock.storage[id.kind]
                    .free_list
                    .push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }

    fn update_tile(
        &self,
        key: &AtlasKey,
        region: Bounds<DevicePixels>,
        update: AtlasTileUpdate<'_>,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<bool> {
        let mut lock = self.0.lock();
        let queued = lock.queue_update(key, region, update);
        // Even an update that reads nothing later waits: an earlier copy may
        // still read a buffer the caller recycles once this completes.
        lock.completions.extend(on_complete);
        queued
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    fn dmabuf_target(&self) -> Option<DmabufTarget> {
        let (device, format) = {
            let lock = self.0.lock();
            (lock.device.clone(), lock.color_texture_format)
        };
        Some(DmabufTarget {
            render_node: crate::dmabuf::render_node(&device)?,
            // Copies cannot reorder channels into an RGBA atlas.
            modifiers: if format == wgpu::TextureFormat::Bgra8Unorm {
                crate::dmabuf::importable_modifiers(&device)
            } else {
                Vec::new()
            },
        })
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    fn import_dmabuf(&self, dmabuf: &Dmabuf<'_>) -> Result<DmabufImport> {
        let (device, format) = {
            let lock = self.0.lock();
            (lock.device.clone(), lock.color_texture_format)
        };
        ensure!(
            format == wgpu::TextureFormat::Bgra8Unorm,
            "the color atlas is {format:?}, not BGRA"
        );
        ensure!(
            dmabuf.fourcc == DRM_FORMAT_ARGB8888,
            "unsupported dma-buf format {:?}",
            dmabuf.fourcc.to_le_bytes()
        );
        let texture = crate::dmabuf::import(&device, dmabuf)?;
        let mut lock = self.0.lock();
        let id = DmabufId(lock.next_dmabuf);
        lock.next_dmabuf += 1;
        lock.dmabufs.insert(id, texture);
        let state = Arc::downgrade(&self.0);
        Ok(DmabufImport::new(id, move || {
            if let Some(state) = state.upgrade() {
                state.lock().release_dmabuf(id);
            }
        }))
    }
}

impl WgpuAtlasState {
    /// Drops an imported dma-buf once no pending copy reads it.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    fn release_dmabuf(&mut self, id: DmabufId) {
        if self.pending_copies.iter().any(|copy| copy.source == id) {
            self.released_dmabufs.push(id);
        } else {
            // wgpu keeps it alive until submitted copies finish.
            self.dmabufs.remove(&id);
        }
    }

    /// Queues `update`; `false` when `key` has no tile or this format cannot
    /// take it.
    fn queue_update(
        &mut self,
        key: &AtlasKey,
        region: Bounds<DevicePixels>,
        update: AtlasTileUpdate<'_>,
    ) -> Result<bool> {
        let Some(tile) = self.tiles_by_key.get(key).copied() else {
            return Ok(false);
        };
        let Some(format) = self.storage.get(tile.texture_id).map(|texture| texture.format) else {
            return Ok(false);
        };
        let tile_size = tile.bounds.size;
        ensure!(
            region.origin.x.0 >= 0
                && region.origin.y.0 >= 0
                && region.size.width.0 > 0
                && region.size.height.0 > 0
                && region.origin.x.0 + region.size.width.0 <= tile_size.width.0
                && region.origin.y.0 + region.size.height.0 <= tile_size.height.0,
            "region {region:?} lies outside the {tile_size:?} tile"
        );
        let bounds = Bounds {
            origin: tile.bounds.origin + region.origin,
            size: region.size,
        };
        match update {
            AtlasTileUpdate::Pixels {
                bytes,
                bytes_per_row,
            } => {
                let row = region.size.width.0 as usize * 4;
                let rows = region.size.height.0 as usize;
                let stride = bytes_per_row as usize;
                ensure!(
                    stride >= row && bytes.len() >= stride * (rows - 1) + row,
                    "{} bytes cannot hold {rows} rows of {row} bytes, {stride} apart",
                    bytes.len()
                );
                let mut data = Vec::with_capacity(row * rows);
                for line in bytes.chunks(stride).take(rows) {
                    data.extend_from_slice(&line[..row]);
                }
                if format == wgpu::TextureFormat::Rgba8Unorm {
                    for pixel in data.chunks_exact_mut(4) {
                        pixel.swap(0, 2);
                    }
                }
                // Uploads land before this frame's copies: a copy queued
                // earlier would overwrite these newer pixels.
                #[cfg(any(target_os = "linux", target_os = "freebsd"))]
                self.pending_copies.retain(|copy| {
                    copy.id != tile.texture_id || bounds.intersect(&copy.bounds) != copy.bounds
                });
                self.pending_uploads.push(PendingUpload {
                    id: tile.texture_id,
                    bounds,
                    data,
                });
                Ok(true)
            }
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            AtlasTileUpdate::Dmabuf { buffer, origin } => {
                // A GPU copy cannot reorder channels.
                if format != wgpu::TextureFormat::Bgra8Unorm {
                    return Ok(false);
                }
                let Some(source) = self.dmabufs.get(&buffer) else {
                    anyhow::bail!("dma-buf {buffer:?} is not imported");
                };
                ensure!(
                    origin.x.0 >= 0
                        && origin.y.0 >= 0
                        && (origin.x.0 + region.size.width.0) as u32 <= source.width()
                        && (origin.y.0 + region.size.height.0) as u32 <= source.height(),
                    "{:?} at {origin:?} lies outside the {}x{} dma-buf",
                    region.size,
                    source.width(),
                    source.height()
                );
                self.pending_copies.push(PendingCopy {
                    source: buffer,
                    origin,
                    id: tile.texture_id,
                    bounds,
                });
                Ok(true)
            }
        }
    }

    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = &mut self.storage[texture_kind];

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut WgpuAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        let max_texture_size = self.max_texture_size as i32;
        let max_atlas_size = Size {
            width: DevicePixels(max_texture_size),
            height: DevicePixels(max_texture_size),
        };

        let size = min_size.min(&max_atlas_size).max(&DEFAULT_ATLAS_SIZE);
        let format = match kind {
            AtlasTextureKind::Monochrome => wgpu::TextureFormat::R8Unorm,
            AtlasTextureKind::Subpixel | AtlasTextureKind::Polychrome => self.color_texture_format,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                // Tests read tiles back.
                | if cfg!(test) {
                    wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::empty()
                },
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_list = &mut self.storage[kind];
        let index = texture_list.free_list.pop();

        let atlas_texture = WgpuAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            format,
            texture,
            view,
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list
                .textures
                .get_mut(ix)
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list
                .textures
                .last_mut()
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        }
    }

    fn upload_texture(&mut self, id: AtlasTextureId, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let data = self
            .storage
            .get(id)
            .map(|texture| swizzle_upload_data(bytes, texture.format))
            .unwrap_or_else(|| bytes.to_vec());

        self.pending_uploads
            .push(PendingUpload { id, bounds, data });
    }

    fn flush_uploads(&mut self) {
        for upload in self.pending_uploads.drain(..) {
            let Some(texture) = self.storage.get(upload.id) else {
                continue;
            };
            let bytes_per_pixel = texture.bytes_per_pixel();

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.bounds.origin.x.0 as u32,
                        y: upload.bounds.origin.y.0 as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bounds.size.width.0 as u32 * bytes_per_pixel as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: upload.bounds.size.width.0 as u32,
                    height: upload.bounds.size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Encodes queued dma-buf copies. Submitted before the frame's draw, they
    /// run after this frame's texture writes.
    #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
    fn take_copies(&mut self) -> Option<wgpu::CommandBuffer> {
        None
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    fn take_copies(&mut self) -> Option<wgpu::CommandBuffer> {
        let mut commands = None;
        if !self.pending_copies.is_empty() {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dmabuf_copies"),
                });
            for copy in self.pending_copies.drain(..) {
                let (Some(texture), Some(source)) =
                    (self.storage.get(copy.id), self.dmabufs.get(&copy.source))
                else {
                    continue;
                };
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: source,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copy.origin.x.0 as u32,
                            y: copy.origin.y.0 as u32,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copy.bounds.origin.x.0 as u32,
                            y: copy.bounds.origin.y.0 as u32,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: copy.bounds.size.width.0 as u32,
                        height: copy.bounds.size.height.0 as u32,
                        depth_or_array_layers: 1,
                    },
                );
            }
            commands = Some(encoder.finish());
        }
        // wgpu keeps released textures alive until the copies above finish.
        for id in self.released_dmabufs.drain(..) {
            self.dmabufs.remove(&id);
        }
        commands
    }
}

#[derive(Default)]
struct WgpuAtlasStorage {
    monochrome_textures: AtlasTextureList<WgpuAtlasTexture>,
    subpixel_textures: AtlasTextureList<WgpuAtlasTexture>,
    polychrome_textures: AtlasTextureList<WgpuAtlasTexture>,
}

impl ops::Index<AtlasTextureKind> for WgpuAtlasStorage {
    type Output = AtlasTextureList<WgpuAtlasTexture>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl ops::IndexMut<AtlasTextureKind> for WgpuAtlasStorage {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl WgpuAtlasStorage {
    fn get(&self, id: AtlasTextureId) -> Option<&WgpuAtlasTexture> {
        self[id.kind]
            .textures
            .get(id.index as usize)
            .and_then(|t| t.as_ref())
    }
}

impl ops::Index<AtlasTextureId> for WgpuAtlasStorage {
    type Output = WgpuAtlasTexture;
    fn index(&self, id: AtlasTextureId) -> &Self::Output {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        };
        textures[id.index as usize]
            .as_ref()
            .expect("texture must exist")
    }
}

struct WgpuAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    live_atlas_keys: u32,
}

impl WgpuAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn bytes_per_pixel(&self) -> u8 {
        match self.format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => 4,
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn swizzle_upload_data(bytes: &[u8], format: wgpu::TextureFormat) -> Vec<u8> {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => {
            let mut data = bytes.to_vec();
            for pixel in data.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            data
        }
        _ => bytes.to_vec(),
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use gpui::block_on;
    use gpui::{ImageId, RenderImageParams};
    use std::sync::Arc;

    fn test_device_and_queue() -> anyhow::Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request adapter: {error}"))?;
            let descriptor = wgpu::DeviceDescriptor {
                label: Some("wgpu_atlas_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits())
                    .using_alignment(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            };
            // The device GPUI opens, where it can.
            #[cfg(any(target_os = "linux", target_os = "freebsd"))]
            let opened = crate::dmabuf::request_device(&adapter, &descriptor);
            #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
            let opened = None;
            let (device, queue) = match opened {
                Some(opened) => opened,
                None => adapter
                    .request_device(&descriptor)
                    .await
                    .map_err(|error| anyhow::anyhow!("failed to request device: {error}"))?,
            };
            Ok((Arc::new(device), Arc::new(queue)))
        })
    }

    #[test]
    fn before_frame_skips_uploads_for_removed_texture() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;

        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(1),
            frame_index: 0,
        });
        let size = Size {
            width: DevicePixels(1),
            height: DevicePixels(1),
        };
        let mut build = || Ok(Some((size, Cow::Owned(vec![0, 0, 0, 255]))));

        // Regression test: before the fix, this panicked in flush_uploads
        atlas
            .get_or_insert_with(&key, &mut build)?
            .expect("tile should be created");
        atlas.remove(&key);
        atlas.before_frame();
        Ok(())
    }

    #[test]
    fn remove_deallocates_tile_space_for_reuse() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let make_key = |image_id: usize| {
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(image_id),
                frame_index: 0,
            })
        };
        let insert = |key: &AtlasKey, size: Size<DevicePixels>| {
            let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
            atlas
                .get_or_insert_with(key, &mut || {
                    Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
                })
                .expect("allocation should succeed")
                .expect("callback returns Some")
        };

        let keeper_key = make_key(1);
        let big_key_a = make_key(2);
        let big_key_b = make_key(3);

        let keeper_tile = insert(&keeper_key, small);
        let tile_a = insert(&big_key_a, big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert(&big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
        Ok(())
    }

    fn image_key(id: usize) -> AtlasKey {
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(id),
            frame_index: 0,
        })
    }

    fn size(width: i32, height: i32) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(width),
            height: DevicePixels(height),
        }
    }

    fn region(x: i32, y: i32, width: i32, height: i32) -> Bounds<DevicePixels> {
        Bounds {
            origin: Point {
                x: DevicePixels(x),
                y: DevicePixels(y),
            },
            size: size(width, height),
        }
    }

    /// Flushes pending work and returns the tile's pixels, tightly packed.
    fn read_tile(atlas: &WgpuAtlas, tile: AtlasTile) -> Vec<u8> {
        atlas.before_frame();
        let lock = atlas.0.lock();
        let (width, height) = (
            tile.bounds.size.width.0 as u32,
            tile.bounds.size.height.0 as u32,
        );
        let padded = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = lock.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = lock
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &lock.storage[tile.texture_id].texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: tile.bounds.origin.x.0 as u32,
                    y: tile.bounds.origin.y.0 as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        lock.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        lock.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("readback");
        let data = slice.get_mapped_range();
        data.chunks(padded as usize)
            .flat_map(|row| row[..width as usize * 4].to_vec())
            .collect()
    }

    fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * width + x) * 4) as usize;
        pixels[at..at + 4].try_into().unwrap()
    }

    /// A tile away from its texture's corner, so offsets within the atlas
    /// matter.
    fn offset_tile(atlas: &WgpuAtlas, key: &AtlasKey, size: Size<DevicePixels>, bytes: Vec<u8>) -> AtlasTile {
        let corner = image_key(usize::MAX);
        atlas
            .get_or_insert_with(&corner, &mut || Ok(Some((self::size(3, 3), Cow::Owned(vec![0; 36])))))
            .unwrap();
        let tile = atlas
            .get_or_insert_with(key, &mut || Ok(Some((size, Cow::Owned(bytes.clone())))))
            .unwrap()
            .expect("tile");
        assert_ne!(tile.bounds.origin, Point::default(), "the tile sits at its texture's corner");
        tile
    }

    #[test]
    fn update_tile_overwrites_only_the_region() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let key = image_key(7);
        let tile = offset_tile(&atlas, &key, size(8, 6), [10, 20, 30, 255].repeat(48));
        // A 3x2 region at (4, 3), read from rows 16 bytes apart.
        let mut source = vec![0u8; 16 * 2];
        for row in source.chunks_mut(16) {
            row[..12].copy_from_slice(&[1, 2, 3, 255].repeat(3));
        }
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = completed.clone();
        assert!(atlas.update_tile(
            &key,
            region(4, 3, 3, 2),
            AtlasTileUpdate::Pixels {
                bytes: &source,
                bytes_per_row: 16,
            },
            Some(Box::new(move || {
                flag.store(true, std::sync::atomic::Ordering::SeqCst)
            })),
        )?);
        // Only once the GPU finished what was queued before.
        assert!(!completed.load(std::sync::atomic::Ordering::SeqCst));
        let pixels = read_tile(&atlas, tile);
        assert!(completed.load(std::sync::atomic::Ordering::SeqCst));
        for y in 0..6 {
            for x in 0..8 {
                let inside = (4..7).contains(&x) && (3..5).contains(&y);
                let expected = if inside {
                    [1, 2, 3, 255]
                } else {
                    [10, 20, 30, 255]
                };
                assert_eq!(pixel(&pixels, 8, x, y), expected, "pixel ({x}, {y})");
            }
        }
        Ok(())
    }

    #[test]
    fn update_tile_needs_a_cached_tile_inside_its_bounds() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let key = image_key(8);
        let pixels = [0u8; 16];
        let update = || AtlasTileUpdate::Pixels {
            bytes: &pixels,
            bytes_per_row: 8,
        };
        assert!(!atlas.update_tile(&key, region(0, 0, 2, 2), update(), None)?);
        atlas.get_or_insert_with(&key, &mut || {
            Ok(Some((size(4, 4), Cow::Owned(vec![0; 64]))))
        })?;
        assert!(
            atlas
                .update_tile(&key, region(3, 3, 2, 2), update(), None)
                .is_err()
        );
        assert!(
            atlas
                .update_tile(&key, region(0, 0, 2, 3), update(), None)
                .is_err(),
            "16 bytes cannot hold three 8-byte rows"
        );
        atlas.remove(&key);
        assert!(!atlas.update_tile(&key, region(0, 0, 2, 2), update(), None)?);
        Ok(())
    }

    /// Renders into dma-bufs through GBM and GL, as a browser engine does.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    mod producer {
        use anyhow::{Result, ensure};
        use std::{
            ffi::{CStr, c_void},
            os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd},
            ptr::null_mut,
        };

        const GBM_BO_USE_RENDERING: u32 = 1 << 2;
        const EGL_PLATFORM_GBM_KHR: u32 = 0x31D7;
        const EGL_OPENGL_ES_API: u32 = 0x30A0;
        const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
        const EGL_NONE: i32 = 0x3038;
        const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
        const EGL_WIDTH: i32 = 0x3057;
        const EGL_HEIGHT: i32 = 0x3056;
        const EGL_LINUX_DRM_FOURCC_EXT: i32 = 0x3271;
        const EGL_DMA_BUF_PLANE0_FD_EXT: i32 = 0x3272;
        const EGL_DMA_BUF_PLANE0_OFFSET_EXT: i32 = 0x3273;
        const EGL_DMA_BUF_PLANE0_PITCH_EXT: i32 = 0x3274;
        const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: i32 = 0x3443;
        const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: i32 = 0x3444;
        const GL_FRAMEBUFFER: u32 = 0x8D40;
        const GL_RENDERBUFFER: u32 = 0x8D41;
        const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
        const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
        const GL_SCISSOR_TEST: u32 = 0x0C11;
        const GL_COLOR_BUFFER_BIT: u32 = 0x4000;

        #[link(name = "gbm")]
        unsafe extern "C" {
            fn gbm_create_device(fd: i32) -> *mut c_void;
            fn gbm_device_destroy(device: *mut c_void);
            fn gbm_bo_create_with_modifiers2(
                device: *mut c_void,
                width: u32,
                height: u32,
                format: u32,
                modifiers: *const u64,
                count: u32,
                flags: u32,
            ) -> *mut c_void;
            fn gbm_bo_destroy(bo: *mut c_void);
            fn gbm_bo_get_fd(bo: *mut c_void) -> i32;
            fn gbm_bo_get_stride(bo: *mut c_void) -> u32;
            fn gbm_bo_get_offset(bo: *mut c_void, plane: i32) -> u32;
            fn gbm_bo_get_modifier(bo: *mut c_void) -> u64;
            fn gbm_bo_get_plane_count(bo: *mut c_void) -> i32;
        }

        #[link(name = "EGL")]
        unsafe extern "C" {
            fn eglGetPlatformDisplay(
                platform: u32,
                native: *mut c_void,
                attributes: *const isize,
            ) -> *mut c_void;
            fn eglInitialize(display: *mut c_void, major: *mut i32, minor: *mut i32) -> u32;
            fn eglTerminate(display: *mut c_void) -> u32;
            fn eglBindAPI(api: u32) -> u32;
            fn eglCreateContext(
                display: *mut c_void,
                config: *mut c_void,
                share: *mut c_void,
                attributes: *const i32,
            ) -> *mut c_void;
            fn eglDestroyContext(display: *mut c_void, context: *mut c_void) -> u32;
            fn eglMakeCurrent(
                display: *mut c_void,
                draw: *mut c_void,
                read: *mut c_void,
                context: *mut c_void,
            ) -> u32;
            fn eglGetProcAddress(name: *const i8) -> *mut c_void;
        }

        #[link(name = "GLESv2")]
        unsafe extern "C" {
            fn glGenRenderbuffers(n: i32, names: *mut u32);
            fn glDeleteRenderbuffers(n: i32, names: *const u32);
            fn glBindRenderbuffer(target: u32, name: u32);
            fn glGenFramebuffers(n: i32, names: *mut u32);
            fn glDeleteFramebuffers(n: i32, names: *const u32);
            fn glBindFramebuffer(target: u32, name: u32);
            fn glFramebufferRenderbuffer(target: u32, attachment: u32, rb_target: u32, rb: u32);
            fn glCheckFramebufferStatus(target: u32) -> u32;
            fn glEnable(cap: u32);
            fn glScissor(x: i32, y: i32, width: i32, height: i32);
            fn glClearColor(r: f32, g: f32, b: f32, a: f32);
            fn glClear(mask: u32);
            fn glFinish();
        }

        type CreateImage = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u32,
            *mut c_void,
            *const i32,
        ) -> *mut c_void;
        type DestroyImage = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
        type ImageStorage = unsafe extern "C" fn(u32, *mut c_void);

        /// A GL context on one render node.
        pub struct Producer {
            _node: std::fs::File,
            gbm: *mut c_void,
            display: *mut c_void,
            context: *mut c_void,
        }

        /// A buffer GBM allocated for rendering.
        pub struct Buffer {
            bo: *mut c_void,
            pub fd: OwnedFd,
            pub modifier: u64,
            pub offset: u32,
            pub stride: u32,
            pub width: u32,
            pub height: u32,
        }

        impl Drop for Buffer {
            fn drop(&mut self) {
                unsafe { gbm_bo_destroy(self.bo) };
            }
        }

        fn proc<T>(name: &CStr) -> Result<T> {
            let address = unsafe { eglGetProcAddress(name.as_ptr()) };
            ensure!(!address.is_null(), "no {name:?}");
            Ok(unsafe { std::mem::transmute_copy(&address) })
        }

        impl Producer {
            pub fn new(node: &std::path::Path) -> Result<Self> {
                let node = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(node)?;
                let gbm = unsafe { gbm_create_device(node.as_raw_fd()) };
                ensure!(!gbm.is_null(), "no GBM device");
                let mut producer = Self {
                    _node: node,
                    gbm,
                    display: null_mut(),
                    context: null_mut(),
                };
                producer.display =
                    unsafe { eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm, std::ptr::null()) };
                ensure!(
                    !producer.display.is_null()
                        && unsafe { eglInitialize(producer.display, null_mut(), null_mut()) } != 0,
                    "no EGL display"
                );
                let attributes = [EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE];
                producer.context = unsafe {
                    eglBindAPI(EGL_OPENGL_ES_API);
                    eglCreateContext(producer.display, null_mut(), null_mut(), attributes.as_ptr())
                };
                ensure!(!producer.context.is_null(), "no GLES context");
                Ok(producer)
            }

            /// A buffer in exactly `modifier`, or `None` when the GPU cannot
            /// render into that layout.
            pub fn buffer(&self, width: u32, height: u32, modifier: u64) -> Option<Buffer> {
                let bo = unsafe {
                    gbm_bo_create_with_modifiers2(
                        self.gbm,
                        width,
                        height,
                        super::DRM_FORMAT_ARGB8888,
                        &modifier,
                        1,
                        GBM_BO_USE_RENDERING,
                    )
                };
                if bo.is_null() {
                    return None;
                }
                let fd = unsafe { gbm_bo_get_fd(bo) };
                if fd < 0 || unsafe { gbm_bo_get_plane_count(bo) } != 1 {
                    unsafe { gbm_bo_destroy(bo) };
                    return None;
                }
                Some(Buffer {
                    bo,
                    fd: unsafe { OwnedFd::from_raw_fd(fd) },
                    modifier: unsafe { gbm_bo_get_modifier(bo) },
                    offset: unsafe { gbm_bo_get_offset(bo, 0) },
                    stride: unsafe { gbm_bo_get_stride(bo) },
                    width,
                    height,
                })
            }

            /// Clears each `(x, y, width, height)` rectangle of `buffer`, in
            /// rows from the first in memory, to its BGRA color, then waits
            /// for the GPU.
            pub fn paint(&self, buffer: &Buffer, rects: &[([i32; 4], [u8; 4])]) -> Result<()> {
                let create: CreateImage = proc(c"eglCreateImageKHR")?;
                let destroy: DestroyImage = proc(c"eglDestroyImageKHR")?;
                let storage: ImageStorage = proc(c"glEGLImageTargetRenderbufferStorageOES")?;
                let attributes = [
                    EGL_WIDTH,
                    buffer.width as i32,
                    EGL_HEIGHT,
                    buffer.height as i32,
                    EGL_LINUX_DRM_FOURCC_EXT,
                    super::DRM_FORMAT_ARGB8888 as i32,
                    EGL_DMA_BUF_PLANE0_FD_EXT,
                    buffer.fd.as_raw_fd(),
                    EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                    buffer.offset as i32,
                    EGL_DMA_BUF_PLANE0_PITCH_EXT,
                    buffer.stride as i32,
                    EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                    buffer.modifier as u32 as i32,
                    EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                    (buffer.modifier >> 32) as u32 as i32,
                    EGL_NONE,
                ];
                unsafe {
                    ensure!(
                        eglMakeCurrent(self.display, null_mut(), null_mut(), self.context) != 0,
                        "cannot make the context current"
                    );
                    let image = create(
                        self.display,
                        null_mut(),
                        EGL_LINUX_DMA_BUF_EXT,
                        null_mut(),
                        attributes.as_ptr(),
                    );
                    ensure!(!image.is_null(), "EGL cannot import the buffer");
                    let (mut renderbuffer, mut framebuffer) = (0, 0);
                    glGenRenderbuffers(1, &mut renderbuffer);
                    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
                    storage(GL_RENDERBUFFER, image);
                    glGenFramebuffers(1, &mut framebuffer);
                    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
                    glFramebufferRenderbuffer(
                        GL_FRAMEBUFFER,
                        GL_COLOR_ATTACHMENT0,
                        GL_RENDERBUFFER,
                        renderbuffer,
                    );
                    let complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
                    if complete {
                        glEnable(GL_SCISSOR_TEST);
                        for ([x, y, width, height], [b, g, r, a]) in rects {
                            glScissor(*x, *y, *width, *height);
                            let unit = |v: &u8| *v as f32 / 255.;
                            glClearColor(unit(r), unit(g), unit(b), unit(a));
                            glClear(GL_COLOR_BUFFER_BIT);
                        }
                        glFinish();
                    }
                    glDeleteFramebuffers(1, &framebuffer);
                    glDeleteRenderbuffers(1, &renderbuffer);
                    destroy(self.display, image);
                    eglMakeCurrent(self.display, null_mut(), null_mut(), null_mut());
                    ensure!(complete, "the buffer is not a complete framebuffer");
                }
                Ok(())
            }
        }

        impl Drop for Producer {
            fn drop(&mut self) {
                unsafe {
                    if !self.context.is_null() {
                        eglDestroyContext(self.display, self.context);
                    }
                    if !self.display.is_null() {
                        eglTerminate(self.display);
                    }
                    gbm_device_destroy(self.gbm);
                }
            }
        }
    }

    /// Pages a GPU rendered with GL, in every layout the device imports,
    /// copied region by region into a tile without touching the CPU. Runs on
    /// every Vulkan GPU, as hybrid laptops render on either.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn dmabuf_regions_copy_into_the_tile() -> anyhow::Result<()> {
        use std::{os::fd::AsFd as _, sync::atomic::Ordering::SeqCst};
        const BLUE: [u8; 4] = [255, 0, 0, 255];
        const RED: [u8; 4] = [0, 0, 255, 255];
        const GREEN: [u8; 4] = [0, 255, 0, 255];
        const NONE: [u8; 4] = [0, 0, 0, 0];
        // Odd sizes, so tiled layouts end mid-tile.
        let (width, height) = (321u32, 201u32);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let mut tested = 0;
        for adapter in block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN)) {
            let name = adapter.get_info().name;
            let descriptor = wgpu::DeviceDescriptor {
                label: Some("wgpu_atlas_dmabuf_test"),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits())
                    .using_alignment(adapter.limits()),
                ..Default::default()
            };
            let Some((device, queue)) = crate::dmabuf::request_device(&adapter, &descriptor) else {
                eprintln!("{name}: skipped, cannot import tiled dma-bufs");
                continue;
            };
            let atlas = WgpuAtlas::new(
                Arc::new(device),
                Arc::new(queue),
                wgpu::TextureFormat::Bgra8Unorm,
            );
            let Some(target) = atlas.dmabuf_target() else {
                eprintln!("{name}: skipped, no DRM render node");
                continue;
            };
            let producer = producer::Producer::new(&target.render_node)?;
            let mut layouts = Vec::new();
            for (index, &modifier) in target.modifiers.iter().enumerate() {
                let Some(buffer) = producer.buffer(width, height, modifier) else {
                    continue;
                };
                assert_eq!(buffer.modifier, modifier);
                producer.paint(
                    &buffer,
                    &[
                        ([0, 0, 321, 201], BLUE),
                        ([160, 0, 161, 150], RED),
                        ([0, 150, 321, 51], GREEN),
                    ],
                )?;
                let import = atlas.import_dmabuf(&Dmabuf {
                    fd: buffer.fd.as_fd(),
                    width,
                    height,
                    fourcc: DRM_FORMAT_ARGB8888,
                    modifier,
                    offset: buffer.offset,
                    stride: buffer.stride,
                })?;
                let key = image_key(100 + index);
                let tile = offset_tile(
                    &atlas,
                    &key,
                    size(width as i32, height as i32),
                    vec![0; (width * height * 4) as usize],
                );
                // A frame's damage: the red right half, a band of the green
                // bottom, and a corner of the green moved to the top left.
                let done = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                for (to, from) in [
                    (region(160, 0, 161, 150), (160, 0)),
                    (region(0, 190, 321, 11), (0, 190)),
                    (region(0, 0, 16, 8), (305, 193)),
                ] {
                    let done = done.clone();
                    assert!(atlas.update_tile(
                        &key,
                        to,
                        AtlasTileUpdate::Dmabuf {
                            buffer: import.id(),
                            origin: Point {
                                x: DevicePixels(from.0),
                                y: DevicePixels(from.1),
                            },
                        },
                        Some(Box::new(move || {
                            done.fetch_add(1, SeqCst);
                        })),
                    )?);
                }
                let at = format!("{name}, modifier {modifier:#018x}");
                assert_eq!(done.load(SeqCst), 0, "{at}: done before the GPU copied");
                // Released from another thread while its copies are queued.
                std::thread::spawn(move || drop(import)).join().unwrap();
                let pixels = read_tile(&atlas, tile);
                assert_eq!(done.load(SeqCst), 3, "{at}");
                assert!(atlas.0.lock().dmabufs.is_empty(), "{at}: released after its copies");
                for (x, y, expected, what) in [
                    (5, 5, GREEN, "moved corner"),
                    (15, 7, GREEN, "moved corner's last pixel"),
                    (16, 7, NONE, "beside the moved corner"),
                    (10, 10, NONE, "untouched"),
                    (159, 100, NONE, "left of the red"),
                    (160, 0, RED, "red's first pixel"),
                    (320, 149, RED, "red's last pixel"),
                    (200, 160, NONE, "outside the damage"),
                    (10, 195, GREEN, "green band"),
                    (320, 200, GREEN, "last pixel"),
                ] {
                    assert_eq!(pixel(&pixels, width, x, y), expected, "{at}: {what} ({x}, {y})");
                }
                // Pixels queued after a copy into the same place win, and a
                // removed tile's copies spare the tile that takes its place.
                let import = atlas.import_dmabuf(&Dmabuf {
                    fd: buffer.fd.as_fd(),
                    width,
                    height,
                    fourcc: DRM_FORMAT_ARGB8888,
                    modifier,
                    offset: buffer.offset,
                    stride: buffer.stride,
                })?;
                let whole = region(0, 0, width as i32, height as i32);
                let copy = || AtlasTileUpdate::Dmabuf {
                    buffer: import.id(),
                    origin: Point::default(),
                };
                let white = [255u8; 4].repeat((width * height) as usize);
                assert!(atlas.update_tile(&key, whole, copy(), None)?);
                let pixels = AtlasTileUpdate::Pixels {
                    bytes: &white,
                    bytes_per_row: width * 4,
                };
                assert!(atlas.update_tile(&key, whole, pixels, None)?);
                let pixels = read_tile(&atlas, tile);
                assert_eq!(pixel(&pixels, width, 200, 100), [255; 4], "{at}: pixels after a copy");
                assert!(atlas.update_tile(&key, whole, copy(), None)?);
                atlas.remove(&key);
                let next = image_key(200 + index);
                let replacement = atlas
                    .get_or_insert_with(&next, &mut || {
                        Ok(Some((size(width as i32, height as i32), Cow::Owned(white.clone()))))
                    })?
                    .expect("tile");
                assert_eq!(replacement.bounds, tile.bounds, "{at}: a tile in the freed place");
                let pixels = read_tile(&atlas, replacement);
                assert_eq!(pixel(&pixels, width, 200, 100), [255; 4], "{at}: a removed tile's copy");
                atlas.remove(&next);
                layouts.push(format!("{modifier:#018x}"));
            }
            eprintln!("{name}: copied pages in {} layouts: {}", layouts.len(), layouts.join(" "));
            tested += layouts.len();
        }
        if tested == 0 {
            eprintln!("skipped: no GPU imports dma-bufs it renders");
        }
        Ok(())
    }

    /// Layouts and sizes the device does not support fail the import before
    /// Vulkan sees them, as creating such images is undefined behavior.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn dmabuf_imports_refuse_unsupported_layouts_and_sizes() -> anyhow::Result<()> {
        use std::os::fd::AsFd as _;
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let file = std::fs::File::open("/dev/null")?;
        let import = |width: u32, modifier: u64| {
            atlas.import_dmabuf(&Dmabuf {
                fd: file.as_fd(),
                width,
                height: 64,
                fourcc: DRM_FORMAT_ARGB8888,
                modifier,
                offset: 0,
                stride: width * 4,
            })
        };
        // DRM_FORMAT_MOD_INVALID names no layout at all.
        let error = import(256, 0x00ff_ffff_ffff_ffff).unwrap_err();
        assert!(format!("{error:#}").contains("modifier"), "{error:#}");
        let modifiers = atlas.dmabuf_target().map(|target| target.modifiers);
        if let Some(&modifier) = modifiers.as_deref().and_then(|m| m.first()) {
            let error = import(1 << 20, modifier).unwrap_err();
            assert!(format!("{error:#}").contains("at most"), "{error:#}");
        }
        Ok(())
    }

    #[test]
    fn swizzle_upload_data_preserves_bgra_uploads() {
        let input = vec![0x10, 0x20, 0x30, 0x40];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Bgra8Unorm),
            input
        );
    }

    #[test]
    fn swizzle_upload_data_converts_bgra_to_rgba() {
        let input = vec![0x10, 0x20, 0x30, 0x40, 0xAA, 0xBB, 0xCC, 0xDD];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Rgba8Unorm),
            vec![0x30, 0x20, 0x10, 0x40, 0xCC, 0xBB, 0xAA, 0xDD]
        );
    }
}
