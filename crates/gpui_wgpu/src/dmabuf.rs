//! Imports dma-bufs other processes render into as Vulkan images, linear or
//! tiled, so the atlas can copy their damaged regions into its textures on
//! the GPU.

use anyhow::{Context as _, Result, anyhow, ensure};
use ash::vk;
use gpui::Dmabuf;
use std::{
    io::{Seek as _, SeekFrom},
    os::fd::{AsRawFd as _, IntoRawFd as _},
    path::PathBuf,
};
use wgpu::hal::api::Vulkan;

/// DRM's `AR24`, the only format the atlas imports: B, G, R, A bytes.
const FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// Lets images take a dma-buf's layout, tiled ones included; wgpu leaves it off.
const MODIFIERS: &std::ffi::CStr = ash::ext::image_drm_format_modifier::NAME;

const IMPORT_EXTENSIONS: [&std::ffi::CStr; 3] = [
    ash::khr::external_memory_fd::NAME,
    ash::ext::external_memory_dma_buf::NAME,
    MODIFIERS,
];

/// Opens a device on `adapter` that can import dma-bufs in any layout the GPU
/// renders, not only linear ones, which some GPUs cannot render into. `None`
/// when the adapter cannot, so the caller opens a plain device instead.
pub(crate) fn request_device(
    adapter: &wgpu::Adapter,
    desc: &wgpu::DeviceDescriptor<'_>,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    // wgpu checks these when it opens the device itself, not from here: let
    // that path report what the adapter lacks.
    if !adapter.features().contains(desc.required_features)
        || !desc.required_limits.check_limits(&adapter.limits())
    {
        return None;
    }
    let open = {
        let hal = unsafe { adapter.as_hal::<Vulkan>() }?;
        let capabilities = hal.physical_device_capabilities();
        let version = capabilities
            .properties()
            .api_version
            .min(hal.shared_instance().instance_api_version());
        // Vulkan 1.2 holds every extension this one depends on.
        if version < vk::API_VERSION_1_2 || !capabilities.supports_extension(MODIFIERS) {
            return None;
        }
        let callback: Box<wgpu::hal::vulkan::CreateDeviceCallback> = Box::new(|args| {
            if !args.extensions.contains(&MODIFIERS) {
                args.extensions.push(MODIFIERS);
            }
        });
        let open = unsafe {
            hal.open_with_callback(
                desc.required_features,
                &desc.required_limits,
                &desc.memory_hints,
                Some(callback),
            )
        };
        match open {
            Ok(open) => open,
            Err(error) => {
                log::warn!("no device that imports tiled dma-bufs: {error}");
                return None;
            }
        }
    };
    unsafe { adapter.create_device_from_hal(open, desc) }
        .inspect_err(|error| log::warn!("no device that imports tiled dma-bufs: {error}"))
        .ok()
}

/// The DRM render node of the physical GPU behind `device`.
pub(crate) fn render_node(device: &wgpu::Device) -> Option<PathBuf> {
    let hal = unsafe { device.as_hal::<Vulkan>() }?;
    let instance = hal.shared_instance();
    if instance.instance_api_version() < vk::API_VERSION_1_1 {
        return None;
    }
    let raw = instance.raw_instance();
    let physical = hal.raw_physical_device();
    let supported = unsafe { raw.enumerate_device_extension_properties(physical) }
        .ok()?
        .iter()
        .any(|extension| {
            extension.extension_name_as_c_str() == Ok(ash::ext::physical_device_drm::NAME)
        });
    if !supported {
        return None;
    }
    let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
    unsafe { raw.get_physical_device_properties2(physical, &mut properties) };
    if drm.has_render == vk::FALSE {
        return None;
    }
    device_node(drm.render_major as u64, drm.render_minor as u64)
}

fn device_node(major: u64, minor: u64) -> Option<PathBuf> {
    use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _};
    // Linux dev_t layout (see major()/minor() in sys/sysmacros.h).
    let split = |dev: u64| {
        (
            ((dev >> 32) & 0xffff_f000) | ((dev >> 8) & 0x0000_0fff),
            ((dev >> 12) & 0xffff_ff00) | (dev & 0x0000_00ff),
        )
    };
    std::fs::read_dir("/dev/dri")
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            std::fs::metadata(path).is_ok_and(|metadata| {
                metadata.file_type().is_char_device() && split(metadata.rdev()) == (major, minor)
            })
        })
}

/// The format modifiers of the `AR24` dma-bufs [`import`] takes.
pub(crate) fn importable_modifiers(device: &wgpu::Device) -> Vec<u64> {
    unsafe { device.as_hal::<Vulkan>() }
        .map(|hal| layouts(&hal).into_iter().map(|(modifier, _)| modifier).collect())
        .unwrap_or_default()
}

/// The `AR24` layouts the device imports from dma-bufs, each with the largest
/// image it takes: uncompressed, single-plane ones the GPU can copy from, as
/// producers share one fd per buffer.
fn layouts(hal: &wgpu::hal::vulkan::Device) -> Vec<(u64, vk::Extent3D)> {
    let extensions = hal.enabled_device_extensions();
    if !IMPORT_EXTENSIONS.iter().all(|name| extensions.contains(name)) {
        return Vec::new();
    }
    let instance = hal.shared_instance().raw_instance();
    let physical = hal.raw_physical_device();
    let query = |list: &mut vk::DrmFormatModifierPropertiesListEXT| {
        let mut properties = vk::FormatProperties2::default().push_next(list);
        unsafe {
            instance.get_physical_device_format_properties2(physical, FORMAT, &mut properties)
        };
    };
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
    query(&mut list);
    let mut layouts =
        vec![vk::DrmFormatModifierPropertiesEXT::default(); list.drm_format_modifier_count as usize];
    let mut list =
        vk::DrmFormatModifierPropertiesListEXT::default().drm_format_modifier_properties(&mut layouts);
    query(&mut list);
    let count = list.drm_format_modifier_count as usize;
    layouts.truncate(count);
    layouts
        .iter()
        .filter(|layout| {
            layout.drm_format_modifier_plane_count == 1
                && uncompressed(layout.drm_format_modifier)
                && layout
                    .drm_format_modifier_tiling_features
                    .contains(vk::FormatFeatureFlags::TRANSFER_SRC)
        })
        .filter_map(|layout| {
            let modifier = layout.drm_format_modifier;
            Some((modifier, max_extent(instance, physical, modifier)?))
        })
        .collect()
}

/// Whether `modifier` lays pixels out without compression. wgpu moves an
/// imported image out of an undefined layout, with no acquire from the
/// producer's queue, which lets a driver reinitialize compression metadata
/// and lose the pixels; some compressed layouts keep that metadata in their
/// only plane. Layouts of other vendors, which this cannot decode, are out.
fn uncompressed(modifier: u64) -> bool {
    let code = modifier & ((1 << 56) - 1);
    match modifier >> 56 {
        // Linear only: the rest is `DRM_FORMAT_MOD_INVALID`.
        0 => modifier == 0,
        // Intel: X, Y, Yf and 4 tiling; the others add CCS.
        0x01 => matches!(code, 1 | 2 | 3 | 9),
        // AMD: without DCC.
        0x02 => code >> 13 & 1 == 0,
        // NVIDIA: without framebuffer compression.
        0x03 => code >> 23 & 7 == 0,
        _ => false,
    }
}

/// The largest image in `modifier`'s layout the device imports from dma-bufs.
fn max_extent(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    modifier: u64,
) -> Option<vk::Extent3D> {
    let mut layout = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(modifier)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut handle = vk::PhysicalDeviceExternalImageFormatInfo::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(FORMAT)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::TRANSFER_SRC)
        .push_next(&mut layout)
        .push_next(&mut handle);
    let mut external = vk::ExternalImageFormatProperties::default();
    let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external);
    unsafe { instance.get_physical_device_image_format_properties2(physical, &info, &mut properties) }
        .ok()?;
    let extent = properties.image_format_properties.max_extent;
    external
        .external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        .then_some(extent)
}

/// Imports `dmabuf`, an `AR24` buffer in one of [`importable_modifiers`], as a
/// texture the GPU can copy from.
pub(crate) fn import(device: &wgpu::Device, dmabuf: &Dmabuf<'_>) -> Result<wgpu::Texture> {
    let available = std::fs::File::from(dmabuf.fd.try_clone_to_owned()?)
        .seek(SeekFrom::End(0))
        .context("dma-buf size")?;
    let size = wgpu::Extent3d {
        width: dmabuf.width,
        height: dmabuf.height,
        depth_or_array_layers: 1,
    };
    let hal_texture = {
        let hal = unsafe { device.as_hal::<Vulkan>() }
            .ok_or_else(|| anyhow!("dma-buf import needs the Vulkan backend"))?;
        // Creating an image in a layout or size the device does not support
        // is undefined behavior, not an error.
        let max = layouts(&hal)
            .into_iter()
            .find_map(|(modifier, max)| (modifier == dmabuf.modifier).then_some(max))
            .ok_or_else(|| {
                anyhow!("the GPU cannot import dma-bufs with modifier {:#018x}", dmabuf.modifier)
            })?;
        ensure!(
            (1..=max.width).contains(&dmabuf.width) && (1..=max.height).contains(&dmabuf.height),
            "the GPU imports at most {}x{} dma-bufs, not {}x{}",
            max.width,
            max.height,
            dmabuf.width,
            dmabuf.height
        );
        let raw = hal.raw_device();
        let external_fd =
            ash::khr::external_memory_fd::Device::new(hal.shared_instance().raw_instance(), raw);
        let plane = [vk::SubresourceLayout {
            offset: dmabuf.offset.into(),
            size: 0,
            row_pitch: dmabuf.stride.into(),
            array_pitch: 0,
            depth_pitch: 0,
        }];
        let mut layout = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(dmabuf.modifier)
            .plane_layouts(&plane);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
            .extent(vk::Extent3D {
                width: dmabuf.width,
                height: dmabuf.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut layout)
            .push_next(&mut external);
        let image = unsafe { raw.create_image(&info, None) }.with_context(|| {
            format!(
                "no {}x{} image with modifier {:#018x}, stride {}",
                dmabuf.width, dmabuf.height, dmabuf.modifier, dmabuf.stride
            )
        })?;
        let memory = (|| -> Result<vk::DeviceMemory> {
            let requirements = unsafe { raw.get_image_memory_requirements(image) };
            ensure!(
                available >= requirements.size,
                "dma-buf holds {available} bytes, the image needs {}",
                requirements.size
            );
            let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
            unsafe {
                external_fd.get_memory_fd_properties(
                    vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                    dmabuf.fd.as_raw_fd(),
                    &mut fd_properties,
                )
            }?;
            let memory_types = requirements.memory_type_bits & fd_properties.memory_type_bits;
            ensure!(memory_types != 0, "no memory type can hold this dma-buf");
            // A successful import takes ownership of the duplicate.
            let duplicate = dmabuf.fd.try_clone_to_owned()?;
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                .fd(duplicate.as_raw_fd());
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_types.trailing_zeros())
                .push_next(&mut import)
                .push_next(&mut dedicated);
            let memory = unsafe { raw.allocate_memory(&allocate, None) }?;
            let _ = duplicate.into_raw_fd();
            if let Err(error) = unsafe { raw.bind_image_memory(image, memory, 0) } {
                unsafe { raw.free_memory(memory, None) };
                return Err(error.into());
            }
            Ok(memory)
        })();
        match memory {
            // wgpu destroys the image and frees its memory with the texture.
            Ok(memory) => unsafe {
                hal.texture_from_raw(
                    image,
                    &wgpu::hal::TextureDescriptor {
                        label: Some("dmabuf"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Bgra8Unorm,
                        usage: wgpu::TextureUses::COPY_SRC,
                        memory_flags: wgpu::hal::MemoryFlags::empty(),
                        view_formats: Vec::new(),
                    },
                    None,
                    wgpu::hal::vulkan::TextureMemory::Dedicated(memory),
                )
            },
            Err(error) => {
                unsafe { raw.destroy_image(image, None) };
                return Err(error);
            }
        }
    };
    // wgpu's first barrier starts from an undefined layout, which keeps the
    // contents of the uncompressed layouts `layouts` lists: drivers move no
    // pixels for them.
    Ok(unsafe {
        device.create_texture_from_hal::<Vulkan>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("dmabuf"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::uncompressed;

    #[test]
    fn only_uncompressed_layouts_are_imported() {
        for (modifier, expected, what) in [
            (0, true, "linear"),
            (0x00ff_ffff_ffff_ffff, false, "invalid"),
            (0x0100_0000_0000_0009, true, "Intel 4 tiling"),
            (0x0100_0000_0000_000a, false, "Intel DG2 render CCS"),
            (0x0100_0000_0000_0010, false, "Intel Lunar Lake CCS"),
            (0x0200_0000_0040_1b03, true, "AMD without DCC"),
            (0x0200_0000_0056_bb03, false, "AMD DCC"),
            (0x0300_0000_0060_6014, true, "NVIDIA block linear"),
            (0x0300_0000_00e0_8014, false, "NVIDIA compressed block linear"),
            (0x0800_0000_0000_0001, false, "Arm AFBC"),
        ] {
            assert_eq!(uncompressed(modifier), expected, "{what}: {modifier:#018x}");
        }
    }
}
