<p align="center"><img src="https://raw.githubusercontent.com/go-virtio/brand/main/social/go-virtio-gpu.png" alt="go-virtio/gpu" width="720"></p>

# go-virtio/gpu

[![Go Reference](https://pkg.go.dev/badge/github.com/go-virtio/gpu.svg)](https://pkg.go.dev/github.com/go-virtio/gpu)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://github.com/go-virtio/gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/go-virtio/gpu/actions/workflows/ci.yml)

Pure-Go virtio-gpu (2D framebuffer + virgl 3D + software rasterizer)
driver targeting the `go-virtio/common` transport interfaces. Implements
the modern-transport (Virtio 1.0+) init sequence and the control-queue
command path for the standard PCI-bound virtio-gpu device (VID 0x1AF4,
DID 0x1050).

## Scope

"Pure-Go 3D" is three different things, and this module makes the split
explicit:

- **2D framebuffer.** The base path: negotiates exactly
  `VIRTIO_F_VERSION_1`, keeping the device in plain 2D mode (Virtio 1.1
  §5.7). `OpenVirtioGPU` owns device bring-up, both virtqueues — the
  **control queue** (controlq, carrying every command) and the **cursor
  queue** (cursorq, set up for spec-completeness but unused) — and the
  on-the-wire virtio-gpu control protocol. Every command is a
  2-descriptor chain (a read-only request followed by a device-writable
  response), built with `common.AddChain`. It exposes a
  scanout-enumeration + framebuffer API:

  - `DisplayInfo` lists the device's scanouts (`GET_DISPLAY_INFO`).
  - `SetupFramebuffer` creates a host 2D resource, attaches guest backing,
    and binds it to a scanout (`RESOURCE_CREATE_2D` +
    `RESOURCE_ATTACH_BACKING` + `SET_SCANOUT`). The returned
    `Framebuffer.Pix` is a BGRA byte buffer the caller draws into.
  - `Framebuffer.Flush` pushes the drawn pixels to the host and refreshes
    the scanout (`TRANSFER_TO_HOST_2D` + `RESOURCE_FLUSH`).

- **Software (CPU), `gpu/soft3d`.** A dependency-free, z-buffered
  triangle rasterizer that renders straight into the 2D
  `Framebuffer.Pix` — no `VIRTIO_GPU_F_VIRGL`, no host-GPU involvement,
  works on any host. `soft3d.RenderCube` draws a rotating, flat-shaded,
  perspective-projected cube; validated end-to-end on a real device via
  [`go-virtio/validate`](https://github.com/go-virtio/validate).

- **virgl (host GPU), `OpenVirtioGPU3D`.** Negotiates
  `VIRTIO_F_VERSION_1 | VIRTIO_GPU_F_VIRGL` and hand-encodes the virgl
  command stream (shaders shipped as TGSI *text*, per
  `virgl_encode_shader_state`) so a host `virglrenderer` does the
  drawing — still pure Go, CGO=0. Three milestones, each additive and
  independently validated against a real `virglrenderer`
  (software llvmpipe) via `go-virtio/validate/vtest`:

  - `ClearScreen` (M1) — clear a scanout to a solid colour.
  - `DrawTriangle` (M2) — one flat-shaded triangle (two shaders, four
    pipeline-state objects, a vertex buffer, `DRAW_VBO`).
  - `DrawTexturedTriangle` (M3) — M2 plus a sampled texture (sampler
    view/state, perspective-interpolated texcoord).

  `gpu` also carries the blob-resource + `context_init` wire encoders
  (`EncodeResourceCreateBlob`, `EncodeCtxCreateVenus`,
  `NegotiateVenusFeatures`, …) that the separate
  [`venus`](https://github.com/go-virtio/venus) module needs to stand up
  a Venus (Vulkan-over-virtio) context on top of this same device.

- **Vulkan / Venus.** Handled by the separate
  [`venus`](https://github.com/go-virtio/venus) module, which reuses this
  package's blob/context-init primitives.

## Quick start

```go
import (
    virtiogpu "github.com/go-virtio/gpu"
)

// transport is any value that implements go-virtio/common.Transport.
g, err := virtiogpu.OpenVirtioGPU(transport)
if err != nil {
    return err
}

displays, err := g.DisplayInfo()
if err != nil {
    return err
}
d := displays[0] // first scanout

fb, err := g.SetupFramebuffer(d.ScanoutID, d.Width, d.Height)
if err != nil {
    return err
}

// Draw BGRA pixels into fb.Pix, then push to the display.
for i := 0; i+3 < len(fb.Pix); i += 4 {
    fb.Pix[i+0] = 0x20 // B
    fb.Pix[i+1] = 0x80 // G
    fb.Pix[i+2] = 0xFF // R
    fb.Pix[i+3] = 0xFF // A
}
err = fb.Flush()
```

## Sibling packages

  - [`github.com/go-virtio/common`](https://github.com/go-virtio/common)
    — transport-agnostic infrastructure (PCI cap walker, modern config
    layout, split-virtqueue + descriptor-chain impl, transport interfaces).
  - [`github.com/go-virtio/net`](https://github.com/go-virtio/net) —
    pure-Go virtio-net driver.
  - [`github.com/go-virtio/rng`](https://github.com/go-virtio/rng) —
    pure-Go virtio-rng driver.
  - [`github.com/go-virtio/vsock`](https://github.com/go-virtio/vsock) —
    pure-Go virtio-vsock driver.
  - [`github.com/go-virtio/blk`](https://github.com/go-virtio/blk) —
    pure-Go virtio-blk driver (the descriptor-chain reference this package
    mirrors).
  - [`github.com/go-virtio/venus`](https://github.com/go-virtio/venus) —
    Vulkan-over-virtio (Venus), built on this package's blob/context-init
    primitives.
  - [`github.com/go-virtio/validate`](https://github.com/go-virtio/validate)
    — real-hardware validation harness that exercises this package's 2D,
    soft3d and virgl 3D paths against real devices and renderers.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
