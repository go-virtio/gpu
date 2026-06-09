# go-virtio/gpu

Pure-Go virtio-gpu (2D framebuffer) driver targeting the
`go-virtio/common` transport interfaces. Implements the modern-transport
(Virtio 1.0+) init sequence and the control-queue command path for the
standard PCI-bound virtio-gpu device (VID 0x1AF4, DID 0x1050).

## Scope

**2D framebuffer only.** 3D / virgl is explicitly out of scope: it
requires a Mesa-class GL/Vulkan stack to generate the OpenGL command
streams the host consumes, which does not exist in Go. The driver
negotiates exactly `VIRTIO_F_VERSION_1` and deliberately does **not**
negotiate `VIRTIO_GPU_F_VIRGL` (bit 0), keeping the device in plain 2D
mode (Virtio 1.1 §5.7).

Like the sibling drivers this package owns device bring-up, both
virtqueues — the **control queue** (controlq, carrying every 2D command)
and the **cursor queue** (cursorq, set up for spec-completeness but
unused) — and the on-the-wire virtio-gpu control protocol. Every command
is a 2-descriptor chain (a read-only request followed by a
device-writable response), built with `common.AddChain`.

It exposes a scanout-enumeration + framebuffer API:

  - `DisplayInfo` lists the device's scanouts (`GET_DISPLAY_INFO`).
  - `SetupFramebuffer` creates a host 2D resource, attaches guest backing,
    and binds it to a scanout (`RESOURCE_CREATE_2D` +
    `RESOURCE_ATTACH_BACKING` + `SET_SCANOUT`). The returned
    `Framebuffer.Pix` is a BGRA byte buffer the caller draws into.
  - `Framebuffer.Flush` pushes the drawn pixels to the host and refreshes
    the scanout (`TRANSFER_TO_HOST_2D` + `RESOURCE_FLUSH`).

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

## License

BSD-3-Clause. See [LICENSE](LICENSE).
