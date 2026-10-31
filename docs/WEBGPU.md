# virtio-gpu × WebGPU

Status: **design draft, unvalidated.** No host implements either option
yet (see "Where is the host?"). Nothing below has run against a real
device. The guest-side framing in `wgpubatch/` is tested offline only,
and offline self-consistency is not validation.

This document covers two ways to connect go-virtio/gpu with
[go-gfx/webgpu](https://github.com/go-gfx/webgpu):

- **Option 2: 2D scanout on WebGPU.** A virtio-gpu *device* presents
  the guest's 2D resources through WebGPU, so a VM or emulator that runs
  in a browser gets an accelerated display.
- **Option 4: the `webgpu-batch` capset.** A private context type
  carries go-gfx/webgpu Recorder streams from a guest to a host that
  replays them on WebGPU. It works like Venus (Vulkan over virtio), but
  for WebGPU.

## Where is the host?

go-virtio is guest-side only. A search on 2026-10-31 found no
virtio-gpu **device model** anywhere in the fleet:

- local clones: every Go file that mentions `virtio-gpu` or
  `RESOURCE_FLUSH` outside go-virtio is a guest-side consumer. That
  covers go-doom and go-quake1 (their tamago backends), cloud-boot,
  and go-gfx/webgpu/timeline (a comment);
- `openweft/weft-driver-vz` attaches **Apple's** `VZVirtioGraphicsDevice`
  (`builtin/vmrun.go:672`). It is not our code. nano-container-linux
  uses QEMU;
- `gh search repos` in go-wasmcontainers, openweft, go-desktop and
  wasmdesk found no VM or emulator with a virtio-gpu device.

The host side of both options is therefore **new code**. The proposed
home, which is pending a decision, is under "Proposed packaging" at the
end.

## Bibliography (opened, with file:line)

| Source | What it settles |
|---|---|
| virtio spec, `device-types/gpu/description.tex` (oasis-tcs/virtio-spec master) | |
| :173-179 | fencing: with `VIRTIO_GPU_FLAG_FENCE` the device MUST finish processing before it returns the command |
| :255-297 | `ctrl_hdr`. `ring_idx` is 0..63. A fence is a sequence number on the (ctx, ring) timeline, and completing it retires every earlier fence on that timeline |
| :523-545 | `capset_id` "MUST be one of the following": VIRGL, VIRGL2, GFXSTREAM, VENUS, CROSS_DOMAIN. **The spec defines no private or experimental range** |
| :727-750 | `CTX_CREATE.context_init`: the low 8 bits select the capset when `F_CONTEXT_INIT` is negotiated |
| :764-766 | `SUBMIT_3D` carries an opaque stream whose type the context fixes |
| Linux `include/uapi/linux/virtio_gpu.h` | |
| :66 | `VIRTIO_GPU_F_CONTEXT_INIT 4` |
| :143, :150 | `FLAG_INFO_RING_IDX`, `ring_idx` |
| :291 | `VIRTIO_GPU_CONTEXT_INIT_CAPSET_ID_MASK 0xff` |
| :318-323 | capsets 1..6 (VIRGL … DRM) |
| :381-390 | 2D formats (B8G8R8A8=1 … R8G8B8X8=134) |
| Linux `drivers/gpu/drm/virtio/virtgpu_drv.h:59-60` | `MAX_CAPSET_ID 63`, `MAX_RINGS 64` |
| Linux `drivers/gpu/drm/virtio/virtgpu_kms.c:92-110` | one advertised capset id of 0 or above 63 makes the driver **discard every capset** |
| Linux `virtgpu_kms.c:112` | `capset_id_mask \|= 1 << id`: an `int` shift, so ids ≥ 31 are undefined behaviour and corrupt the 64-bit mask |
| Linux `drivers/gpu/drm/virtio/virtgpu_ioctl.c:667-676` | `CONTEXT_INIT` refuses an id above 63 or one outside `capset_id_mask` |
| Linux `include/uapi/drm/virtgpu_drm.h:167-172, :210` | the DRM capset ids, `VIRTGPU_CONTEXT_PARAM_CAPSET_ID` |
| rutabaga_gfx `src/rutabaga_utils.rs:213-222` (magma-gpu/rutabaga_gfx) | ids already in use: 1..10 (…, MAGMA 7, GFXSTREAM_GLES 8, GFXSTREAM_COMPOSER 9, ROCM 10) |
| QEMU `hw/display/virtio-gpu.c:264-279` | `calc_image_hostmem`: the stride and size are computed in u64, refused above u32 |
| QEMU `virtio-gpu.c:281-330` | `RESOURCE_CREATE_2D`: rejects id 0, a duplicate id or an unknown format, and enforces the `max_hostmem` quota |
| QEMU `virtio-gpu.c:526-596` | `TRANSFER_TO_HOST_2D`: checks the rect against the resource, then copies row by row from the backing iov |
| QEMU `virtio-gpu.c:598-685` | `RESOURCE_FLUSH`: checks the bounds, intersects the rect with each scanout, updates the console |
| crosvm `devices/src/virtio/gpu/protocol.rs:418-422` | the same capset ids, seen from the device side |
| Mesa `src/virtio/vulkan/vn_ring.c:92-145` | the Venus ring: the guest stores `tail` (seq_cst) and the renderer stores `head` (release) |
| Mesa `vn_ring.c:257-283` | ring layout: head, tail and status each 64-byte aligned, then a power-of-two buffer, then extra |
| virglrenderer `src/venus/vkr_ring.c:84-101, :302-317` | the host refuses `cmd_size > buffer size` and **copies** the command out of shared memory before decoding it, so a guest cannot rewrite it mid-decode |
| go-gfx/webgpu `record.go:20-44` | the Recorder opcodes. "The stream format is internal" (:17-18) |
| go-gfx/webgpu `decode.go:37-191` | the Go mirror of replay.js: argument counts, placement, object checks, data bounds |
| go-gfx/webgpu `quota.go`, `timeline/timeline.go:3` | per-device memory and pipeline quotas; the timeline is the virtio-gpu fence model |
| go-virtio/venus `ring/ring.go` | our existing Venus ring layout (`ShmemSize`, `Submit`, `AdvanceHead`) |

## Option 4: the `webgpu-batch` capset

### Capset id: 30

The spec reserves no private range (`description.tex:531-545` says
"MUST be one of the following"). Any id that is not listed is
non-conforming by construction. The honest choice is an id that every
deployed guest can carry and that is unlikely to collide:

1. **8-bit wire field:** `context_init & 0xff` (`virtio_gpu.h:291`).
2. **Linux accepts 1..63:** any other advertised id makes the driver
   drop *all* capsets, including virgl and Venus (`virtgpu_kms.c:95-109`).
   A bad choice therefore breaks unrelated contexts on the same device.
3. **Linux records 1..30 correctly:** `1 << id` is an `int` shift
   (`virtgpu_kms.c:112`, `capset_id_mask` is `uint64_t` at
   `virtgpu_drv.h:306`). Id 31 sign-extends into bits 31..63, and ids of
   32 or more are undefined behaviour. In both cases the mask names
   capsets the host never offered.
4. **Taken:** 1..6 in the kernel uapi, 1..10 in rutabaga.

We choose **30**: the highest id that Linux handles correctly, as far
as possible from the allocation front, which grows upward from 10. It
is a Go constant (`wgpubatch.CapsetID`), so a host and guest that need
another id change one line. If WebGPU over virtio is ever proposed
upstream, an assigned id replaces it. The protocol carries its own magic
and version, so a host can tell a misrouted stream apart.

### Capset blob (GET_CAPSET, version 1)

`GET_CAPSET_INFO` reports `capset_id=30`, `max_version=1` and
`max_size=40`. `GET_CAPSET(30, 1)` returns, little-endian:

| off | field | meaning |
|---|---|---|
| 0 | `magic` u32 | `0x42504757` ("WGPB") |
| 4 | `version` u32 | protocol version, 1 |
| 8 | `stream_version` u32 | version of the go-gfx/webgpu Recorder stream the host replays |
| 12 | `max_frame` u32 | largest SUBMIT_3D payload accepted, in bytes |
| 16 | `max_objects` u32 | live guest objects per context |
| 20 | `max_rings` u32 | timelines (ring_idx values) per context, ≤ 64 |
| 24 | `features` u32 | bit 0: shared-memory ring (v2 transport) |
| 28 | reserved u32 | 0 |
| 32 | `max_memory` u64 | bytes of buffers and textures per context (go-gfx/webgpu Quota) |

The decoder refuses a short blob, a bad magic, version 0, a `max_frame`
below one frame header, or `max_rings` above 64.

### Context creation

`CTX_CREATE` with `context_init = 30` (the ring count is not part of
`context_init`, which carries only the capset). It requires
`VERSION_1 | VIRGL | CONTEXT_INIT`. VIRGL is still the "3D available"
gate that hosts advertise, and BLOB is needed only for the v2 ring.

### Transport v1: framed SUBMIT_3D

A batch rides as the opaque payload of `SUBMIT_3D`
(`description.tex:764`). The payload is a **frame**: a 16-byte header
followed by messages.

```
frame   := magic u32 ("WGPB") | version u16 (1) | flags u16 (0)
           | size u32 (whole frame, bytes) | count u32 (messages)
message := type u32 | size u32 (payload bytes, multiple of 4) | payload
```

| type | payload | purpose |
|---|---|---|
| 1 CREATE | `id u32, kind u32, descriptor[]` | create a guest-numbered object; the descriptor is go-gfx/webgpu's id-based descriptor form (see "What go-gfx/webgpu must export") |
| 2 DESTROY | `ids u32[]` | release guest objects |
| 3 STREAM | `nwords u32, ndata u32, words[nwords], data[ndata], pad to 4` | one go-gfx/webgpu Recorder submit, **verbatim**, with object ids in the guest's namespace |
| 4 BIND_RESOURCE | `id u32, resource_id u32, offset u64, size u64` | expose part of a virtio resource (attached with `CTX_ATTACH_RESOURCE`) as a readback or upload window for a guest buffer |

Object ids are **guest-chosen**, nonzero and below `max_objects`, and
private to the context. The STREAM's words are never interpreted by
go-virtio. Only go-gfx/webgpu reads them (the rule is one implementation
per format).

### Fencing

A frame that ends a frame-worth of work is submitted with
`VIRTIO_GPU_FLAG_FENCE | VIRTIO_GPU_FLAG_INFO_RING_IDX`. `fence_id` is
the next number on that ring's timeline and `ring_idx` names the
timeline (0 by default; `description.tex:284-297`). The host answers
once the replayed submit's `onSubmittedWorkDone` resolves. Completing
fence n retires every fence ≤ n on that ring, which is exactly the
contract of go-gfx/webgpu/timeline (`timeline.go:3`). The host maps it
one to one: `Device.WaitSubmitted(seq)` drives the virtio response.

### Transport v2: shared ring (feature bit 0)

Large or frequent batches pay one virtqueue round trip per SUBMIT_3D. v2
moves frames into a guest-allocated blob (`BLOB_MEM_GUEST`), using the
**go-virtio/venus/ring layout unchanged**: head, tail and status, each
64-byte aligned, then a power-of-two buffer (`vn_ring.c:257-283`). The
same memory ordering applies: the guest stores the tail seq_cst and the
host stores the head with release semantics (`vn_ring.c:95-113`). A
SUBMIT_3D control message `RING_CREATE{resource_id, buffer_size}` sets
it up, and `RING_NOTIFY` wakes an idle host. Like virglrenderer, the
host copies each frame **out** of the ring before it decodes anything
(`vkr_ring.c:302-317`). v2 is designed here and not implemented.

### Quotas

These are per context and enforced by the host before any WebGPU call:

- frame size ≤ `max_frame`; message count ≤ frame size / 8;
- live objects ≤ `max_objects`; an id already live is refused on CREATE;
- buffer and texture bytes are charged to a go-gfx/webgpu `Quota`
  (`quota.go`), one Quota per context. Shader source length is checked
  by `Quota.checkShader`, pipelines by `chargePipeline`;
- unfinished fences per ring are bounded, and the guest blocks (no
  response) instead of growing a host queue;
- GPU time: a submit whose fence does not complete within a deadline
  marks the context lost. WebGPU device loss is mapped to **context**
  loss, so other contexts and the host's own display survive.

### Security model

The guest is hostile. Every value it supplies is checked by the host in
64-bit arithmetic before use:

1. **Copy, then decode.** The frame is copied out of guest memory into
   host memory once (`vkr_ring.c:312`), so no check can be bypassed by
   rewriting shared memory between the check and the use.
2. **Frame framing**: `wgpubatch.Decode` checks the magic, the version,
   `size == len`, that each message fits, that payloads are 4-aligned,
   and the count. It is fuzzed.
3. **Handles**: guest ids are looked up in the context's table. An
   unknown id, id 0 (except where the stream allows "none") or an id
   from another context is an error. The host's own objects, such as
   its canvas texture, are never in the table. go-gfx/webgpu must remap
   stream ids (guest → host) with the same placement knowledge as
   `decode.go`. A STREAM is validated by go-gfx/webgpu's decoder first
   (`decode.go:37-191`: argument counts, placement, data bounds
   `a[3]+a[4] ≤ ndata` in u64).
4. **Sizes and offsets**: CREATE descriptors pass through go-gfx/webgpu's
   `validate.go` checks (the same as the direct path). BIND_RESOURCE
   requires `offset+size ≤ resource size` in u64 and a resource
   attached to *this* context.
5. **WebGPU validation is the second line, not the first.** Each replay
   runs inside an error scope. A validation error marks the context
   lost and is never turned into a host panic.
6. **Information flow**: readback (`MapAsync` → BIND_RESOURCE window →
   `TRANSFER_FROM_HOST_3D`) only writes into resources this context
   owns. Uninitialised GPU memory is zeroed by WebGPU itself.

Integer overflow on 32-bit hosts (GOOS=js/wasm is 32-bit `int`): all
size arithmetic uses `uint64`, and nothing converts a guest u32 or u64
to `int` before it is bounded.

### What go-gfx/webgpu must export (one implementation per format)

The Recorder stream is internal today (`record.go:17-18`). A virtio
transport needs, **in go-gfx/webgpu** and not copied here:

1. a portable `wire` package with the opcodes, `Decode` (today's
   `decode.go`, exported) and `Remap(words, f)`, which rewrites every
   object-id position in place through `f`. Only the format owner knows
   where the ids sit;
2. a `StreamVersion` constant, advertised in the capset blob;
3. `(*Device).Replay(words, data, objects)`, which submits an external,
   already-remapped stream through replay.js;
4. an id-based descriptor encoding for CREATE. Descriptors hold
   `*Buffer` and `*BindGroupLayout` pointers today, so a guest needs a
   form that names objects by id;
5. for guest programs: a portable "remote device" that implements the
   go-gfx/webgpu API by recording into this transport instead of JS.
   Without it, only hand-built streams reach the guest side.

These belong to a go-gfx/webgpu PR series. The `wgpubatch` package in
go-virtio treats STREAM words and CREATE descriptors as opaque until
`wire` exists.

## Option 2: 2D scanout on WebGPU

### Host data path

| guest command | host action | cost |
|---|---|---|
| `RESOURCE_CREATE_2D(id, fmt, w, h)` | check id≠0, not live, format known, `w*h*4 ≤ max_hostmem` in u64 (QEMU :264-330); `Device.CreateTexture(w, h, bgra8unorm or rgba8unorm, COPY_DST\|COPY_SRC\|TEXTURE_BINDING)` | one GPU allocation |
| `RESOURCE_ATTACH_BACKING(id, entries)` | record the guest-physical ranges and check that each lies inside guest RAM in u64 | none |
| `TRANSFER_TO_HOST_2D(id, rect, offset)` | check the rect against the resource (QEMU :561-573) and `offset + stride*(h-1) + w*bpp ≤ backing length` in u64. QEMU truncates `offset` to `uint32_t` at :531/:582; we do not. Then `Queue.WriteTexture` straight from the guest-RAM view, using `bytesPerRow = stride`, so there is **no shadow copy** | one copy, guest RAM → GPU |
| `SET_SCANOUT(scanout, id, rect)` | bind the resource to the canvas `Surface` | none |
| `RESOURCE_FLUSH(id, rect)` | intersect with the scanout (QEMU :666-684); `copyTextureToTexture(resource → Surface.CurrentTexture())` when the formats match, otherwise one full-screen blit pass; present | one GPU copy |

Formats: B8G8R8A8 and B8G8R8X8 map to `bgra8unorm`, and R8G8B8A8 and
R8G8B8X8 map to `rgba8unorm`. The X variants rely on the canvas
`alphaMode: "opaque"`. A8R8G8B8, X8R8G8B8, A8B8G8R8 and X8B8G8R8 need a
swizzle pass, which go-simd provides on the CPU or a shader on the GPU.

Scatter-gather backing: when the backing is one contiguous range (what
go-virtio/gpu's `SetupFramebuffer` allocates), the write is a single
`writeTexture`. When the backing is fragmented, the host uses one
`writeTexture` per contiguous run of rows. As a fallback it packs
fragmented rows into a staging slice (one extra CPU copy).

Cost estimate (**unmeasured**): a 1920×1080 BGRA frame is 8.3 MB.
`writeTexture` is a memcpy into a GPU staging buffer, so a full frame
should cost a few ms and a damage rectangle proportionally less. The
FLUSH copy is GPU-side and negligible. The reference to beat is QEMU's
pixman path (`:580-595` plus a console update) on the same machine. It
will be measured once a host exists.

The `RESOURCE_FLUSH` copy needs `copyTextureToTexture` in a batch. The
Recorder has no such opcode today (`record.go:20-44`), so the flush uses
the direct `CommandEncoder` path: a handful of JS calls per frame. That
is acceptable at display rates. A Recorder opcode is a possible
go-gfx/webgpu addition.

### Consumers

- **Option 2**: any guest that already speaks 2D virtio-gpu, unchanged.
  That includes the go-virtio/gpu `SetupFramebuffer`/`Flush` path, the
  tamago backends of go-doom and go-quake1, and the Linux `virtio_gpu`
  KMS driver. On the host side, the consumer is a browser-hosted VM or
  emulator. **None exists in the fleet yet.**
- **Option 4**: guest programs that want GPU compute or rendering
  through the go-gfx/webgpu API (go-wasmcontainers/gpu workloads run in
  a VM rather than in a page, and tamago games). It needs item 5 of the
  export list above.

## Proposed packaging (pending decision)

- `go-virtio/gpu/wgpubatch` (**this repository**): the capset id, the
  capset blob, and the frame encoder and decoder. Guest and host both
  import it, and it has no GOOS constraint.
- **Host device model: a new repository**, for example
  `go-virtio/gpudev`. It holds the virtio-gpu control queue and resource
  table, the 2D commands, and a webgpu-batch context, with two backends:
  go-gfx/webgpu (GOOS=js) and an in-memory backend for tests and fuzzing.
  It is not created until the decision is made.

## Guest API (this repository)

- `wgpubatch`: `CapsetID`, `Caps` / `DecodeCaps`, `Encoder` (Create,
  Destroy, Stream, BindResource, Frame), `Decode` with `Limits`, and
  the typed `Message` parsers.
- `gpu`: `OpenVirtioGPUWebGPUBatch`, `(*VirtioGPU).WebGPUBatchCaps`,
  `CreateWebGPUBatchContext` and `SubmitWebGPUBatch` (3-descriptor
  SUBMIT_3D whose pages are cached and grown, with optional
  FENCE/RING_IDX, and a check that the device echoes the fence). It also
  has the byte builders and parsers: `EncodeSubmit3DHeader`,
  `EncodeGetCapsetInfo`/`EncodeGetCapset`, `DecodeRespCapsetInfo`, and
  `DecodeRespCapset`, which caps the data at the announced `max_size`.
- An existing defect was fixed on the way: `submit3D` copied any
  length into its one page. A virgl buffer above 1024 bytes overlapped
  the response slot, and one above 3072 bytes ran past the page while
  the descriptor still claimed the full length. It now refuses with
  `ErrVirglBufferTooLarge`.

## Validation status

| piece | status |
|---|---|
| `wgpubatch` encode/decode | offline unit tests (100%) and fuzzing (`FuzzDecode`, `FuzzDecodeCaps`) only |
| capset query and response parsing | offline, fuzzed (`FuzzCapsetResponses`) |
| CTX_CREATE / SUBMIT_3D with capset 30 | **unvalidated**: no host advertises it; tested against the in-process fake device only |
| Option 2 host | not written |
| Option 4 host | not written |
