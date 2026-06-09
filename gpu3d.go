// 3D / virgl extension of the virtio-gpu driver (Milestone 1). This file
// is purely additive: it does NOT touch the 2D OpenVirtioGPU /
// SetupFramebuffer / Flush / DisplayInfo path. It adds a second
// constructor, OpenVirtioGPU3D, that negotiates VIRTIO_GPU_F_VIRGL in
// addition to VERSION_1, and a single high-level operation, ClearScreen,
// that clears a scanout to a solid colour using the HOST GPU via
// virglrenderer.
//
// M1 scope is deliberately tiny: stand up a virgl context, create a 3D
// render-target resource, submit a 3-command virgl buffer (CREATE
// SURFACE + SET_FRAMEBUFFER_STATE + CLEAR), pull the result back into the
// guest backing, and bind+flush it to a scanout. No Mesa is needed because
// the virgl command buffer for a flat clear is short enough to hand-encode.
//
// References:
//
//   - Linux uapi linux/virtio_gpu.h — the 3D control structs and command
//     codes (VIRTIO_GPU_CMD_*_3D, VIRTIO_GPU_F_VIRGL).
//   - virgl_protocol.h — VIRGL_CMD0 header layout, VIRGL_OBJECT_SURFACE,
//     VIRGL_CCMD_* opcodes, VIRGL_OBJ_SURFACE_* / clear field order.
//   - virgl_encode.c — the canonical encoders this hand-encoding mirrors.
package gpu

import (
	"encoding/binary"
	"math"

	"github.com/go-virtio/common"
)

// VIRTIO_GPU_F_VIRGL is feature bit index 0 (Virtio 1.1 §5.7.3): the host
// advertises it when virglrenderer-backed 3D is available.
const FeatureVirgl uint64 = 1 << 0

// AcceptedFeatures3D is the feature mask the 3D constructor negotiates ON:
// VERSION_1 (non-negotiable) plus VIRGL (bit 0).
const AcceptedFeatures3D uint64 = common.FeatureVersion1 | FeatureVirgl

// 3D control-command request type codes (Linux uapi virtio_gpu.h,
// sequential from 0x0200). Success responses are OK_NODATA (0x1100), same
// as the 2D commands.
const (
	CmdCtxCreate          uint32 = 0x0200
	CmdCtxDestroy         uint32 = 0x0201
	CmdCtxAttachResource  uint32 = 0x0202
	CmdCtxDetachResource  uint32 = 0x0203
	CmdResourceCreate3D   uint32 = 0x0204
	CmdTransferToHost3D   uint32 = 0x0205
	CmdTransferFromHost3D uint32 = 0x0206
	CmdSubmit3D           uint32 = 0x0207
)

// ctxID is the single virgl context id the M1 path uses (1; 0 is
// reserved). It is written into ctrl_hdr.ctx_id (offset 16) of every 3D
// command except CTX_CREATE / CTX_DESTROY.
const ctxID uint32 = 1

// hdrCtxIDOffset is the byte offset of ctrl_hdr.ctx_id within the 24-byte
// virtio_gpu_ctrl_hdr (le32 type@0, le32 flags@4, le64 fence_id@8, le32
// ctx_id@16, le32 padding@20).
const hdrCtxIDOffset = 16

// RESOURCE_CREATE_3D parameter constants (pipe / virgl enums baked in for
// the M1 flat-colour render target).
const (
	// PIPE_TEXTURE_2D — a plain 2D texture target.
	pipeTexture2D uint32 = 2
	// VIRGL_FORMAT_B8G8R8A8_UNORM — matches the 2D BGRA framebuffer
	// format so the scanout consumes the resource unchanged.
	virglFormatB8G8R8A8Unorm uint32 = 1
	// VIRGL_BIND_RENDER_TARGET — the resource is bound as a colour
	// render target (required to CLEAR into it).
	virglBindRenderTarget uint32 = 2
)

// Virgl command-buffer opcodes and object types (virgl_protocol.h). A
// command word is VIRGL_CMD0(cmd, obj, len) = cmd | obj<<8 | len<<16,
// followed by `len` payload dwords.
const (
	// VIRGL_CCMD_CREATE_OBJECT.
	virglCmdCreateObject uint32 = 1
	// VIRGL_CCMD_SET_FRAMEBUFFER_STATE.
	virglCmdSetFramebufferState uint32 = 5
	// VIRGL_CCMD_CLEAR.
	virglCmdClear uint32 = 7
	// VIRGL_OBJECT_SURFACE = 8. The enum is NULL=0, BLEND, RASTERIZER,
	// DSA, SHADER, VERTEX_ELEMENTS, SAMPLER_VIEW, SAMPLER_STATE=7,
	// SURFACE=8 — confirmed verbatim against virglrenderer
	// src/virgl_protocol.h (the easy-to-miss SAMPLER_STATE slot sits at 7).
	virglObjectSurface uint32 = 8
	// PIPE_CLEAR_COLOR0 — clear the first colour buffer.
	pipeClearColor0 uint32 = 4
	// surfaceHandle is the virgl object handle assigned to the surface
	// and referenced by SET_FRAMEBUFFER_STATE's cbuf0.
	surfaceHandle uint32 = 1
)

// virglCmd0 builds a virgl command header dword:
// cmd | (obj<<8) | (len<<16), where len is the number of payload dwords.
func virglCmd0(cmd, obj, length uint32) uint32 {
	return cmd | (obj << 8) | (length << 16)
}

// OpenVirtioGPU3D drives the same bring-up as OpenVirtioGPU but additionally
// negotiates VIRTIO_GPU_F_VIRGL. If the host does not advertise virgl, the
// driver-feature write masks the bit off and FEATURES_OK still latches —
// but the resulting device has no 3D, so callers must check the negotiated
// mask. More importantly, if FEATURES_OK fails to latch after writing the
// driver features, the host has rejected the requested set: this is read as
// "no virglrenderer" and surfaced as ErrVirglUnavailable.
func OpenVirtioGPU3D(t common.Transport) (*VirtioGPU, error) {
	g, err := bringUp(t, AcceptedFeatures3D, ErrVirglUnavailable)
	if err != nil {
		return nil, err
	}
	if g.NegotiatedFeatures&FeatureVirgl == 0 {
		return nil, ErrVirglUnavailable
	}
	return g, nil
}

// ClearScreen clears scanout scanoutID to the solid RGBA colour (r,g,b,a)
// using the host GPU, executing the full M1 virgl sequence:
//
//	(a) DisplayInfo to resolve the scanout's width/height (and reject a
//	    disabled or out-of-range scanout with ErrNoScanout).
//	(b) CTX_CREATE(ctx=1).
//	(c) RESOURCE_CREATE_3D(res=1, w, h) — a BGRA render target.
//	(d) RESOURCE_ATTACH_BACKING(res=1, one mem_entry{phys, w*h*4}).
//	(e) CTX_ATTACH_RESOURCE(ctx=1, res=1).
//	(f) SUBMIT_3D(ctx=1, <CREATE SURFACE + SET_FRAMEBUFFER_STATE + CLEAR>).
//	(g) TRANSFER_TO_HOST_3D(ctx=1, box{0,0,0,w,h,1}, res=1).
//	(h) SET_SCANOUT(scanoutID, res=1, rect{0,0,w,h}).
//	(i) RESOURCE_FLUSH(res=1, rect{0,0,w,h}).
//
// Every step checks its response is OK_NODATA; any other response type is
// ErrGPUCommandFailed.
func (g *VirtioGPU) ClearScreen(scanoutID uint32, r, gr, b, a float32) error {
	// (a) Resolve scanout dimensions.
	if scanoutID >= g.NumScanouts {
		return ErrNoScanout
	}
	displays, err := g.DisplayInfo()
	if err != nil {
		return err
	}
	d := displays[scanoutID]
	if !d.Enabled {
		return ErrNoScanout
	}
	width, height := d.Width, d.Height

	// (b) CTX_CREATE — 96 bytes: le32 nlen@24, le32 context_init@28,
	// byte debug_name[64]@32. nlen=0, context_init=0, name all-zero.
	ctxCreate := make([]byte, ctrlHdrSize+72)
	binary.LittleEndian.PutUint32(ctxCreate[0:4], CmdCtxCreate)
	binary.LittleEndian.PutUint32(ctxCreate[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	// nlen@24, context_init@28, debug_name[64]@32 all left zero.
	if _, err := g.sendCommand(ctxCreate); err != nil {
		return err
	}

	// (c) RESOURCE_CREATE_3D — 72 bytes (48-byte body after the hdr).
	create := make([]byte, ctrlHdrSize+48)
	binary.LittleEndian.PutUint32(create[0:4], CmdResourceCreate3D)
	binary.LittleEndian.PutUint32(create[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	b3 := ctrlHdrSize
	binary.LittleEndian.PutUint32(create[b3+0:b3+4], resourceID)                // resource_id@24
	binary.LittleEndian.PutUint32(create[b3+4:b3+8], pipeTexture2D)             // target@28
	binary.LittleEndian.PutUint32(create[b3+8:b3+12], virglFormatB8G8R8A8Unorm) // format@32
	binary.LittleEndian.PutUint32(create[b3+12:b3+16], virglBindRenderTarget)   // bind@36
	binary.LittleEndian.PutUint32(create[b3+16:b3+20], width)                   // width@40
	binary.LittleEndian.PutUint32(create[b3+20:b3+24], height)                  // height@44
	binary.LittleEndian.PutUint32(create[b3+24:b3+28], 1)                       // depth@48
	binary.LittleEndian.PutUint32(create[b3+28:b3+32], 1)                       // array_size@52
	binary.LittleEndian.PutUint32(create[b3+32:b3+36], 0)                       // last_level@56
	binary.LittleEndian.PutUint32(create[b3+36:b3+40], 0)                       // nr_samples@60
	binary.LittleEndian.PutUint32(create[b3+40:b3+44], 0)                       // flags@64
	binary.LittleEndian.PutUint32(create[b3+44:b3+48], 0)                       // padding@68
	if _, err := g.sendCommand(create); err != nil {
		return err
	}

	// (d) RESOURCE_ATTACH_BACKING — allocate the guest backing run and
	// attach a single mem_entry{phys, w*h*4}.
	size := uint64(width) * uint64(height) * 4
	pages := int((size + uint64(common.PageSize) - 1) / uint64(common.PageSize))
	phys, _, err := g.transport.AllocatePages(pages)
	if err != nil {
		return err
	}
	if phys == 0 {
		return common.ErrAllocReturnedZero
	}
	attach := make([]byte, ctrlHdrSize+8+memEntrySize)
	binary.LittleEndian.PutUint32(attach[0:4], cmdResourceAttachBacking)
	binary.LittleEndian.PutUint32(attach[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	binary.LittleEndian.PutUint32(attach[ctrlHdrSize+0:ctrlHdrSize+4], resourceID)
	binary.LittleEndian.PutUint32(attach[ctrlHdrSize+4:ctrlHdrSize+8], 1) // nr_entries
	me := ctrlHdrSize + 8
	binary.LittleEndian.PutUint64(attach[me+0:me+8], phys)
	binary.LittleEndian.PutUint32(attach[me+8:me+12], uint32(size))
	binary.LittleEndian.PutUint32(attach[me+12:me+16], 0) // padding
	if _, err := g.sendCommand(attach); err != nil {
		return err
	}

	// (e) CTX_ATTACH_RESOURCE — 32 bytes: le32 resource_id@24, le32 pad@28.
	ctxAttach := make([]byte, ctrlHdrSize+8)
	binary.LittleEndian.PutUint32(ctxAttach[0:4], CmdCtxAttachResource)
	binary.LittleEndian.PutUint32(ctxAttach[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	binary.LittleEndian.PutUint32(ctxAttach[ctrlHdrSize+0:ctrlHdrSize+4], resourceID)
	binary.LittleEndian.PutUint32(ctxAttach[ctrlHdrSize+4:ctrlHdrSize+8], 0) // padding
	if _, err := g.sendCommand(ctxAttach); err != nil {
		return err
	}

	// (f) SUBMIT_3D — the virgl command buffer rides as a separate
	// read-only descriptor (a 3-descriptor chain).
	vbuf := buildClearVirglBuffer(r, gr, b, a)
	if err := g.submit3D(vbuf); err != nil {
		return err
	}

	// (g) TRANSFER_TO_HOST_3D — 72 bytes: box{0,0,0,w,h,1}@24 (24B),
	// le64 offset@48, le32 resource_id@56, le32 level@60, le32 stride@64,
	// le32 layer_stride@68.
	transfer := make([]byte, ctrlHdrSize+48)
	binary.LittleEndian.PutUint32(transfer[0:4], CmdTransferToHost3D)
	binary.LittleEndian.PutUint32(transfer[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	bx := ctrlHdrSize
	binary.LittleEndian.PutUint32(transfer[bx+0:bx+4], 0)            // box.x
	binary.LittleEndian.PutUint32(transfer[bx+4:bx+8], 0)            // box.y
	binary.LittleEndian.PutUint32(transfer[bx+8:bx+12], 0)           // box.z
	binary.LittleEndian.PutUint32(transfer[bx+12:bx+16], width)      // box.w
	binary.LittleEndian.PutUint32(transfer[bx+16:bx+20], height)     // box.h
	binary.LittleEndian.PutUint32(transfer[bx+20:bx+24], 1)          // box.d
	binary.LittleEndian.PutUint64(transfer[bx+24:bx+32], 0)          // offset@48
	binary.LittleEndian.PutUint32(transfer[bx+32:bx+36], resourceID) // resource_id@56
	binary.LittleEndian.PutUint32(transfer[bx+36:bx+40], 0)          // level@60
	binary.LittleEndian.PutUint32(transfer[bx+40:bx+44], 0)          // stride@64
	binary.LittleEndian.PutUint32(transfer[bx+44:bx+48], 0)          // layer_stride@68
	if _, err := g.sendCommand(transfer); err != nil {
		return err
	}

	// (h) SET_SCANOUT — bind the resource to the scanout (reuses the 2D
	// command encoding).
	setScanout := make([]byte, ctrlHdrSize+rectSize+8)
	binary.LittleEndian.PutUint32(setScanout[0:4], cmdSetScanout)
	putRect(setScanout[ctrlHdrSize:], 0, 0, width, height)
	ss := ctrlHdrSize + rectSize
	binary.LittleEndian.PutUint32(setScanout[ss+0:ss+4], scanoutID)
	binary.LittleEndian.PutUint32(setScanout[ss+4:ss+8], resourceID)
	if _, err := g.sendCommand(setScanout); err != nil {
		return err
	}

	// (i) RESOURCE_FLUSH — refresh the scanout (reuses the 2D encoding).
	flush := make([]byte, ctrlHdrSize+rectSize+8)
	binary.LittleEndian.PutUint32(flush[0:4], cmdResourceFlush)
	putRect(flush[ctrlHdrSize:], 0, 0, width, height)
	fo := ctrlHdrSize + rectSize
	binary.LittleEndian.PutUint32(flush[fo+0:fo+4], resourceID)
	binary.LittleEndian.PutUint32(flush[fo+4:fo+8], 0) // padding
	if _, err := g.sendCommand(flush); err != nil {
		return err
	}
	return nil
}

// buildClearVirglBuffer hand-encodes the 19-dword (76-byte) virgl command
// buffer for the M1 clear: CREATE_OBJECT(SURFACE) +
// SET_FRAMEBUFFER_STATE + CLEAR. Colour components are float32 raw bits;
// the clear depth (0.0) is a float64 split into two le32 halves.
func buildClearVirglBuffer(r, gr, b, a float32) []byte {
	dwords := []uint32{
		// 1. CREATE_OBJECT(SURFACE): cmd=1, obj=7, len=5 -> 6 dwords.
		virglCmd0(virglCmdCreateObject, virglObjectSurface, 5),
		surfaceHandle,            // d1: surface handle = 1
		resourceID,               // d2: resource id = 1
		virglFormatB8G8R8A8Unorm, // d3: format = 1
		0,                        // d4: level = 0
		0,                        // d5: first_layer | last_layer<<16 = 0

		// 2. SET_FRAMEBUFFER_STATE: cmd=5, obj=0, len=3 -> 4 dwords.
		virglCmd0(virglCmdSetFramebufferState, 0, 3),
		1,             // d1: nr_cbufs = 1
		0,             // d2: zsurf handle = 0
		surfaceHandle, // d3: cbuf0 handle = surface handle = 1

		// 3. CLEAR: cmd=7, obj=0, len=8 -> 9 dwords.
		virglCmd0(virglCmdClear, 0, 8),
		pipeClearColor0,      // d1: buffers = PIPE_CLEAR_COLOR0
		math.Float32bits(r),  // d2: color[0]
		math.Float32bits(gr), // d3: color[1]
		math.Float32bits(b),  // d4: color[2]
		math.Float32bits(a),  // d5: color[3]
		0,                    // d6: depth low32  (0.0)
		0,                    // d7: depth high32 (0.0)
		0,                    // d8: stencil = 0
	}
	// depth = 0.0; spelled out via Float64bits to mirror the encoder.
	depthBits := math.Float64bits(0.0)
	dwords[16] = uint32(depthBits)
	dwords[17] = uint32(depthBits >> 32)

	buf := make([]byte, len(dwords)*4)
	for i, dw := range dwords {
		binary.LittleEndian.PutUint32(buf[i*4:i*4+4], dw)
	}
	return buf
}

// submit3D issues a SUBMIT_3D command. The 32-byte submit struct (le32
// size@24, le32 padding@28) carries the BYTE length of the virgl buffer,
// and the virgl buffer rides as a SEPARATE read-only descriptor, making
// this a 3-descriptor chain: [0] submit struct (RO), [1] virgl bytes (RO),
// [2] response (device-writable). The three descriptors are laid out in a
// single allocated page (submit@0, virgl@1024, response@2048).
func (g *VirtioGPU) submit3D(vbuf []byte) error {
	phys, mem, err := g.transport.AllocatePages(1)
	if err != nil {
		return err
	}
	if phys == 0 {
		return common.ErrAllocReturnedZero
	}

	const submitSize = ctrlHdrSize + 8 // 32 bytes
	submit := mem[0:submitSize]
	for i := range submit {
		submit[i] = 0
	}
	binary.LittleEndian.PutUint32(submit[0:4], CmdSubmit3D)
	binary.LittleEndian.PutUint32(submit[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	binary.LittleEndian.PutUint32(submit[ctrlHdrSize+0:ctrlHdrSize+4], uint32(len(vbuf))) // size@24
	binary.LittleEndian.PutUint32(submit[ctrlHdrSize+4:ctrlHdrSize+8], 0)                 // padding@28

	const virglOffset = 1024
	copy(mem[virglOffset:], vbuf)

	submitPhys := phys + 0
	virglPhys := phys + virglOffset
	respPhys := phys + responseOffset
	chain := []common.ChainBuffer{
		{Addr: uintptr(submitPhys), Phys: submitPhys, Len: submitSize, Writable: false},
		{Addr: uintptr(virglPhys), Phys: virglPhys, Len: uint32(len(vbuf)), Writable: false},
		{Addr: uintptr(respPhys), Phys: respPhys, Len: responseOffset, Writable: true},
	}

	head, err := g.controlq.AddChain(chain)
	if err != nil {
		return err
	}
	if err := g.Cfg.NotifyQueue(ControlQueueIdx, g.controlq.NotifyOff); err != nil {
		return err
	}

	respBuf := mem[responseOffset:]
	for spin := 0; spin < CommandPollIterations; spin++ {
		_, _, ok := g.controlq.PollUsed()
		if !ok {
			continue
		}
		_ = g.controlq.ReclaimChain(head)
		respType := binary.LittleEndian.Uint32(respBuf[0:4])
		if respType == respOKNoData {
			return nil
		}
		return ErrGPUCommandFailed
	}
	_ = g.controlq.ReclaimChain(head)
	return ErrRequestTimeout
}

// ErrVirglUnavailable is returned by OpenVirtioGPU3D when the host does not
// support VIRTIO_GPU_F_VIRGL (no virglrenderer): the FEATURES_OK bit does
// not latch after the VIRGL driver-feature write, or the negotiated mask
// comes back without the VIRGL bit.
var ErrVirglUnavailable = commonGPUError("go-virtio/gpu: host does not support VIRTIO_GPU_F_VIRGL (no virglrenderer)")
