// 3D / virgl extension of the virtio-gpu driver (Milestone 2). This file is
// purely additive: it does NOT touch the 2D path (OpenVirtioGPU /
// SetupFramebuffer / Flush / DisplayInfo) nor the M1 ClearScreen path in
// gpu3d.go. It adds a single high-level operation, DrawTriangle, that
// renders one flat-shaded triangle into a virtio-gpu scanout using the HOST
// GPU via virglrenderer — still pure Go, CGO=0.
//
// Why this is the "hard part". A clear (M1) only needs three virgl commands
// and no shaders. A *draw* needs the whole Gallium draw pipeline encoded by
// hand: two shaders, four pipeline-state objects (vertex-elements,
// rasterizer, blend, depth-stencil-alpha), a vertex buffer with guest
// backing, and the DRAW_VBO itself — all in one SUBMIT_3D virgl command
// buffer. The crux is the shader encoding. virglrenderer does NOT take
// binary TGSI tokens over the wire: virgl_encode_shader_state() in Mesa
// (gallium/drivers/virgl) calls tgsi_dump_str() and ships the shader as a
// NUL-terminated ASCII TGSI *text* dump; the host (vrend_decode_create_shader
// -> vrend_create_shader -> tgsi_text_translate) re-parses that text into
// tokens. So this driver hand-authors TGSI text, which the host parser
// accepts (str_match_nocase_whole on "VERT"/"FRAG", "DCL", "IMM", "MOV",
// "END"; see gallium/auxiliary/tgsi/tgsi_text.c). This sidesteps the binary
// token format entirely.
//
// Authoritative sources (every encoding below is annotated with the file +
// symbol it was derived from):
//
//   - virgl_protocol.h (virglrenderer src/) — enum virgl_context_cmd, enum
//     virgl_object_type, and the VIRGL_OBJ_*/VIRGL_SET_*/VIRGL_DRAW_VBO_*
//     field-offset + size macros.
//   - virgl_encode.c (Mesa gallium/drivers/virgl) — virgl_encode_shader_state,
//     virgl_emit_shader_header, virgl_emit_shader_streamout,
//     virgl_encoder_create_vertex_elements, virgl_encode_rasterizer_state,
//     virgl_encode_blend_state, virgl_encode_dsa_state,
//     virgl_encoder_set_vertex_buffers, virgl_encoder_draw_vbo,
//     virgl_encoder_set_viewport_states, virgl_encoder_create_surface,
//     virgl_encoder_set_framebuffer_state, virgl_encode_bind_object.
//   - tgsi_text.c (Mesa gallium/auxiliary/tgsi) — accepted TGSI text syntax.
//   - p_defines.h (Mesa gallium/include/pipe) — PIPE_SHADER_*, PIPE_BUFFER,
//     PIPE_PRIM_TRIANGLES.
//   - virgl_hw.h (virglrenderer src/) — VIRGL_BIND_*, VIRGL_FORMAT_*.
//
// See the gpu3d_draw_test.go header and the package REPORT for the per-dword
// derivation and the UNCERTAINTIES that still need hardware validation.
package gpu

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/go-virtio/common"
)

// --- virgl command opcodes used by the draw pipeline ------------------
//
// enum virgl_context_cmd (virgl_protocol.h): NOP=0, CREATE_OBJECT=1,
// BIND_OBJECT=2, DESTROY_OBJECT=3, SET_VIEWPORT_STATE=4,
// SET_FRAMEBUFFER_STATE=5, SET_VERTEX_BUFFERS=6, CLEAR=7, DRAW_VBO=8, ...,
// BIND_SHADER=32. A command word is VIRGL_CMD0(cmd, obj, len) =
// cmd | obj<<8 | len<<16 (reused from gpu3d.go's virglCmd0).
const (
	// VIRGL_CCMD_CREATE_OBJECT = 1.
	ccmdCreateObject uint32 = 1
	// VIRGL_CCMD_BIND_OBJECT = 2.
	ccmdBindObject uint32 = 2
	// VIRGL_CCMD_SET_VIEWPORT_STATE = 4.
	ccmdSetViewportState uint32 = 4
	// VIRGL_CCMD_SET_FRAMEBUFFER_STATE = 5.
	ccmdSetFramebufferState uint32 = 5
	// VIRGL_CCMD_SET_VERTEX_BUFFERS = 6.
	ccmdSetVertexBuffers uint32 = 6
	// VIRGL_CCMD_DRAW_VBO = 8.
	ccmdDrawVBO uint32 = 8
	// VIRGL_CCMD_BIND_SHADER = 32 (33rd entry, 0-based from NOP=0).
	ccmdBindShader uint32 = 32
)

// --- virgl object types (enum virgl_object_type, virgl_protocol.h) ----
//
// NULL=0, BLEND=1, RASTERIZER=2, DSA=3, SHADER=4, VERTEX_ELEMENTS=5,
// SAMPLER_VIEW=6, SAMPLER_STATE=7, SURFACE=8, ...
//
// objectSurface=8 here matches gpu3d.go's virglObjectSurface (both
// authoritative per virglrenderer src/virgl_protocol.h — the easy-to-miss
// SAMPLER_STATE slot is 7, SURFACE is 8).
const (
	objectBlend          uint32 = 1
	objectRasterizer     uint32 = 2
	objectDSA            uint32 = 3
	objectShader         uint32 = 4
	objectVertexElements uint32 = 5
	objectSurface        uint32 = 8
)

// --- pipe / virgl enum values baked into the draw -------------------------
const (
	// PIPE_SHADER_VERTEX = 0, PIPE_SHADER_FRAGMENT = 1 (p_defines.h,
	// enum pipe_shader_type). Written into VIRGL_OBJ_SHADER_TYPE.
	pipeShaderVertex   uint32 = 0
	pipeShaderFragment uint32 = 1

	// PIPE_PRIM_TRIANGLES = 4 (p_defines.h, enum pipe_prim_type). Written
	// into VIRGL_DRAW_VBO_MODE.
	pipePrimTriangles uint32 = 4

	// PIPE_BUFFER = 0 (p_defines.h, enum pipe_texture_target). The vertex
	// buffer resource's RESOURCE_CREATE_3D target.
	pipeBuffer uint32 = 0

	// VIRGL_BIND_VERTEX_BUFFER = (1<<4) = 16 (virgl_hw.h). The bind flag
	// for the vertex-buffer resource.
	virglBindVertexBuffer uint32 = 1 << 4

	// VIRGL_FORMAT_R32G32B32_FLOAT = 30 (virgl_hw.h, enum virgl_formats).
	// The vertex-element source format: 3 floats (x,y,z) per vertex.
	virglFormatR32G32B32Float uint32 = 30
)

// --- distinct handles for the draw -------------------------------------
//
// Object/resource handles are a single per-context namespace on the host.
// M1's ClearScreen uses resourceID=1 and surfaceHandle=1 but runs in its own
// invocation; DrawTriangle picks its own non-overlapping ids so the two paths
// never collide even if a caller mixed them in one context (DrawTriangle
// creates a fresh context regardless — see ctxID reuse note below).
const (
	rtResourceID   uint32 = 1 // render-target texture resource
	vbufResourceID uint32 = 2 // vertex-buffer resource

	drawSurfaceHandle uint32 = 10 // VIRGL_OBJECT_SURFACE handle
	vsHandle          uint32 = 11 // vertex-shader object handle
	fsHandle          uint32 = 12 // fragment-shader object handle
	veHandle          uint32 = 13 // VERTEX_ELEMENTS object handle
	rasterHandle      uint32 = 14 // RASTERIZER object handle
	blendHandle       uint32 = 15 // BLEND object handle
	dsaHandle         uint32 = 16 // DSA object handle
)

// vertexStride is the byte stride of one vertex: 3 float32 (x,y,z).
const vertexStride uint32 = 12

// shaderTokenBudget is the value written into VIRGL_OBJ_SHADER_NUM_TOKENS.
// The host (vrend_create_shader) allocates a tgsi_token output array of this
// many entries before calling tgsi_text_translate. Mesa's own driver derives
// it from tgsi_num_tokens() on the binary tokens; since this driver ships
// text, it has no binary tokens to count, so it advertises a generous fixed
// budget that comfortably exceeds the token count of these tiny shaders.
// (A 3-instruction shader is well under ~50 tokens; 4096 is safely large.)
// See the REPORT UNCERTAINTIES — the exact lower bound the host enforces was
// not confirmed from source.
const shaderTokenBudget uint32 = 4096

// vsText is the vertex shader as TGSI text (tgsi_text.c accepts this).
// Passthrough: the clip-space position arrives in IN[0] and is copied to the
// POSITION output. Declaring OUT[0] with the POSITION semantic makes it the
// clip-space position the rasterizer consumes.
const vsText = "VERT\n" +
	"DCL IN[0]\n" +
	"DCL OUT[0], POSITION\n" +
	"MOV OUT[0], IN[0]\n" +
	"END\n"

// fsTextFor builds the fragment shader as TGSI text with the flat colour
// baked into an FLT32 immediate. DCL OUT[0], COLOR declares the colour
// output (render target 0); MOV OUT[0], IMM[0] writes the constant colour.
// Baking the colour into the IMM avoids needing a constant buffer.
func fsTextFor(color [4]float32) string {
	return "FRAG\n" +
		"DCL OUT[0], COLOR\n" +
		fmt.Sprintf("IMM[0] FLT32 { %s, %s, %s, %s }\n",
			ftoa(color[0]), ftoa(color[1]), ftoa(color[2]), ftoa(color[3])) +
		"MOV OUT[0], IMM[0]\n" +
		"END\n"
}

// ftoa formats a float32 as a decimal literal the TGSI text parser accepts
// (parse_float uses atof). %g gives a compact round-trippable form; a bare
// integer like "1" is widened to "1.0" so it is unambiguously a float.
func ftoa(f float32) string {
	s := fmt.Sprintf("%g", float64(f))
	hasDotOrExp := false
	for i := 0; i < len(s); i++ {
		if s[i] == '.' || s[i] == 'e' || s[i] == 'E' {
			hasDotOrExp = true
			break
		}
	}
	if !hasDotOrExp {
		s += ".0"
	}
	return s
}

// EXPERIMENTAL — NOT VALIDATED ON REAL HARDWARE. Every command and TGSI
// shader below is hand-encoded against the Mesa virgl encoder + virglrenderer
// sources and is exercised only by an in-process fake device; it has NOT been
// run against a real virglrenderer/GPU. Several field encodings are inferred
// rather than confirmed from a known-good capture — in particular: the
// per-shader num_tokens budget, the minimal rasterizer state bits, the
// viewport depth half-range, the DRAW_VBO trailing fields, the PIPE_BUFFER
// RESOURCE_CREATE_3D semantics, and the absence of capset negotiation. Treat a
// working triangle as a hypothesis until validated on QEMU -device
// virtio-gpu-gl (or equivalent). The shaders are shipped as TGSI *text*
// (virglrenderer re-parses tgsi_dump_str output), not binary tokens.
//
// DrawTriangle renders one flat-shaded triangle to scanout scanoutID using
// the host GPU. verts is 3 clip-space vertices laid out x0,y0,z0, x1,y1,z1,
// x2,y2,z2 (the vertex shader passes them straight through to POSITION, so
// they must already be in clip space — e.g. each component in [-1,1]). color
// is the flat RGBA the fragment shader emits for every covered pixel.
//
// The sequence mirrors ClearScreen's host-GPU shape but adds the draw
// pipeline:
//
//	(a) resolve scanout dimensions (DisplayInfo).
//	(b) CTX_CREATE(ctx).
//	(c) RESOURCE_CREATE_3D(rt, w, h, BGRA, RENDER_TARGET) + backing +
//	    CTX_ATTACH_RESOURCE.
//	(d) RESOURCE_CREATE_3D(vbuf, PIPE_BUFFER, VERTEX_BUFFER, w=byteSize) +
//	    backing (the 9 floats) + CTX_ATTACH_RESOURCE.
//	(e) TRANSFER_TO_HOST_3D(vbuf) so the host sees the vertices.
//	(f) one SUBMIT_3D carrying the whole draw: create+bind VS, create+bind
//	    FS, create surface, create+bind vertex-elements / rasterizer / blend
//	    / dsa, SET_FRAMEBUFFER_STATE, SET_VIEWPORT_STATE,
//	    SET_VERTEX_BUFFERS, DRAW_VBO.
//	(g) TRANSFER_TO_HOST_3D(rt) to pull the rendered pixels into the backing.
//	(h) SET_SCANOUT(scanoutID, rt) + RESOURCE_FLUSH.
//
// Every step checks its response is OK_NODATA; any other type is
// ErrGPUCommandFailed.
func (g *VirtioGPU) DrawTriangle(scanoutID uint32, verts [9]float32, color [4]float32) error {
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

	// (b) CTX_CREATE — identical encoding to M1 (96 bytes total).
	if err := g.ctxCreate(); err != nil {
		return err
	}

	// (c) Render-target resource (BGRA, RENDER_TARGET) + backing + attach.
	if err := g.createRenderTarget(width, height); err != nil {
		return err
	}

	// (d) Vertex-buffer resource (PIPE_BUFFER, VERTEX_BUFFER) + backing.
	if err := g.createVertexBuffer(verts); err != nil {
		return err
	}

	// (f) Build + submit the per-draw virgl command buffer.
	vbuf := g.buildDrawVirglBuffer(width, height, color)
	if err := g.submit3D(vbuf); err != nil {
		return err
	}

	// (g) Pull the rendered pixels back into the render-target backing.
	if err := g.transferToHost3D(rtResourceID, width, height); err != nil {
		return err
	}

	// (h) SET_SCANOUT + RESOURCE_FLUSH (reuse the 2D encodings).
	if err := g.setScanout(scanoutID, rtResourceID, width, height); err != nil {
		return err
	}
	return g.resourceFlush(rtResourceID, width, height)
}

// ctxCreate issues CTX_CREATE(ctxID) — the same 96-byte struct M1 builds:
// le32 nlen@24, le32 context_init@28, byte debug_name[64]@32, all zero.
func (g *VirtioGPU) ctxCreate() error {
	ctxCreate := make([]byte, ctrlHdrSize+72)
	binary.LittleEndian.PutUint32(ctxCreate[0:4], CmdCtxCreate)
	binary.LittleEndian.PutUint32(ctxCreate[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	_, err := g.sendCommand(ctxCreate)
	return err
}

// createRenderTarget creates the BGRA render-target texture, allocates and
// attaches its guest backing, and attaches it to the context.
func (g *VirtioGPU) createRenderTarget(width, height uint32) error {
	if err := g.resourceCreate3D(rtResourceID, pipeTexture2D,
		virglFormatB8G8R8A8Unorm, virglBindRenderTarget, width, height, 1); err != nil {
		return err
	}
	size := uint64(width) * uint64(height) * 4
	if _, err := g.attachBacking(rtResourceID, size); err != nil {
		return err
	}
	return g.ctxAttachResource(rtResourceID)
}

// createVertexBuffer creates the PIPE_BUFFER vertex-buffer resource, writes
// the 9 float32 vertices into its guest backing, attaches it to the context,
// and TRANSFER_TO_HOST_3D's the bytes so the host GPU sees them.
func (g *VirtioGPU) createVertexBuffer(verts [9]float32) error {
	// RESOURCE_CREATE_3D for a buffer: target=PIPE_BUFFER, format=0, bind=
	// VERTEX_BUFFER, width=byte size, height=1, depth=1 (virgl_encode.c
	// virgl_encoder_create_resource / the kernel's create_3d struct).
	byteSize := uint32(9 * 4)
	if err := g.resourceCreate3D(vbufResourceID, pipeBuffer,
		0, virglBindVertexBuffer, byteSize, 1, 1); err != nil {
		return err
	}
	// Allocate backing and write the vertices into it directly.
	phys, mem, err := g.transport.AllocatePages(1)
	if err != nil {
		return err
	}
	if phys == 0 {
		return common.ErrAllocReturnedZero
	}
	for i := 0; i < 9; i++ {
		binary.LittleEndian.PutUint32(mem[i*4:i*4+4], math.Float32bits(verts[i]))
	}
	if err := g.attachBackingAt(vbufResourceID, phys, uint64(byteSize)); err != nil {
		return err
	}
	if err := g.ctxAttachResource(vbufResourceID); err != nil {
		return err
	}
	// TRANSFER_TO_HOST_3D for a buffer: box = {x=0, w=byteSize, rest 1/0}.
	return g.transferBufferToHost3D(vbufResourceID, byteSize)
}

// resourceCreate3D issues RESOURCE_CREATE_3D with the given parameters. The
// 48-byte body layout matches M1's: resource_id@24, target@28, format@32,
// bind@36, width@40, height@44, depth@48, array_size@52=1, last_level@56=0,
// nr_samples@60=0, flags@64=0, padding@68=0.
func (g *VirtioGPU) resourceCreate3D(resID, target, format, bind, width, height, depth uint32) error {
	create := make([]byte, ctrlHdrSize+48)
	binary.LittleEndian.PutUint32(create[0:4], CmdResourceCreate3D)
	binary.LittleEndian.PutUint32(create[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	b := ctrlHdrSize
	binary.LittleEndian.PutUint32(create[b+0:b+4], resID)    // resource_id@24
	binary.LittleEndian.PutUint32(create[b+4:b+8], target)   // target@28
	binary.LittleEndian.PutUint32(create[b+8:b+12], format)  // format@32
	binary.LittleEndian.PutUint32(create[b+12:b+16], bind)   // bind@36
	binary.LittleEndian.PutUint32(create[b+16:b+20], width)  // width@40
	binary.LittleEndian.PutUint32(create[b+20:b+24], height) // height@44
	binary.LittleEndian.PutUint32(create[b+24:b+28], depth)  // depth@48
	binary.LittleEndian.PutUint32(create[b+28:b+32], 1)      // array_size@52
	binary.LittleEndian.PutUint32(create[b+32:b+36], 0)      // last_level@56
	binary.LittleEndian.PutUint32(create[b+36:b+40], 0)      // nr_samples@60
	binary.LittleEndian.PutUint32(create[b+40:b+44], 0)      // flags@64
	binary.LittleEndian.PutUint32(create[b+44:b+48], 0)      // padding@68
	_, err := g.sendCommand(create)
	return err
}

// attachBacking allocates ceil(size/PageSize) pages and RESOURCE_ATTACH_BACKING's
// the run as one mem_entry. Returns the backing physical address.
func (g *VirtioGPU) attachBacking(resID uint32, size uint64) (uint64, error) {
	pages := int((size + uint64(common.PageSize) - 1) / uint64(common.PageSize))
	phys, _, err := g.transport.AllocatePages(pages)
	if err != nil {
		return 0, err
	}
	if phys == 0 {
		return 0, common.ErrAllocReturnedZero
	}
	if err := g.attachBackingAt(resID, phys, size); err != nil {
		return 0, err
	}
	return phys, nil
}

// attachBackingAt RESOURCE_ATTACH_BACKING's a single mem_entry{phys,size} to
// resID (same encoding as M1: resource_id@24, nr_entries@28=1, then mem_entry
// {le64 addr, le32 length, le32 padding}).
func (g *VirtioGPU) attachBackingAt(resID uint32, phys, size uint64) error {
	attach := make([]byte, ctrlHdrSize+8+memEntrySize)
	binary.LittleEndian.PutUint32(attach[0:4], cmdResourceAttachBacking)
	binary.LittleEndian.PutUint32(attach[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	binary.LittleEndian.PutUint32(attach[ctrlHdrSize+0:ctrlHdrSize+4], resID)
	binary.LittleEndian.PutUint32(attach[ctrlHdrSize+4:ctrlHdrSize+8], 1) // nr_entries
	me := ctrlHdrSize + 8
	binary.LittleEndian.PutUint64(attach[me+0:me+8], phys)
	binary.LittleEndian.PutUint32(attach[me+8:me+12], uint32(size))
	binary.LittleEndian.PutUint32(attach[me+12:me+16], 0) // padding
	_, err := g.sendCommand(attach)
	return err
}

// ctxAttachResource issues CTX_ATTACH_RESOURCE(ctxID, resID) — 32 bytes:
// resource_id@24, padding@28 (same as M1).
func (g *VirtioGPU) ctxAttachResource(resID uint32) error {
	ctxAttach := make([]byte, ctrlHdrSize+8)
	binary.LittleEndian.PutUint32(ctxAttach[0:4], CmdCtxAttachResource)
	binary.LittleEndian.PutUint32(ctxAttach[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	binary.LittleEndian.PutUint32(ctxAttach[ctrlHdrSize+0:ctrlHdrSize+4], resID)
	binary.LittleEndian.PutUint32(ctxAttach[ctrlHdrSize+4:ctrlHdrSize+8], 0) // padding
	_, err := g.sendCommand(ctxAttach)
	return err
}

// transferToHost3D pulls a 2D texture's pixels into its backing —
// box{0,0,0,w,h,1}, offset@48=0, resource_id@56, level@60=0, stride@64=0,
// layer_stride@68=0 (same as M1's transfer).
func (g *VirtioGPU) transferToHost3D(resID, width, height uint32) error {
	transfer := make([]byte, ctrlHdrSize+48)
	binary.LittleEndian.PutUint32(transfer[0:4], CmdTransferToHost3D)
	binary.LittleEndian.PutUint32(transfer[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	b := ctrlHdrSize
	binary.LittleEndian.PutUint32(transfer[b+0:b+4], 0)        // box.x
	binary.LittleEndian.PutUint32(transfer[b+4:b+8], 0)        // box.y
	binary.LittleEndian.PutUint32(transfer[b+8:b+12], 0)       // box.z
	binary.LittleEndian.PutUint32(transfer[b+12:b+16], width)  // box.w
	binary.LittleEndian.PutUint32(transfer[b+16:b+20], height) // box.h
	binary.LittleEndian.PutUint32(transfer[b+20:b+24], 1)      // box.d
	binary.LittleEndian.PutUint64(transfer[b+24:b+32], 0)      // offset@48
	binary.LittleEndian.PutUint32(transfer[b+32:b+36], resID)  // resource_id@56
	binary.LittleEndian.PutUint32(transfer[b+36:b+40], 0)      // level@60
	binary.LittleEndian.PutUint32(transfer[b+40:b+44], 0)      // stride@64
	binary.LittleEndian.PutUint32(transfer[b+44:b+48], 0)      // layer_stride@68
	_, err := g.sendCommand(transfer)
	return err
}

// transferBufferToHost3D pushes a PIPE_BUFFER resource's bytes to the host —
// for a buffer the box is linear: x=0, w=byteSize, y=z=0, h=d=1.
func (g *VirtioGPU) transferBufferToHost3D(resID, byteSize uint32) error {
	transfer := make([]byte, ctrlHdrSize+48)
	binary.LittleEndian.PutUint32(transfer[0:4], CmdTransferToHost3D)
	binary.LittleEndian.PutUint32(transfer[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	b := ctrlHdrSize
	binary.LittleEndian.PutUint32(transfer[b+0:b+4], 0)          // box.x
	binary.LittleEndian.PutUint32(transfer[b+4:b+8], 0)          // box.y
	binary.LittleEndian.PutUint32(transfer[b+8:b+12], 0)         // box.z
	binary.LittleEndian.PutUint32(transfer[b+12:b+16], byteSize) // box.w = byte size
	binary.LittleEndian.PutUint32(transfer[b+16:b+20], 1)        // box.h = 1
	binary.LittleEndian.PutUint32(transfer[b+20:b+24], 1)        // box.d = 1
	binary.LittleEndian.PutUint64(transfer[b+24:b+32], 0)        // offset@48
	binary.LittleEndian.PutUint32(transfer[b+32:b+36], resID)    // resource_id@56
	binary.LittleEndian.PutUint32(transfer[b+36:b+40], 0)        // level@60
	binary.LittleEndian.PutUint32(transfer[b+40:b+44], 0)        // stride@64
	binary.LittleEndian.PutUint32(transfer[b+44:b+48], 0)        // layer_stride@68
	_, err := g.sendCommand(transfer)
	return err
}

// setScanout binds resID to scanoutID (reuses the 2D SET_SCANOUT encoding).
func (g *VirtioGPU) setScanout(scanoutID, resID, width, height uint32) error {
	setScanout := make([]byte, ctrlHdrSize+rectSize+8)
	binary.LittleEndian.PutUint32(setScanout[0:4], cmdSetScanout)
	putRect(setScanout[ctrlHdrSize:], 0, 0, width, height)
	ss := ctrlHdrSize + rectSize
	binary.LittleEndian.PutUint32(setScanout[ss+0:ss+4], scanoutID)
	binary.LittleEndian.PutUint32(setScanout[ss+4:ss+8], resID)
	_, err := g.sendCommand(setScanout)
	return err
}

// resourceFlush refreshes the scanout (reuses the 2D RESOURCE_FLUSH encoding).
func (g *VirtioGPU) resourceFlush(resID, width, height uint32) error {
	flush := make([]byte, ctrlHdrSize+rectSize+8)
	binary.LittleEndian.PutUint32(flush[0:4], cmdResourceFlush)
	putRect(flush[ctrlHdrSize:], 0, 0, width, height)
	fo := ctrlHdrSize + rectSize
	binary.LittleEndian.PutUint32(flush[fo+0:fo+4], resID)
	binary.LittleEndian.PutUint32(flush[fo+4:fo+8], 0) // padding
	_, err := g.sendCommand(flush)
	return err
}

// --- the per-draw virgl command buffer ---------------------------------

// virglBuilder accumulates dwords into a virgl command buffer.
type virglBuilder struct {
	dw []uint32
}

func (b *virglBuilder) dword(v uint32)  { b.dw = append(b.dw, v) }
func (b *virglBuilder) float(f float32) { b.dw = append(b.dw, math.Float32bits(f)) }

// cmd writes a VIRGL_CMD0 header for `length` payload dwords.
func (b *virglBuilder) cmd(cmd, obj, length uint32) {
	b.dword(virglCmd0(cmd, obj, length))
}

// bytesLE serialises the accumulated dwords little-endian.
func (b *virglBuilder) bytesLE() []byte {
	buf := make([]byte, len(b.dw)*4)
	for i, v := range b.dw {
		binary.LittleEndian.PutUint32(buf[i*4:i*4+4], v)
	}
	return buf
}

// emitShader appends a CREATE_OBJECT(SHADER) command carrying TGSI text.
//
// virgl_emit_shader_header (virgl_encode.c): the header is
// CREATE_OBJECT|SHADER with len payload dwords, then
//
//	d1 handle
//	d2 type            (VIRGL_OBJ_SHADER_TYPE)
//	d3 offlen          (VIRGL_OBJ_SHADER_OFFSET) = byte length of text incl.
//	                    trailing NUL, CONT bit (0x80000000) clear for a
//	                    single-chunk shader
//	d4 num_tokens      (VIRGL_OBJ_SHADER_NUM_TOKENS)
//
// then virgl_emit_shader_streamout appends
//
//	d5 num_outputs = 0 (no stream-out)
//
// then the NUL-terminated text, padded with zero bytes to a dword boundary.
//
// len = hdr(5: cmd+handle+type+offlen+num_tokens is 5 dwords incl. cmd, so
// the payload after the cmd dword is handle+type+offlen+num_tokens(4) +
// num_outputs(1) + textDwords) = 5 + textDwords. This matches
// virgl_encode_shader_state's `len = ((length+3)/4) + hdr_len` with
// hdr_len=5 and no stream-out.
func (b *virglBuilder) emitShader(handle, shaderType uint32, text string) {
	raw := append([]byte(text), 0) // NUL terminator
	shaderLen := uint32(len(raw))  // byte length incl. NUL -> offlen
	textDwords := (shaderLen + 3) / 4
	length := 5 + textDwords // payload dwords after the cmd header

	b.cmd(ccmdCreateObject, objectShader, length)
	b.dword(handle)            // VIRGL_OBJ_SHADER_HANDLE
	b.dword(shaderType)        // VIRGL_OBJ_SHADER_TYPE
	b.dword(shaderLen)         // VIRGL_OBJ_SHADER_OFFSET (CONT clear)
	b.dword(shaderTokenBudget) // VIRGL_OBJ_SHADER_NUM_TOKENS
	b.dword(0)                 // streamout num_outputs = 0

	// Append the text padded to a dword boundary with zero bytes.
	padded := make([]byte, textDwords*4)
	copy(padded, raw)
	for i := 0; i < len(padded); i += 4 {
		b.dword(binary.LittleEndian.Uint32(padded[i : i+4]))
	}
}

// bindShader appends BIND_SHADER(type, handle).
//
// virgl_protocol.h: VIRGL_BIND_SHADER_SIZE=2, _HANDLE=1, _TYPE=2. The Mesa
// encoder writes CMD0(BIND_SHADER,0,2) then dword handle, dword type.
func (b *virglBuilder) bindShader(handle, shaderType uint32) {
	b.cmd(ccmdBindShader, 0, 2)
	b.dword(handle)     // VIRGL_BIND_SHADER_HANDLE
	b.dword(shaderType) // VIRGL_BIND_SHADER_TYPE
}

// bindObject appends BIND_OBJECT(obj, handle).
//
// virgl_encode_bind_object (virgl_encode.c): CMD0(BIND_OBJECT, object, 1)
// then dword handle.
func (b *virglBuilder) bindObject(obj, handle uint32) {
	b.cmd(ccmdBindObject, obj, 1)
	b.dword(handle)
}

// buildDrawVirglBuffer hand-encodes the entire per-draw virgl command buffer:
// create+bind both shaders, create the surface, create+bind the four pipeline
// state objects, set framebuffer/viewport/vertex-buffer state, and DRAW_VBO.
func (g *VirtioGPU) buildDrawVirglBuffer(width, height uint32, color [4]float32) []byte {
	b := &virglBuilder{}

	// 1. Vertex + fragment shaders (CREATE_OBJECT SHADER + BIND_SHADER).
	b.emitShader(vsHandle, pipeShaderVertex, vsText)
	b.bindShader(vsHandle, pipeShaderVertex)
	b.emitShader(fsHandle, pipeShaderFragment, fsTextFor(color))
	b.bindShader(fsHandle, pipeShaderFragment)

	// 2. Surface over the render target (CREATE_OBJECT SURFACE).
	//    virgl_encoder_create_surface (non-MSAA): VIRGL_OBJ_SURFACE_SIZE=5.
	//    d1 handle, d2 res, d3 format, d4 level, d5 first|last<<16.
	b.cmd(ccmdCreateObject, objectSurface, 5)
	b.dword(drawSurfaceHandle)        // VIRGL_OBJ_SURFACE_HANDLE
	b.dword(rtResourceID)             // VIRGL_OBJ_SURFACE_RES_HANDLE
	b.dword(virglFormatB8G8R8A8Unorm) // VIRGL_OBJ_SURFACE_FORMAT
	b.dword(0)                        // texture level
	b.dword(0)                        // first_layer | (last_layer<<16) = 0

	// 3. Vertex elements: one element, src_offset=0, instance_divisor=0,
	//    vertex_buffer_index=0, src_format=R32G32B32_FLOAT.
	//    virgl_encoder_create_vertex_elements: SIZE(1)=1*4+1=5.
	//    d1 handle, then per element: src_offset, instance_divisor,
	//    vertex_buffer_index, src_format.
	b.cmd(ccmdCreateObject, objectVertexElements, 5)
	b.dword(veHandle)                  // VIRGL_OBJ_VERTEX_ELEMENTS_HANDLE
	b.dword(0)                         // element 0 src_offset
	b.dword(0)                         // element 0 instance_divisor
	b.dword(0)                         // element 0 vertex_buffer_index
	b.dword(virglFormatR32G32B32Float) // element 0 src_format
	b.bindObject(objectVertexElements, veHandle)

	// 4. Rasterizer (minimal sane state). virgl_encode_rasterizer_state:
	//    VIRGL_OBJ_RS_SIZE=9. d1 handle, d2 S0, d3 point_size(fui),
	//    d4 sprite_coord_enable, d5 S3, d6 line_width(fui), d7 offset_units,
	//    d8 offset_scale, d9 offset_clamp.
	//    S0: only DEPTH_CLIP(bit1) and HALF_PIXEL_CENTER(bit29) set — a
	//    conventional minimal raster state with no culling so winding order
	//    cannot drop the triangle. point_size/line_width = 1.0.
	b.cmd(ccmdCreateObject, objectRasterizer, 9)
	b.dword(rasterHandle)         // handle
	b.dword((1 << 1) | (1 << 29)) // S0: DEPTH_CLIP | HALF_PIXEL_CENTER
	b.float(1.0)                  // point_size
	b.dword(0)                    // sprite_coord_enable
	b.dword(0)                    // S3 (line stipple / clip planes)
	b.float(1.0)                  // line_width
	b.float(0.0)                  // offset_units
	b.float(0.0)                  // offset_scale
	b.float(0.0)                  // offset_clamp
	b.bindObject(objectRasterizer, rasterHandle)

	// 5. Blend (one RT, writemask 0xf, no blending). virgl_encode_blend_state:
	//    VIRGL_OBJ_BLEND_SIZE = VIRGL_MAX_COLOR_BUFS+3 = 8+3 = 11.
	//    d1 handle, d2 S0=0, d3 S1=0, then 8 per-RT S2 dwords. RT0 has
	//    COLORMASK=0xf at bits 27..30; RT1..7 zero.
	b.cmd(ccmdCreateObject, objectBlend, 11)
	b.dword(blendHandle) // handle
	b.dword(0)           // S0
	b.dword(0)           // S1
	b.dword(0xf << 27)   // RT0 S2: COLORMASK=0xf (VIRGL_OBJ_BLEND_S2_RT_COLORMASK)
	for i := 0; i < 7; i++ {
		b.dword(0) // RT1..RT7 S2 = 0
	}
	b.bindObject(objectBlend, blendHandle)

	// 6. DSA (depth test off). virgl_encode_dsa_state: VIRGL_OBJ_DSA_SIZE=5.
	//    d1 handle, d2 S0=0 (depth disabled), d3 stencil[0]=0,
	//    d4 stencil[1]=0, d5 alpha_ref(fui)=0.
	b.cmd(ccmdCreateObject, objectDSA, 5)
	b.dword(dsaHandle) // handle
	b.dword(0)         // S0: depth/alpha all disabled
	b.dword(0)         // stencil[0]
	b.dword(0)         // stencil[1]
	b.dword(0)         // alpha_ref_value (fui(0.0) == 0)
	b.bindObject(objectDSA, dsaHandle)

	// 7. SET_FRAMEBUFFER_STATE: nr_cbufs=1, zsurf=0, cbuf0=surface.
	//    virgl_encoder_set_framebuffer_state: SIZE(1)=1+2=3.
	b.cmd(ccmdSetFramebufferState, 0, 3)
	b.dword(1)                 // nr_cbufs
	b.dword(0)                 // zsurf handle
	b.dword(drawSurfaceHandle) // cbuf0 handle

	// 8. SET_VIEWPORT_STATE (full surface). virgl_encoder_set_viewport_states:
	//    SIZE(1) = 6*1+1 = 7. d1 start_slot=0, then scale[0..2],
	//    translate[0..2]. For an OpenGL [-1,1]->[0,W]x[0,H] window transform:
	//    scale = (W/2, H/2, 0.5), translate = (W/2, H/2, 0.5).
	b.cmd(ccmdSetViewportState, 0, 7)
	b.dword(0) // start_slot
	hw := float32(width) / 2
	hh := float32(height) / 2
	b.float(hw)  // scale[0]
	b.float(hh)  // scale[1]
	b.float(0.5) // scale[2]
	b.float(hw)  // translate[0]
	b.float(hh)  // translate[1]
	b.float(0.5) // translate[2]

	// 9. SET_VERTEX_BUFFERS (one buffer). virgl_encoder_set_vertex_buffers:
	//    SIZE(1)=1*3=3. Per buffer: stride, buffer_offset, res handle.
	b.cmd(ccmdSetVertexBuffers, 0, 3)
	b.dword(vertexStride)   // stride = 12
	b.dword(0)              // buffer_offset
	b.dword(vbufResourceID) // resource handle

	// 10. DRAW_VBO. virgl_encoder_draw_vbo: VIRGL_DRAW_VBO_SIZE=12.
	//     start, count, mode, indexed, instance_count, index_bias,
	//     start_instance, primitive_restart, restart_index, min_index,
	//     max_index, count_from_so.
	b.cmd(ccmdDrawVBO, 0, 12)
	b.dword(0)                 // start
	b.dword(3)                 // count = 3 vertices
	b.dword(pipePrimTriangles) // mode = PIPE_PRIM_TRIANGLES
	b.dword(0)                 // indexed = 0
	b.dword(1)                 // instance_count = 1
	b.dword(0)                 // index_bias
	b.dword(0)                 // start_instance
	b.dword(0)                 // primitive_restart
	b.dword(0)                 // restart_index
	b.dword(0)                 // min_index
	b.dword(0xFFFFFFFF)        // max_index = ~0 (index_bounds invalid)
	b.dword(0)                 // count_from_stream_output = 0

	return b.bytesLE()
}
