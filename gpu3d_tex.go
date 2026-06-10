// 3D / virgl extension of the virtio-gpu driver (Milestone 3). This file is
// purely additive: it does NOT touch the 2D path (OpenVirtioGPU /
// SetupFramebuffer / Flush / DisplayInfo), the M1 ClearScreen path in
// gpu3d.go, nor the M2 DrawTriangle path in gpu3d_draw.go. It adds a single
// high-level operation, DrawTexturedTriangle, that renders one triangle whose
// fragments are sampled from a host-side texture, into a virtio-gpu scanout
// using the HOST GPU via virglrenderer — still pure Go, CGO=0.
//
// M3 = M2 (DrawTriangle) + a sampled texture. On top of the M2 draw pipeline
// it adds, per Mesa's virgl encoder + virglrenderer protocol:
//
//   - a sampled-texture resource (RGBA8, VIRGL_BIND_SAMPLER_VIEW) with a guest
//     backing the caller's texels are written into + TRANSFER_TO_HOST_3D;
//   - CREATE_OBJECT(SAMPLER_VIEW=6) over that texture (format + texture-layer +
//     texture-level + swizzle dwords);
//   - CREATE_OBJECT(SAMPLER_STATE=7) (wrap/filter packing);
//   - SET_SAMPLER_VIEWS(FRAGMENT, slot 0) + BIND_SAMPLER_STATES(FRAGMENT,
//     slot 0);
//   - a 2-element VERTEX_ELEMENTS layout (position R32G32B32_FLOAT @0 +
//     texcoord R32G32_FLOAT @12, stride 20);
//   - texturing TGSI shaders: the VS forwards a GENERIC[0] texcoord, the FS
//     declares SAMP[0]/SVIEW[0] and TEX-samples OUT[0] COLOR.
//
// Authoritative sources (each encoding below cites the file + symbol it was
// derived from; line numbers are from the notaz/mesa mirror of Mesa, branch
// master, fetched 2026-06-10):
//
//   - virgl_protocol.h (Mesa src/gallium/drivers/virgl) — enum
//     virgl_object_type (SAMPLER_VIEW=6, SAMPLER_STATE=7), enum
//     virgl_context_cmd (SET_SAMPLER_VIEWS=10, BIND_SAMPLER_STATES=18), and
//     the VIRGL_OBJ_SAMPLER_VIEW_* / VIRGL_OBJ_SAMPLE_STATE_S0_* /
//     VIRGL_SET_SAMPLER_VIEWS_* / VIRGL_BIND_SAMPLER_STATES_* macros.
//   - virgl_encode.c (Mesa src/gallium/drivers/virgl) —
//     virgl_encode_sampler_view, virgl_encode_sampler_state,
//     virgl_encode_set_sampler_views, virgl_encode_bind_sampler_states.
//   - virgl_hw.h (Mesa src/gallium/drivers/virgl) — VIRGL_FORMAT_R32G32_FLOAT
//     =29, VIRGL_FORMAT_R8G8B8A8_UNORM=67, VIRGL_BIND_SAMPLER_VIEW=(1<<3).
//   - tgsi_text.c / tgsi_strings.c / tgsi_info.c (Mesa
//     src/gallium/auxiliary/tgsi) — the DCL SAMP / DCL SVIEW grammar, the TEX
//     instruction operand order, and the "2D"/"FLOAT"/"GENERIC" token names.
//   - p_defines.h (Mesa src/gallium/include/pipe) — PIPE_SWIZZLE_*,
//     PIPE_TEX_WRAP_*, PIPE_TEX_FILTER_*, PIPE_TEX_MIPFILTER_*.
//
// See the gpu3d_tex_test.go header and the package REPORT for the per-dword
// derivation and the LOUD UNCERTAINTIES that still need hardware validation.
package gpu

import (
	"encoding/binary"
	"math"

	"github.com/go-virtio/common"
)

// --- M3-specific virgl command opcodes --------------------------------
//
// enum virgl_context_cmd (virgl_protocol.h), counting from NOP=0:
// ... CLEAR=7, DRAW_VBO=8, RESOURCE_INLINE_WRITE=9, SET_SAMPLER_VIEWS=10,
// SET_INDEX_BUFFER=11, SET_CONSTANT_BUFFER=12, SET_STENCIL_REF=13,
// SET_BLEND_COLOR=14, SET_SCISSOR_STATE=15, BLIT=16, RESOURCE_COPY_REGION=17,
// BIND_SAMPLER_STATES=18, ...
const (
	// VIRGL_CCMD_SET_SAMPLER_VIEWS = 10.
	ccmdSetSamplerViews uint32 = 10
	// VIRGL_CCMD_BIND_SAMPLER_STATES = 18.
	ccmdBindSamplerStates uint32 = 18
)

// --- M3-specific virgl object types (enum virgl_object_type) ----------
//
// NULL=0, BLEND=1, RASTERIZER=2, DSA=3, SHADER=4, VERTEX_ELEMENTS=5,
// SAMPLER_VIEW=6, SAMPLER_STATE=7, SURFACE=8 — confirmed verbatim against
// virgl_protocol.h. objectSurface (=8) and the others are already defined in
// gpu3d_draw.go; M3 only needs the two sampler object ids.
const (
	objectSamplerView  uint32 = 6
	objectSamplerState uint32 = 7
)

// --- M3-specific pipe / virgl enum values -----------------------------
const (
	// VIRGL_FORMAT_R32G32_FLOAT = 29 (virgl_hw.h). The texcoord
	// vertex-element source format: 2 floats (u,v) per vertex.
	virglFormatR32G32Float uint32 = 29

	// VIRGL_FORMAT_R8G8B8A8_UNORM = 67 (virgl_hw.h). The sampled texture's
	// pixel format; the caller supplies RGBA8 texels in this byte order.
	virglFormatR8G8B8A8Unorm uint32 = 67

	// VIRGL_BIND_SAMPLER_VIEW = (1<<3) = 8 (virgl_hw.h). The bind flag for
	// a resource that will back a sampler view.
	virglBindSamplerView uint32 = 1 << 3
)

// --- M3-specific distinct handles -------------------------------------
//
// DrawTexturedTriangle creates its own context (ctxID) like DrawTriangle, and
// picks handles/resource-ids that do not collide with M1/M2 within that
// context. M2 used resource ids 1..2 and object handles 10..16; M3 reuses the
// same render-target + vertex-buffer ids (it builds a fresh context) and adds
// the texture resource + the two sampler-object handles + the sampler-view
// handle in a fresh range.
const (
	texResourceID uint32 = 3 // sampled-texture resource

	samplerViewHandle  uint32 = 20 // VIRGL_OBJECT_SAMPLER_VIEW handle
	samplerStateHandle uint32 = 21 // VIRGL_OBJECT_SAMPLER_STATE handle
)

// texVertexStride is the byte stride of one M3 vertex: 3 float32 position
// (x,y,z) + 2 float32 texcoord (u,v) = 5 float32 = 20 bytes.
const texVertexStride uint32 = 20

// --- M3 sampler-state field values (p_defines.h pipe enums) -----------
//
// A conventional "sample a colour texture" state: CLAMP_TO_EDGE wrap on all
// axes, LINEAR min/mag image filter, MIPFILTER_NONE (single level), no
// shadow compare. These are encoded into the VIRGL_OBJ_SAMPLE_STATE_S0 dword.
const (
	pipeTexWrapClampToEdge uint32 = 2 // PIPE_TEX_WRAP_CLAMP_TO_EDGE
	pipeTexFilterLinear    uint32 = 1 // PIPE_TEX_FILTER_LINEAR
	pipeTexMipFilterNone   uint32 = 2 // PIPE_TEX_MIPFILTER_NONE
)

// samplerStateS0 packs the VIRGL_OBJ_SAMPLE_STATE_S0 dword exactly as
// virgl_encode_sampler_state does:
//
//	WRAP_S(x)      = (x & 0x7) << 0
//	WRAP_T(x)      = (x & 0x7) << 3
//	WRAP_R(x)      = (x & 0x7) << 6
//	MIN_IMG_FILTER = (x & 0x3) << 9
//	MIN_MIP_FILTER = (x & 0x3) << 11
//	MAG_IMG_FILTER = (x & 0x3) << 13
//	COMPARE_MODE   = (x & 0x1) << 15  (0 = none)
//	COMPARE_FUNC   = (x & 0x7) << 16  (0)
func samplerStateS0() uint32 {
	return (pipeTexWrapClampToEdge << 0) |
		(pipeTexWrapClampToEdge << 3) |
		(pipeTexWrapClampToEdge << 6) |
		(pipeTexFilterLinear << 9) | // min_img_filter
		(pipeTexMipFilterNone << 11) | // min_mip_filter
		(pipeTexFilterLinear << 13) // mag_img_filter
	// compare_mode (bit15) and compare_func (bits16..18) left 0.
}

// --- M3 sampler-view swizzle (p_defines.h PIPE_SWIZZLE_*) -------------
//
// Identity swizzle: R<-RED(0), G<-GREEN(1), B<-BLUE(2), A<-ALPHA(3), packed by
// virgl_encode_sampler_view as:
//
//	SWIZZLE_R(x) = (x & 0x7) << 0
//	SWIZZLE_G(x) = (x & 0x7) << 3
//	SWIZZLE_B(x) = (x & 0x7) << 6
//	SWIZZLE_A(x) = (x & 0x7) << 9
const (
	pipeSwizzleRed   uint32 = 0
	pipeSwizzleGreen uint32 = 1
	pipeSwizzleBlue  uint32 = 2
	pipeSwizzleAlpha uint32 = 3
)

func samplerViewSwizzle() uint32 {
	return (pipeSwizzleRed << 0) |
		(pipeSwizzleGreen << 3) |
		(pipeSwizzleBlue << 6) |
		(pipeSwizzleAlpha << 9)
}

// vsTexText is the M3 vertex shader as TGSI text. It forwards the clip-space
// position (IN[0] -> OUT[0] POSITION) and passes the texcoord through a
// GENERIC[0] varying (IN[1] GENERIC[0] -> OUT[1] GENERIC[0]). The rasterizer
// interpolates GENERIC[0] across the triangle for the fragment shader.
const vsTexText = "VERT\n" +
	"DCL IN[0]\n" +
	"DCL IN[1]\n" +
	"DCL OUT[0], POSITION\n" +
	"DCL OUT[1], GENERIC[0]\n" +
	"MOV OUT[0], IN[0]\n" +
	"MOV OUT[1], IN[1]\n" +
	"END\n"

// fsTexText is the M3 fragment shader as TGSI text. It declares the
// interpolated texcoord varying (IN[0] GENERIC[0]), the colour output, a
// sampler (SAMP[0]) and a 2D float sampler view (SVIEW[0], 2D, FLOAT), then
// TEX-samples the texture at the interpolated coordinate into OUT[0].
//
// Grammar confirmed against tgsi_text.c:
//   - "DCL IN[0], GENERIC[0], PERSPECTIVE" : a fragment input MUST carry an
//     interpolation token. tgsi_text.c's parse_declaration defaults an
//     unqualified input to TGSI_INTERPOLATE_CONSTANT (flat), so "DCL IN[0],
//     GENERIC[0]" makes the texcoord constant at the provoking vertex — real
//     virglrenderer validation showed the whole triangle sampling a single
//     texel. PERSPECTIVE gives perspective-correct interpolation across the
//     primitive, which is what a texcoord varying needs.
//   - "DCL SAMP[0]" : file=SAMP, no extra tokens (parse_declaration falls
//     through for the SAMPLER file).
//   - "DCL SVIEW[0], 2D, FLOAT" : file=SVIEW, then a texture-target name
//     ("2D", tgsi_texture_names) and a return-type name ("FLOAT",
//     tgsi_return_type_names).
//   - "TEX OUT[0], IN[0], SAMP[0], 2D" : TGSI_OPCODE_TEX has num_dst=1,
//     num_src=2, is_tex=1 (tgsi_info.c), so the operands are dst=OUT[0],
//     src0=IN[0], src1=SAMP[0], then the texture target token "2D".
const fsTexText = "FRAG\n" +
	"DCL IN[0], GENERIC[0], PERSPECTIVE\n" +
	"DCL OUT[0], COLOR\n" +
	"DCL SAMP[0]\n" +
	"DCL SVIEW[0], 2D, FLOAT\n" +
	"TEX OUT[0], IN[0], SAMP[0], 2D\n" +
	"END\n"

// DrawTexturedTriangle renders one triangle textured from the host GPU to
// scanout scanoutID.
//
// VALIDATED against a real virglrenderer (software llvmpipe, via the
// go-virtio/validate vtest harness): the full textured-triangle stream is
// accepted and the texture is sampled and perspective-correctly interpolated
// across the primitive — a 2×2 red/green/blue/white texture yields a smooth
// multi-texel gradient in the framebuffer readback (24 distinct interior
// colours), no renderer error. The texture-specific commands (SAMPLER_VIEW /
// SAMPLER_STATE creation, SET_SAMPLER_VIEWS, BIND_SAMPLER_STATES) and the
// texcoord varying are all confirmed. This validation caught a real shader bug:
// the fragment texcoord input defaulted to CONSTANT (flat) interpolation until
// declared PERSPECTIVE (see fsTexText) — until then the whole triangle sampled
// a single texel. NOT yet pixel-asserted for exact texel placement. The shaders
// are shipped as TGSI *text* (virglrenderer re-parses tgsi_dump_str output),
// not binary tokens.
//
// NEW INFERRED FIELDS in M3 (LOUD — byte-review these before publishing):
//
//   - SAMPLER_VIEW packing: for a non-PIPE_BUFFER target, virgl_encode.c
//     writes dword4 = first_layer | (last_layer<<16) and dword5 = first_level
//     | (last_level<<8). M3 uses all-zero (single layer, single level), so
//     both dwords are 0 — but the *shape* (which field lands in which dword)
//     is inferred from source, not from a known-good capture.
//   - SAMPLER_STATE S0 bits: the wrap/filter/compare bit packing
//     (samplerStateS0). The chosen values (CLAMP_TO_EDGE/LINEAR/NONE) are a
//     conventional default, not a captured one.
//   - The swizzle dword (samplerViewSwizzle = identity RGBA = 0x688). Bit
//     layout is from VIRGL_OBJ_SAMPLER_VIEW_SWIZZLE_* macros.
//   - The texcoord vertex-element format VIRGL_FORMAT_R32G32_FLOAT (=29) and
//     the 2-element layout / stride-20 vertex buffer.
//   - The texturing TGSI grammar (DCL SAMP / DCL SVIEW / TEX with the "2D"
//     target token) — parsed by tgsi_text.c, but never exercised here against
//     a real host parser.
//   - The sampled texture is created as RGBA8 (R8G8B8A8_UNORM) with the
//     caller's bytes uploaded verbatim; no row-padding / stride handling is
//     attempted (tightly packed texW*texH*4 bytes, stride=0 in the transfer).
//
// verts is 3 vertices, each x,y,z,u,v (5 float32) = 15 float32: clip-space
// position (already in clip space, e.g. each xyz component in [-1,1]) plus a
// texture coordinate (u,v in [0,1]). tex is texW*texH*4 RGBA8 bytes.
//
// The sequence mirrors DrawTriangle's host-GPU shape but adds the texture:
//
//	(a) resolve scanout dimensions (DisplayInfo).
//	(b) CTX_CREATE(ctx).
//	(c) RESOURCE_CREATE_3D(rt, BGRA, RENDER_TARGET) + backing + attach.
//	(d) RESOURCE_CREATE_3D(vbuf, PIPE_BUFFER, VERTEX_BUFFER) + backing
//	    (15 floats) + attach + TRANSFER_TO_HOST_3D.
//	(e) RESOURCE_CREATE_3D(tex, 2D, RGBA8, SAMPLER_VIEW) + backing (texels) +
//	    attach + TRANSFER_TO_HOST_3D.
//	(f) one SUBMIT_3D carrying the whole textured draw: create+bind VS/FS,
//	    create surface, create+bind 2-element vertex-elements / rasterizer /
//	    blend / dsa, create sampler-view + sampler-state, SET_SAMPLER_VIEWS,
//	    BIND_SAMPLER_STATES, SET_FRAMEBUFFER_STATE, SET_VIEWPORT_STATE,
//	    SET_VERTEX_BUFFERS, DRAW_VBO.
//	(g) TRANSFER_TO_HOST_3D(rt) to pull the rendered pixels into the backing.
//	(h) SET_SCANOUT(scanoutID, rt) + RESOURCE_FLUSH.
//
// Every step checks its response is OK_NODATA; any other type is
// ErrGPUCommandFailed.
func (g *VirtioGPU) DrawTexturedTriangle(scanoutID uint32, verts [15]float32, tex []byte, texW, texH uint32) error {
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

	// (b) CTX_CREATE — identical encoding to M1/M2.
	if err := g.ctxCreate(); err != nil {
		return err
	}

	// (c) Render-target resource (BGRA, RENDER_TARGET) + backing + attach.
	if err := g.createRenderTarget(width, height); err != nil {
		return err
	}

	// (d) Vertex-buffer resource (PIPE_BUFFER, VERTEX_BUFFER) + backing.
	if err := g.createTexVertexBuffer(verts); err != nil {
		return err
	}

	// (e) Sampled-texture resource (2D, RGBA8, SAMPLER_VIEW) + texels.
	if err := g.createTexture(tex, texW, texH); err != nil {
		return err
	}

	// (f) Build + submit the per-draw virgl command buffer.
	vbuf := g.buildTexDrawVirglBuffer(width, height)
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

// createTexVertexBuffer creates the PIPE_BUFFER vertex-buffer resource, writes
// the 15 float32 vertices (3 × x,y,z,u,v) into its guest backing, attaches it
// to the context, and TRANSFER_TO_HOST_3D's the bytes so the host GPU sees
// them. Same shape as M2's createVertexBuffer but 15 floats instead of 9.
func (g *VirtioGPU) createTexVertexBuffer(verts [15]float32) error {
	byteSize := uint32(15 * 4)
	if err := g.resourceCreate3D(vbufResourceID, pipeBuffer,
		0, virglBindVertexBuffer, byteSize, 1, 1); err != nil {
		return err
	}
	phys, mem, err := g.transport.AllocatePages(1)
	if err != nil {
		return err
	}
	if phys == 0 {
		return common.ErrAllocReturnedZero
	}
	for i := 0; i < 15; i++ {
		binary.LittleEndian.PutUint32(mem[i*4:i*4+4], math.Float32bits(verts[i]))
	}
	if err := g.attachBackingAt(vbufResourceID, phys, uint64(byteSize)); err != nil {
		return err
	}
	if err := g.ctxAttachResource(vbufResourceID); err != nil {
		return err
	}
	return g.transferBufferToHost3D(vbufResourceID, byteSize)
}

// createTexture creates the sampled 2D texture resource (RGBA8,
// VIRGL_BIND_SAMPLER_VIEW), copies the caller's RGBA8 texels into a fresh
// guest backing, attaches it to the context, and TRANSFER_TO_HOST_3D's the
// full 2D box so the host GPU sees the texels.
//
// The texels are uploaded tightly packed (texW*texH*4 bytes, no row padding);
// the transfer's box is the full 2D image and stride/layer_stride are left 0
// (host infers them from the resource for a tightly packed level-0 transfer).
func (g *VirtioGPU) createTexture(tex []byte, texW, texH uint32) error {
	if err := g.resourceCreate3D(texResourceID, pipeTexture2D,
		virglFormatR8G8B8A8Unorm, virglBindSamplerView, texW, texH, 1); err != nil {
		return err
	}
	size := uint64(texW) * uint64(texH) * 4
	phys, mem, err := g.allocPagesFor(size)
	if err != nil {
		return err
	}
	if phys == 0 {
		return common.ErrAllocReturnedZero
	}
	copy(mem, tex)
	if err := g.attachBackingAt(texResourceID, phys, size); err != nil {
		return err
	}
	if err := g.ctxAttachResource(texResourceID); err != nil {
		return err
	}
	return g.transferTextureToHost3D(texResourceID, texW, texH)
}

// allocPagesFor allocates ceil(size/PageSize) pages and returns the backing
// physical address + its byte slice. Mirrors attachBacking's page math but
// also hands back the memory so callers can populate it (texture texels).
func (g *VirtioGPU) allocPagesFor(size uint64) (uint64, []byte, error) {
	pages := int((size + uint64(common.PageSize) - 1) / uint64(common.PageSize))
	return g.transport.AllocatePages(pages)
}

// transferTextureToHost3D pushes a 2D texture's level-0 texels to the host —
// box{0,0,0,w,h,1}, offset@48=0, resource_id@56, level@60=0, stride@64=0,
// layer_stride@68=0. Same wire encoding as transferToHost3D; named distinctly
// for clarity (this one PUSHES texels up, the render-target one PULLS pixels).
func (g *VirtioGPU) transferTextureToHost3D(resID, width, height uint32) error {
	return g.transferToHost3D(resID, width, height)
}

// --- the per-draw virgl command buffer (textured) ----------------------

// buildTexDrawVirglBuffer hand-encodes the entire per-draw virgl command
// buffer for the textured triangle. It is M2's buildDrawVirglBuffer with the
// flat-colour FS replaced by a texturing FS, a 2-element vertex layout, and
// the sampler-view/sampler-state create + set/bind commands inserted before
// SET_FRAMEBUFFER_STATE.
func (g *VirtioGPU) buildTexDrawVirglBuffer(width, height uint32) []byte {
	b := &virglBuilder{}

	// 1. Vertex + fragment shaders (CREATE_OBJECT SHADER + BIND_SHADER).
	//    The texturing VS/FS replace M2's passthrough/flat-colour pair.
	b.emitShader(vsHandle, pipeShaderVertex, vsTexText)
	b.bindShader(vsHandle, pipeShaderVertex)
	b.emitShader(fsHandle, pipeShaderFragment, fsTexText)
	b.bindShader(fsHandle, pipeShaderFragment)

	// 2. Surface over the render target (CREATE_OBJECT SURFACE).
	//    virgl_encoder_create_surface (non-MSAA): VIRGL_OBJ_SURFACE_SIZE=5.
	b.cmd(ccmdCreateObject, objectSurface, 5)
	b.dword(drawSurfaceHandle)        // VIRGL_OBJ_SURFACE_HANDLE
	b.dword(rtResourceID)             // VIRGL_OBJ_SURFACE_RES_HANDLE
	b.dword(virglFormatB8G8R8A8Unorm) // VIRGL_OBJ_SURFACE_FORMAT
	b.dword(0)                        // texture level
	b.dword(0)                        // first_layer | (last_layer<<16) = 0

	// 3. Vertex elements: TWO elements.
	//    virgl_encoder_create_vertex_elements: SIZE(n)=n*4+1.  For n=2 => 9.
	//    d1 handle, then per element: src_offset, instance_divisor,
	//    vertex_buffer_index, src_format.
	//    elem0 = position R32G32B32_FLOAT @ offset 0.
	//    elem1 = texcoord R32G32_FLOAT    @ offset 12.
	b.cmd(ccmdCreateObject, objectVertexElements, 9)
	b.dword(veHandle)                  // VIRGL_OBJ_VERTEX_ELEMENTS_HANDLE
	b.dword(0)                         // elem0 src_offset = 0
	b.dword(0)                         // elem0 instance_divisor
	b.dword(0)                         // elem0 vertex_buffer_index
	b.dword(virglFormatR32G32B32Float) // elem0 src_format (position)
	b.dword(12)                        // elem1 src_offset = 12 (after xyz)
	b.dword(0)                         // elem1 instance_divisor
	b.dword(0)                         // elem1 vertex_buffer_index
	b.dword(virglFormatR32G32Float)    // elem1 src_format (texcoord uv)
	b.bindObject(objectVertexElements, veHandle)

	// 4. Rasterizer (minimal sane state) — identical to M2.
	b.cmd(ccmdCreateObject, objectRasterizer, 9)
	b.dword(rasterHandle)         // handle
	b.dword((1 << 1) | (1 << 29)) // S0: DEPTH_CLIP | HALF_PIXEL_CENTER
	b.float(1.0)                  // point_size
	b.dword(0)                    // sprite_coord_enable
	b.dword(0)                    // S3
	b.float(1.0)                  // line_width
	b.float(0.0)                  // offset_units
	b.float(0.0)                  // offset_scale
	b.float(0.0)                  // offset_clamp
	b.bindObject(objectRasterizer, rasterHandle)

	// 5. Blend (one RT, writemask 0xf, no blending) — identical to M2.
	b.cmd(ccmdCreateObject, objectBlend, 11)
	b.dword(blendHandle) // handle
	b.dword(0)           // S0
	b.dword(0)           // S1
	b.dword(0xf << 27)   // RT0 S2: COLORMASK=0xf
	for i := 0; i < 7; i++ {
		b.dword(0) // RT1..RT7 S2 = 0
	}
	b.bindObject(objectBlend, blendHandle)

	// 6. DSA (depth test off) — identical to M2.
	b.cmd(ccmdCreateObject, objectDSA, 5)
	b.dword(dsaHandle) // handle
	b.dword(0)         // S0
	b.dword(0)         // stencil[0]
	b.dword(0)         // stencil[1]
	b.dword(0)         // alpha_ref_value
	b.bindObject(objectDSA, dsaHandle)

	// 7. SAMPLER_VIEW over the texture (CREATE_OBJECT SAMPLER_VIEW=6).
	//    virgl_encode_sampler_view: VIRGL_OBJ_SAMPLER_VIEW_SIZE=6.
	//    d1 handle, d2 res (write_res = resource id, one dword),
	//    d3 format, then (non-PIPE_BUFFER target):
	//      d4 = first_layer | last_layer<<16   (=0)
	//      d5 = first_level | last_level<<8     (=0)
	//    d6 swizzle (identity RGBA).
	b.cmd(ccmdCreateObject, objectSamplerView, 6)
	b.dword(samplerViewHandle)        // handle
	b.dword(texResourceID)            // write_res -> resource id
	b.dword(virglFormatR8G8B8A8Unorm) // format
	b.dword(0)                        // first_layer | last_layer<<16
	b.dword(0)                        // first_level | last_level<<8
	b.dword(samplerViewSwizzle())     // swizzle (identity = 0x688)

	// 8. SAMPLER_STATE (CREATE_OBJECT SAMPLER_STATE=7).
	//    virgl_encode_sampler_state: VIRGL_OBJ_SAMPLER_STATE_SIZE=9.
	//    d1 handle, d2 S0 (wrap/filter/compare), d3 lod_bias(fui),
	//    d4 min_lod(fui), d5 max_lod(fui), d6..d9 border_color[4] (ui).
	b.cmd(ccmdCreateObject, objectSamplerState, 9)
	b.dword(samplerStateHandle) // handle
	b.dword(samplerStateS0())   // S0: wrap=CLAMP_TO_EDGE, filter=LINEAR, mip=NONE
	b.float(0.0)                // lod_bias
	b.float(0.0)                // min_lod
	b.float(0.0)                // max_lod
	b.dword(0)                  // border_color[0]
	b.dword(0)                  // border_color[1]
	b.dword(0)                  // border_color[2]
	b.dword(0)                  // border_color[3]

	// 9. SET_SAMPLER_VIEWS(FRAGMENT, start_slot=0, [samplerView]).
	//    virgl_encode_set_sampler_views: VIRGL_SET_SAMPLER_VIEWS_SIZE(n)=n+2.
	//    For n=1 => 3. d1 shader_type, d2 start_slot, d3 view handle.
	b.cmd(ccmdSetSamplerViews, 0, 3)
	b.dword(pipeShaderFragment) // shader_type = FRAGMENT
	b.dword(0)                  // start_slot
	b.dword(samplerViewHandle)  // view[0] handle

	// 10. BIND_SAMPLER_STATES(FRAGMENT, start_slot=0, [samplerState]).
	//     virgl_encode_bind_sampler_states: VIRGL_BIND_SAMPLER_STATES(n)=n+2.
	//     For n=1 => 3. d1 shader_type, d2 start_slot, d3 state handle.
	b.cmd(ccmdBindSamplerStates, 0, 3)
	b.dword(pipeShaderFragment) // shader_type = FRAGMENT
	b.dword(0)                  // start_slot
	b.dword(samplerStateHandle) // state[0] handle

	// 11. SET_FRAMEBUFFER_STATE: nr_cbufs=1, zsurf=0, cbuf0=surface.
	b.cmd(ccmdSetFramebufferState, 0, 3)
	b.dword(1)                 // nr_cbufs
	b.dword(0)                 // zsurf handle
	b.dword(drawSurfaceHandle) // cbuf0 handle

	// 12. SET_VIEWPORT_STATE (full surface) — identical to M2.
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

	// 13. SET_VERTEX_BUFFERS (one buffer, stride 20).
	b.cmd(ccmdSetVertexBuffers, 0, 3)
	b.dword(texVertexStride) // stride = 20
	b.dword(0)               // buffer_offset
	b.dword(vbufResourceID)  // resource handle

	// 14. DRAW_VBO — 3 vertices, PIPE_PRIM_TRIANGLES — identical to M2.
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
	b.dword(0xFFFFFFFF)        // max_index = ~0
	b.dword(0)                 // count_from_stream_output = 0

	return b.bytesLE()
}
