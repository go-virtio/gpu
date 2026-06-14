// Tests for the Milestone 3 textured-draw path (DrawTexturedTriangle + the
// per-draw virgl command-buffer builder buildTexDrawVirglBuffer). These reuse
// the fakeGPUDevice + injectTransport harness from gpu_test.go. The fake's
// default response branch already ACKs every M3 command (RESOURCE_CREATE_3D,
// ATTACH_BACKING, CTX_ATTACH_RESOURCE, TRANSFER_TO_HOST_3D, etc.) with
// OK_NODATA, and its 3-descriptor SUBMIT_3D handler captures the (now larger)
// virgl byte stream into d.submitVirgl, so the textured command buffer can be
// asserted dword-for-dword — no fake changes are needed beyond what M2 added.

package gpu

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/go-virtio/common"
)

// texDrawCommandCount is the number of control commands DrawTexturedTriangle
// issues: DisplayInfo(1) + CTX_CREATE(1) + RT[create+backing+attach](3) +
// VBUF[create+backing+attach+transfer](4) + TEX[create+backing+attach+transfer]
// (4) + SUBMIT_3D(1) + TRANSFER_TO_HOST_3D(1) + SET_SCANOUT(1) +
// RESOURCE_FLUSH(1) = 17.
const texDrawCommandCount = 17

var (
	// 3 vertices, each x,y,z,u,v.
	texTriVerts = [15]float32{
		0.0, 0.5, 0.0, 0.5, 1.0, // top, uv (0.5,1)
		-0.5, -0.5, 0.0, 0.0, 0.0, // bottom-left, uv (0,0)
		0.5, -0.5, 0.0, 1.0, 0.0, // bottom-right, uv (1,0)
	}
	// A tiny 2x2 RGBA8 texture (4 texels * 4 bytes).
	texW2, texH2 = uint32(2), uint32(2)
	texData2     = []byte{
		0xFF, 0x00, 0x00, 0xFF, // red
		0x00, 0xFF, 0x00, 0xFF, // green
		0x00, 0x00, 0xFF, 0xFF, // blue
		0xFF, 0xFF, 0xFF, 0xFF, // white
	}
)

// --- constant / enum sanity (authoritative-value pinning) -------------

func TestTexConstants(t *testing.T) {
	cases := []struct {
		name string
		got  uint32
		want uint32
	}{
		{"ccmdSetSamplerViews", ccmdSetSamplerViews, 10},
		{"ccmdBindSamplerStates", ccmdBindSamplerStates, 18},
		{"objectSamplerView", objectSamplerView, 6},
		{"objectSamplerState", objectSamplerState, 7},
		{"virglFormatR32G32Float", virglFormatR32G32Float, 29},
		{"virglFormatR8G8B8A8Unorm", virglFormatR8G8B8A8Unorm, 67},
		{"virglBindSamplerView", virglBindSamplerView, 8},
		{"texVertexStride", texVertexStride, 20},
		{"pipeTexWrapClampToEdge", pipeTexWrapClampToEdge, 2},
		{"pipeTexFilterLinear", pipeTexFilterLinear, 1},
		{"pipeTexMipFilterNone", pipeTexMipFilterNone, 2},
		{"pipeSwizzleRed", pipeSwizzleRed, 0},
		{"pipeSwizzleGreen", pipeSwizzleGreen, 1},
		{"pipeSwizzleBlue", pipeSwizzleBlue, 2},
		{"pipeSwizzleAlpha", pipeSwizzleAlpha, 3},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got %d, want %d", c.name, c.got, c.want)
		}
	}
}

// samplerStateS0 must pack to the documented bit layout:
// WRAP_S/T/R=CLAMP_TO_EDGE(2), MIN_IMG=LINEAR(1), MIN_MIP=NONE(2),
// MAG_IMG=LINEAR(1), compare 0.
func TestSamplerStateS0(t *testing.T) {
	want := uint32((2 << 0) | (2 << 3) | (2 << 6) | (1 << 9) | (2 << 11) | (1 << 13))
	if got := samplerStateS0(); got != want {
		t.Errorf("samplerStateS0: got 0x%08x, want 0x%08x", got, want)
	}
	// Spell the expected literal out independently as a cross-check.
	if want != 0x3292 {
		t.Errorf("S0 literal cross-check: 0x%08x != 0x3292", want)
	}
}

// samplerViewSwizzle must be the identity RGBA swizzle = 0x688.
func TestSamplerViewSwizzle(t *testing.T) {
	want := uint32((0 << 0) | (1 << 3) | (2 << 6) | (3 << 9))
	if got := samplerViewSwizzle(); got != want {
		t.Errorf("samplerViewSwizzle: got 0x%08x, want 0x%08x", got, want)
	}
	if want != 0x688 {
		t.Errorf("swizzle literal cross-check: 0x%08x != 0x688", want)
	}
}

// --- shader text -------------------------------------------------------

func TestTexShaderText(t *testing.T) {
	if !strings.HasPrefix(vsTexText, "VERT\n") {
		t.Errorf("vsTexText must start with VERT header: %q", vsTexText)
	}
	for _, frag := range []string{
		"DCL IN[0]", "DCL IN[1]", "DCL OUT[0], POSITION", "DCL OUT[1], GENERIC[0]",
		"MOV OUT[0], IN[0]", "MOV OUT[1], IN[1]", "END",
	} {
		if !strings.Contains(vsTexText, frag) {
			t.Errorf("vsTexText missing %q", frag)
		}
	}
	if !strings.HasPrefix(fsTexText, "FRAG\n") {
		t.Errorf("fsTexText must start with FRAG header: %q", fsTexText)
	}
	for _, frag := range []string{
		"DCL IN[0], GENERIC[0]", "DCL OUT[0], COLOR", "DCL SAMP[0]",
		"DCL SVIEW[0], 2D, FLOAT", "TEX OUT[0], IN[0], SAMP[0], 2D", "END",
	} {
		if !strings.Contains(fsTexText, frag) {
			t.Errorf("fsTexText missing %q", frag)
		}
	}
}

// --- DrawTexturedTriangle happy path ----------------------------------

func TestDrawTexturedTriangle_Success(t *testing.T) {
	d, g := openGPU3D(t)
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); err != nil {
		t.Fatalf("DrawTexturedTriangle: %v", err)
	}
	if d.ctrlConsumed != texDrawCommandCount {
		t.Errorf("ctrlConsumed: got %d, want %d", d.ctrlConsumed, texDrawCommandCount)
	}
	// The captured virgl buffer must byte-match the builder's output for the
	// resolved scanout dimensions (1024x768 from the fake's DisplayInfo).
	want := g.buildTexDrawVirglBuffer(1024, 768)
	if len(d.submitVirgl) != len(want) {
		t.Fatalf("virgl buffer length: got %d, want %d", len(d.submitVirgl), len(want))
	}
	for i := range want {
		if d.submitVirgl[i] != want[i] {
			t.Fatalf("virgl byte %d: got 0x%02x, want 0x%02x", i, d.submitVirgl[i], want[i])
		}
	}
}

func TestDrawTexturedTriangle_ScanoutOutOfRange(t *testing.T) {
	_, g := openGPU3D(t)
	if err := g.DrawTexturedTriangle(5, texTriVerts, texData2, texW2, texH2); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v, want ErrNoScanout", err)
	}
}

func TestDrawTexturedTriangle_DisplayInfoNoneEnabled(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 2)
	g, err := OpenVirtioGPU3D(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	if err := g.DrawTexturedTriangle(1, texTriVerts, texData2, texW2, texH2); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v, want ErrNoScanout", err)
	}
}

func TestDrawTexturedTriangle_DisplayInfoFailed(t *testing.T) {
	d, g := openGPU3D(t)
	d.dropDisplayInfo = true
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v, want ErrNoScanout", err)
	}
}

// --- per-step ErrGPUCommandFailed coverage ----------------------------

func TestDrawTexturedTriangle_StepFailures(t *testing.T) {
	// failAfter = N makes the command at 0-based index N (and onward) reply
	// with an error type; DrawTexturedTriangle aborts at the first failure, so
	// each N exercises exactly one step's error branch.
	cases := []struct {
		name       string
		failAfter  int
		forceError bool
	}{
		{"DisplayInfo", 0, true}, // first command; force-error all
		{"CtxCreate", 1, false},
		{"RTCreate", 2, false},
		{"RTBacking", 3, false},
		{"RTCtxAttach", 4, false},
		{"VBufCreate", 5, false},
		{"VBufBacking", 6, false},
		{"VBufCtxAttach", 7, false},
		{"VBufTransfer", 8, false},
		{"TexCreate", 9, false},
		{"TexBacking", 10, false},
		{"TexCtxAttach", 11, false},
		{"TexTransfer", 12, false},
		{"Submit3D", 13, false},
		{"RTTransfer", 14, false},
		{"SetScanout", 15, false},
		{"ResourceFlush", 16, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, g := openGPU3D(t)
			d.failAfter = c.failAfter
			d.forceError = c.forceError
			if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); !errors.Is(err, ErrGPUCommandFailed) {
				t.Errorf("%s: got %v, want ErrGPUCommandFailed", c.name, err)
			}
		})
	}
}

// --- alloc / zero-phys branches ---------------------------------------
//
// After the R-doom1c sendCommand-page cache fix, the only AllocatePages
// calls during DrawTexturedTriangle (while armed) are:
//
//	#1 ensureCmdPage (lazy alloc on the FIRST sendCommand — DisplayInfo)
//	#2 RT backing page (attachBacking's own AllocatePages in gpu3d_draw.go)
//	#3 VBUF backing page (createTexVertexBuffer's own AllocatePages)
//	#4 TEX backing page (createTexture's own allocPagesFor)
//	#5 SUBMIT_3D page (submit3D's own AllocatePages — not a sendCommand)
//
// Every other DrawTexturedTriangle step issues its command via sendCommand,
// which reuses the cached command page and allocates nothing.

func TestDrawTexturedTriangle_RTBackingAllocFail(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.fp = failPoint{"AllocatePages", 2} // #1 ensureCmdPage, #2 RT backing fails
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); err == nil {
		t.Error("expected RT backing alloc error")
	}
}

func TestDrawTexturedTriangle_RTBackingZeroPhys(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.zeroPhys = true
	it.zeroPhysAfter = 1 // #1 ensureCmdPage real; RT backing (#2) returns zero phys
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v, want ErrAllocReturnedZero", err)
	}
}

func TestDrawTexturedTriangle_VBufBackingAllocFail(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.fp = failPoint{"AllocatePages", 3} // #1 ensureCmdPage, #2 RT backing, #3 VBUF backing fails
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); err == nil {
		t.Error("expected vbuf backing alloc error")
	}
}

func TestDrawTexturedTriangle_VBufBackingZeroPhys(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.zeroPhys = true
	it.zeroPhysAfter = 2 // #1..#2 real; VBUF backing (#3) zero phys
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v, want ErrAllocReturnedZero", err)
	}
}

func TestDrawTexturedTriangle_TexBackingAllocFail(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.fp = failPoint{"AllocatePages", 4} // #1..#3 real; TEX backing (#4) fails
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); err == nil {
		t.Error("expected tex backing alloc error")
	}
}

func TestDrawTexturedTriangle_TexBackingZeroPhys(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.zeroPhys = true
	it.zeroPhysAfter = 3 // #1..#3 real; TEX backing (#4) zero phys
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v, want ErrAllocReturnedZero", err)
	}
}

func TestDrawTexturedTriangle_Submit3DAllocFail(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.fp = failPoint{"AllocatePages", 5} // SUBMIT_3D page (#5) alloc fails
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); err == nil {
		t.Error("expected submit3D alloc error")
	}
}

func TestDrawTexturedTriangle_Submit3DTimeout(t *testing.T) {
	d, g := openGPU3D(t)
	// Let the first 13 commands complete (DisplayInfo, CTX, RT x3, VBUF x4,
	// TEX x4 = 1+1+3+4+4 = 13), then stop so SUBMIT_3D's poll loop times out.
	d.completesUntil = 13
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); !errors.Is(err, ErrRequestTimeout) {
		t.Errorf("got %v, want ErrRequestTimeout", err)
	}
}

// --- builder-level dword assertions -----------------------------------

func TestBuildTexDrawVirglBuffer_Layout(t *testing.T) {
	_, g := openGPU3D(t)
	buf := g.buildTexDrawVirglBuffer(1024, 768)
	if len(buf)%4 != 0 {
		t.Fatalf("virgl buffer not dword-aligned: %d bytes", len(buf))
	}
	dw := make([]uint32, len(buf)/4)
	for i := range dw {
		dw[i] = binary.LittleEndian.Uint32(buf[i*4 : i*4+4])
	}

	i := 0

	// VS CREATE_OBJECT(SHADER) — texturing VS.
	vsRaw := append([]byte(vsTexText), 0)
	vsTextDw := (uint32(len(vsRaw)) + 3) / 4
	vsLen := 5 + vsTextDw
	if dw[i] != virglCmd0(ccmdCreateObject, objectShader, vsLen) {
		t.Errorf("VS shader header: got 0x%08x", dw[i])
	}
	if dw[i+1] != vsHandle || dw[i+2] != pipeShaderVertex || dw[i+3] != uint32(len(vsRaw)) {
		t.Errorf("VS hdr fields: %v", dw[i+1:i+4])
	}
	if gotVS := decodeShaderText(dw[i+6:i+6+int(vsTextDw)], uint32(len(vsRaw))); gotVS != vsTexText {
		t.Errorf("VS text roundtrip mismatch: got %q", gotVS)
	}
	i += 1 + int(vsLen)

	// BIND_SHADER(VS).
	if dw[i] != virglCmd0(ccmdBindShader, 0, 2) || dw[i+1] != vsHandle || dw[i+2] != pipeShaderVertex {
		t.Errorf("BIND_SHADER VS: 0x%08x %d %d", dw[i], dw[i+1], dw[i+2])
	}
	i += 3

	// FS CREATE_OBJECT(SHADER) — texturing FS.
	fsRaw := append([]byte(fsTexText), 0)
	fsTextDw := (uint32(len(fsRaw)) + 3) / 4
	fsLen := 5 + fsTextDw
	if dw[i] != virglCmd0(ccmdCreateObject, objectShader, fsLen) {
		t.Errorf("FS shader header: got 0x%08x", dw[i])
	}
	if dw[i+2] != pipeShaderFragment || dw[i+3] != uint32(len(fsRaw)) {
		t.Errorf("FS hdr fields: %v", dw[i+2:i+4])
	}
	if gotFS := decodeShaderText(dw[i+6:i+6+int(fsTextDw)], uint32(len(fsRaw))); gotFS != fsTexText {
		t.Errorf("FS text roundtrip mismatch: got %q", gotFS)
	}
	i += 1 + int(fsLen)

	// BIND_SHADER(FS).
	if dw[i] != virglCmd0(ccmdBindShader, 0, 2) || dw[i+1] != fsHandle || dw[i+2] != pipeShaderFragment {
		t.Errorf("BIND_SHADER FS: 0x%08x %d %d", dw[i], dw[i+1], dw[i+2])
	}
	i += 3

	// CREATE SURFACE.
	if dw[i] != virglCmd0(ccmdCreateObject, objectSurface, 5) {
		t.Errorf("SURFACE header: got 0x%08x", dw[i])
	}
	if dw[i+1] != drawSurfaceHandle || dw[i+2] != rtResourceID ||
		dw[i+3] != virglFormatB8G8R8A8Unorm || dw[i+4] != 0 || dw[i+5] != 0 {
		t.Errorf("SURFACE fields: %v", dw[i+1:i+6])
	}
	i += 6

	// CREATE VERTEX_ELEMENTS — TWO elements (SIZE 9).
	if dw[i] != virglCmd0(ccmdCreateObject, objectVertexElements, 9) {
		t.Errorf("VERTEX_ELEMENTS header: got 0x%08x", dw[i])
	}
	// elem0: offset 0, div 0, vb 0, R32G32B32_FLOAT.
	if dw[i+1] != veHandle || dw[i+2] != 0 || dw[i+3] != 0 || dw[i+4] != 0 ||
		dw[i+5] != virglFormatR32G32B32Float {
		t.Errorf("VERTEX_ELEMENTS elem0: %v", dw[i+1:i+6])
	}
	// elem1: offset 12, div 0, vb 0, R32G32_FLOAT.
	if dw[i+6] != 12 || dw[i+7] != 0 || dw[i+8] != 0 || dw[i+9] != virglFormatR32G32Float {
		t.Errorf("VERTEX_ELEMENTS elem1: %v", dw[i+6:i+10])
	}
	i += 10
	if dw[i] != virglCmd0(ccmdBindObject, objectVertexElements, 1) || dw[i+1] != veHandle {
		t.Errorf("BIND VERTEX_ELEMENTS: 0x%08x %d", dw[i], dw[i+1])
	}
	i += 2

	// RASTERIZER.
	if dw[i] != virglCmd0(ccmdCreateObject, objectRasterizer, 9) {
		t.Errorf("RASTERIZER header: got 0x%08x", dw[i])
	}
	if dw[i+1] != rasterHandle || dw[i+2] != (1<<1)|(1<<29) || dw[i+3] != math.Float32bits(1.0) {
		t.Errorf("RASTERIZER fields: %v", dw[i+1:i+4])
	}
	i += 10
	if dw[i] != virglCmd0(ccmdBindObject, objectRasterizer, 1) || dw[i+1] != rasterHandle {
		t.Errorf("BIND RASTERIZER: 0x%08x %d", dw[i], dw[i+1])
	}
	i += 2

	// BLEND.
	if dw[i] != virglCmd0(ccmdCreateObject, objectBlend, 11) {
		t.Errorf("BLEND header: got 0x%08x", dw[i])
	}
	if dw[i+1] != blendHandle || dw[i+2] != 0 || dw[i+3] != 0 || dw[i+4] != (0xf<<27) {
		t.Errorf("BLEND fields: %v", dw[i+1:i+5])
	}
	i += 12
	if dw[i] != virglCmd0(ccmdBindObject, objectBlend, 1) || dw[i+1] != blendHandle {
		t.Errorf("BIND BLEND: 0x%08x %d", dw[i], dw[i+1])
	}
	i += 2

	// DSA.
	if dw[i] != virglCmd0(ccmdCreateObject, objectDSA, 5) {
		t.Errorf("DSA header: got 0x%08x", dw[i])
	}
	if dw[i+1] != dsaHandle || dw[i+2] != 0 || dw[i+3] != 0 || dw[i+4] != 0 || dw[i+5] != 0 {
		t.Errorf("DSA fields: %v", dw[i+1:i+6])
	}
	i += 6
	if dw[i] != virglCmd0(ccmdBindObject, objectDSA, 1) || dw[i+1] != dsaHandle {
		t.Errorf("BIND DSA: 0x%08x %d", dw[i], dw[i+1])
	}
	i += 2

	// CREATE SAMPLER_VIEW (SIZE 6).
	if dw[i] != virglCmd0(ccmdCreateObject, objectSamplerView, 6) {
		t.Errorf("SAMPLER_VIEW header: got 0x%08x", dw[i])
	}
	if dw[i+1] != samplerViewHandle || dw[i+2] != texResourceID ||
		dw[i+3] != virglFormatR8G8B8A8Unorm || dw[i+4] != 0 || dw[i+5] != 0 ||
		dw[i+6] != samplerViewSwizzle() {
		t.Errorf("SAMPLER_VIEW fields: %v", dw[i+1:i+7])
	}
	i += 7

	// CREATE SAMPLER_STATE (SIZE 9).
	if dw[i] != virglCmd0(ccmdCreateObject, objectSamplerState, 9) {
		t.Errorf("SAMPLER_STATE header: got 0x%08x", dw[i])
	}
	if dw[i+1] != samplerStateHandle || dw[i+2] != samplerStateS0() {
		t.Errorf("SAMPLER_STATE handle/S0: %d 0x%08x", dw[i+1], dw[i+2])
	}
	// lod_bias, min_lod, max_lod (fui(0)=0), border_color[0..3]=0.
	for k := 3; k <= 9; k++ {
		if dw[i+k] != 0 {
			t.Errorf("SAMPLER_STATE dword %d not zero: 0x%08x", k, dw[i+k])
		}
	}
	i += 10

	// SET_SAMPLER_VIEWS (FRAGMENT, slot 0, [view]).
	if dw[i] != virglCmd0(ccmdSetSamplerViews, 0, 3) {
		t.Errorf("SET_SAMPLER_VIEWS header: got 0x%08x", dw[i])
	}
	if dw[i+1] != pipeShaderFragment || dw[i+2] != 0 || dw[i+3] != samplerViewHandle {
		t.Errorf("SET_SAMPLER_VIEWS fields: %v", dw[i+1:i+4])
	}
	i += 4

	// BIND_SAMPLER_STATES (FRAGMENT, slot 0, [state]).
	if dw[i] != virglCmd0(ccmdBindSamplerStates, 0, 3) {
		t.Errorf("BIND_SAMPLER_STATES header: got 0x%08x", dw[i])
	}
	if dw[i+1] != pipeShaderFragment || dw[i+2] != 0 || dw[i+3] != samplerStateHandle {
		t.Errorf("BIND_SAMPLER_STATES fields: %v", dw[i+1:i+4])
	}
	i += 4

	// SET_FRAMEBUFFER_STATE.
	if dw[i] != virglCmd0(ccmdSetFramebufferState, 0, 3) {
		t.Errorf("SET_FRAMEBUFFER header: got 0x%08x", dw[i])
	}
	if dw[i+1] != 1 || dw[i+2] != 0 || dw[i+3] != drawSurfaceHandle {
		t.Errorf("SET_FRAMEBUFFER fields: %v", dw[i+1:i+4])
	}
	i += 4

	// SET_VIEWPORT_STATE.
	if dw[i] != virglCmd0(ccmdSetViewportState, 0, 7) {
		t.Errorf("SET_VIEWPORT header: got 0x%08x", dw[i])
	}
	if dw[i+1] != 0 ||
		dw[i+2] != math.Float32bits(512) || dw[i+3] != math.Float32bits(384) || dw[i+4] != math.Float32bits(0.5) ||
		dw[i+5] != math.Float32bits(512) || dw[i+6] != math.Float32bits(384) || dw[i+7] != math.Float32bits(0.5) {
		t.Errorf("SET_VIEWPORT fields: %v", dw[i+1:i+8])
	}
	i += 8

	// SET_VERTEX_BUFFERS — stride 20.
	if dw[i] != virglCmd0(ccmdSetVertexBuffers, 0, 3) {
		t.Errorf("SET_VERTEX_BUFFERS header: got 0x%08x", dw[i])
	}
	if dw[i+1] != texVertexStride || dw[i+2] != 0 || dw[i+3] != vbufResourceID {
		t.Errorf("SET_VERTEX_BUFFERS fields: %v", dw[i+1:i+4])
	}
	i += 4

	// DRAW_VBO.
	if dw[i] != virglCmd0(ccmdDrawVBO, 0, 12) {
		t.Errorf("DRAW_VBO header: got 0x%08x", dw[i])
	}
	wantDraw := []uint32{0, 3, pipePrimTriangles, 0, 1, 0, 0, 0, 0, 0, 0xFFFFFFFF, 0}
	for k, w := range wantDraw {
		if dw[i+1+k] != w {
			t.Errorf("DRAW_VBO dword %d: got 0x%08x, want 0x%08x", k+1, dw[i+1+k], w)
		}
	}
	i += 13

	if i != len(dw) {
		t.Errorf("stream length mismatch: consumed %d of %d dwords", i, len(dw))
	}
}

// --- texture upload: the texels reach the texture backing -------------

// TestDrawTexturedTriangle_TexelUpload checks createTexture copies the caller's
// RGBA8 bytes into the texture's guest backing verbatim. The fake records each
// ATTACH_BACKING's mem_entry; here we instead verify via the inject harness
// that the texture backing page (alloc #4 after the R-doom1c sendCommand-page
// cache fix) holds the texels after the call.
func TestDrawTexturedTriangle_TexelUpload(t *testing.T) {
	d, g, it := openGPU3DInject(t)
	it.enable = true
	// Capture the 4th armed AllocatePages slice (the texture backing):
	// #1 ensureCmdPage, #2 RT backing, #3 VBUF backing, #4 TEX backing.
	var texBacking []byte
	it.onAlloc = func(n int, mem []byte) {
		if n == 4 {
			texBacking = mem
		}
	}
	if err := g.DrawTexturedTriangle(0, texTriVerts, texData2, texW2, texH2); err != nil {
		t.Fatalf("DrawTexturedTriangle: %v", err)
	}
	if texBacking == nil {
		t.Fatal("texture backing not captured")
	}
	for i := range texData2 {
		if texBacking[i] != texData2[i] {
			t.Fatalf("texel byte %d: got 0x%02x, want 0x%02x", i, texBacking[i], texData2[i])
		}
	}
	_ = d
}
