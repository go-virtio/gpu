// Tests for the Milestone 2 draw path (DrawTriangle + the per-draw virgl
// command-buffer builder). These reuse the fakeGPUDevice + injectTransport
// harness from gpu_test.go. The fake's default response branch already ACKs
// every new command with OK_NODATA, and its 3-descriptor SUBMIT_3D handler
// already captures the virgl byte stream into d.submitVirgl, so the draw
// command buffer can be asserted dword-for-dword.

package gpu

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/go-virtio/common"
)

// drawCommandCount is the number of control commands DrawTriangle issues:
// DisplayInfo(1) + CTX_CREATE(1) + RT[create+backing+attach](3) +
// VBUF[create+backing+attach+transfer](4) + SUBMIT_3D(1) +
// TRANSFER_TO_HOST_3D(1) + SET_SCANOUT(1) + RESOURCE_FLUSH(1) = 13.
const drawCommandCount = 13

var (
	triVerts = [9]float32{
		0.0, 0.5, 0.0, // top
		-0.5, -0.5, 0.0, // bottom-left
		0.5, -0.5, 0.0, // bottom-right
	}
	triColor = [4]float32{0.25, 0.5, 0.75, 1.0}
)

// --- constant / enum sanity (authoritative-value pinning) -------------

func TestDrawConstants(t *testing.T) {
	cases := []struct {
		name string
		got  uint32
		want uint32
	}{
		{"ccmdCreateObject", ccmdCreateObject, 1},
		{"ccmdBindObject", ccmdBindObject, 2},
		{"ccmdSetViewportState", ccmdSetViewportState, 4},
		{"ccmdSetFramebufferState", ccmdSetFramebufferState, 5},
		{"ccmdSetVertexBuffers", ccmdSetVertexBuffers, 6},
		{"ccmdDrawVBO", ccmdDrawVBO, 8},
		{"ccmdBindShader", ccmdBindShader, 31},
		{"objectBlend", objectBlend, 1},
		{"objectRasterizer", objectRasterizer, 2},
		{"objectDSA", objectDSA, 3},
		{"objectShader", objectShader, 4},
		{"objectVertexElements", objectVertexElements, 5},
		{"objectSurface", objectSurface, 8},
		{"pipeShaderVertex", pipeShaderVertex, 0},
		{"pipeShaderFragment", pipeShaderFragment, 1},
		{"pipePrimTriangles", pipePrimTriangles, 4},
		{"pipeBuffer", pipeBuffer, 0},
		{"virglBindVertexBuffer", virglBindVertexBuffer, 16},
		{"virglFormatR32G32B32Float", virglFormatR32G32B32Float, 30},
		{"vertexStride", vertexStride, 12},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got %d, want %d", c.name, c.got, c.want)
		}
	}
}

// --- ftoa: TGSI-text float literal formatting -------------------------

func TestFtoa(t *testing.T) {
	cases := []struct {
		in   float32
		want string
	}{
		{1.0, "1.0"},   // bare integer widened to ".0"
		{0.0, "0.0"},   // zero widened
		{0.25, "0.25"}, // already has a dot
		{-0.5, "-0.5"}, // negative
		{2.0, "2.0"},   // integer-valued -> widened
	}
	for _, c := range cases {
		if got := ftoa(c.in); got != c.want {
			t.Errorf("ftoa(%v): got %q, want %q", c.in, got, c.want)
		}
	}
	// Exponent form keeps its 'e' (no spurious ".0"): a tiny value.
	if got := ftoa(1e-20); !strings.ContainsAny(got, "eE.") {
		t.Errorf("ftoa(1e-20)=%q: expected a float-shaped literal", got)
	}
}

// --- shader text -------------------------------------------------------

func TestShaderText(t *testing.T) {
	if !strings.HasPrefix(vsText, "VERT\n") {
		t.Errorf("vsText must start with VERT header: %q", vsText)
	}
	for _, frag := range []string{"DCL IN[0]", "DCL OUT[0], POSITION", "MOV OUT[0], IN[0]", "END"} {
		if !strings.Contains(vsText, frag) {
			t.Errorf("vsText missing %q", frag)
		}
	}
	fs := fsTextFor(triColor)
	if !strings.HasPrefix(fs, "FRAG\n") {
		t.Errorf("fsText must start with FRAG header: %q", fs)
	}
	for _, frag := range []string{"DCL OUT[0], COLOR", "IMM[0] FLT32 {", "MOV OUT[0], IMM[0]", "END"} {
		if !strings.Contains(fs, frag) {
			t.Errorf("fsText missing %q", frag)
		}
	}
	// The colour components must appear in the IMM literal.
	if !strings.Contains(fs, "0.25") || !strings.Contains(fs, "0.5") ||
		!strings.Contains(fs, "0.75") || !strings.Contains(fs, "1.0") {
		t.Errorf("fsText IMM missing colour components: %q", fs)
	}
}

// --- DrawTriangle happy path ------------------------------------------

func TestDrawTriangle_Success(t *testing.T) {
	d, g := openGPU3D(t)
	if err := g.DrawTriangle(0, triVerts, triColor); err != nil {
		t.Fatalf("DrawTriangle: %v", err)
	}
	if d.ctrlConsumed != drawCommandCount {
		t.Errorf("ctrlConsumed: got %d, want %d", d.ctrlConsumed, drawCommandCount)
	}
	// The captured virgl buffer must byte-match the builder's output for the
	// resolved scanout dimensions (1024x768 from the fake's DisplayInfo).
	want := g.buildDrawVirglBuffer(1024, 768, triColor)
	if len(d.submitVirgl) != len(want) {
		t.Fatalf("virgl buffer length: got %d, want %d", len(d.submitVirgl), len(want))
	}
	for i := range want {
		if d.submitVirgl[i] != want[i] {
			t.Fatalf("virgl byte %d: got 0x%02x, want 0x%02x", i, d.submitVirgl[i], want[i])
		}
	}
}

func TestDrawTriangle_ScanoutOutOfRange(t *testing.T) {
	_, g := openGPU3D(t)
	if err := g.DrawTriangle(5, triVerts, triColor); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v, want ErrNoScanout", err)
	}
}

func TestDrawTriangle_DisplayInfoNoneEnabled(t *testing.T) {
	// numScanouts=2: scanout 1 is in range but DisplayInfo reports only
	// scanout 0 enabled -> the !Enabled branch.
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 2)
	g, err := OpenVirtioGPU3D(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	if err := g.DrawTriangle(1, triVerts, triColor); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v, want ErrNoScanout", err)
	}
}

func TestDrawTriangle_DisplayInfoFailed(t *testing.T) {
	d, g := openGPU3D(t)
	d.dropDisplayInfo = true
	if err := g.DrawTriangle(0, triVerts, triColor); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v, want ErrNoScanout", err)
	}
}

// --- per-step ErrGPUCommandFailed coverage ----------------------------

func TestDrawTriangle_StepFailures(t *testing.T) {
	// failAfter = N makes the command at 0-based index N (and onward) reply
	// with an error type; DrawTriangle aborts at the first failure, so each N
	// exercises exactly one step's error branch.
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
		{"Submit3D", 9, false},
		{"RTTransfer", 10, false},
		{"SetScanout", 11, false},
		{"ResourceFlush", 12, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, g := openGPU3D(t)
			d.failAfter = c.failAfter
			d.forceError = c.forceError
			if err := g.DrawTriangle(0, triVerts, triColor); !errors.Is(err, ErrGPUCommandFailed) {
				t.Errorf("%s: got %v, want ErrGPUCommandFailed", c.name, err)
			}
		})
	}
}

// --- alloc / zero-phys branches ---------------------------------------

// allocSeq: during DrawTriangle the AllocatePages calls (while armed) are:
//
//	#1 DisplayInfo (sendCommand page)
//	#2 CTX_CREATE (sendCommand page)
//	#3 RT RESOURCE_CREATE_3D (sendCommand page)
//	#4 RT backing (attachBacking)
//	#5 RT ATTACH_BACKING (sendCommand page)
//	#6 RT CTX_ATTACH (sendCommand page)
//	#7 VBUF RESOURCE_CREATE_3D (sendCommand page)
//	#8 VBUF backing page (createVertexBuffer's own AllocatePages)
//	#9 VBUF ATTACH_BACKING (sendCommand page)
//	... etc.
//
// Each sendCommand allocates one page; attachBacking and the vertex-buffer
// backing allocate their own page too.

func TestDrawTriangle_RTBackingAllocFail(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.fp = failPoint{"AllocatePages", 4} // RT backing alloc fails
	if err := g.DrawTriangle(0, triVerts, triColor); err == nil {
		t.Error("expected RT backing alloc error")
	}
}

func TestDrawTriangle_RTBackingZeroPhys(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.zeroPhys = true
	it.zeroPhysAfter = 3 // #1..#3 real; RT backing (#4) returns zero phys
	if err := g.DrawTriangle(0, triVerts, triColor); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v, want ErrAllocReturnedZero", err)
	}
}

func TestDrawTriangle_VBufBackingAllocFail(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	// #1 DisplayInfo, #2 CTX, #3 RT create, #4 RT backing, #5 RT attach,
	// #6 RT ctxattach, #7 VBUF create, #8 VBUF backing alloc fails.
	it.fp = failPoint{"AllocatePages", 8}
	if err := g.DrawTriangle(0, triVerts, triColor); err == nil {
		t.Error("expected vbuf backing alloc error")
	}
}

func TestDrawTriangle_VBufBackingZeroPhys(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.zeroPhys = true
	it.zeroPhysAfter = 7 // #1..#7 real; VBUF backing (#8) zero phys
	if err := g.DrawTriangle(0, triVerts, triColor); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v, want ErrAllocReturnedZero", err)
	}
}

func TestDrawTriangle_Submit3DAllocFail(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	// Allocs through the VBUF transfer: #1 DisplayInfo, #2 CTX, #3 RT create,
	// #4 RT backing, #5 RT attach, #6 RT ctxattach, #7 VBUF create, #8 VBUF
	// backing, #9 VBUF attach, #10 VBUF ctxattach, #11 VBUF transfer, #12
	// SUBMIT_3D page alloc fails.
	it.fp = failPoint{"AllocatePages", 12}
	if err := g.DrawTriangle(0, triVerts, triColor); err == nil {
		t.Error("expected submit3D alloc error")
	}
}

func TestDrawTriangle_Submit3DZeroPhys(t *testing.T) {
	_, g, it := openGPU3DInject(t)
	it.enable = true
	it.zeroPhys = true
	it.zeroPhysAfter = 11 // first 11 allocs real; SUBMIT_3D page (#12) zero
	if err := g.DrawTriangle(0, triVerts, triColor); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v, want ErrAllocReturnedZero", err)
	}
}

func TestDrawTriangle_Submit3DTimeout(t *testing.T) {
	d, g := openGPU3D(t)
	// Let the first 9 commands complete (DisplayInfo, CTX, RT x3, VBUF x4),
	// then stop completing so SUBMIT_3D's poll loop times out.
	d.completesUntil = 9
	if err := g.DrawTriangle(0, triVerts, triColor); !errors.Is(err, ErrRequestTimeout) {
		t.Errorf("got %v, want ErrRequestTimeout", err)
	}
}

// --- builder-level dword assertions -----------------------------------

func TestBuildDrawVirglBuffer_Layout(t *testing.T) {
	_, g := openGPU3D(t)
	buf := g.buildDrawVirglBuffer(1024, 768, triColor)
	if len(buf)%4 != 0 {
		t.Fatalf("virgl buffer not dword-aligned: %d bytes", len(buf))
	}
	dw := make([]uint32, len(buf)/4)
	for i := range dw {
		dw[i] = binary.LittleEndian.Uint32(buf[i*4 : i*4+4])
	}

	// Walk the stream command-by-command, asserting headers and key fields.
	i := 0

	// VS CREATE_OBJECT(SHADER).
	vsRaw := append([]byte(vsText), 0)
	vsTextDw := (uint32(len(vsRaw)) + 3) / 4
	vsLen := 5 + vsTextDw
	if dw[i] != virglCmd0(ccmdCreateObject, objectShader, vsLen) {
		t.Errorf("VS shader header: got 0x%08x", dw[i])
	}
	if dw[i+1] != vsHandle {
		t.Errorf("VS handle: got %d", dw[i+1])
	}
	if dw[i+2] != pipeShaderVertex {
		t.Errorf("VS type: got %d", dw[i+2])
	}
	if dw[i+3] != uint32(len(vsRaw)) {
		t.Errorf("VS offlen: got %d, want %d", dw[i+3], len(vsRaw))
	}
	if dw[i+4] != shaderTokenBudget {
		t.Errorf("VS num_tokens: got %d", dw[i+4])
	}
	if dw[i+5] != 0 {
		t.Errorf("VS streamout num_outputs: got %d", dw[i+5])
	}
	// Decode the VS text payload back and compare.
	gotVS := decodeShaderText(dw[i+6:i+6+int(vsTextDw)], uint32(len(vsRaw)))
	if gotVS != vsText {
		t.Errorf("VS text roundtrip mismatch: got %q", gotVS)
	}
	i += 1 + int(vsLen)

	// BIND_SHADER(VS).
	if dw[i] != virglCmd0(ccmdBindShader, 0, 2) || dw[i+1] != vsHandle || dw[i+2] != pipeShaderVertex {
		t.Errorf("BIND_SHADER VS: 0x%08x %d %d", dw[i], dw[i+1], dw[i+2])
	}
	i += 3

	// FS CREATE_OBJECT(SHADER).
	fs := fsTextFor(triColor)
	fsRaw := append([]byte(fs), 0)
	fsTextDw := (uint32(len(fsRaw)) + 3) / 4
	fsLen := 5 + fsTextDw
	if dw[i] != virglCmd0(ccmdCreateObject, objectShader, fsLen) {
		t.Errorf("FS shader header: got 0x%08x", dw[i])
	}
	if dw[i+2] != pipeShaderFragment {
		t.Errorf("FS type: got %d", dw[i+2])
	}
	if dw[i+3] != uint32(len(fsRaw)) {
		t.Errorf("FS offlen: got %d, want %d", dw[i+3], len(fsRaw))
	}
	gotFS := decodeShaderText(dw[i+6:i+6+int(fsTextDw)], uint32(len(fsRaw)))
	if gotFS != fs {
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

	// CREATE VERTEX_ELEMENTS.
	if dw[i] != virglCmd0(ccmdCreateObject, objectVertexElements, 5) {
		t.Errorf("VERTEX_ELEMENTS header: got 0x%08x", dw[i])
	}
	if dw[i+1] != veHandle || dw[i+2] != 0 || dw[i+3] != 0 || dw[i+4] != 0 ||
		dw[i+5] != virglFormatR32G32B32Float {
		t.Errorf("VERTEX_ELEMENTS fields: %v", dw[i+1:i+6])
	}
	i += 6
	// BIND_OBJECT(VERTEX_ELEMENTS).
	if dw[i] != virglCmd0(ccmdBindObject, objectVertexElements, 1) || dw[i+1] != veHandle {
		t.Errorf("BIND VERTEX_ELEMENTS: 0x%08x %d", dw[i], dw[i+1])
	}
	i += 2

	// RASTERIZER.
	if dw[i] != virglCmd0(ccmdCreateObject, objectRasterizer, 9) {
		t.Errorf("RASTERIZER header: got 0x%08x", dw[i])
	}
	if dw[i+1] != rasterHandle {
		t.Errorf("RASTERIZER handle: got %d", dw[i+1])
	}
	if dw[i+2] != (1<<1)|(1<<29) {
		t.Errorf("RASTERIZER S0: got 0x%08x", dw[i+2])
	}
	if dw[i+3] != math.Float32bits(1.0) {
		t.Errorf("RASTERIZER point_size: got 0x%08x", dw[i+3])
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
	if dw[i+1] != blendHandle || dw[i+2] != 0 || dw[i+3] != 0 {
		t.Errorf("BLEND handle/S0/S1: %v", dw[i+1:i+4])
	}
	if dw[i+4] != (0xf << 27) {
		t.Errorf("BLEND RT0 colormask: got 0x%08x", dw[i+4])
	}
	for k := 5; k < 12; k++ {
		if dw[i+k] != 0 {
			t.Errorf("BLEND RT%d not zero: 0x%08x", k-4, dw[i+k])
		}
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
	if dw[i+1] != 0 {
		t.Errorf("SET_VIEWPORT start_slot: got %d", dw[i+1])
	}
	if dw[i+2] != math.Float32bits(512) || dw[i+3] != math.Float32bits(384) ||
		dw[i+4] != math.Float32bits(0.5) {
		t.Errorf("SET_VIEWPORT scale: %v", dw[i+2:i+5])
	}
	if dw[i+5] != math.Float32bits(512) || dw[i+6] != math.Float32bits(384) ||
		dw[i+7] != math.Float32bits(0.5) {
		t.Errorf("SET_VIEWPORT translate: %v", dw[i+5:i+8])
	}
	i += 8

	// SET_VERTEX_BUFFERS.
	if dw[i] != virglCmd0(ccmdSetVertexBuffers, 0, 3) {
		t.Errorf("SET_VERTEX_BUFFERS header: got 0x%08x", dw[i])
	}
	if dw[i+1] != vertexStride || dw[i+2] != 0 || dw[i+3] != vbufResourceID {
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

// decodeShaderText reconstructs the NUL-terminated text from textDwords
// dwords and trims at the offlen-1 NUL boundary.
func decodeShaderText(dwords []uint32, offlen uint32) string {
	raw := make([]byte, len(dwords)*4)
	for i, v := range dwords {
		binary.LittleEndian.PutUint32(raw[i*4:i*4+4], v)
	}
	// offlen includes the trailing NUL; the text is offlen-1 bytes.
	return string(raw[:offlen-1])
}

// --- inject-harness helper for the draw path --------------------------

func openGPU3DInject(t *testing.T) (*fakeGPUDevice, *VirtioGPU, *injectTransport) {
	t.Helper()
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	it := newInject(d, false)
	g, err := OpenVirtioGPU3D(it)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	return d, g, it
}
