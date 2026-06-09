// Tests for the 3D / virgl extension: OpenVirtioGPU3D bring-up and the
// ClearScreen M1 sequence. These reuse the fakeGPUDevice + injectTransport
// harness from gpu_test.go; the fake's generic chain-walk already handles
// the 3-descriptor SUBMIT_3D chain, and its default response branch returns
// OK_NODATA for the new 3D command codes.

package gpu

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/go-virtio/common"
)

// --- 3D command-code + constant sanity --------------------------------

func TestVirgl3DConstants(t *testing.T) {
	cases := []struct {
		name string
		got  uint32
		want uint32
	}{
		{"CmdCtxCreate", CmdCtxCreate, 0x0200},
		{"CmdCtxDestroy", CmdCtxDestroy, 0x0201},
		{"CmdCtxAttachResource", CmdCtxAttachResource, 0x0202},
		{"CmdCtxDetachResource", CmdCtxDetachResource, 0x0203},
		{"CmdResourceCreate3D", CmdResourceCreate3D, 0x0204},
		{"CmdTransferToHost3D", CmdTransferToHost3D, 0x0205},
		{"CmdTransferFromHost3D", CmdTransferFromHost3D, 0x0206},
		{"CmdSubmit3D", CmdSubmit3D, 0x0207},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got 0x%x, want 0x%x", c.name, c.got, c.want)
		}
	}
	if FeatureVirgl != 1 {
		t.Errorf("FeatureVirgl: got 0x%x, want 1", FeatureVirgl)
	}
	if AcceptedFeatures3D != common.FeatureVersion1|1 {
		t.Errorf("AcceptedFeatures3D: got 0x%x", AcceptedFeatures3D)
	}
}

func TestVirglCmd0(t *testing.T) {
	if got := virglCmd0(virglCmdCreateObject, virglObjectSurface, 5); got != 0x00050801 {
		t.Errorf("CREATE_OBJECT(SURFACE): got 0x%08x, want 0x00050801", got)
	}
	if got := virglCmd0(virglCmdSetFramebufferState, 0, 3); got != 0x00030005 {
		t.Errorf("SET_FRAMEBUFFER_STATE: got 0x%08x, want 0x00030005", got)
	}
	if got := virglCmd0(virglCmdClear, 0, 8); got != 0x00080007 {
		t.Errorf("CLEAR: got 0x%08x, want 0x00080007", got)
	}
}

// --- OpenVirtioGPU3D bring-up ------------------------------------------

func TestOpenVirtioGPU3D_Success(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	g, err := OpenVirtioGPU3D(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	if g.NegotiatedFeatures&FeatureVirgl == 0 {
		t.Errorf("VIRGL not negotiated: 0x%x", g.NegotiatedFeatures)
	}
	if g.NegotiatedFeatures&common.FeatureVersion1 == 0 {
		t.Errorf("VERSION_1 not negotiated: 0x%x", g.NegotiatedFeatures)
	}
}

func TestOpenVirtioGPU3D_VirglUnavailable_FeaturesNotLatched(t *testing.T) {
	// Host offers VIRGL but rejects it at FEATURES_OK time.
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	d.rejectVirgl = true
	if _, err := OpenVirtioGPU3D(d); !errors.Is(err, ErrVirglUnavailable) {
		t.Errorf("got %v, want ErrVirglUnavailable", err)
	}
}

func TestOpenVirtioGPU3D_VirglUnavailable_NotOffered(t *testing.T) {
	// Host does NOT offer VIRGL: bring-up succeeds (VERSION_1 only) but
	// the negotiated mask lacks VIRGL -> ErrVirglUnavailable.
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	if _, err := OpenVirtioGPU3D(d); !errors.Is(err, ErrVirglUnavailable) {
		t.Errorf("got %v, want ErrVirglUnavailable", err)
	}
}

func TestOpenVirtioGPU3D_BringUpError(t *testing.T) {
	// A transport error during bring-up propagates unchanged.
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	le.PutUint16(d.cfg[2:], common.PCIDeviceIDModernNet) // wrong device id
	if _, err := OpenVirtioGPU3D(d); !errors.Is(err, ErrInitWrongDeviceID) {
		t.Errorf("got %v, want ErrInitWrongDeviceID", err)
	}
}

// --- ClearScreen happy path + virgl byte assertion --------------------

func openGPU3D(t *testing.T) (*fakeGPUDevice, *VirtioGPU) {
	t.Helper()
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	g, err := OpenVirtioGPU3D(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	return d, g
}

func TestClearScreen_Success(t *testing.T) {
	d, g := openGPU3D(t)
	if err := g.ClearScreen(0, 0.25, 0.5, 0.75, 1.0); err != nil {
		t.Fatalf("ClearScreen: %v", err)
	}
	// 9 commands: DisplayInfo, CTX_CREATE, RESOURCE_CREATE_3D,
	// ATTACH_BACKING, CTX_ATTACH_RESOURCE, SUBMIT_3D, TRANSFER_TO_HOST_3D,
	// SET_SCANOUT, RESOURCE_FLUSH.
	if d.ctrlConsumed != 9 {
		t.Errorf("ctrlConsumed: got %d, want 9", d.ctrlConsumed)
	}

	// Assert the captured virgl command buffer byte-for-byte.
	want := buildClearVirglBuffer(0.25, 0.5, 0.75, 1.0)
	if len(d.submitVirgl) != 76 {
		t.Fatalf("virgl buffer length: got %d, want 76", len(d.submitVirgl))
	}
	for i := range want {
		if d.submitVirgl[i] != want[i] {
			t.Fatalf("virgl buffer byte %d: got 0x%02x, want 0x%02x", i, d.submitVirgl[i], want[i])
		}
	}

	// Decode the captured dwords and check each field.
	dw := make([]uint32, 19)
	for i := range dw {
		dw[i] = binary.LittleEndian.Uint32(d.submitVirgl[i*4 : i*4+4])
	}
	expect := []uint32{
		0x00050801, 1, 1, 1, 0, 0, // CREATE_OBJECT(SURFACE)
		0x00030005, 1, 0, 1, // SET_FRAMEBUFFER_STATE
		0x00080007, 4, // CLEAR header + buffers
		math.Float32bits(0.25), math.Float32bits(0.5),
		math.Float32bits(0.75), math.Float32bits(1.0),
		0, 0, 0, // depth low, depth high, stencil
	}
	for i := range expect {
		if dw[i] != expect[i] {
			t.Errorf("virgl dword %d: got 0x%08x, want 0x%08x", i, dw[i], expect[i])
		}
	}
}

func TestClearScreen_ScanoutOutOfRange(t *testing.T) {
	_, g := openGPU3D(t)
	if err := g.ClearScreen(5, 0, 0, 0, 1); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v, want ErrNoScanout", err)
	}
}

func TestClearScreen_DisplayInfoNoneEnabled(t *testing.T) {
	// numScanouts=2 so scanout 1 is in range, but DisplayInfo reports only
	// scanout 0 enabled -> scanout 1 hits the !Enabled branch.
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 2)
	g, err := OpenVirtioGPU3D(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	if err := g.ClearScreen(1, 0, 0, 0, 1); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v, want ErrNoScanout", err)
	}
}

func TestClearScreen_DisplayInfoFailed(t *testing.T) {
	d, g := openGPU3D(t)
	d.dropDisplayInfo = true // GET_DISPLAY_INFO returns none enabled
	if err := g.ClearScreen(0, 0, 0, 0, 1); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v, want ErrNoScanout", err)
	}
}

// --- per-step ErrGPUCommandFailed coverage ----------------------------

func TestClearScreen_StepFailures(t *testing.T) {
	// failAfter = N makes the command at 0-based index N (and onward)
	// respond with an error type. The ClearScreen sequence aborts at the
	// first failure, so each N exercises exactly one step's error branch.
	cases := []struct {
		name       string
		failAfter  int
		forceError bool
	}{
		{"DisplayInfo", 0, true}, // first command; force-error all
		{"CtxCreate", 1, false},
		{"ResourceCreate3D", 2, false},
		{"AttachBacking", 3, false},
		{"CtxAttachResource", 4, false},
		{"Submit3D", 5, false},
		{"TransferToHost3D", 6, false},
		{"SetScanout", 7, false},
		{"ResourceFlush", 8, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, g := openGPU3D(t)
			d.failAfter = c.failAfter
			d.forceError = c.forceError
			if err := g.ClearScreen(0, 0, 0, 0, 1); !errors.Is(err, ErrGPUCommandFailed) {
				t.Errorf("%s: got %v, want ErrGPUCommandFailed", c.name, err)
			}
		})
	}
}

// --- transport / alloc / timeout branches -----------------------------

func TestClearScreen_BackingAllocFail(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	it := newInject(d, false)
	g, err := OpenVirtioGPU3D(it)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	it.enable = true
	// DisplayInfo alloc (#1), CTX_CREATE alloc (#2), RESOURCE_CREATE_3D
	// alloc (#3), then ATTACH_BACKING alloc (#4) fails.
	it.fp = failPoint{"AllocatePages", 4}
	if err := g.ClearScreen(0, 64, 64, 0, 1); err == nil {
		t.Error("expected backing alloc error")
	}
}

func TestClearScreen_BackingZeroPhys(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	it := newInject(d, false)
	g, err := OpenVirtioGPU3D(it)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	it.enable = true
	it.zeroPhys = true
	it.zeroPhysAfter = 3 // DisplayInfo(#1), CTX_CREATE(#2), CREATE_3D(#3) real; backing(#4) zero
	if err := g.ClearScreen(0, 64, 64, 0, 1); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v, want ErrAllocReturnedZero", err)
	}
}

func TestClearScreen_Submit3DAllocFail(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	it := newInject(d, false)
	g, err := OpenVirtioGPU3D(it)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	it.enable = true
	// DisplayInfo(#1), CTX_CREATE(#2), CREATE_3D(#3), ATTACH_BACKING
	// backing(#4), ATTACH_BACKING cmd(#5), CTX_ATTACH(#6), SUBMIT_3D
	// alloc(#7) fails.
	it.fp = failPoint{"AllocatePages", 7}
	if err := g.ClearScreen(0, 64, 64, 0, 1); err == nil {
		t.Error("expected submit3D alloc error")
	}
}

func TestClearScreen_Submit3DZeroPhys(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	it := newInject(d, false)
	g, err := OpenVirtioGPU3D(it)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	it.enable = true
	it.zeroPhys = true
	it.zeroPhysAfter = 6 // first 6 allocs real; SUBMIT_3D alloc (#7) zero
	if err := g.ClearScreen(0, 64, 64, 0, 1); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v, want ErrAllocReturnedZero", err)
	}
}

func TestClearScreen_Submit3DNotifyFail(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1)
	it := newInject(d, false)
	g, err := OpenVirtioGPU3D(it)
	if err != nil {
		t.Fatalf("OpenVirtioGPU3D: %v", err)
	}
	it.enable = true
	// Doorbell writes go through Write32 (notify region 0x1000..0x2000).
	// DisplayInfo(1), CTX_CREATE(2), CREATE_3D(3), ATTACH(4),
	// CTX_ATTACH(5), SUBMIT_3D notify(6).
	it.fp = failPoint{"Write32", 6}
	if err := g.ClearScreen(0, 64, 64, 0, 1); err == nil {
		t.Error("expected submit3D notify error")
	}
}

func TestClearScreen_Submit3DTimeout(t *testing.T) {
	d, g := openGPU3D(t)
	// Let the first five commands (DisplayInfo + CTX_CREATE +
	// RESOURCE_CREATE_3D + ATTACH_BACKING + CTX_ATTACH_RESOURCE) complete,
	// then stop completing so the SUBMIT_3D poll loop hits the timeout.
	d.completesUntil = 5
	if err := g.ClearScreen(0, 64, 64, 0, 1); !errors.Is(err, ErrRequestTimeout) {
		t.Errorf("got %v, want ErrRequestTimeout", err)
	}
}

func TestClearScreen_Submit3DAddChainFull(t *testing.T) {
	d, g := openGPU3D(t)
	// SUBMIT_3D is a 3-descriptor chain. After the 5th ClearScreen command
	// (CTX_ATTACH_RESOURCE) completes, the driver is about to issue
	// SUBMIT_3D; mark all but 2 of the control ring's descriptors in-use so
	// AddChain (needing 3 free slots) returns ErrQueueFull. Using the
	// device's after-command hook keeps the earlier 2-descriptor commands
	// unaffected and the failure deterministic.
	q := g.ControlQueue()
	d.afterCmd = func(seen int) {
		if seen != 5 {
			return
		}
		// Mark every descriptor in-use. The CTX_ATTACH chain that just
		// completed is reclaimed right after this hook returns, freeing
		// its 2 slots — leaving exactly 2 free. SUBMIT_3D needs 3, so
		// AddChain returns ErrQueueFull.
		for i := uint16(0); i < q.Layout.Size; i++ {
			q.Buffers[i].InUse = true
		}
	}
	if err := g.ClearScreen(0, 64, 64, 0, 1); err == nil {
		t.Error("expected submit3D AddChain error")
	}
}
