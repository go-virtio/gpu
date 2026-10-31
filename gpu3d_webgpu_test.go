// Tests of the webgpu-batch guest path against the in-process fake device.
// They check what this driver sends and how it reads replies; they do NOT
// validate the protocol against a host (none exists yet, docs/WEBGPU.md).
package gpu

import (
	"bytes"
	"errors"
	"testing"

	"github.com/go-virtio/common"
	"github.com/go-virtio/gpu/wgpubatch"
)

const wgpuFeats = common.FeatureVersion1 | FeatureVirgl | FeatureContextInit | FeatureResourceBlob

func testCaps() wgpubatch.Caps {
	return wgpubatch.Caps{Version: 1, StreamVersion: 1, MaxFrame: 1 << 16, MaxObjects: 1024, MaxRings: 2, MaxMemory: 1 << 28}
}

func TestWebGPUBatchConstants(t *testing.T) {
	if CapsetWebGPUBatch != 30 {
		t.Fatalf("capset %d", CapsetWebGPUBatch)
	}
	if CapsetWebGPUBatch&0xff != CapsetWebGPUBatch || CapsetWebGPUBatch >= 31 {
		t.Fatal("capset id must fit context_init's 8 bits and Linux's int shift (<31)")
	}
}

func TestEncodeCtxCreateWebGPUBatch(t *testing.T) {
	b := EncodeCtxCreateWebGPUBatch(7, "wgpu")
	if len(b) != 96 {
		t.Fatalf("len %d", len(b))
	}
	for _, c := range []struct{ off, want uint32 }{{0, CmdCtxCreate}, {16, 7}, {24, 4}, {28, 30}} {
		if got := le.Uint32(b[c.off:]); got != c.want {
			t.Errorf("@%d = %d, want %d", c.off, got, c.want)
		}
	}
	if string(b[32:36]) != "wgpu" {
		t.Errorf("debug_name %q", b[32:36])
	}
}

func TestEncodeSubmit3DHeader(t *testing.T) {
	b, err := EncodeSubmit3DHeader(3, 100, Fence{On: true, ID: 0x1122334455, Ringed: true, Ring: 63})
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{
		0x07, 0x02, 0, 0, // SUBMIT_3D
		3, 0, 0, 0, // FENCE | INFO_RING_IDX
		0x55, 0x44, 0x33, 0x22, 0x11, 0, 0, 0, // fence_id
		3, 0, 0, 0, // ctx_id
		63, 0, 0, 0, // ring_idx + padding
		100, 0, 0, 0, 0, 0, 0, 0, // size, padding
	}
	if !bytes.Equal(b, want) {
		t.Fatalf("\n got %x\nwant %x", b, want)
	}
	b, _ = EncodeSubmit3DHeader(3, 8, Fence{})
	if le.Uint32(b[4:]) != 0 || le.Uint64(b[8:]) != 0 || b[20] != 0 {
		t.Fatalf("unfenced header %x", b)
	}
	if _, err := EncodeSubmit3DHeader(3, 8, Fence{Ringed: true, Ring: 64}); !errors.Is(err, ErrRingIndex) {
		t.Fatalf("ring 64: %v", err)
	}
}

func TestCapsetQueries(t *testing.T) {
	b := EncodeGetCapsetInfo(5)
	if le.Uint32(b[0:]) != CmdGetCapsetInfo || le.Uint32(b[24:]) != 5 || len(b) != 32 {
		t.Fatalf("GET_CAPSET_INFO %x", b)
	}
	b = EncodeGetCapset(30, 1)
	if le.Uint32(b[0:]) != CmdGetCapset || le.Uint32(b[24:]) != 30 || le.Uint32(b[28:]) != 1 {
		t.Fatalf("GET_CAPSET %x", b)
	}

	resp := make([]byte, 40)
	le.PutUint32(resp[0:], RespOKCapsetInfo)
	le.PutUint32(resp[24:], 30)
	le.PutUint32(resp[28:], 1)
	le.PutUint32(resp[32:], 40)
	info, ok := DecodeRespCapsetInfo(resp)
	if !ok || info != (CapsetInfo{30, 1, 40}) {
		t.Fatalf("info %+v %v", info, ok)
	}
	if _, ok := DecodeRespCapsetInfo(resp[:39]); ok {
		t.Error("short info accepted")
	}
	le.PutUint32(resp[0:], RespOKCapset)
	if _, ok := DecodeRespCapsetInfo(resp); ok {
		t.Error("wrong type accepted")
	}

	blob := testCaps().Encode()
	r := append(make([]byte, 24), blob...)
	r = append(r, 0xEE, 0xEE) // past max_size: must be cut off
	le.PutUint32(r[0:], RespOKCapset)
	data, ok := DecodeRespCapset(r, uint32(len(blob)))
	if !ok || !bytes.Equal(data, blob) {
		t.Fatalf("capset data %x", data)
	}
	if data, _ := DecodeRespCapset(r, 1000); len(data) != len(blob)+2 {
		t.Fatalf("short response not kept whole: %d", len(data))
	}
	if _, ok := DecodeRespCapset(r[:23], 40); ok {
		t.Error("short capset accepted")
	}
	le.PutUint32(r[0:], RespOKCapsetInfo)
	if _, ok := DecodeRespCapset(r, 40); ok {
		t.Error("wrong type accepted")
	}

	if _, err := DecodeWebGPUBatchCaps(r, CapsetInfo{ID: 4, MaxSize: 40}); !errors.Is(err, ErrNotWebGPUBatch) {
		t.Errorf("venus capset: %v", err)
	}
	if _, err := DecodeWebGPUBatchCaps(r, CapsetInfo{ID: 30, MaxSize: 40}); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("wrong response type: %v", err)
	}
	le.PutUint32(r[0:], RespOKCapset)
	if c, err := DecodeWebGPUBatchCaps(r, CapsetInfo{ID: 30, MaxSize: 40}); err != nil || c != testCaps() {
		t.Errorf("caps %+v %v", c, err)
	}
	if _, err := DecodeWebGPUBatchCaps(r, CapsetInfo{ID: 30, MaxSize: 39}); !errors.Is(err, wgpubatch.ErrFrame) {
		t.Errorf("truncated by max_size: %v", err)
	}
}

func TestNegotiateWebGPUBatchFeatures(t *testing.T) {
	if n, ok := NegotiateWebGPUBatchFeatures(wgpuFeats | 1<<40); !ok || n != wgpuFeats {
		t.Errorf("full: %#x %v", n, ok)
	}
	if n, ok := NegotiateWebGPUBatchFeatures(wgpuFeats &^ FeatureResourceBlob); !ok || n&FeatureResourceBlob != 0 {
		t.Errorf("blob is optional: %#x %v", n, ok)
	}
	for _, missing := range []uint64{common.FeatureVersion1, FeatureVirgl, FeatureContextInit} {
		if _, ok := NegotiateWebGPUBatchFeatures(wgpuFeats &^ missing); ok {
			t.Errorf("accepted without %#x", missing)
		}
	}
}

func openWGPU(t *testing.T, capsets ...fakeCapset) (*fakeGPUDevice, *VirtioGPU) {
	t.Helper()
	d := newFakeGPUDevice(wgpuFeats, 1)
	d.capsets = capsets
	g, err := OpenVirtioGPUWebGPUBatch(d)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	return d, g
}

func TestOpenVirtioGPUWebGPUBatch(t *testing.T) {
	_, g := openWGPU(t)
	if g.NegotiatedFeatures != wgpuFeats {
		t.Errorf("negotiated %#x", g.NegotiatedFeatures)
	}
	d := newFakeGPUDevice(common.FeatureVersion1|FeatureVirgl, 1) // no CONTEXT_INIT
	if _, err := OpenVirtioGPUWebGPUBatch(d); !errors.Is(err, ErrWebGPUBatchUnavailable) {
		t.Errorf("no CONTEXT_INIT: %v", err)
	}
	d = newFakeGPUDevice(wgpuFeats, 1)
	d.rejectVirgl = true
	if _, err := OpenVirtioGPUWebGPUBatch(d); !errors.Is(err, ErrWebGPUBatchUnavailable) {
		t.Errorf("FEATURES_OK refused: %v", err)
	}
}

func TestWebGPUBatchCaps(t *testing.T) {
	venus := fakeCapset{id: CapsetVenus, maxVersion: 0, blob: make([]byte, 64)}
	ours := fakeCapset{id: 30, maxVersion: 3, blob: testCaps().Encode()}
	_, g := openWGPU(t, venus, ours)
	c, err := g.WebGPUBatchCaps()
	if err != nil || c != testCaps() {
		t.Fatalf("caps %+v %v", c, err)
	}

	_, g = openWGPU(t, venus)
	if _, err := g.WebGPUBatchCaps(); !errors.Is(err, ErrWebGPUBatchUnavailable) {
		t.Errorf("not offered: %v", err)
	}

	d, g := openWGPU(t, venus, ours)
	d.forceError = true
	if _, err := g.WebGPUBatchCaps(); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("info error: %v", err)
	}

	d, g = openWGPU(t, venus, ours)
	d.respType = respOKNoData // an OK the driver accepts, but not CAPSET_INFO
	if _, err := g.WebGPUBatchCaps(); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("info of the wrong type: %v", err)
	}

	d, g = openWGPU(t, venus, ours)
	d.failAfter = 2 // both CAPSET_INFOs answer, GET_CAPSET fails
	if _, err := g.WebGPUBatchCaps(); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("GET_CAPSET error: %v", err)
	}

	d, g = openWGPU(t)
	d.numCapsets = 1000 // a hostile count: the first missing index errors out
	if _, err := g.WebGPUBatchCaps(); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("hostile num_capsets: %v", err)
	}

	d = newFakeGPUDevice(wgpuFeats, 1)
	d.capsets = []fakeCapset{ours}
	it := newInject(d, false)
	g, err = OpenVirtioGPUWebGPUBatch(it)
	if err != nil {
		t.Fatal(err)
	}
	it.enable = true
	it.fp = failPoint{"Read32", 1} // num_capsets
	if _, err := g.WebGPUBatchCaps(); !errors.Is(err, errInjected) {
		t.Errorf("config read: %v", err)
	}
}

func TestCreateWebGPUBatchContext(t *testing.T) {
	d, g := openWGPU(t)
	if err := g.CreateWebGPUBatchContext(2, "guest"); err != nil {
		t.Fatal(err)
	}
	req := g.cmdPageMem[requestOffset:]
	if le.Uint32(req[0:]) != CmdCtxCreate || le.Uint32(req[16:]) != 2 || le.Uint32(req[28:]) != 30 {
		t.Errorf("request %x", req[:32])
	}
	if err := g.CreateWebGPUBatchContext(0, ""); !errors.Is(err, ErrBadContext) {
		t.Errorf("ctx 0: %v", err)
	}
	d.forceError = true
	if err := g.CreateWebGPUBatchContext(2, ""); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("device error: %v", err)
	}
	plain := newFakeGPUDevice(wgpuFeats, 1)
	g2, err := OpenVirtioGPU(plain) // negotiates VERSION_1 only
	if err != nil {
		t.Fatal(err)
	}
	if err := g2.CreateWebGPUBatchContext(2, ""); !errors.Is(err, ErrWebGPUBatchUnavailable) {
		t.Errorf("without CONTEXT_INIT: %v", err)
	}
}

func testFrame(t *testing.T, n int) []byte {
	t.Helper()
	e := wgpubatch.NewEncoder(1 << 20)
	e.Stream(make([]uint32, n), []byte{1, 2, 3})
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Clone(f)
}

func TestSubmitWebGPUBatch(t *testing.T) {
	d := newFakeGPUDevice(wgpuFeats, 1)
	it := newInject(d, true) // counts allocations, injects nothing
	g, err := OpenVirtioGPUWebGPUBatch(it)
	if err != nil {
		t.Fatal(err)
	}
	f := testFrame(t, 2000) // > 1 page
	fence := Fence{On: true, ID: 41, Ringed: true, Ring: 1}
	if err := g.SubmitWebGPUBatch(5, f, fence); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(d.submitVirgl, f) {
		t.Fatalf("device saw %d bytes, sent %d", len(d.submitVirgl), len(f))
	}
	want, _ := EncodeSubmit3DHeader(5, uint32(len(f)), fence)
	if !bytes.Equal(g.batchMem[:32], want) {
		t.Errorf("header %x", g.batchMem[:32])
	}
	allocs := it.allocCalls
	// A smaller frame reuses the pages; no fence, no ring.
	small := testFrame(t, 4)
	if err := g.SubmitWebGPUBatch(5, small, Fence{}); err != nil {
		t.Fatal(err)
	}
	if it.allocCalls != allocs {
		t.Errorf("pages not reused: %d allocations, then %d", allocs, it.allocCalls)
	}
	if !bytes.Equal(d.submitVirgl, small) {
		t.Errorf("small frame not delivered")
	}
}

func TestSubmitWebGPUBatchRefuses(t *testing.T) {
	_, g := openWGPU(t)
	f := testFrame(t, 4)
	if err := g.SubmitWebGPUBatch(0, f, Fence{}); !errors.Is(err, ErrBadContext) {
		t.Errorf("ctx 0: %v", err)
	}
	if err := g.SubmitWebGPUBatch(1, nil, Fence{}); !errors.Is(err, ErrEmptyFrame) {
		t.Errorf("empty: %v", err)
	}
	if err := g.SubmitWebGPUBatch(1, f, Fence{Ringed: true, Ring: 99}); !errors.Is(err, ErrRingIndex) {
		t.Errorf("ring: %v", err)
	}
	old := maxBatchFrame
	maxBatchFrame = uint64(len(f)) - 1
	err := g.SubmitWebGPUBatch(1, f, Fence{})
	maxBatchFrame = old
	if !errors.Is(err, ErrFrameTooLarge) {
		t.Errorf("too large: %v", err)
	}
}

func TestSubmitWebGPUBatchDeviceFaults(t *testing.T) {
	f := testFrame(t, 4)
	fenced := Fence{On: true, ID: 9}
	for name, c := range map[string]struct {
		set  func(*fakeGPUDevice)
		want error
	}{
		"error":       {func(d *fakeGPUDevice) { d.forceError = true }, ErrGPUCommandFailed},
		"no-echo":     {func(d *fakeGPUDevice) { d.noFenceEcho = true }, ErrFenceNotEchoed},
		"bad-fence":   {func(d *fakeGPUDevice) { d.badFenceID = true }, ErrFenceNotEchoed},
		"timeout":     {func(d *fakeGPUDevice) { d.completes = false }, ErrRequestTimeout},
		"stale-first": {func(d *fakeGPUDevice) { d.injectStaleBefore = 2 }, nil},
	} {
		d, g := openWGPU(t)
		c.set(d)
		if err := g.SubmitWebGPUBatch(1, f, fenced); !errors.Is(err, c.want) {
			t.Errorf("%s: %v, want %v", name, err, c.want)
		}
	}

	// A late completion left in the used ring is drained before the next
	// submit (R-doom1f).
	d, g := openWGPU(t)
	d.completes = false
	if err := g.SubmitWebGPUBatch(1, f, fenced); !errors.Is(err, ErrRequestTimeout) {
		t.Fatalf("timeout: %v", err)
	}
	d.completes = true
	d.mu.Lock()
	d.handleCommand() // the device finally answers the timed-out submit
	d.mu.Unlock()
	if err := g.SubmitWebGPUBatch(1, f, fenced); err != nil {
		t.Errorf("after a late completion: %v", err)
	}
}

// submit3D used to copy any length into its page: a buffer above 1024
// bytes overlapped the response slot, above 3072 it ran past the page.
func TestSubmit3DRefusesOversizeVirgl(t *testing.T) {
	_, g := openGPU3D(t)
	if err := g.submit3D(make([]byte, 1025)); !errors.Is(err, ErrVirglBufferTooLarge) {
		t.Errorf("1025 bytes: %v", err)
	}
	if err := g.submit3D(make([]byte, 1024)); err != nil {
		t.Errorf("1024 bytes: %v", err)
	}
}

func TestSubmitWebGPUBatchTransportFaults(t *testing.T) {
	f := testFrame(t, 4)
	for name, set := range map[string]func(*injectTransport){
		"alloc":  func(it *injectTransport) { it.fp = failPoint{"AllocatePages", 1} },
		"zero":   func(it *injectTransport) { it.zeroPhys = true },
		"notify": func(it *injectTransport) { it.fp = failPoint{"Write32", 1} },
	} {
		d := newFakeGPUDevice(wgpuFeats, 1)
		it := newInject(d, false)
		g, err := OpenVirtioGPUWebGPUBatch(it)
		if err != nil {
			t.Fatal(err)
		}
		it.enable = true
		set(it)
		if err := g.SubmitWebGPUBatch(1, f, Fence{}); err == nil {
			t.Errorf("%s: no error", name)
		}
	}

	d, g := openWGPU(t)
	q := g.ControlQueue()
	for i := uint16(0); i < q.Layout.Size; i++ {
		q.Buffers[i].InUse = true
	}
	_ = d
	if err := g.SubmitWebGPUBatch(1, f, Fence{}); err == nil {
		t.Error("full queue: no error")
	}
}
