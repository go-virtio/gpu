// Tests for the OpenVirtioGPU driver path and the DisplayInfo /
// SetupFramebuffer / Flush command path. fakeGPUDevice is a minimal
// in-memory virtio-gpu device that, on a control-queue doorbell, walks the
// 2-descriptor chain (request + response), reads the request hdr.type,
// writes a response into the response descriptor's buffer, and publishes a
// used-ring entry.
//
// The driver itself needs no unsafe (it reads the DMA []byte it holds);
// the test does, to play the device side that reads/writes guest memory by
// physical address.

package gpu

import (
	"encoding/binary"
	"errors"
	"sync"
	"testing"
	"unsafe"

	"github.com/go-virtio/common"
)

var le = binary.LittleEndian

func uintptrFromSlice(b []byte) uintptr {
	if len(b) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&b[0]))
}

// sliceAt reconstructs a guest-memory byte view from a physical address —
// the device side of the DMA contract (in this fake, phys is a real Go
// pointer produced by AllocatePages).
func sliceAt(phys uint64, n int) []byte {
	if phys == 0 || n <= 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(uintptr(phys))), n)
}

func TestDeviceType(t *testing.T) {
	if DeviceType != 16 {
		t.Errorf("DeviceType: got %d, want 16", DeviceType)
	}
}

type fakeGPUDevice struct {
	mu sync.Mutex

	cfg []byte

	deviceFeatureSelect uint32
	deviceFeatures      uint64
	driverFeatures      uint64
	deviceStatus        uint8
	currentQueue        uint16

	qsize      map[uint16]uint16
	qenable    map[uint16]uint16
	qdesc      map[uint16]uint64
	qdriver    map[uint16]uint64
	qdevice    map[uint16]uint64
	qnotifyOff map[uint16]uint16

	bar map[uint64]uint64

	numScanouts     uint32
	clearFeaturesOK bool
	completes       bool
	forceError      bool   // make EVERY command respond with an error type
	dropDisplayInfo bool   // GET_DISPLAY_INFO returns all scanouts disabled
	failAfter       int    // >0: fail the command at this 0-based index onward
	cmdSeen         int    // number of commands processed (for failAfter)
	ctrlConsumed    uint16 // used-ring bookkeeping for controlq

	heldPages [][]byte
	allocFail bool
}

func newFakeGPUDevice(deviceFeats uint64, numScanouts uint32) *fakeGPUDevice {
	d := &fakeGPUDevice{
		deviceFeatures: deviceFeats,
		qsize:          map[uint16]uint16{0: 32, 1: 32},
		qenable:        map[uint16]uint16{},
		qdesc:          map[uint16]uint64{},
		qdriver:        map[uint16]uint64{},
		qdevice:        map[uint16]uint64{},
		qnotifyOff:     map[uint16]uint16{0: 0, 1: 1},
		bar:            map[uint64]uint64{},
		numScanouts:    numScanouts,
		completes:      true,
	}
	d.cfg = buildVirtioGPUCfgSpace()
	return d
}

func barKey(bar uint8, off uint64) uint64 { return uint64(bar)<<48 | off }

func (d *fakeGPUDevice) ReadConfig8(off uint8) (uint8, error) {
	if int(off) >= len(d.cfg) {
		return 0, errors.New("read past cfg")
	}
	return d.cfg[off], nil
}
func (d *fakeGPUDevice) ReadConfig16(off uint8) (uint16, error) {
	if int(off)+2 > len(d.cfg) {
		return 0, errors.New("read past cfg")
	}
	return le.Uint16(d.cfg[off : off+2]), nil
}
func (d *fakeGPUDevice) ReadConfig32(off uint8) (uint32, error) {
	if int(off)+4 > len(d.cfg) {
		return 0, errors.New("read past cfg")
	}
	return le.Uint32(d.cfg[off : off+4]), nil
}

func (d *fakeGPUDevice) AllocatePages(count int) (uint64, []byte, error) {
	if d.allocFail {
		return 0, nil, errors.New("alloc fail")
	}
	mem := make([]byte, count*int(common.PageSize))
	addr := uintptr(0)
	if len(mem) > 0 {
		d.heldPages = append(d.heldPages, mem)
		addr = uintptrFromSlice(mem)
	}
	return uint64(addr), mem, nil
}

func (d *fakeGPUDevice) commonCfgBAR() uint8     { return 0 }
func (d *fakeGPUDevice) commonCfgOffset() uint64 { return 0 }

const deviceCfgOff = 0x8000

func (d *fakeGPUDevice) Read8(bar uint8, off uint64) (uint8, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if bar == d.commonCfgBAR() {
		switch off - d.commonCfgOffset() {
		case common.CfgDeviceStatus:
			return d.deviceStatus, nil
		case common.CfgConfigGeneration:
			return 0, nil
		}
	}
	return uint8(d.bar[barKey(bar, off)] & 0xFF), nil
}

func (d *fakeGPUDevice) Read16(bar uint8, off uint64) (uint16, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if bar == d.commonCfgBAR() {
		switch off - d.commonCfgOffset() {
		case common.CfgNumQueues:
			return 2, nil
		case common.CfgQueueSelect:
			return d.currentQueue, nil
		case common.CfgQueueSize:
			return d.qsize[d.currentQueue], nil
		case common.CfgQueueEnable:
			return d.qenable[d.currentQueue], nil
		case common.CfgQueueNotifyOff:
			return d.qnotifyOff[d.currentQueue], nil
		}
	}
	return uint16(d.bar[barKey(bar, off)] & 0xFFFF), nil
}

func (d *fakeGPUDevice) Read32(bar uint8, off uint64) (uint32, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if bar == d.commonCfgBAR() {
		switch off - d.commonCfgOffset() {
		case common.CfgDeviceFeatureSelect:
			return d.deviceFeatureSelect, nil
		case common.CfgDeviceFeature:
			if d.deviceFeatureSelect == 0 {
				return uint32(d.deviceFeatures & 0xFFFFFFFF), nil
			}
			return uint32(d.deviceFeatures >> 32), nil
		}
	}
	// virtio_gpu_config: serve the whole 16-byte region.
	// events_read@0, events_clear@4, num_scanouts@8, num_capsets@12.
	if bar == 0 && off >= deviceCfgOff && off < deviceCfgOff+16 {
		switch off - deviceCfgOff {
		case uint64(cfgEventsRead):
			return 0, nil
		case uint64(cfgEventsClear):
			return 0, nil
		case uint64(cfgNumScanouts):
			return d.numScanouts, nil
		case uint64(cfgNumCapsets):
			return 0, nil
		}
	}
	return uint32(d.bar[barKey(bar, off)] & 0xFFFFFFFF), nil
}

func (d *fakeGPUDevice) Read64(bar uint8, off uint64) (uint64, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if bar == d.commonCfgBAR() {
		switch off - d.commonCfgOffset() {
		case common.CfgQueueDesc:
			return d.qdesc[d.currentQueue], nil
		case common.CfgQueueDriver:
			return d.qdriver[d.currentQueue], nil
		case common.CfgQueueDevice:
			return d.qdevice[d.currentQueue], nil
		}
	}
	return d.bar[barKey(bar, off)], nil
}

func (d *fakeGPUDevice) Write8(bar uint8, off uint64, v uint8) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if bar == d.commonCfgBAR() && off-d.commonCfgOffset() == common.CfgDeviceStatus {
		if v&common.StatusFeaturesOK != 0 {
			if d.clearFeaturesOK || d.driverFeatures&common.FeatureVersion1 == 0 {
				v &^= common.StatusFeaturesOK
			}
		}
		d.deviceStatus = v
		return nil
	}
	d.bar[barKey(bar, off)] = uint64(v)
	return nil
}

func (d *fakeGPUDevice) Write16(bar uint8, off uint64, v uint16) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if bar == d.commonCfgBAR() {
		switch off - d.commonCfgOffset() {
		case common.CfgQueueSelect:
			d.currentQueue = v
			return nil
		case common.CfgQueueSize:
			d.qsize[d.currentQueue] = v
			return nil
		case common.CfgQueueEnable:
			d.qenable[d.currentQueue] = v
			return nil
		}
	}
	if off >= 0x1000 && off < 0x2000 {
		d.handleCommand()
	}
	d.bar[barKey(bar, off)] = uint64(v)
	return nil
}

func (d *fakeGPUDevice) Write32(bar uint8, off uint64, v uint32) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if bar == d.commonCfgBAR() {
		switch off - d.commonCfgOffset() {
		case common.CfgDeviceFeatureSelect:
			d.deviceFeatureSelect = v
			return nil
		case common.CfgDriverFeatureSelect:
			d.bar[barKey(bar, off)] = uint64(v)
			return nil
		case common.CfgDriverFeature:
			sel := d.bar[barKey(bar, common.CfgDriverFeatureSelect)]
			if sel == 0 {
				d.driverFeatures = (d.driverFeatures &^ 0xFFFFFFFF) | uint64(v)
			} else {
				d.driverFeatures = (d.driverFeatures & 0xFFFFFFFF) | (uint64(v) << 32)
			}
			return nil
		}
	}
	if off >= 0x1000 && off < 0x2000 {
		d.handleCommand()
	}
	d.bar[barKey(bar, off)] = uint64(v)
	return nil
}

func (d *fakeGPUDevice) Write64(bar uint8, off uint64, v uint64) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if bar == d.commonCfgBAR() {
		switch off - d.commonCfgOffset() {
		case common.CfgQueueDesc:
			d.qdesc[d.currentQueue] = v
			return nil
		case common.CfgQueueDriver:
			d.qdriver[d.currentQueue] = v
			return nil
		case common.CfgQueueDevice:
			d.qdevice[d.currentQueue] = v
			return nil
		}
	}
	d.bar[barKey(bar, off)] = v
	return nil
}

type fakeDesc struct {
	addr   uint64
	length uint32
	flags  uint16
	next   uint16
}

// handleCommand is the device side of one control command: walk the
// 2-descriptor chain (request + response), read the request hdr.type,
// write a response, publish used. Called from Write32/Write16 with d.mu
// held.
func (d *fakeGPUDevice) handleCommand() {
	if !d.completes {
		return
	}
	const q = ControlQueueIdx
	availAddr := d.qdriver[q]
	usedAddr := d.qdevice[q]
	descAddr := d.qdesc[q]
	if availAddr == 0 || usedAddr == 0 || descAddr == 0 {
		return
	}
	size := d.qsize[q]
	availSlice := sliceAt(availAddr, 4+2*int(size))
	availIdx := le.Uint16(availSlice[2:4])
	if d.ctrlConsumed >= availIdx {
		return
	}
	slot := d.ctrlConsumed % size
	head := le.Uint16(availSlice[4+slot*2 : 4+slot*2+2])

	descSlice := sliceAt(descAddr, 16*int(size))
	var descs []fakeDesc
	idx := head
	for i := 0; i < int(size); i++ {
		o := int(idx) * 16
		dd := fakeDesc{
			addr:   le.Uint64(descSlice[o : o+8]),
			length: le.Uint32(descSlice[o+8 : o+12]),
			flags:  le.Uint16(descSlice[o+12 : o+14]),
			next:   le.Uint16(descSlice[o+14 : o+16]),
		}
		descs = append(descs, dd)
		if dd.flags&common.VirtqDescFNext == 0 {
			break
		}
		idx = dd.next
	}

	reqDesc := descs[0]
	respDesc := descs[len(descs)-1]
	reqBuf := sliceAt(reqDesc.addr, int(reqDesc.length))
	reqType := le.Uint32(reqBuf[0:4])
	respBuf := sliceAt(respDesc.addr, int(respDesc.length))

	fail := d.forceError || (d.failAfter > 0 && d.cmdSeen >= d.failAfter)
	d.cmdSeen++
	d.writeResponse(respBuf, reqType, fail)

	usedSlice := sliceAt(usedAddr, 4+8*int(size))
	usedIdx := le.Uint16(usedSlice[2:4])
	uslot := usedIdx % size
	uo := 4 + int(uslot)*8
	le.PutUint32(usedSlice[uo:uo+4], uint32(head))
	le.PutUint32(usedSlice[uo+4:uo+8], respDesc.length)
	le.PutUint16(usedSlice[2:4], usedIdx+1)
	d.ctrlConsumed++
}

// writeResponse fills the response buffer for the given request type.
func (d *fakeGPUDevice) writeResponse(resp []byte, reqType uint32, fail bool) {
	for i := range resp {
		resp[i] = 0
	}
	if fail {
		le.PutUint32(resp[0:4], respErrBase) // 0x1200 — a failure type
		return
	}
	switch reqType {
	case cmdGetDisplayInfo:
		le.PutUint32(resp[0:4], respOKDisplayInfo)
		if d.dropDisplayInfo {
			// All 16 display_one entries stay zeroed -> none enabled.
			return
		}
		// One enabled scanout (index 0, rect 0,0,1024,768), rest disabled.
		base := ctrlHdrSize // first virtio_gpu_display_one
		// rect: x@0 y@4 width@8 height@12
		le.PutUint32(resp[base+0:base+4], 0)
		le.PutUint32(resp[base+4:base+8], 0)
		le.PutUint32(resp[base+8:base+12], 1024)
		le.PutUint32(resp[base+12:base+16], 768)
		le.PutUint32(resp[base+16:base+20], 1) // enabled
		le.PutUint32(resp[base+20:base+24], 0) // flags
	default:
		le.PutUint32(resp[0:4], respOKNoData)
	}
}

func buildVirtioGPUCfgSpace() []byte {
	cfg := make([]byte, 256)
	le.PutUint16(cfg[0:], common.PCIVendorID)
	le.PutUint16(cfg[2:], common.PCIDeviceIDModernGPU)
	le.PutUint16(cfg[6:], common.PCIStatusCapabilityList)
	cfg[0x34] = 0x40

	cfg[0x40] = common.PCICapIDVendorSpecific
	cfg[0x41] = 0x50
	cfg[0x42] = 16
	cfg[0x43] = common.PCICapCommonCfg
	le.PutUint32(cfg[0x48:], 0)
	le.PutUint32(cfg[0x4C:], 0x38)

	cfg[0x50] = common.PCICapIDVendorSpecific
	cfg[0x51] = 0x68
	cfg[0x52] = 20
	cfg[0x53] = common.PCICapNotifyCfg
	le.PutUint32(cfg[0x58:], 0x1000)
	le.PutUint32(cfg[0x5C:], 0x100)
	le.PutUint32(cfg[0x60:], 4) // notify multiplier = 4

	cfg[0x68] = common.PCICapIDVendorSpecific
	cfg[0x69] = 0x00
	cfg[0x6A] = 16
	cfg[0x6B] = common.PCICapDeviceCfg
	le.PutUint32(cfg[0x70:], deviceCfgOff)
	le.PutUint32(cfg[0x74:], 16) // virtio_gpu_config is 16 bytes

	return cfg
}

// --- happy path + semantics -------------------------------------------

func TestOpenVirtioGPU_Success(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, err := OpenVirtioGPU(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU: %v", err)
	}
	if g.NumScanouts != 1 {
		t.Errorf("NumScanouts: got %d, want 1", g.NumScanouts)
	}
	if g.NegotiatedFeatures != common.FeatureVersion1 {
		t.Errorf("NegotiatedFeatures: got 0x%x", g.NegotiatedFeatures)
	}
	if g.ControlQueue() == nil {
		t.Error("ControlQueue nil")
	}
}

func TestAcceptFeatures(t *testing.T) {
	// VIRGL (bit 0) must NOT be accepted — only VERSION_1.
	if got, err := AcceptFeatures(common.FeatureVersion1 | 1); err != nil || got != common.FeatureVersion1 {
		t.Errorf("modern: got 0x%x, %v", got, err)
	}
	if _, err := AcceptFeatures(1); !errors.Is(err, ErrNotModernDevice) {
		t.Errorf("legacy: got %v", err)
	}
}

func TestOpenVirtioGPU_WrongDeviceID(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	le.PutUint16(d.cfg[2:], common.PCIDeviceIDModernNet)
	if _, err := OpenVirtioGPU(d); !errors.Is(err, ErrInitWrongDeviceID) {
		t.Errorf("got %v", err)
	}
}

func TestOpenVirtioGPU_LegacyDevice(t *testing.T) {
	d := newFakeGPUDevice(0, 1) // no VERSION_1
	if _, err := OpenVirtioGPU(d); !errors.Is(err, ErrNotModernDevice) {
		t.Errorf("got %v", err)
	}
}

func TestOpenVirtioGPU_FeaturesNotOK(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	d.clearFeaturesOK = true
	if _, err := OpenVirtioGPU(d); !errors.Is(err, ErrFeaturesNotOK) {
		t.Errorf("got %v", err)
	}
}

func TestOpenVirtioGPU_ControlQueueZeroSize(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	d.qsize[0] = 0
	if _, err := OpenVirtioGPU(d); !errors.Is(err, ErrQueueNotAvailable) {
		t.Errorf("got %v", err)
	}
}

func TestOpenVirtioGPU_CursorQueueZeroSize(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	d.qsize[1] = 0
	if _, err := OpenVirtioGPU(d); !errors.Is(err, ErrQueueNotAvailable) {
		t.Errorf("got %v", err)
	}
}

func TestOpenVirtioGPU_QueueSizeClampAndRound(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	d.qsize[0] = 6 // clamp 16->6, round 6->4
	g, err := OpenVirtioGPU(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU: %v", err)
	}
	if got := g.ControlQueue().Layout.Size; got != 4 {
		t.Errorf("queue size: got %d, want 4", got)
	}
}

// --- DisplayInfo -------------------------------------------------------

func TestDisplayInfo_Success(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, err := OpenVirtioGPU(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU: %v", err)
	}
	displays, err := g.DisplayInfo()
	if err != nil {
		t.Fatalf("DisplayInfo: %v", err)
	}
	if len(displays) != maxScanouts {
		t.Fatalf("len: got %d, want %d", len(displays), maxScanouts)
	}
	if !displays[0].Enabled || displays[0].Width != 1024 || displays[0].Height != 768 {
		t.Errorf("scanout 0: %+v", displays[0])
	}
	if displays[0].ScanoutID != 0 {
		t.Errorf("ScanoutID: got %d", displays[0].ScanoutID)
	}
	if displays[1].Enabled {
		t.Error("scanout 1 should be disabled")
	}
}

func TestDisplayInfo_NoScanout(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	d.dropDisplayInfo = true
	if _, err := g.DisplayInfo(); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v", err)
	}
}

func TestDisplayInfo_CommandFailed(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	d.forceError = true
	if _, err := g.DisplayInfo(); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("got %v", err)
	}
}

func TestDisplayInfo_AllocFail(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	d.allocFail = true
	if _, err := g.DisplayInfo(); err == nil {
		t.Error("expected alloc error")
	}
}

func TestDisplayInfo_AllocZeroPhys(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	it := newInject(d, false)
	g, _ := OpenVirtioGPU(it)
	it.enable = true
	it.zeroPhys = true // first command alloc returns 0
	if _, err := g.DisplayInfo(); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v", err)
	}
}

func TestDisplayInfo_NotifyFail(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	it := newInject(d, false)
	g, _ := OpenVirtioGPU(it)
	it.enable = true
	it.fp = failPoint{"Write32", 1} // controlq doorbell
	if _, err := g.DisplayInfo(); err == nil {
		t.Error("expected notify error")
	}
}

func TestDisplayInfo_Timeout(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	d.completes = false
	if _, err := g.DisplayInfo(); !errors.Is(err, ErrRequestTimeout) {
		t.Errorf("got %v", err)
	}
}

func TestDisplayInfo_AddChainFull(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, err := OpenVirtioGPU(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU: %v", err)
	}
	q := g.ControlQueue()
	phys, _, _ := d.AllocatePages(1)
	for i := uint16(0); i < q.Layout.Size; i++ {
		if _, err := q.AddBuffer(uintptr(phys), phys, 16, false); err != nil {
			t.Fatalf("saturate[%d]: %v", i, err)
		}
	}
	if _, err := g.DisplayInfo(); err == nil {
		t.Error("expected AddChain queue-full error")
	}
}

// --- SetupFramebuffer + Flush -----------------------------------------

func TestSetupFramebuffer_Success(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, err := OpenVirtioGPU(d)
	if err != nil {
		t.Fatalf("OpenVirtioGPU: %v", err)
	}
	fb, err := g.SetupFramebuffer(0, 1024, 768)
	if err != nil {
		t.Fatalf("SetupFramebuffer: %v", err)
	}
	if got, want := len(fb.Pix), 1024*768*4; got != want {
		t.Errorf("Pix length: got %d, want %d", got, want)
	}
	if fb.Width != 1024 || fb.Height != 768 {
		t.Errorf("dims: %dx%d", fb.Width, fb.Height)
	}
	if fb.ResourceID != resourceID {
		t.Errorf("ResourceID: got %d", fb.ResourceID)
	}
	// Three commands consumed: create, attach, scanout.
	if d.ctrlConsumed != 3 {
		t.Errorf("ctrlConsumed: got %d, want 3", d.ctrlConsumed)
	}
	if err := fb.Flush(); err != nil {
		t.Errorf("Flush: %v", err)
	}
	if d.ctrlConsumed != 5 { // + transfer + flush
		t.Errorf("ctrlConsumed after flush: got %d, want 5", d.ctrlConsumed)
	}
}

func TestSetupFramebuffer_ScanoutOutOfRange(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	if _, err := g.SetupFramebuffer(5, 64, 64); !errors.Is(err, ErrNoScanout) {
		t.Errorf("got %v", err)
	}
}

func TestSetupFramebuffer_ZeroSize(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	if _, err := g.SetupFramebuffer(0, 0, 0); !errors.Is(err, ErrFramebufferTooLarge) {
		t.Errorf("got %v", err)
	}
}

func TestSetupFramebuffer_TooLarge(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	if _, err := g.SetupFramebuffer(0, 1<<16, 1<<16); !errors.Is(err, ErrFramebufferTooLarge) {
		t.Errorf("got %v", err)
	}
}

func TestSetupFramebuffer_CreateFailed(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	d.forceError = true // RESOURCE_CREATE_2D fails
	if _, err := g.SetupFramebuffer(0, 64, 64); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("got %v", err)
	}
}

func TestSetupFramebuffer_BackingAllocFail(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	it := newInject(d, false)
	g, _ := OpenVirtioGPU(it)
	it.enable = true
	it.fp = failPoint{"AllocatePages", 2} // create cmd ok, backing alloc fails
	if _, err := g.SetupFramebuffer(0, 64, 64); err == nil {
		t.Error("expected backing alloc error")
	}
}

func TestSetupFramebuffer_BackingZeroPhys(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	it := newInject(d, false)
	g, _ := OpenVirtioGPU(it)
	it.enable = true
	it.zeroPhys = true
	it.zeroPhysAfter = 1 // create cmd alloc (#1) real, backing (#2) zero
	if _, err := g.SetupFramebuffer(0, 64, 64); !errors.Is(err, common.ErrAllocReturnedZero) {
		t.Errorf("got %v", err)
	}
}

func TestSetupFramebuffer_AttachFailed(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	d.failAfter = 1 // create ok; attach fails
	if _, err := g.SetupFramebuffer(0, 64, 64); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("got %v", err)
	}
}

func TestSetupFramebuffer_ScanoutFailed(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	d.failAfter = 2 // create + attach ok; set_scanout fails
	if _, err := g.SetupFramebuffer(0, 64, 64); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("got %v", err)
	}
}

func TestFlush_TransferFailed(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	fb, err := g.SetupFramebuffer(0, 64, 64)
	if err != nil {
		t.Fatalf("SetupFramebuffer: %v", err)
	}
	d.forceError = true // TRANSFER_TO_HOST_2D fails
	if err := fb.Flush(); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("got %v", err)
	}
}

func TestFlush_FlushFailed(t *testing.T) {
	d := newFakeGPUDevice(common.FeatureVersion1, 1)
	g, _ := OpenVirtioGPU(d)
	fb, err := g.SetupFramebuffer(0, 64, 64)
	if err != nil {
		t.Fatalf("SetupFramebuffer: %v", err)
	}
	d.cmdSeen = 0
	d.failAfter = 1 // transfer (index 0) ok; flush (index 1) fails
	if err := fb.Flush(); !errors.Is(err, ErrGPUCommandFailed) {
		t.Errorf("got %v", err)
	}
}

func TestSentinelError(t *testing.T) {
	if got := ErrGPUCommandFailed.Error(); got != string(ErrGPUCommandFailed) {
		t.Errorf("Error(): %q", got)
	}
}

// --- injection harness + Open transport-error coverage ----------------

var errInjected = errors.New("injected transport failure")

type failPoint struct {
	method string
	nth    int
}

type injectTransport struct {
	*fakeGPUDevice
	fp            failPoint
	counts        map[string]int
	enable        bool
	zeroPhys      bool
	zeroPhysAfter int
	allocCalls    int
}

func newInject(d *fakeGPUDevice, enable bool) *injectTransport {
	return &injectTransport{fakeGPUDevice: d, counts: map[string]int{}, enable: enable}
}

func (t *injectTransport) fail(m string) bool {
	if !t.enable || t.fp.method != m {
		return false
	}
	t.counts[m]++
	return t.counts[m] == t.fp.nth
}

func (t *injectTransport) ReadConfig16(o uint8) (uint16, error) {
	if t.fail("ReadConfig16") {
		return 0, errInjected
	}
	return t.fakeGPUDevice.ReadConfig16(o)
}
func (t *injectTransport) Read8(b uint8, o uint64) (uint8, error) {
	if t.fail("Read8") {
		return 0, errInjected
	}
	return t.fakeGPUDevice.Read8(b, o)
}
func (t *injectTransport) Read16(b uint8, o uint64) (uint16, error) {
	if t.fail("Read16") {
		return 0, errInjected
	}
	return t.fakeGPUDevice.Read16(b, o)
}
func (t *injectTransport) Read32(b uint8, o uint64) (uint32, error) {
	if t.fail("Read32") {
		return 0, errInjected
	}
	return t.fakeGPUDevice.Read32(b, o)
}
func (t *injectTransport) Read64(b uint8, o uint64) (uint64, error) {
	if t.fail("Read64") {
		return 0, errInjected
	}
	return t.fakeGPUDevice.Read64(b, o)
}
func (t *injectTransport) Write8(b uint8, o uint64, v uint8) error {
	if t.fail("Write8") {
		return errInjected
	}
	return t.fakeGPUDevice.Write8(b, o, v)
}
func (t *injectTransport) Write16(b uint8, o uint64, v uint16) error {
	if t.fail("Write16") {
		return errInjected
	}
	return t.fakeGPUDevice.Write16(b, o, v)
}
func (t *injectTransport) Write32(b uint8, o uint64, v uint32) error {
	if t.fail("Write32") {
		return errInjected
	}
	return t.fakeGPUDevice.Write32(b, o, v)
}
func (t *injectTransport) Write64(b uint8, o uint64, v uint64) error {
	if t.fail("Write64") {
		return errInjected
	}
	return t.fakeGPUDevice.Write64(b, o, v)
}
func (t *injectTransport) AllocatePages(c int) (uint64, []byte, error) {
	if t.fail("AllocatePages") {
		return 0, nil, errInjected
	}
	phys, mem, err := t.fakeGPUDevice.AllocatePages(c)
	// Count only while armed so zeroPhysAfter is relative to the
	// operation under test, not to the queue allocs done during Open.
	if t.enable {
		t.allocCalls++
		if t.zeroPhys && t.allocCalls > t.zeroPhysAfter {
			return 0, mem, nil
		}
	}
	return phys, mem, err
}

func TestOpenVirtioGPU_TransportErrors(t *testing.T) {
	cases := []struct {
		name string
		fp   failPoint
	}{
		{"DIDRead", failPoint{"ReadConfig16", 1}},
		{"InitModernConfig", failPoint{"ReadConfig16", 2}},
		{"ResetStatus", failPoint{"Write8", 1}},
		{"PostResetStatusRead", failPoint{"Read8", 1}},
		{"AckStatus", failPoint{"Write8", 2}},
		{"DriverStatus", failPoint{"Write8", 3}},
		{"DeviceFeatures", failPoint{"Write32", 1}},
		{"DriverFeatures", failPoint{"Write32", 3}},
		{"FeaturesOKStatus", failPoint{"Write8", 4}},
		{"PostFeaturesStatusRead", failPoint{"Read8", 2}},
		// controlq (queue 0) setup.
		{"SelectControlQueue", failPoint{"Write16", 1}},
		{"ControlQueueSize", failPoint{"Read16", 1}},
		{"SetControlQueueSize", failPoint{"Write16", 2}},
		{"ControlQueueNotifyOff", failPoint{"Read16", 2}},
		{"AllocControlVirtqueue", failPoint{"AllocatePages", 1}},
		{"SetControlQueueDesc", failPoint{"Write64", 1}},
		{"SetControlQueueDriver", failPoint{"Write64", 2}},
		{"SetControlQueueDevice", failPoint{"Write64", 3}},
		{"SetControlQueueEnable", failPoint{"Write16", 3}},
		// cursorq (queue 1) setup.
		{"SelectCursorQueue", failPoint{"Write16", 4}},
		{"CursorQueueSize", failPoint{"Read16", 3}},
		{"SetCursorQueueSize", failPoint{"Write16", 5}},
		{"CursorQueueNotifyOff", failPoint{"Read16", 4}},
		{"AllocCursorVirtqueue", failPoint{"AllocatePages", 2}},
		{"SetCursorQueueDesc", failPoint{"Write64", 4}},
		{"SetCursorQueueDriver", failPoint{"Write64", 5}},
		{"SetCursorQueueDevice", failPoint{"Write64", 6}},
		{"SetCursorQueueEnable", failPoint{"Write16", 6}},
		{"DriverOKStatus", failPoint{"Write8", 5}},
		{"NumScanoutsRead", failPoint{"Read32", 3}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := newFakeGPUDevice(common.FeatureVersion1, 1)
			it := newInject(d, true)
			it.fp = tc.fp
			if _, err := OpenVirtioGPU(it); err == nil {
				t.Fatalf("%s: expected error at %+v", tc.name, tc.fp)
			}
		})
	}
}
