// Package gpu is a pure-Go virtio-gpu (2D framebuffer) driver. It drives
// a modern (Virtio 1.0+) PCI virtio-gpu device through the transport
// interfaces defined in github.com/go-virtio/common; the same code drives
// a UEFI-backed device, a bare-metal device, or a virtio-mmio device
// depending on which common.Transport implementation the caller supplies.
//
// Scope — 2D FRAMEBUFFER ONLY. The driver negotiates exactly
// VIRTIO_F_VERSION_1 and deliberately does NOT negotiate VIRTIO_GPU_F_VIRGL
// (bit 0), so the device operates in plain 2D mode (Virtio 1.1 §5.7).
// 3D / virgl acceleration is out of scope: it requires a Mesa-class
// GL/Vulkan stack to generate the OpenGL command streams the host
// consumes, and no such stack exists in Go.
//
// The driver owns device bring-up, the control virtqueue (controlq), the
// cursor virtqueue (cursorq, set up but unused), and the on-the-wire
// virtio-gpu control protocol (every command is a 2-descriptor chain: a
// read-only request followed by a device-writable response). It exposes a
// scanout-enumeration + framebuffer API:
//
//   - DisplayInfo lists the device's scanouts (GET_DISPLAY_INFO).
//   - SetupFramebuffer creates a host resource, attaches guest backing,
//     and binds it to a scanout (RESOURCE_CREATE_2D +
//     RESOURCE_ATTACH_BACKING + SET_SCANOUT). The returned Framebuffer's
//     Pix is a BGRA byte buffer the caller draws into.
//   - Framebuffer.Flush pushes the drawn pixels to the host and refreshes
//     the scanout (TRANSFER_TO_HOST_2D + RESOURCE_FLUSH).
//
// References:
//
//   - Virtio 1.1 §5.7   "GPU Device" — device-type 16 binding.
//   - Virtio 1.1 §5.7.4 "Device configuration layout" — virtio_gpu_config.
//   - Virtio 1.1 §5.7.6 "Device Operation" — control + cursor queues,
//     struct virtio_gpu_ctrl_hdr and the 2D command structs.
//   - Virtio 1.1 §3.1.1 "Device Initialization".
package gpu

import (
	"encoding/binary"

	"github.com/go-virtio/common"
)

// DeviceType is the virtio device-type encoding for virtio-gpu
// (Virtio 1.1 §5.7.1). Retained for callers enumerating PCI devices that
// want a stable name.
const DeviceType uint16 = 16

// Queue indices (Virtio 1.1 §5.7.2). The control queue carries every 2D
// command; the cursor queue carries cursor updates and is set up for
// spec-completeness but otherwise unused by this driver.
const (
	ControlQueueIdx uint16 = 0
	CursorQueueIdx  uint16 = 1
)

// ControlQueueSize / CursorQueueSize are the desired ring sizes (clamped +
// rounded during setup). Commands are issued one at a time, so a small
// ring is plenty.
const (
	ControlQueueSize uint16 = 16
	CursorQueueSize  uint16 = 16
)

// CommandPollIterations is the busy-poll budget spent waiting for the
// device to complete one control command.
const CommandPollIterations = 200000

// Device-config (virtio_gpu_config) field offsets (Virtio 1.1 §5.7.4):
// le32 events_read @0, events_clear @4, num_scanouts @8, num_capsets @12.
const (
	cfgEventsRead  uint32 = 0
	cfgEventsClear uint32 = 4
	cfgNumScanouts uint32 = 8
	cfgNumCapsets  uint32 = 12
)

// ctrlHdrSize is the on-the-wire size of struct virtio_gpu_ctrl_hdr, the
// 24-byte header that prefixes EVERY request and response (Virtio 1.1
// §5.7.6.1): le32 type, le32 flags, le64 fence_id, le32 ctx_id, le32
// padding.
const ctrlHdrSize = 24

// rectSize is the on-the-wire size of struct virtio_gpu_rect: le32 x,
// le32 y, le32 width, le32 height.
const rectSize = 16

// memEntrySize is the on-the-wire size of struct virtio_gpu_mem_entry:
// le64 addr, le32 length, le32 padding.
const memEntrySize = 16

// displayOneSize is the on-the-wire size of one entry of the
// GET_DISPLAY_INFO response array (struct virtio_gpu_display_one):
// virtio_gpu_rect (16) + le32 enabled + le32 flags.
const displayOneSize = 24

// maxScanouts is the fixed number of virtio_gpu_display_one entries in a
// GET_DISPLAY_INFO response (Virtio 1.1 §5.7.6.8 —
// VIRTIO_GPU_MAX_SCANOUTS).
const maxScanouts = 16

// requestOffset / responseOffset are the byte offsets within the single
// command page where the request and response structs live. One
// AllocatePages(1) per command serves both (phys+0 request, phys+2048
// response).
const (
	requestOffset  = 0
	responseOffset = 2048
)

// Control-command request type codes (Virtio 1.1 §5.7.6.7).
const (
	cmdGetDisplayInfo        uint32 = 0x0100
	cmdResourceCreate2D      uint32 = 0x0101
	cmdResourceUnref         uint32 = 0x0102
	cmdSetScanout            uint32 = 0x0103
	cmdResourceFlush         uint32 = 0x0104
	cmdTransferToHost2D      uint32 = 0x0105
	cmdResourceAttachBacking uint32 = 0x0106
)

// Control-command response type codes (Virtio 1.1 §5.7.6.7). A response
// is a success iff its hdr.type is one of these; any other value (the
// error responses start at 0x1200) is a failure.
const (
	respOKNoData      uint32 = 0x1100
	respOKDisplayInfo uint32 = 0x1101
	respErrBase       uint32 = 0x1200
)

// VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM = 1 (Virtio 1.1 §5.7.6.7 —
// virtio_gpu_formats). The driver always creates BGRA resources.
const formatB8G8R8A8Unorm uint32 = 1

// resourceID is the single 2D resource id the driver uses for its
// framebuffer (1 is the conventional first valid id; 0 is reserved).
const resourceID uint32 = 1

// maxFramebufferBytes bounds the framebuffer backing size SetupFramebuffer
// will allocate, guarding against an absurd width*height that would try to
// allocate a runaway number of pages.
const maxFramebufferBytes = 256 << 20 // 256 MiB

// AcceptedFeatures is the feature mask the driver negotiates ON — only the
// non-negotiable VIRTIO_F_VERSION_1. VIRTIO_GPU_F_VIRGL (bit 0) is
// deliberately NOT accepted, keeping the device in 2D mode.
const AcceptedFeatures uint64 = common.FeatureVersion1

// AcceptFeatures returns the negotiated mask (requires VERSION_1).
func AcceptFeatures(deviceFeatures uint64) (uint64, error) {
	if deviceFeatures&common.FeatureVersion1 == 0 {
		return 0, ErrNotModernDevice
	}
	return deviceFeatures & AcceptedFeatures, nil
}

// VirtioGPU wraps one initialised virtio-gpu device in 2D mode.
type VirtioGPU struct {
	// Cfg is the modern-transport handle.
	Cfg *common.ModernConfig

	// NumScanouts is the device's advertised scanout count, read from
	// virtio_gpu_config.num_scanouts at Open.
	NumScanouts uint32

	// NegotiatedFeatures records the driver-feature handshake result.
	NegotiatedFeatures uint64

	transport common.Transport
	controlq  *common.Virtqueue
	cursorq   *common.Virtqueue
}

// OpenVirtioGPU drives the full bring-up of one virtio-gpu device:
//
//  1. Verify the PCI device ID is 0x1050 (modern GPU).
//  2. InitModernConfig walks PCI caps + populates the BAR locators.
//  3. Reset → ACK → DRIVER status progression.
//  4. Read DeviceFeature, require VERSION_1, mask to VERSION_1 (NOT
//     VIRGL), write DriverFeature.
//  5. Set FEATURES_OK, verify it stuck.
//  6. Allocate + publish controlq (queue 0) and cursorq (queue 1).
//  7. DRIVER_OK status.
//  8. Read num_scanouts (le32) from DeviceCfg offset 8.
func OpenVirtioGPU(t common.Transport) (*VirtioGPU, error) {
	did, err := t.ReadConfig16(common.PCICfgDeviceID)
	if err != nil {
		return nil, err
	}
	if did != common.PCIDeviceIDModernGPU {
		return nil, ErrInitWrongDeviceID
	}

	cfg, err := common.InitModernConfig(t)
	if err != nil {
		return nil, err
	}

	if err := cfg.SetDeviceStatus(0); err != nil {
		return nil, err
	}
	if _, err := cfg.DeviceStatus(); err != nil {
		return nil, err
	}
	if err := cfg.SetDeviceStatus(common.StatusAcknowledge); err != nil {
		return nil, err
	}
	if err := cfg.SetDeviceStatus(common.StatusAcknowledge | common.StatusDriver); err != nil {
		return nil, err
	}

	deviceFeats, err := cfg.DeviceFeatures64()
	if err != nil {
		return nil, err
	}
	if deviceFeats&common.FeatureVersion1 == 0 {
		return nil, ErrNotModernDevice
	}
	negotiated := deviceFeats & AcceptedFeatures
	if err := cfg.SetDriverFeatures64(negotiated); err != nil {
		return nil, err
	}

	if err := cfg.SetDeviceStatus(common.StatusAcknowledge | common.StatusDriver | common.StatusFeaturesOK); err != nil {
		return nil, err
	}
	status, err := cfg.DeviceStatus()
	if err != nil {
		return nil, err
	}
	if status&common.StatusFeaturesOK == 0 {
		return nil, ErrFeaturesNotOK
	}

	controlq, err := setupQueue(cfg, t, ControlQueueIdx, ControlQueueSize)
	if err != nil {
		return nil, err
	}
	cursorq, err := setupQueue(cfg, t, CursorQueueIdx, CursorQueueSize)
	if err != nil {
		return nil, err
	}

	if err := cfg.SetDeviceStatus(common.StatusAcknowledge | common.StatusDriver | common.StatusFeaturesOK | common.StatusDriverOK); err != nil {
		return nil, err
	}

	// num_scanouts: le32 at DeviceCfg offset 8 (Virtio 1.1 §5.7.4).
	numScanouts, err := cfg.DeviceCfgRead32(cfgNumScanouts)
	if err != nil {
		return nil, err
	}

	return &VirtioGPU{
		Cfg:                cfg,
		NumScanouts:        numScanouts,
		NegotiatedFeatures: negotiated,
		transport:          t,
		controlq:           controlq,
		cursorq:            cursorq,
	}, nil
}

// setupQueue performs the per-queue init (select, size, allocate, publish
// addresses, enable).
func setupQueue(cfg *common.ModernConfig, t common.Transport, queueIdx uint16, desiredSize uint16) (*common.Virtqueue, error) {
	if err := cfg.SelectQueue(queueIdx); err != nil {
		return nil, err
	}
	maxSize, err := cfg.QueueSize()
	if err != nil {
		return nil, err
	}
	if maxSize == 0 {
		return nil, ErrQueueNotAvailable
	}
	size := desiredSize
	if size > maxSize {
		size = maxSize
	}
	for size&(size-1) != 0 {
		size &= size - 1
	}
	if err := cfg.SetQueueSize(size); err != nil {
		return nil, err
	}
	notifyOff, err := cfg.QueueNotifyOff()
	if err != nil {
		return nil, err
	}
	q, err := common.NewVirtqueue(t, size, queueIdx, notifyOff)
	if err != nil {
		return nil, err
	}
	descAddr := q.BasePhys + uint64(q.Layout.DescTableOffset)
	availAddr := q.BasePhys + uint64(q.Layout.AvailRingOffset)
	usedAddr := q.BasePhys + uint64(q.Layout.UsedRingOffset)
	if err := cfg.SetQueueDesc(descAddr); err != nil {
		return nil, err
	}
	if err := cfg.SetQueueDriver(availAddr); err != nil {
		return nil, err
	}
	if err := cfg.SetQueueDevice(usedAddr); err != nil {
		return nil, err
	}
	if err := cfg.SetQueueEnable(1); err != nil {
		return nil, err
	}
	return q, nil
}

// ControlQueue exposes the control virtqueue handle for diagnostics.
func (g *VirtioGPU) ControlQueue() *common.Virtqueue { return g.controlq }

// Display describes one scanout reported by GET_DISPLAY_INFO.
type Display struct {
	ScanoutID     uint32
	Width, Height uint32
	Enabled       bool
}

// DisplayInfo issues GET_DISPLAY_INFO and parses the fixed array of 16
// virtio_gpu_display_one entries into one Display per scanout index.
func (g *VirtioGPU) DisplayInfo() ([]Display, error) {
	// Request: bare 24-byte header. Response: header + 16*24 bytes.
	req := make([]byte, ctrlHdrSize)
	binary.LittleEndian.PutUint32(req[0:4], cmdGetDisplayInfo)

	resp, err := g.sendCommand(req)
	if err != nil {
		return nil, err
	}

	out := make([]Display, maxScanouts)
	anyEnabled := false
	for i := 0; i < maxScanouts; i++ {
		base := ctrlHdrSize + i*displayOneSize
		// rect: x@0 y@4 width@8 height@12; enabled@16 flags@20.
		width := binary.LittleEndian.Uint32(resp[base+8 : base+12])
		height := binary.LittleEndian.Uint32(resp[base+12 : base+16])
		enabled := binary.LittleEndian.Uint32(resp[base+16:base+20]) != 0
		out[i] = Display{
			ScanoutID: uint32(i),
			Width:     width,
			Height:    height,
			Enabled:   enabled,
		}
		if enabled {
			anyEnabled = true
		}
	}
	if !anyEnabled {
		return nil, ErrNoScanout
	}
	return out, nil
}

// Framebuffer is a host 2D resource bound to a scanout, backed by a guest
// BGRA pixel buffer the caller draws into and then pushes via Flush.
type Framebuffer struct {
	// Pix is the BGRA pixel buffer, width*height*4 bytes. The caller
	// writes pixels here, then calls Flush to display them.
	Pix []byte

	Width, Height uint32
	ResourceID    uint32

	pixPhys   uint64
	gpu       *VirtioGPU
	scanoutID uint32
}

// SetupFramebuffer creates a 2D BGRA resource, attaches a guest-allocated
// backing store, and binds it to scanoutID:
//
//  1. RESOURCE_CREATE_2D(resource_id=1, format=BGRA, width, height).
//  2. Allocate ceil(width*height*4 / PageSize) contiguous pages; hold the
//     []byte as Pix (sliced to width*height*4).
//  3. RESOURCE_ATTACH_BACKING(resource_id=1, one mem_entry = {phys, size}).
//  4. SET_SCANOUT(scanout_id, resource_id=1, rect{0,0,width,height}).
func (g *VirtioGPU) SetupFramebuffer(scanoutID, width, height uint32) (*Framebuffer, error) {
	if scanoutID >= g.NumScanouts {
		return nil, ErrNoScanout
	}
	size := uint64(width) * uint64(height) * 4
	if size == 0 || size > maxFramebufferBytes {
		return nil, ErrFramebufferTooLarge
	}

	// 1. RESOURCE_CREATE_2D.
	create := make([]byte, ctrlHdrSize+16)
	binary.LittleEndian.PutUint32(create[0:4], cmdResourceCreate2D)
	binary.LittleEndian.PutUint32(create[ctrlHdrSize+0:ctrlHdrSize+4], resourceID)
	binary.LittleEndian.PutUint32(create[ctrlHdrSize+4:ctrlHdrSize+8], formatB8G8R8A8Unorm)
	binary.LittleEndian.PutUint32(create[ctrlHdrSize+8:ctrlHdrSize+12], width)
	binary.LittleEndian.PutUint32(create[ctrlHdrSize+12:ctrlHdrSize+16], height)
	if _, err := g.sendCommand(create); err != nil {
		return nil, err
	}

	// 2. Allocate backing.
	pages := int((size + uint64(common.PageSize) - 1) / uint64(common.PageSize))
	pixPhys, pixMem, err := g.transport.AllocatePages(pages)
	if err != nil {
		return nil, err
	}
	if pixPhys == 0 {
		return nil, common.ErrAllocReturnedZero
	}
	pix := pixMem[:size]

	// 3. RESOURCE_ATTACH_BACKING with one mem_entry.
	attach := make([]byte, ctrlHdrSize+8+memEntrySize)
	binary.LittleEndian.PutUint32(attach[0:4], cmdResourceAttachBacking)
	binary.LittleEndian.PutUint32(attach[ctrlHdrSize+0:ctrlHdrSize+4], resourceID)
	binary.LittleEndian.PutUint32(attach[ctrlHdrSize+4:ctrlHdrSize+8], 1) // nr_entries
	me := ctrlHdrSize + 8
	binary.LittleEndian.PutUint64(attach[me+0:me+8], pixPhys)
	binary.LittleEndian.PutUint32(attach[me+8:me+12], uint32(size))
	binary.LittleEndian.PutUint32(attach[me+12:me+16], 0) // padding
	if _, err := g.sendCommand(attach); err != nil {
		return nil, err
	}

	// 4. SET_SCANOUT.
	setScanout := make([]byte, ctrlHdrSize+rectSize+8)
	binary.LittleEndian.PutUint32(setScanout[0:4], cmdSetScanout)
	putRect(setScanout[ctrlHdrSize:], 0, 0, width, height)
	ss := ctrlHdrSize + rectSize
	binary.LittleEndian.PutUint32(setScanout[ss+0:ss+4], scanoutID)
	binary.LittleEndian.PutUint32(setScanout[ss+4:ss+8], resourceID)
	if _, err := g.sendCommand(setScanout); err != nil {
		return nil, err
	}

	return &Framebuffer{
		Pix:        pix,
		Width:      width,
		Height:     height,
		ResourceID: resourceID,
		pixPhys:    pixPhys,
		gpu:        g,
		scanoutID:  scanoutID,
	}, nil
}

// Flush pushes the drawn pixels to the host resource and refreshes the
// scanout: TRANSFER_TO_HOST_2D(rect{0,0,W,H}, offset=0) then
// RESOURCE_FLUSH(rect{0,0,W,H}).
func (fb *Framebuffer) Flush() error {
	// TRANSFER_TO_HOST_2D: rect(16); le64 offset; le32 resource_id; le32 padding.
	transfer := make([]byte, ctrlHdrSize+rectSize+16)
	binary.LittleEndian.PutUint32(transfer[0:4], cmdTransferToHost2D)
	putRect(transfer[ctrlHdrSize:], 0, 0, fb.Width, fb.Height)
	to := ctrlHdrSize + rectSize
	binary.LittleEndian.PutUint64(transfer[to+0:to+8], 0) // offset
	binary.LittleEndian.PutUint32(transfer[to+8:to+12], fb.ResourceID)
	binary.LittleEndian.PutUint32(transfer[to+12:to+16], 0) // padding
	if _, err := fb.gpu.sendCommand(transfer); err != nil {
		return err
	}

	// RESOURCE_FLUSH: rect(16); le32 resource_id; le32 padding.
	flush := make([]byte, ctrlHdrSize+rectSize+8)
	binary.LittleEndian.PutUint32(flush[0:4], cmdResourceFlush)
	putRect(flush[ctrlHdrSize:], 0, 0, fb.Width, fb.Height)
	fo := ctrlHdrSize + rectSize
	binary.LittleEndian.PutUint32(flush[fo+0:fo+4], fb.ResourceID)
	binary.LittleEndian.PutUint32(flush[fo+4:fo+8], 0) // padding
	if _, err := fb.gpu.sendCommand(flush); err != nil {
		return err
	}
	return nil
}

// putRect encodes a virtio_gpu_rect (le32 x, y, width, height) into the
// first 16 bytes of dst.
func putRect(dst []byte, x, y, width, height uint32) {
	binary.LittleEndian.PutUint32(dst[0:4], x)
	binary.LittleEndian.PutUint32(dst[4:8], y)
	binary.LittleEndian.PutUint32(dst[8:12], width)
	binary.LittleEndian.PutUint32(dst[12:16], height)
}

// sendCommand sends one control command over controlq as a 2-descriptor
// chain — request (read-only) + response (device-writable) — rings the
// doorbell, busy-polls for completion, checks the response hdr.type, and
// returns a copy of the response bytes on success.
//
// The request and response share one allocated page (request at offset 0,
// response at offset 2048); the driver holds the page []byte so it reads
// the response directly without unsafe.
func (g *VirtioGPU) sendCommand(req []byte) ([]byte, error) {
	phys, mem, err := g.transport.AllocatePages(1)
	if err != nil {
		return nil, err
	}
	if phys == 0 {
		return nil, common.ErrAllocReturnedZero
	}
	copy(mem[requestOffset:], req)

	reqPhys := phys + requestOffset
	respPhys := phys + responseOffset
	chain := []common.ChainBuffer{
		{Addr: uintptr(reqPhys), Phys: reqPhys, Len: uint32(len(req)), Writable: false},
		{Addr: uintptr(respPhys), Phys: respPhys, Len: responseOffset, Writable: true},
	}

	head, err := g.controlq.AddChain(chain)
	if err != nil {
		return nil, err
	}
	if err := g.Cfg.NotifyQueue(ControlQueueIdx, g.controlq.NotifyOff); err != nil {
		return nil, err
	}

	respBuf := mem[responseOffset:]
	for spin := 0; spin < CommandPollIterations; spin++ {
		_, _, ok := g.controlq.PollUsed()
		if !ok {
			continue
		}
		_ = g.controlq.ReclaimChain(head)
		respType := binary.LittleEndian.Uint32(respBuf[0:4])
		if respType == respOKNoData || respType == respOKDisplayInfo {
			out := make([]byte, len(respBuf))
			copy(out, respBuf)
			return out, nil
		}
		return nil, ErrGPUCommandFailed
	}
	_ = g.controlq.ReclaimChain(head)
	return nil, ErrRequestTimeout
}

// Sentinel errors for the virtio-gpu path.
var (
	ErrNotModernDevice     = commonGPUError("go-virtio/gpu: device doesn't offer VIRTIO_F_VERSION_1 (legacy-only)")
	ErrFeaturesNotOK       = commonGPUError("go-virtio/gpu: FEATURES_OK status bit didn't stick after DriverFeature write")
	ErrInitWrongDeviceID   = commonGPUError("go-virtio/gpu: PCI device ID is not 0x1050 (modern GPU device)")
	ErrQueueNotAvailable   = commonGPUError("go-virtio/gpu: device reports QueueSize=0 for a required queue")
	ErrRequestTimeout      = commonGPUError("go-virtio/gpu: command poll timeout (device did not complete the command)")
	ErrGPUCommandFailed    = commonGPUError("go-virtio/gpu: device returned an error response to a control command")
	ErrNoScanout           = commonGPUError("go-virtio/gpu: no usable scanout (none enabled, or scanout id out of range)")
	ErrFramebufferTooLarge = commonGPUError("go-virtio/gpu: framebuffer dimensions are zero or exceed the supported bound")
)

// commonGPUError is the package's tiny sentinel-error type.
type commonGPUError string

func (e commonGPUError) Error() string { return string(e) }
