// webgpu-batch context primitives: the guest side of the private virtio-gpu
// context type that carries go-gfx/webgpu Recorder streams to a host that
// replays them on WebGPU (see docs/WEBGPU.md and package wgpubatch).
//
// Like gpu3d_blob.go these are pure byte builders and parsers, asserted
// byte-for-byte in tests. No host implements this capset yet, so nothing
// here has been validated against a device.
//
// References:
//
//   - Linux uapi include/uapi/linux/virtio_gpu.h: VIRTIO_GPU_FLAG_FENCE /
//     FLAG_INFO_RING_IDX (:143), ctrl_hdr.ring_idx (:150), GET_CAPSET_INFO /
//     GET_CAPSET and their responses (:325-352), virtio_gpu_cmd_submit (:312).
//   - virtio spec device-types/gpu/description.tex :255-297 (ctrl_hdr,
//     ring_idx 0..63, fence timelines), :523-582 (capset queries).
package gpu

import (
	"encoding/binary"
	"math"

	"github.com/go-virtio/common"
	"github.com/go-virtio/gpu/wgpubatch"
)

// CapsetWebGPUBatch is the private webgpu-batch capset id (30), defined
// by package wgpubatch. It is NOT a spec-assigned id; see docs/WEBGPU.md.
const CapsetWebGPUBatch = wgpubatch.CapsetID

// Capset query command and response codes (virtio_gpu.h enum
// virtio_gpu_ctrl_type: 9th and 10th 2D entries; 3rd and 4th OK responses).
const (
	CmdGetCapsetInfo uint32 = 0x0108
	CmdGetCapset     uint32 = 0x0109
	RespOKCapsetInfo uint32 = 0x1102
	RespOKCapset     uint32 = 0x1103
)

// Control-header flags (virtio_gpu.h).
const (
	FlagFence       uint32 = 1 << 0
	FlagInfoRingIdx uint32 = 1 << 1
)

// hdrRingIdxOffset is ctrl_hdr.ring_idx (u8 after le32 ctx_id@16).
const hdrRingIdxOffset = 20

// AcceptedFeaturesWebGPUBatch is what a webgpu-batch bring-up negotiates:
// VERSION_1, VIRGL (the "3D available" gate hosts advertise) and
// CONTEXT_INIT (to select the capset). RESOURCE_BLOB is only needed by the
// v2 ring transport, so it is requested but not required.
const AcceptedFeaturesWebGPUBatch uint64 = common.FeatureVersion1 |
	FeatureVirgl | FeatureContextInit | FeatureResourceBlob

// NegotiateWebGPUBatchFeatures returns the subset of
// AcceptedFeaturesWebGPUBatch the device offers, and whether the required
// set (VERSION_1 + VIRGL + CONTEXT_INIT) is present. Pure mask arithmetic.
func NegotiateWebGPUBatchFeatures(deviceFeatures uint64) (negotiated uint64, ok bool) {
	negotiated = deviceFeatures & AcceptedFeaturesWebGPUBatch
	ok = negotiated&common.FeatureVersion1 != 0 &&
		negotiated&FeatureVirgl != 0 &&
		negotiated&FeatureContextInit != 0
	return negotiated, ok
}

// EncodeCtxCreateWebGPUBatch builds the 96-byte CTX_CREATE request of a
// webgpu-batch context: context_init = CapsetWebGPUBatch.
func EncodeCtxCreateWebGPUBatch(ctxID uint32, debugName string) []byte {
	return encodeCtxCreate(ctxID, CapsetWebGPUBatch, debugName)
}

// putFencedHdr fills a ctrl_hdr in b[0:24]. A fenced command sets
// FLAG_FENCE and fence_id; a ringed one also sets FLAG_INFO_RING_IDX and
// ring_idx, making fenceID a sequence number on the (ctx, ring) timeline.
func putFencedHdr(b []byte, typ, ctx uint32, fence Fence) error {
	var flags uint32
	if fence.On {
		flags |= FlagFence
		binary.LittleEndian.PutUint64(b[8:16], fence.ID)
	}
	if fence.Ringed {
		if fence.Ring >= wgpubatch.MaxRings {
			return ErrRingIndex
		}
		flags |= FlagInfoRingIdx
		b[hdrRingIdxOffset] = fence.Ring
	}
	binary.LittleEndian.PutUint32(b[0:4], typ)
	binary.LittleEndian.PutUint32(b[4:8], flags)
	binary.LittleEndian.PutUint32(b[hdrCtxIDOffset:hdrCtxIDOffset+4], ctx)
	return nil
}

// Fence selects the fencing of a command: On sets VIRTIO_GPU_FLAG_FENCE
// with ID; Ringed sets VIRTIO_GPU_FLAG_INFO_RING_IDX with Ring (0..63).
type Fence struct {
	On     bool
	ID     uint64
	Ringed bool
	Ring   uint8
}

// ErrRingIndex: a ring index above 63 (description.tex:289).
var ErrRingIndex = commonGPUError("go-virtio/gpu: ring index above 63")

// EncodeSubmit3DHeader builds the 32-byte virtio_gpu_cmd_submit that
// precedes a SUBMIT_3D payload of size bytes (a wgpubatch frame), with the
// given fencing. The payload rides in its own descriptor(s).
func EncodeSubmit3DHeader(ctx uint32, size uint32, fence Fence) ([]byte, error) {
	b := make([]byte, ctrlHdrSize+8)
	if err := putFencedHdr(b, CmdSubmit3D, ctx, fence); err != nil {
		return nil, err
	}
	binary.LittleEndian.PutUint32(b[ctrlHdrSize:], size)
	return b, nil
}

// EncodeGetCapsetInfo builds GET_CAPSET_INFO for capset_index (which must
// be below the config's num_capsets).
func EncodeGetCapsetInfo(index uint32) []byte {
	b := make([]byte, ctrlHdrSize+8)
	binary.LittleEndian.PutUint32(b[0:4], CmdGetCapsetInfo)
	binary.LittleEndian.PutUint32(b[ctrlHdrSize:], index)
	return b
}

// CapsetInfo is a RESP_OK_CAPSET_INFO body.
type CapsetInfo struct {
	ID, MaxVersion, MaxSize uint32
}

// DecodeRespCapsetInfo parses RESP_OK_CAPSET_INFO (ctrl_hdr + le32 id,
// max_version, max_size, padding).
func DecodeRespCapsetInfo(resp []byte) (CapsetInfo, bool) {
	if len(resp) < ctrlHdrSize+16 || binary.LittleEndian.Uint32(resp[0:4]) != RespOKCapsetInfo {
		return CapsetInfo{}, false
	}
	o := resp[ctrlHdrSize:]
	return CapsetInfo{
		ID:         binary.LittleEndian.Uint32(o[0:]),
		MaxVersion: binary.LittleEndian.Uint32(o[4:]),
		MaxSize:    binary.LittleEndian.Uint32(o[8:]),
	}, true
}

// EncodeGetCapset builds GET_CAPSET for (id, version).
func EncodeGetCapset(id, version uint32) []byte {
	b := make([]byte, ctrlHdrSize+8)
	binary.LittleEndian.PutUint32(b[0:4], CmdGetCapset)
	binary.LittleEndian.PutUint32(b[ctrlHdrSize:], id)
	binary.LittleEndian.PutUint32(b[ctrlHdrSize+4:], version)
	return b
}

// DecodeRespCapset returns the capset_data of RESP_OK_CAPSET, at most
// maxSize bytes (the CapsetInfo.MaxSize the device announced): a device
// cannot hand the guest more than it promised. The slice aliases resp.
func DecodeRespCapset(resp []byte, maxSize uint32) ([]byte, bool) {
	if len(resp) < ctrlHdrSize || binary.LittleEndian.Uint32(resp[0:4]) != RespOKCapset {
		return nil, false
	}
	data := resp[ctrlHdrSize:]
	if uint64(len(data)) > uint64(maxSize) {
		data = data[:maxSize]
	}
	return data, true
}

// DecodeWebGPUBatchCaps reads the webgpu-batch capset blob from a
// RESP_OK_CAPSET response.
func DecodeWebGPUBatchCaps(resp []byte, info CapsetInfo) (wgpubatch.Caps, error) {
	if info.ID != CapsetWebGPUBatch {
		return wgpubatch.Caps{}, ErrNotWebGPUBatch
	}
	data, ok := DecodeRespCapset(resp, info.MaxSize)
	if !ok {
		return wgpubatch.Caps{}, ErrGPUCommandFailed
	}
	return wgpubatch.DecodeCaps(data)
}

// Errors of the webgpu-batch path.
var (
	ErrNotWebGPUBatch         = commonGPUError("go-virtio/gpu: capset is not webgpu-batch")
	ErrWebGPUBatchUnavailable = commonGPUError("go-virtio/gpu: device lacks VIRGL or CONTEXT_INIT, or does not offer the webgpu-batch capset")
	ErrFenceNotEchoed         = commonGPUError("go-virtio/gpu: fenced response without VIRTIO_GPU_FLAG_FENCE and the request's fence_id")
	ErrBadContext             = commonGPUError("go-virtio/gpu: context id 0")
	ErrEmptyFrame             = commonGPUError("go-virtio/gpu: empty webgpu-batch frame")
)

// OpenVirtioGPUWebGPUBatch brings the device up negotiating
// AcceptedFeaturesWebGPUBatch and fails unless VIRGL and CONTEXT_INIT
// were granted. It does not yet query the capset (WebGPUBatchCaps does).
func OpenVirtioGPUWebGPUBatch(t common.Transport) (*VirtioGPU, error) {
	g, err := bringUp(t, AcceptedFeaturesWebGPUBatch, ErrWebGPUBatchUnavailable)
	if err != nil {
		return nil, err
	}
	if _, ok := NegotiateWebGPUBatchFeatures(g.NegotiatedFeatures); !ok {
		return nil, ErrWebGPUBatchUnavailable
	}
	return g, nil
}

// WebGPUBatchCaps walks the device's capsets (num_capsets from the device
// config, GET_CAPSET_INFO per index) and returns the webgpu-batch capset
// blob, or ErrWebGPUBatchUnavailable when the device does not offer it.
func (g *VirtioGPU) WebGPUBatchCaps() (wgpubatch.Caps, error) {
	n, err := g.Cfg.DeviceCfgRead32(cfgNumCapsets)
	if err != nil {
		return wgpubatch.Caps{}, err
	}
	// A hostile num_capsets costs one command per index; bound it by the
	// id space (8-bit context_init field), well above any real device.
	n = min(n, 256)
	for i := uint32(0); i < n; i++ {
		resp, err := g.sendCommand(EncodeGetCapsetInfo(i))
		if err != nil {
			return wgpubatch.Caps{}, err
		}
		info, ok := DecodeRespCapsetInfo(resp)
		if !ok {
			return wgpubatch.Caps{}, ErrGPUCommandFailed
		}
		if info.ID != CapsetWebGPUBatch {
			continue
		}
		resp, err = g.sendCommand(EncodeGetCapset(info.ID, min(info.MaxVersion, wgpubatch.Version)))
		if err != nil {
			return wgpubatch.Caps{}, err
		}
		return DecodeWebGPUBatchCaps(resp, info)
	}
	return wgpubatch.Caps{}, ErrWebGPUBatchUnavailable
}

// CreateWebGPUBatchContext creates context ctxID with context_init =
// CapsetWebGPUBatch. It needs a device opened with
// OpenVirtioGPUWebGPUBatch (CONTEXT_INIT negotiated): without it the
// device would ignore context_init and create a default context.
func (g *VirtioGPU) CreateWebGPUBatchContext(ctxID uint32, debugName string) error {
	if ctxID == 0 {
		return ErrBadContext
	}
	if g.NegotiatedFeatures&FeatureContextInit == 0 {
		return ErrWebGPUBatchUnavailable
	}
	_, err := g.sendCommand(EncodeCtxCreateWebGPUBatch(ctxID, debugName))
	return err
}

// Layout of a SubmitWebGPUBatch chain inside the cached batch pages:
// submit header at 0, frame at batchFrameOffset, response after the frame
// on the next 64-byte boundary.
const (
	batchFrameOffset = 64
	batchRespSize    = 64
)

// maxBatchFrame bounds a frame so that every descriptor length and the
// submit header's size field fit in 32 bits. A variable so a test can
// lower it.
var maxBatchFrame uint64 = math.MaxUint32 - batchFrameOffset - 2*batchRespSize

// ErrFrameTooLarge: a frame above maxBatchFrame.
var ErrFrameTooLarge = commonGPUError("go-virtio/gpu: webgpu-batch frame too large for one SUBMIT_3D")

// SubmitWebGPUBatch sends one wgpubatch frame as the payload of SUBMIT_3D
// on context ctxID, as a 3-descriptor chain (header RO, frame RO, response
// W) like submit3D, and waits for the response. With fence.On the device
// answers only once the batch has executed (description.tex:173-179) and
// must echo FLAG_FENCE and fence_id, which is checked. The frame is
// copied into driver-owned pages first, so the caller may reuse it.
func (g *VirtioGPU) SubmitWebGPUBatch(ctxID uint32, frame []byte, fence Fence) error {
	if ctxID == 0 {
		return ErrBadContext
	}
	if len(frame) == 0 {
		return ErrEmptyFrame
	}
	if uint64(len(frame)) > maxBatchFrame {
		return ErrFrameTooLarge
	}
	hdr, err := EncodeSubmit3DHeader(ctxID, uint32(len(frame)), fence)
	if err != nil {
		return err
	}
	respOff := (batchFrameOffset + uint64(len(frame)) + 63) &^ 63
	need := respOff + batchRespSize
	if uint64(len(g.batchMem)) < need {
		ps := uint64(common.PageSize)
		pages := int((need + ps - 1) / ps)
		phys, mem, err := g.transport.AllocatePages(pages)
		if err != nil {
			return err
		}
		if phys == 0 {
			return common.ErrAllocReturnedZero
		}
		g.batchPhys, g.batchMem = phys, mem
	}
	mem := g.batchMem
	copy(mem[0:], hdr)
	copy(mem[batchFrameOffset:], frame)
	resp := mem[respOff : respOff+batchRespSize]
	clear(resp)

	p := g.batchPhys
	chain := []common.ChainBuffer{
		{Addr: uintptr(p), Phys: p, Len: uint32(len(hdr))},
		{Addr: uintptr(p + batchFrameOffset), Phys: p + batchFrameOffset, Len: uint32(len(frame))},
		{Addr: uintptr(p + respOff), Phys: p + respOff, Len: batchRespSize, Writable: true},
	}
	// Late completions of a timed-out command are drained first, as in
	// sendCommand (R-doom1f).
	for drain := 0; drain < int(ControlQueueSize); drain++ {
		if _, _, ok := g.controlq.PollUsed(); !ok {
			break
		}
	}
	head, err := g.controlq.AddChain(chain)
	if err != nil {
		return err
	}
	if err := g.Cfg.NotifyQueue(ControlQueueIdx, g.controlq.NotifyOff); err != nil {
		return err
	}
	for spin := 0; spin < CommandPollIterations; spin++ {
		descIdx, _, ok := g.controlq.PollUsed()
		if !ok || descIdx != head {
			continue
		}
		_ = g.controlq.ReclaimChain(head)
		if binary.LittleEndian.Uint32(resp[0:4]) != respOKNoData {
			return ErrGPUCommandFailed
		}
		if fence.On && (binary.LittleEndian.Uint32(resp[4:8])&FlagFence == 0 ||
			binary.LittleEndian.Uint64(resp[8:16]) != fence.ID) {
			return ErrFenceNotEchoed
		}
		return nil
	}
	_ = g.controlq.ReclaimChain(head)
	return ErrRequestTimeout
}
