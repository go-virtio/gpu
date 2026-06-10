// Blob-resource + context-init primitives for the virtio-gpu 3D path
// (Milestone toward Venus). This file is PURELY additive: it adds the
// guest-side WIRE ENCODINGS for the blob-resource control commands and the
// context_init-bearing CTX_CREATE, plus a Venus-specific feature mask. It
// does NOT add any host round-trip method — every function here is a pure
// byte-builder whose output is asserted byte-for-byte in the unit tests.
//
// Why only encoders, no host calls: the blob path's RUNTIME behaviour
// (whether a HOST3D blob actually becomes guest-mappable, what map_info
// caching type the host returns, whether the Venus ring goes live) is
// host/renderer-dependent and is NOT verifiable on this machine without a
// Venus-capable renderer. The structs below, by contrast, are fixed
// little-endian wire layouts defined in the kernel uapi and are fully
// verifiable offline. See README / the venus repo for the host-dependent
// boundary.
//
// References (every constant/offset below is transcribed from these):
//
//   - Linux uapi include/uapi/linux/virtio_gpu.h:
//   - enum virtio_gpu_ctrl_type — command codes (sequential).
//   - struct virtio_gpu_resource_create_blob.
//   - struct virtio_gpu_resource_map_blob.
//   - struct virtio_gpu_resp_map_info.
//   - struct virtio_gpu_ctx_create (nlen, context_init, debug_name[64]).
//   - VIRTIO_GPU_BLOB_MEM_* / VIRTIO_GPU_BLOB_FLAG_* constants.
//   - VIRTIO_GPU_F_RESOURCE_BLOB (3) / VIRTIO_GPU_F_CONTEXT_INIT (4).
//   - VIRTIO_GPU_CAPSET_VENUS (4).
package gpu

import (
	"encoding/binary"

	"github.com/go-virtio/common"
)

// --- Feature bits (virtio_gpu.h #define VIRTIO_GPU_F_*) -----------------

// FeatureResourceBlob is VIRTIO_GPU_F_RESOURCE_BLOB = feature bit index 3:
// the host supports blob resources (RESOURCE_CREATE_BLOB / *_MAP_BLOB).
// Required by the Venus transport (the command ring is a host-visible
// blob).
const FeatureResourceBlob uint64 = 1 << 3

// FeatureContextInit is VIRTIO_GPU_F_CONTEXT_INIT = feature bit index 4:
// the host honours ctx_create.context_init (the capset selector). Required
// to stand up a Venus (capset 4) context rather than a default virgl one.
const FeatureContextInit uint64 = 1 << 4

// AcceptedFeaturesVenus is the feature mask a Venus-capable bring-up
// negotiates ON: VERSION_1 (non-negotiable) plus VIRGL (bit 0, the virtio
// "3D available" gate the device still advertises), RESOURCE_BLOB (bit 3)
// and CONTEXT_INIT (bit 4). All three of VIRGL/BLOB/CONTEXT_INIT are
// needed before a context_init=VENUS context will come up.
const AcceptedFeaturesVenus uint64 = common.FeatureVersion1 |
	FeatureVirgl | FeatureResourceBlob | FeatureContextInit

// --- Capset selector (virtio_gpu.h #define VIRTIO_GPU_CAPSET_*) --------

// CapsetVirgl / CapsetVirgl2 / CapsetVenus are the context_init capset ids.
// Venus (4) selects the Vulkan-over-virtio protocol for a context created
// with CTX_CREATE.context_init = CapsetVenus.
const (
	CapsetVirgl  uint32 = 1
	CapsetVirgl2 uint32 = 2
	CapsetVenus  uint32 = 4
)

// --- Blob command codes (virtio_gpu.h enum virtio_gpu_ctrl_type) ------
//
// Derived by counting the sequential enum from the explicit anchors. The
// 2D block anchors at 0x0100; RESOURCE_CREATE_BLOB is the 13th 2D entry
// (0-based +0x0C). The 3D block anchors at 0x0200; RESOURCE_MAP_BLOB /
// UNMAP_BLOB follow SUBMIT_3D (0x0207) at +8/+9. RESP_OK_MAP_INFO is the
// 7th success response after the 0x1100 anchor (+0x06).
const (
	// VIRTIO_GPU_CMD_RESOURCE_CREATE_BLOB (2D block, 13th entry).
	CmdResourceCreateBlob uint32 = 0x010C
	// VIRTIO_GPU_CMD_SET_SCANOUT_BLOB (2D block, 14th entry) — included
	// for completeness; not used by the Venus ring path.
	CmdSetScanoutBlob uint32 = 0x010D
	// VIRTIO_GPU_CMD_RESOURCE_MAP_BLOB (3D block, after SUBMIT_3D).
	CmdResourceMapBlob uint32 = 0x0208
	// VIRTIO_GPU_CMD_RESOURCE_UNMAP_BLOB (3D block).
	CmdResourceUnmapBlob uint32 = 0x0209
	// VIRTIO_GPU_RESP_OK_MAP_INFO — the success response carrying
	// virtio_gpu_resp_map_info (map_info caching type).
	RespOKMapInfo uint32 = 0x1106
)

// --- Blob memory + flag constants (virtio_gpu.h) ----------------------

// Blob memory types (VIRTIO_GPU_BLOB_MEM_*). HOST3D is the type used for a
// Venus command ring (host-allocated, host-visible, then guest-mapped via
// RESOURCE_MAP_BLOB).
const (
	BlobMemGuest       uint32 = 0x0001
	BlobMemHost3D      uint32 = 0x0002
	BlobMemHost3DGuest uint32 = 0x0003
)

// Blob usage flags (VIRTIO_GPU_BLOB_FLAG_USE_*). USE_MAPPABLE is required
// for a ring blob the guest will mmap.
const (
	BlobFlagUseMappable    uint32 = 0x0001
	BlobFlagUseShareable   uint32 = 0x0002
	BlobFlagUseCrossDevice uint32 = 0x0004
)

// resourceCreateBlobSize is the on-the-wire size of struct
// virtio_gpu_resource_create_blob: ctrl_hdr(24) + le32 resource_id + le32
// blob_mem + le32 blob_flags + le32 nr_entries + le64 blob_id + le64 size
// = 24 + 4+4+4+4 + 8 + 8 = 56. (mem_entry array, if nr_entries>0, follows.)
const resourceCreateBlobSize = ctrlHdrSize + 32

// resourceMapBlobSize is the on-the-wire size of struct
// virtio_gpu_resource_map_blob: ctrl_hdr(24) + le32 resource_id + le32
// padding + le64 offset = 24 + 4 + 4 + 8 = 40.
const resourceMapBlobSize = ctrlHdrSize + 16

// respMapInfoSize is the on-the-wire size of struct
// virtio_gpu_resp_map_info: ctrl_hdr(24) + le32 map_info + le32 padding.
const respMapInfoSize = ctrlHdrSize + 8

// BlobCreateParams describes a RESOURCE_CREATE_BLOB request. For a Venus
// command-ring blob the host3d case is used (BlobMem=HOST3D,
// Flags=USE_MAPPABLE, BlobID host-meaningful, NrEntries=0 so no guest
// mem_entry array follows).
type BlobCreateParams struct {
	CtxID      uint32
	ResourceID uint32
	BlobMem    uint32
	BlobFlags  uint32
	NrEntries  uint32
	BlobID     uint64
	Size       uint64
}

// EncodeResourceCreateBlob builds the 56-byte RESOURCE_CREATE_BLOB request
// (without any trailing mem_entry array; NrEntries is encoded but the
// caller appends entries only when BlobMemGuest is used). Field order
// per struct virtio_gpu_resource_create_blob.
//
// Layout (offsets from start of struct):
//
//	hdr.type      @0   = CmdResourceCreateBlob
//	hdr.ctx_id    @16  = CtxID
//	resource_id   @24
//	blob_mem      @28
//	blob_flags    @32
//	nr_entries    @36
//	blob_id       @40  (le64)
//	size          @48  (le64)
func EncodeResourceCreateBlob(p BlobCreateParams) []byte {
	b := make([]byte, resourceCreateBlobSize)
	binary.LittleEndian.PutUint32(b[0:4], CmdResourceCreateBlob)
	binary.LittleEndian.PutUint32(b[hdrCtxIDOffset:hdrCtxIDOffset+4], p.CtxID)
	o := ctrlHdrSize
	binary.LittleEndian.PutUint32(b[o+0:o+4], p.ResourceID)
	binary.LittleEndian.PutUint32(b[o+4:o+8], p.BlobMem)
	binary.LittleEndian.PutUint32(b[o+8:o+12], p.BlobFlags)
	binary.LittleEndian.PutUint32(b[o+12:o+16], p.NrEntries)
	binary.LittleEndian.PutUint64(b[o+16:o+24], p.BlobID)
	binary.LittleEndian.PutUint64(b[o+24:o+32], p.Size)
	return b
}

// EncodeResourceMapBlob builds the 40-byte RESOURCE_MAP_BLOB request.
// Field order per struct virtio_gpu_resource_map_blob.
//
// Layout:
//
//	hdr.type     @0   = CmdResourceMapBlob
//	hdr.ctx_id   @16  = ctxID
//	resource_id  @24
//	padding      @28  = 0
//	offset       @32  (le64) — the host shmem window offset to map at.
func EncodeResourceMapBlob(ctxID, resourceID uint32, offset uint64) []byte {
	b := make([]byte, resourceMapBlobSize)
	binary.LittleEndian.PutUint32(b[0:4], CmdResourceMapBlob)
	binary.LittleEndian.PutUint32(b[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	o := ctrlHdrSize
	binary.LittleEndian.PutUint32(b[o+0:o+4], resourceID)
	binary.LittleEndian.PutUint32(b[o+4:o+8], 0) // padding
	binary.LittleEndian.PutUint64(b[o+8:o+16], offset)
	return b
}

// DecodeRespMapInfo extracts the map_info caching type from a
// VIRTIO_GPU_RESP_OK_MAP_INFO response (struct virtio_gpu_resp_map_info).
// It returns (map_info, true) iff resp is at least respMapInfoSize bytes
// and hdr.type == RespOKMapInfo. The map_info value's MEANING (the host's
// chosen caching mode) is host-defined and not interpreted here.
func DecodeRespMapInfo(resp []byte) (mapInfo uint32, ok bool) {
	if len(resp) < respMapInfoSize {
		return 0, false
	}
	if binary.LittleEndian.Uint32(resp[0:4]) != RespOKMapInfo {
		return 0, false
	}
	return binary.LittleEndian.Uint32(resp[ctrlHdrSize : ctrlHdrSize+4]), true
}

// EncodeCtxCreateVenus builds the 96-byte CTX_CREATE request for a Venus
// context: context_init = CapsetVenus (4). Struct virtio_gpu_ctx_create:
// ctrl_hdr(24) + le32 nlen@24 + le32 context_init@28 + char debug_name[64]@32.
//
// debugName is copied into debug_name[64] (truncated/zero-padded to 64),
// and nlen is set to its byte length (capped at 64), mirroring how the
// kernel's virtio-gpu DRM ioctl populates the field from a context name.
// For an anonymous Venus context pass debugName="" (nlen=0).
func EncodeCtxCreateVenus(ctxID uint32, debugName string) []byte {
	return encodeCtxCreate(ctxID, CapsetVenus, debugName)
}

// encodeCtxCreate is the general CTX_CREATE encoder with an explicit
// context_init capset selector (0 = legacy/virgl default).
func encodeCtxCreate(ctxID, contextInit uint32, debugName string) []byte {
	b := make([]byte, ctrlHdrSize+72) // 24 + 4 + 4 + 64 = 96
	binary.LittleEndian.PutUint32(b[0:4], CmdCtxCreate)
	binary.LittleEndian.PutUint32(b[hdrCtxIDOffset:hdrCtxIDOffset+4], ctxID)
	nlen := len(debugName)
	if nlen > 64 {
		nlen = 64
	}
	binary.LittleEndian.PutUint32(b[ctrlHdrSize+0:ctrlHdrSize+4], uint32(nlen)) // nlen@24
	binary.LittleEndian.PutUint32(b[ctrlHdrSize+4:ctrlHdrSize+8], contextInit)  // context_init@28
	copy(b[ctrlHdrSize+8:ctrlHdrSize+8+64], debugName)                          // debug_name[64]@32
	return b
}

// NegotiateVenusFeatures returns the subset of AcceptedFeaturesVenus the
// device actually offers, AND a bool reporting whether the full Venus
// prerequisite set (VERSION_1 + VIRGL + RESOURCE_BLOB + CONTEXT_INIT) is
// present. A Venus context CANNOT be stood up unless ok is true. This is a
// pure mask computation; it performs no device I/O.
func NegotiateVenusFeatures(deviceFeatures uint64) (negotiated uint64, ok bool) {
	negotiated = deviceFeatures & AcceptedFeaturesVenus
	ok = negotiated&common.FeatureVersion1 != 0 &&
		negotiated&FeatureVirgl != 0 &&
		negotiated&FeatureResourceBlob != 0 &&
		negotiated&FeatureContextInit != 0
	return negotiated, ok
}
