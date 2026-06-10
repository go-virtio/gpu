// Byte-exact unit tests for the blob-resource + Venus context_init wire
// encodings. Every expected value here is cross-checked against the kernel
// uapi struct layouts cited in gpu3d_blob.go — these are offline-verifiable
// fixed wire structs, so the tests assert the FULL byte slice, not just a
// length.

package gpu

import (
	"encoding/binary"
	"testing"

	"github.com/go-virtio/common"
)

func le32(b []byte, off int) uint32 { return binary.LittleEndian.Uint32(b[off : off+4]) }
func le64(b []byte, off int) uint64 { return binary.LittleEndian.Uint64(b[off : off+8]) }

// --- command-code + constant sanity -----------------------------------

func TestBlobConstants(t *testing.T) {
	cases := []struct {
		name string
		got  uint32
		want uint32
	}{
		{"CmdResourceCreateBlob", CmdResourceCreateBlob, 0x010C},
		{"CmdSetScanoutBlob", CmdSetScanoutBlob, 0x010D},
		{"CmdResourceMapBlob", CmdResourceMapBlob, 0x0208},
		{"CmdResourceUnmapBlob", CmdResourceUnmapBlob, 0x0209},
		{"RespOKMapInfo", RespOKMapInfo, 0x1106},
		{"BlobMemGuest", BlobMemGuest, 0x0001},
		{"BlobMemHost3D", BlobMemHost3D, 0x0002},
		{"BlobMemHost3DGuest", BlobMemHost3DGuest, 0x0003},
		{"BlobFlagUseMappable", BlobFlagUseMappable, 0x0001},
		{"BlobFlagUseShareable", BlobFlagUseShareable, 0x0002},
		{"BlobFlagUseCrossDevice", BlobFlagUseCrossDevice, 0x0004},
		{"CapsetVirgl", CapsetVirgl, 1},
		{"CapsetVirgl2", CapsetVirgl2, 2},
		{"CapsetVenus", CapsetVenus, 4},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got 0x%x, want 0x%x", c.name, c.got, c.want)
		}
	}
}

func TestVenusFeatureBits(t *testing.T) {
	if FeatureResourceBlob != 1<<3 {
		t.Errorf("FeatureResourceBlob: got 0x%x, want 0x8", FeatureResourceBlob)
	}
	if FeatureContextInit != 1<<4 {
		t.Errorf("FeatureContextInit: got 0x%x, want 0x10", FeatureContextInit)
	}
	want := common.FeatureVersion1 | FeatureVirgl | FeatureResourceBlob | FeatureContextInit
	if AcceptedFeaturesVenus != want {
		t.Errorf("AcceptedFeaturesVenus: got 0x%x, want 0x%x", AcceptedFeaturesVenus, want)
	}
}

// --- RESOURCE_CREATE_BLOB byte layout ---------------------------------

func TestEncodeResourceCreateBlob(t *testing.T) {
	p := BlobCreateParams{
		CtxID:      1,
		ResourceID: 2,
		BlobMem:    BlobMemHost3D,
		BlobFlags:  BlobFlagUseMappable,
		NrEntries:  0,
		BlobID:     0xCAFEF00D,
		Size:       0x1000,
	}
	b := EncodeResourceCreateBlob(p)
	if len(b) != 56 {
		t.Fatalf("length: got %d, want 56", len(b))
	}
	checks := []struct {
		name string
		off  int
		got  uint32
		want uint32
	}{
		{"hdr.type", 0, le32(b, 0), CmdResourceCreateBlob},
		{"hdr.ctx_id", 16, le32(b, 16), 1},
		{"resource_id", 24, le32(b, 24), 2},
		{"blob_mem", 28, le32(b, 28), BlobMemHost3D},
		{"blob_flags", 32, le32(b, 32), BlobFlagUseMappable},
		{"nr_entries", 36, le32(b, 36), 0},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s@%d: got 0x%x, want 0x%x", c.name, c.off, c.got, c.want)
		}
	}
	if le64(b, 40) != 0xCAFEF00D {
		t.Errorf("blob_id@40: got 0x%x, want 0xCAFEF00D", le64(b, 40))
	}
	if le64(b, 48) != 0x1000 {
		t.Errorf("size@48: got 0x%x, want 0x1000", le64(b, 48))
	}
	// hdr.flags@4, fence_id@8, padding@20 must be zero.
	for _, off := range []int{4, 8, 12, 20} {
		if le32(b, off) != 0 {
			t.Errorf("reserved hdr byte @%d nonzero: 0x%x", off, le32(b, off))
		}
	}
}

// --- RESOURCE_MAP_BLOB byte layout ------------------------------------

func TestEncodeResourceMapBlob(t *testing.T) {
	b := EncodeResourceMapBlob(1, 2, 0xDEAD0000)
	if len(b) != 40 {
		t.Fatalf("length: got %d, want 40", len(b))
	}
	if le32(b, 0) != CmdResourceMapBlob {
		t.Errorf("hdr.type: got 0x%x, want 0x%x", le32(b, 0), CmdResourceMapBlob)
	}
	if le32(b, 16) != 1 {
		t.Errorf("hdr.ctx_id: got %d, want 1", le32(b, 16))
	}
	if le32(b, 24) != 2 {
		t.Errorf("resource_id: got %d, want 2", le32(b, 24))
	}
	if le32(b, 28) != 0 {
		t.Errorf("padding@28: got 0x%x, want 0", le32(b, 28))
	}
	if le64(b, 32) != 0xDEAD0000 {
		t.Errorf("offset@32: got 0x%x, want 0xDEAD0000", le64(b, 32))
	}
}

// --- RESP_OK_MAP_INFO decode ------------------------------------------

func TestDecodeRespMapInfo(t *testing.T) {
	resp := make([]byte, respMapInfoSize)
	binary.LittleEndian.PutUint32(resp[0:4], RespOKMapInfo)
	binary.LittleEndian.PutUint32(resp[ctrlHdrSize:ctrlHdrSize+4], 3) // arbitrary map_info
	mi, ok := DecodeRespMapInfo(resp)
	if !ok || mi != 3 {
		t.Errorf("DecodeRespMapInfo: got (%d,%v), want (3,true)", mi, ok)
	}
}

func TestDecodeRespMapInfo_TooShort(t *testing.T) {
	if _, ok := DecodeRespMapInfo(make([]byte, respMapInfoSize-1)); ok {
		t.Error("expected ok=false for short buffer")
	}
}

func TestDecodeRespMapInfo_WrongType(t *testing.T) {
	resp := make([]byte, respMapInfoSize)
	binary.LittleEndian.PutUint32(resp[0:4], respOKNoData) // not RESP_OK_MAP_INFO
	if _, ok := DecodeRespMapInfo(resp); ok {
		t.Error("expected ok=false for wrong hdr.type")
	}
}

// --- CTX_CREATE (context_init = VENUS) byte layout --------------------

func TestEncodeCtxCreateVenus(t *testing.T) {
	b := EncodeCtxCreateVenus(1, "venus")
	if len(b) != 96 {
		t.Fatalf("length: got %d, want 96", len(b))
	}
	if le32(b, 0) != CmdCtxCreate {
		t.Errorf("hdr.type: got 0x%x, want 0x%x", le32(b, 0), CmdCtxCreate)
	}
	if le32(b, 16) != 1 {
		t.Errorf("hdr.ctx_id: got %d, want 1", le32(b, 16))
	}
	if le32(b, 24) != 5 { // nlen = len("venus")
		t.Errorf("nlen@24: got %d, want 5", le32(b, 24))
	}
	if le32(b, 28) != CapsetVenus {
		t.Errorf("context_init@28: got %d, want %d (VENUS)", le32(b, 28), CapsetVenus)
	}
	if string(b[32:37]) != "venus" {
		t.Errorf("debug_name@32: got %q, want %q", string(b[32:37]), "venus")
	}
	// debug_name is exactly 64 bytes; bytes past the name must be zero.
	for i := 37; i < 96; i++ {
		if b[i] != 0 {
			t.Errorf("debug_name padding byte %d nonzero: 0x%x", i, b[i])
		}
	}
}

func TestEncodeCtxCreateVenus_Anonymous(t *testing.T) {
	b := EncodeCtxCreateVenus(7, "")
	if le32(b, 24) != 0 {
		t.Errorf("nlen for anon: got %d, want 0", le32(b, 24))
	}
	if le32(b, 28) != CapsetVenus {
		t.Errorf("context_init: got %d, want VENUS", le32(b, 28))
	}
	if le32(b, 16) != 7 {
		t.Errorf("ctx_id: got %d, want 7", le32(b, 16))
	}
}

func TestEncodeCtxCreateVenus_NameTruncated(t *testing.T) {
	// A name longer than 64 bytes is truncated; nlen caps at 64.
	long := make([]byte, 100)
	for i := range long {
		long[i] = 'x'
	}
	b := EncodeCtxCreateVenus(1, string(long))
	if le32(b, 24) != 64 {
		t.Errorf("nlen for >64 name: got %d, want 64", le32(b, 24))
	}
	// debug_name[64] fully filled with 'x'.
	for i := 32; i < 96; i++ {
		if b[i] != 'x' {
			t.Errorf("debug_name byte %d: got 0x%x, want 'x'", i, b[i])
		}
	}
}

// --- feature negotiation ----------------------------------------------

func TestNegotiateVenusFeatures_AllPresent(t *testing.T) {
	dev := common.FeatureVersion1 | FeatureVirgl | FeatureResourceBlob | FeatureContextInit | (1 << 20)
	neg, ok := NegotiateVenusFeatures(dev)
	if !ok {
		t.Fatal("expected ok=true when all prereqs present")
	}
	if neg != AcceptedFeaturesVenus {
		t.Errorf("negotiated: got 0x%x, want 0x%x (extraneous bit 20 must be masked off)", neg, AcceptedFeaturesVenus)
	}
}

func TestNegotiateVenusFeatures_Missing(t *testing.T) {
	cases := []struct {
		name string
		dev  uint64
	}{
		{"no VERSION_1", FeatureVirgl | FeatureResourceBlob | FeatureContextInit},
		{"no VIRGL", common.FeatureVersion1 | FeatureResourceBlob | FeatureContextInit},
		{"no RESOURCE_BLOB", common.FeatureVersion1 | FeatureVirgl | FeatureContextInit},
		{"no CONTEXT_INIT", common.FeatureVersion1 | FeatureVirgl | FeatureResourceBlob},
		{"none", 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, ok := NegotiateVenusFeatures(c.dev); ok {
				t.Errorf("%s: expected ok=false", c.name)
			}
		})
	}
}
