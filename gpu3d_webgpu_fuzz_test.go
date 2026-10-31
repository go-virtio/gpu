package gpu

import (
	"testing"

	"github.com/go-virtio/gpu/wgpubatch"
)

// FuzzCapsetResponses feeds arbitrary device responses to the capset
// decoders: they must never panic, never return capset data longer than
// the announced max_size, and an accepted webgpu-batch blob must satisfy
// the invariants DecodeCaps promises.
func FuzzCapsetResponses(f *testing.F) {
	ok := append(make([]byte, ctrlHdrSize), testCaps().Encode()...)
	le.PutUint32(ok[0:], RespOKCapset)
	f.Add(ok, uint32(wgpubatch.CapsSize))
	info := make([]byte, ctrlHdrSize+16)
	le.PutUint32(info[0:], RespOKCapsetInfo)
	f.Add(info, uint32(0))
	f.Add([]byte{}, uint32(1<<31))
	f.Fuzz(func(t *testing.T, resp []byte, maxSize uint32) {
		DecodeRespCapsetInfo(resp)
		if data, ok := DecodeRespCapset(resp, maxSize); ok && uint64(len(data)) > uint64(maxSize) {
			t.Fatalf("%d bytes past max_size %d", len(data), maxSize)
		}
		c, err := DecodeWebGPUBatchCaps(resp, CapsetInfo{ID: CapsetWebGPUBatch, MaxSize: maxSize})
		if err != nil {
			return
		}
		if c.Version == 0 || c.MaxRings == 0 || c.MaxRings > wgpubatch.MaxRings ||
			c.MaxFrame < wgpubatch.FrameHeaderSize || c.MaxObjects < 2 {
			t.Fatalf("accepted caps break an invariant: %+v", c)
		}
	})
}
