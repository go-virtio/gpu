package wgpubatch

import (
	"bytes"
	"testing"
)

// FuzzDecode feeds arbitrary frames to Decode. A frame Decode accepts
// must re-encode, through the Encoder, to the same bytes: the decoder
// accepts nothing the encoder could not have written. Unaccepted input
// must fail with an error, never a panic or an allocation the frame
// size does not bound (Decode's capacity is count <= len/8).
func FuzzDecode(f *testing.F) {
	e := NewEncoder(1 << 16)
	e.Create(1, 9, []byte("abcde"))
	e.Stream([]uint32{3, 4, 1, 0, 0, 40}, []byte{1, 2, 3})
	e.BindResource(BindResource{ID: 1, ResourceID: 5, Offset: 64, Size: 4096})
	e.Destroy(1)
	seed, _ := e.Frame()
	f.Add(bytes.Clone(seed))
	f.Add(frame(1, u32s(MsgDestroy, 4, 7)...))
	f.Add(frame(0))
	f.Fuzz(func(t *testing.T, b []byte) {
		lim := Limits{MaxFrame: 1 << 16, MaxObjects: 1 << 10}
		msgs, err := Decode(b, lim)
		if err != nil {
			return
		}
		re := NewEncoder(lim.MaxFrame)
		for _, m := range msgs {
			switch m.Type {
			case MsgCreate:
				c, err := m.Create(lim)
				if err != nil {
					t.Fatal(err)
				}
				re.Create(c.ID, c.Kind, c.Descriptor)
			case MsgDestroy:
				ids, err := m.Destroy(lim)
				if err != nil {
					t.Fatal(err)
				}
				re.Destroy(ids...)
			case MsgStream:
				s, err := m.Stream()
				if err != nil {
					t.Fatal(err)
				}
				re.Stream(s.Words, s.Data)
			case MsgBindResource:
				r, err := m.BindResource(lim)
				if err != nil {
					t.Fatal(err)
				}
				re.BindResource(r)
			}
		}
		out, err := re.Frame()
		if err != nil {
			t.Fatalf("accepted frame does not re-encode: %v", err)
		}
		// Padding bytes are not checked by Decode, so compare after
		// zeroing them in the input: everything else must match.
		if len(out) != len(b) {
			t.Fatalf("re-encoded %d bytes from %d", len(out), len(b))
		}
		if !bytes.Equal(out, zeroPadding(b, msgs)) {
			t.Fatalf("re-encode differs\n in %x\nout %x", b, out)
		}
	})
}

// zeroPadding returns b with the padding of CREATE descriptors and
// STREAM data cleared, the only bytes Decode ignores.
func zeroPadding(b []byte, msgs []Message) []byte {
	c := bytes.Clone(b)
	off := FrameHeaderSize
	for _, m := range msgs {
		off += MessageHeaderSize
		var used int
		switch m.Type {
		case MsgCreate:
			used = 12 + int(le.Uint32(m.Payload[8:]))
		case MsgStream:
			used = 8 + 4*int(le.Uint32(m.Payload[0:])) + int(le.Uint32(m.Payload[4:]))
		default:
			used = len(m.Payload)
		}
		clear(c[off+used : off+len(m.Payload)])
		off += len(m.Payload)
	}
	return c
}

// FuzzDecodeCaps: any blob DecodeCaps accepts re-encodes to its first
// CapsSize bytes, reserved word cleared.
func FuzzDecodeCaps(f *testing.F) {
	f.Add(testCaps().Encode())
	f.Add(make([]byte, CapsSize))
	f.Fuzz(func(t *testing.T, b []byte) {
		c, err := DecodeCaps(b)
		if err != nil {
			return
		}
		want := bytes.Clone(b[:CapsSize])
		clear(want[28:32])
		if got := c.Encode(); !bytes.Equal(got, want) {
			t.Fatalf("re-encode\n got %x\nwant %x", got, want)
		}
	})
}
