package wgpubatch

import (
	"bytes"
	"errors"
	"math"
	"reflect"
	"testing"
)

var testLim = Limits{MaxFrame: 1 << 20, MaxObjects: 1024}

func testCaps() Caps {
	return Caps{Version: 1, StreamVersion: 1, MaxFrame: 1 << 20, MaxObjects: 1024, MaxRings: 4, Features: FeatureRing, MaxMemory: 256 << 20}
}

func TestCapsRoundTrip(t *testing.T) {
	c := testCaps()
	b := c.Encode()
	if len(b) != CapsSize {
		t.Fatalf("len %d", len(b))
	}
	// Byte-for-byte layout of docs/WEBGPU.md "Capset blob".
	want := []byte{
		0x57, 0x47, 0x50, 0x42, // magic "WGPB"
		1, 0, 0, 0, // version
		1, 0, 0, 0, // stream_version
		0, 0, 0x10, 0, // max_frame 1 MiB
		0, 4, 0, 0, // max_objects 1024
		4, 0, 0, 0, // max_rings
		1, 0, 0, 0, // features
		0, 0, 0, 0, // reserved
		0, 0, 0, 0x10, 0, 0, 0, 0, // max_memory 256 MiB
	}
	if !bytes.Equal(b, want) {
		t.Fatalf("blob\n got %x\nwant %x", b, want)
	}
	got, err := DecodeCaps(append(b, 9, 9, 9, 9)) // a longer blob is accepted
	if err != nil || got != c {
		t.Fatalf("DecodeCaps = %+v, %v", got, err)
	}
	if got.Limits() != (Limits{MaxFrame: 1 << 20, MaxObjects: 1024}) {
		t.Fatalf("Limits %+v", got.Limits())
	}
}

func TestDecodeCapsRefuses(t *testing.T) {
	mut := func(f func(*Caps)) []byte { c := testCaps(); f(&c); return c.Encode() }
	badMagic := testCaps().Encode()
	badMagic[0] ^= 1
	for name, b := range map[string][]byte{
		"short":    testCaps().Encode()[:CapsSize-1],
		"magic":    badMagic,
		"version":  mut(func(c *Caps) { c.Version = 0 }),
		"maxframe": mut(func(c *Caps) { c.MaxFrame = FrameHeaderSize - 1 }),
		"rings0":   mut(func(c *Caps) { c.MaxRings = 0 }),
		"rings65":  mut(func(c *Caps) { c.MaxRings = MaxRings + 1 }),
		"objects":  mut(func(c *Caps) { c.MaxObjects = 1 }),
	} {
		if _, err := DecodeCaps(b); !errors.Is(err, ErrFrame) {
			t.Errorf("%s: err %v", name, err)
		}
	}
}

func TestFrameLayout(t *testing.T) {
	e := NewEncoder(1 << 20)
	e.Destroy(7)
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{
		0x57, 0x47, 0x50, 0x42, 1, 0, 0, 0, 28, 0, 0, 0, 1, 0, 0, 0, // header
		2, 0, 0, 0, 4, 0, 0, 0, 7, 0, 0, 0, // DESTROY {7}
	}
	if !bytes.Equal(f, want) {
		t.Fatalf("frame\n got %x\nwant %x", f, want)
	}
}

func TestRoundTrip(t *testing.T) {
	e := NewEncoder(1 << 20)
	e.Create(1, 9, []byte("abcde")) // padded to 8
	e.Create(2, 3, nil)
	e.Stream([]uint32{3, 4, 1, 0, 0, 40}, []byte{1, 2, 3})
	e.Stream(nil, nil)
	e.BindResource(BindResource{ID: 1, ResourceID: 5, Offset: 64, Size: 4096})
	e.Destroy(1, 2)
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	msgs, err := Decode(f, testLim)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 6 {
		t.Fatalf("%d messages", len(msgs))
	}
	c, _ := msgs[0].Create(testLim)
	if c.ID != 1 || c.Kind != 9 || string(c.Descriptor) != "abcde" {
		t.Fatalf("create %+v", c)
	}
	c, _ = msgs[1].Create(testLim)
	if c.ID != 2 || len(c.Descriptor) != 0 {
		t.Fatalf("create %+v", c)
	}
	s, _ := msgs[2].Stream()
	if !reflect.DeepEqual(s.Words, []uint32{3, 4, 1, 0, 0, 40}) || !bytes.Equal(s.Data, []byte{1, 2, 3}) {
		t.Fatalf("stream %+v", s)
	}
	s, _ = msgs[3].Stream()
	if len(s.Words) != 0 || len(s.Data) != 0 {
		t.Fatalf("empty stream %+v", s)
	}
	r, _ := msgs[4].BindResource(testLim)
	if r != (BindResource{ID: 1, ResourceID: 5, Offset: 64, Size: 4096}) {
		t.Fatalf("bind %+v", r)
	}
	ids, _ := msgs[5].Destroy(testLim)
	if !reflect.DeepEqual(ids, []uint32{1, 2}) {
		t.Fatalf("destroy %v", ids)
	}

	// Reset reuses the memory and starts an empty frame.
	e.Reset()
	f, err = e.Frame()
	if err != nil || len(f) != FrameHeaderSize {
		t.Fatalf("empty frame %x %v", f, err)
	}
	if msgs, err := Decode(f, testLim); err != nil || len(msgs) != 0 {
		t.Fatalf("decode empty: %v %v", msgs, err)
	}
}

func frame(count uint32, body ...byte) []byte {
	f := []byte{0x57, 0x47, 0x50, 0x42, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	f = append(f, body...)
	le.PutUint32(f[8:], uint32(len(f)))
	le.PutUint32(f[12:], count)
	return f
}

func u32s(v ...uint32) []byte {
	var b []byte
	for _, x := range v {
		b = le.AppendUint32(b, x)
	}
	return b
}

func TestDecodeRefuses(t *testing.T) {
	good := frame(1, u32s(MsgDestroy, 4, 7)...)
	if _, err := Decode(good, testLim); err != nil {
		t.Fatalf("control frame refused: %v", err)
	}
	set := func(off int, v uint32) []byte {
		b := bytes.Clone(good)
		le.PutUint32(b[off:], v)
		return b
	}
	set16 := func(off int, v uint16) []byte {
		b := bytes.Clone(good)
		le.PutUint16(b[off:], v)
		return b
	}
	bigCount := frame(0)
	le.PutUint32(bigCount[12:], 1)
	cases := map[string][]byte{
		"short":         good[:FrameHeaderSize-1],
		"magic":         set(0, 1),
		"version":       set16(4, 2),
		"flags":         set16(6, 1),
		"size":          set(8, 99),
		"count-too-big": bigCount,
		"msg-trunc":     frame(2, u32s(MsgDestroy, 8, 7, 7, 0)...),
		"msg-unaligned": frame(1, u32s(MsgDestroy, 3, 7)...),
		"msg-overrun":   frame(1, u32s(MsgDestroy, 8, 7)...),
		"trailing":      frame(1, u32s(MsgDestroy, 4, 7, 0)...),
		"type":          frame(1, u32s(99, 0)...),
		"destroy-0":     frame(1, u32s(MsgDestroy, 4, 0)...),
		"destroy-big":   frame(1, u32s(MsgDestroy, 4, 1024)...),
		"destroy-none":  frame(1, u32s(MsgDestroy, 0)...),
		"create-short":  frame(1, u32s(MsgCreate, 8, 1, 1)...),
		"create-id":     frame(1, u32s(MsgCreate, 12, 0, 1, 0)...),
		"create-len":    frame(1, u32s(MsgCreate, 12, 1, 1, 1)...),
		"create-huge":   frame(1, u32s(MsgCreate, 12, 1, 1, math.MaxUint32)...),
		"stream-short":  frame(1, u32s(MsgStream, 4, 0)...),
		"stream-words":  frame(1, u32s(MsgStream, 8, 1, 0)...),
		"stream-huge":   frame(1, u32s(MsgStream, 8, math.MaxUint32, math.MaxUint32)...),
		"bind-len":      frame(1, u32s(MsgBindResource, 4, 1)...),
		"bind-id":       frame(1, u32s(MsgBindResource, 24, 0, 5, 0, 0, 1, 0)...),
		"bind-res":      frame(1, u32s(MsgBindResource, 24, 1, 0, 0, 0, 1, 0)...),
		"bind-size0":    frame(1, u32s(MsgBindResource, 24, 1, 5, 0, 0, 0, 0)...),
		"bind-wrap":     frame(1, u32s(MsgBindResource, 24, 1, 5, 0xffffffff, 0xffffffff, 2, 0)...),
	}
	for name, b := range cases {
		if _, err := Decode(b, testLim); !errors.Is(err, ErrFrame) {
			t.Errorf("%s: err %v", name, err)
		}
	}
	if _, err := Decode(good, Limits{MaxFrame: uint32(len(good)) - 1, MaxObjects: 1024}); !errors.Is(err, ErrFrame) {
		t.Errorf("over limit: err %v", err)
	}
}

func TestParseWrongType(t *testing.T) {
	m := Message{Type: MsgStream, Payload: make([]byte, 24)}
	if _, err := m.Create(testLim); err == nil {
		t.Error("Create on a STREAM")
	}
	if _, err := m.Destroy(testLim); err == nil {
		t.Error("Destroy on a STREAM")
	}
	if _, err := m.BindResource(testLim); err == nil {
		t.Error("BindResource on a STREAM")
	}
	m.Type = MsgCreate
	if _, err := m.Stream(); err == nil {
		t.Error("Stream on a CREATE")
	}
	// Called directly (not through Decode), an unaligned DESTROY is refused.
	if _, err := (Message{Type: MsgDestroy, Payload: []byte{1, 0, 0, 0, 2}}).Destroy(testLim); err == nil {
		t.Error("unaligned DESTROY")
	}
}

func TestEncoderRefuses(t *testing.T) {
	cases := map[string]func(*Encoder){
		"create-0":     func(e *Encoder) { e.Create(0, 1, nil) },
		"destroy-none": func(e *Encoder) { e.Destroy() },
		"destroy-0":    func(e *Encoder) { e.Destroy(1, 0) },
		"bind-id":      func(e *Encoder) { e.BindResource(BindResource{ResourceID: 1, Size: 1}) },
		"bind-res":     func(e *Encoder) { e.BindResource(BindResource{ID: 1, Size: 1}) },
		"bind-size":    func(e *Encoder) { e.BindResource(BindResource{ID: 1, ResourceID: 1}) },
		"bind-wrap": func(e *Encoder) {
			e.BindResource(BindResource{ID: 1, ResourceID: 1, Offset: math.MaxUint64, Size: 1})
		},
		"too-big": func(e *Encoder) { e.Stream(make([]uint32, 64), nil) },
	}
	for name, f := range cases {
		e := NewEncoder(64)
		f(e)
		// After the first error every call is ignored and Frame keeps it.
		e.Destroy(1)
		e.Create(1, 1, nil)
		e.Stream(nil, nil)
		e.BindResource(BindResource{ID: 1, ResourceID: 1, Size: 1})
		if _, err := e.Frame(); !errors.Is(err, ErrEncode) {
			t.Errorf("%s: err %v", name, err)
		}
	}
	// Each message kind refuses to overflow the frame on its own.
	for name, f := range map[string]func(*Encoder){
		"create":  func(e *Encoder) { e.Create(1, 1, make([]byte, 64)) },
		"destroy": func(e *Encoder) { e.Destroy(make1s(16)...) },
		"bind":    func(e *Encoder) { e.BindResource(BindResource{ID: 1, ResourceID: 1, Size: 1}) },
	} {
		e := NewEncoder(FrameHeaderSize + 8)
		f(e)
		if _, err := e.Frame(); !errors.Is(err, ErrEncode) {
			t.Errorf("%s overflow: err %v", name, err)
		}
	}
}

func make1s(n int) []uint32 {
	s := make([]uint32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}
