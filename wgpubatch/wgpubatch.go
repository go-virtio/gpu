// Package wgpubatch is the wire protocol of the private "webgpu-batch"
// virtio-gpu context type: guest programs send go-gfx/webgpu Recorder
// streams to a host that replays them on WebGPU, as Venus does for
// Vulkan. It defines the capset id, the capset blob a host returns from
// GET_CAPSET, and the frame that rides as the opaque payload of
// SUBMIT_3D. Guest and host both import it; it has no GOOS constraint.
//
// The Recorder stream inside a STREAM message, and the descriptor inside
// a CREATE message, belong to go-gfx/webgpu: this package carries them as
// opaque words and bytes and never interprets them (one implementation
// per format).
//
// Status: no host implements this protocol yet. Everything here is
// checked offline (unit tests and fuzzing) only. See docs/WEBGPU.md.
//
// References: virtio spec device-types/gpu/description.tex :523-545
// (capsets), :727-750 (CTX_CREATE.context_init), :764-766 (SUBMIT_3D);
// Linux virtgpu_kms.c:92-112 and virtgpu_drv.h:59 (capset id range).
package wgpubatch

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// CapsetID is the capset id of the webgpu-batch context type, carried in
// the low 8 bits of CTX_CREATE.context_init. The spec reserves no private
// range; 30 is the highest id Linux guests record correctly (virtgpu_kms.c
// shifts an int: ids >= 31 corrupt capset_id_mask) and far from the
// allocated 1..10. See docs/WEBGPU.md, "Capset id".
const CapsetID uint32 = 30

// Magic opens both the capset blob and every frame: "WGPB" little-endian.
const Magic uint32 = 0x42504757

// Version is the protocol version this package encodes and decodes.
const Version = 1

// MaxRings is the number of ring_idx values virtio allows per context
// (description.tex:289, 0..63 inclusive).
const MaxRings = 64

// Sizes on the wire, in bytes.
const (
	CapsSize          = 40
	FrameHeaderSize   = 16
	MessageHeaderSize = 8
)

// Message types.
const (
	MsgCreate       uint32 = 1 // id, kind, descriptor
	MsgDestroy      uint32 = 2 // ids
	MsgStream       uint32 = 3 // one go-gfx/webgpu Recorder submit
	MsgBindResource uint32 = 4 // id, resource_id, offset, size
)

// FeatureRing is the capset feature bit of the shared-memory ring
// transport (v2, designed, not implemented).
const FeatureRing uint32 = 1 << 0

// ErrFrame is wrapped by every decoding error.
var ErrFrame = errors.New("wgpubatch: malformed")

func malformed(format string, a ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrFrame}, a...)...)
}

var le = binary.LittleEndian

// Caps is the capset blob (GET_CAPSET, capset CapsetID, version 1).
type Caps struct {
	Version       uint32 // protocol version, >= 1
	StreamVersion uint32 // go-gfx/webgpu Recorder stream version replayed
	MaxFrame      uint32 // largest SUBMIT_3D payload accepted, bytes
	MaxObjects    uint32 // live guest objects per context; ids are 1..MaxObjects-1
	MaxRings      uint32 // timelines per context, <= MaxRings
	Features      uint32 // FeatureRing ...
	MaxMemory     uint64 // buffer and texture bytes per context
}

// Encode returns the CapsSize-byte blob.
func (c Caps) Encode() []byte {
	b := make([]byte, CapsSize)
	le.PutUint32(b[0:], Magic)
	le.PutUint32(b[4:], c.Version)
	le.PutUint32(b[8:], c.StreamVersion)
	le.PutUint32(b[12:], c.MaxFrame)
	le.PutUint32(b[16:], c.MaxObjects)
	le.PutUint32(b[20:], c.MaxRings)
	le.PutUint32(b[24:], c.Features)
	le.PutUint64(b[32:], c.MaxMemory)
	return b
}

// DecodeCaps reads a capset blob. A longer blob is accepted (a later
// version may append fields); the reserved word is ignored.
func DecodeCaps(b []byte) (Caps, error) {
	if len(b) < CapsSize {
		return Caps{}, malformed("capset blob of %d bytes", len(b))
	}
	if m := le.Uint32(b[0:]); m != Magic {
		return Caps{}, malformed("capset magic %#x", m)
	}
	c := Caps{
		Version:       le.Uint32(b[4:]),
		StreamVersion: le.Uint32(b[8:]),
		MaxFrame:      le.Uint32(b[12:]),
		MaxObjects:    le.Uint32(b[16:]),
		MaxRings:      le.Uint32(b[20:]),
		Features:      le.Uint32(b[24:]),
		MaxMemory:     le.Uint64(b[32:]),
	}
	switch {
	case c.Version == 0:
		return Caps{}, malformed("capset version 0")
	case c.MaxFrame < FrameHeaderSize:
		return Caps{}, malformed("max frame %d below a frame header", c.MaxFrame)
	case c.MaxRings == 0 || c.MaxRings > MaxRings:
		return Caps{}, malformed("%d rings", c.MaxRings)
	case c.MaxObjects < 2:
		return Caps{}, malformed("%d objects", c.MaxObjects)
	}
	return c, nil
}

// Limits bound what Decode accepts. A host takes them from its Caps.
type Limits struct {
	MaxFrame   uint32
	MaxObjects uint32
}

// Limits returns the decoding limits the capset promises.
func (c Caps) Limits() Limits { return Limits{MaxFrame: c.MaxFrame, MaxObjects: c.MaxObjects} }

// Message is one decoded message; Payload aliases the frame.
type Message struct {
	Type    uint32
	Payload []byte
}

// Decode checks a frame and splits it into messages. The host must copy
// the frame out of guest memory before calling it: Payloads alias b.
// Each payload's own layout is checked by the Parse functions, which a
// host calls next; Decode already rejects unknown types and ids out of
// range so that a frame either decodes whole or not at all.
func Decode(b []byte, lim Limits) ([]Message, error) {
	n := uint64(len(b))
	if n < FrameHeaderSize {
		return nil, malformed("frame of %d bytes", n)
	}
	if n > uint64(lim.MaxFrame) {
		return nil, malformed("frame of %d bytes above the %d limit", n, lim.MaxFrame)
	}
	if m := le.Uint32(b[0:]); m != Magic {
		return nil, malformed("frame magic %#x", m)
	}
	if v := le.Uint16(b[4:]); v != Version {
		return nil, malformed("frame version %d", v)
	}
	if f := le.Uint16(b[6:]); f != 0 {
		return nil, malformed("frame flags %#x", f)
	}
	if size := uint64(le.Uint32(b[8:])); size != n {
		return nil, malformed("frame says %d bytes, has %d", size, n)
	}
	count := uint64(le.Uint32(b[12:]))
	if count > (n-FrameHeaderSize)/MessageHeaderSize {
		return nil, malformed("%d messages in %d bytes", count, n)
	}
	msgs := make([]Message, 0, count)
	off := uint64(FrameHeaderSize)
	for i := uint64(0); i < count; i++ {
		if n-off < MessageHeaderSize {
			return nil, malformed("message %d truncated", i)
		}
		typ := le.Uint32(b[off:])
		size := uint64(le.Uint32(b[off+4:]))
		off += MessageHeaderSize
		if size%4 != 0 || size > n-off {
			return nil, malformed("message %d of %d bytes", i, size)
		}
		m := Message{Type: typ, Payload: b[off : off+size]}
		if err := m.check(lim); err != nil {
			return nil, fmt.Errorf("message %d: %w", i, err)
		}
		msgs = append(msgs, m)
		off += size
	}
	if off != n {
		return nil, malformed("%d trailing bytes", n-off)
	}
	return msgs, nil
}

func (m Message) check(lim Limits) error {
	var err error
	switch m.Type {
	case MsgCreate:
		_, err = m.Create(lim)
	case MsgDestroy:
		_, err = m.Destroy(lim)
	case MsgStream:
		_, err = m.Stream()
	case MsgBindResource:
		_, err = m.BindResource(lim)
	default:
		err = malformed("message type %d", m.Type)
	}
	return err
}

func checkID(id uint32, lim Limits) error {
	if id == 0 || id >= lim.MaxObjects {
		return malformed("object id %d outside 1..%d", id, int64(lim.MaxObjects)-1)
	}
	return nil
}

// Create is a CREATE message: a guest-numbered object of a go-gfx/webgpu
// kind, with its descriptor (opaque here).
type Create struct {
	ID, Kind   uint32
	Descriptor []byte // aliases the frame
}

// Create parses a CREATE payload: id u32, kind u32, len u32, bytes, pad.
func (m Message) Create(lim Limits) (Create, error) {
	p := m.Payload
	if m.Type != MsgCreate || len(p) < 12 {
		return Create{}, malformed("CREATE of %d bytes", len(p))
	}
	c := Create{ID: le.Uint32(p[0:]), Kind: le.Uint32(p[4:])}
	if err := checkID(c.ID, lim); err != nil {
		return Create{}, err
	}
	dl := uint64(le.Uint32(p[8:]))
	if pad4(dl) != uint64(len(p))-12 {
		return Create{}, malformed("CREATE descriptor of %d bytes in %d", dl, len(p)-12)
	}
	c.Descriptor = p[12 : 12+dl]
	return c, nil
}

// Destroy parses a DESTROY payload: a list of nonzero ids.
func (m Message) Destroy(lim Limits) ([]uint32, error) {
	p := m.Payload
	if m.Type != MsgDestroy || len(p) == 0 || len(p)%4 != 0 {
		return nil, malformed("DESTROY of %d bytes", len(p))
	}
	ids := make([]uint32, len(p)/4)
	for i := range ids {
		ids[i] = le.Uint32(p[4*i:])
		if err := checkID(ids[i], lim); err != nil {
			return nil, err
		}
	}
	return ids, nil
}

// Stream is a STREAM message: one go-gfx/webgpu Recorder submit, with
// object ids in the guest's namespace. go-gfx/webgpu validates and remaps
// the words; this package only frames them.
type Stream struct {
	Words []uint32
	Data  []byte // aliases the frame
}

// Stream parses a STREAM payload: nwords u32, ndata u32, words, data, pad.
func (m Message) Stream() (Stream, error) {
	p := m.Payload
	if m.Type != MsgStream || len(p) < 8 {
		return Stream{}, malformed("STREAM of %d bytes", len(p))
	}
	nw := uint64(le.Uint32(p[0:]))
	nd := uint64(le.Uint32(p[4:]))
	if 8+4*nw+pad4(nd) != uint64(len(p)) {
		return Stream{}, malformed("STREAM of %d words and %d data bytes in %d", nw, nd, len(p))
	}
	s := Stream{Words: make([]uint32, nw)}
	for i := range s.Words {
		s.Words[i] = le.Uint32(p[8+4*i:])
	}
	d := 8 + 4*nw
	s.Data = p[d : d+nd]
	return s, nil
}

// BindResource exposes part of a virtio resource attached to the context
// as an upload or readback window of a guest buffer.
type BindResource struct {
	ID, ResourceID uint32
	Offset, Size   uint64
}

// BindResource parses a BIND_RESOURCE payload. The host still checks the
// range against the resource it names, which only it knows.
func (m Message) BindResource(lim Limits) (BindResource, error) {
	p := m.Payload
	if m.Type != MsgBindResource || len(p) != 24 {
		return BindResource{}, malformed("BIND_RESOURCE of %d bytes", len(p))
	}
	r := BindResource{
		ID: le.Uint32(p[0:]), ResourceID: le.Uint32(p[4:]),
		Offset: le.Uint64(p[8:]), Size: le.Uint64(p[16:]),
	}
	if err := checkID(r.ID, lim); err != nil {
		return BindResource{}, err
	}
	if r.ResourceID == 0 {
		return BindResource{}, malformed("resource id 0")
	}
	if r.Size == 0 || r.Offset > math.MaxUint64-r.Size {
		return BindResource{}, malformed("window [%d, +%d)", r.Offset, r.Size)
	}
	return r, nil
}

func pad4(n uint64) uint64 { return (n + 3) &^ 3 }

// ErrEncode is wrapped by every encoding error.
var ErrEncode = errors.New("wgpubatch: cannot encode")

// Encoder builds one frame. Methods record the first error and ignore
// later calls; Frame returns it. Reset reuses the memory.
type Encoder struct {
	buf   []byte
	count uint32
	err   error
	max   uint64
}

// NewEncoder returns an Encoder whose frames stay within maxFrame bytes
// (Caps.MaxFrame).
func NewEncoder(maxFrame uint32) *Encoder {
	e := &Encoder{max: uint64(maxFrame)}
	e.Reset()
	return e
}

// Reset starts a new frame, keeping the memory.
func (e *Encoder) Reset() {
	e.buf = append(e.buf[:0], make([]byte, FrameHeaderSize)...)
	e.count, e.err = 0, nil
}

func (e *Encoder) fail(format string, a ...any) {
	if e.err == nil {
		e.err = fmt.Errorf("%w: "+format, append([]any{ErrEncode}, a...)...)
	}
}

// begin appends a message header for a payload of size bytes (already a
// multiple of 4) and reports whether the frame can hold it.
func (e *Encoder) begin(typ uint32, size uint64) bool {
	if e.err != nil {
		return false
	}
	if uint64(len(e.buf))+MessageHeaderSize+size > e.max {
		e.fail("frame would exceed %d bytes", e.max)
		return false
	}
	e.buf = le.AppendUint32(e.buf, typ)
	e.buf = le.AppendUint32(e.buf, uint32(size))
	e.count++
	return true
}

func (e *Encoder) pad() {
	for len(e.buf)%4 != 0 {
		e.buf = append(e.buf, 0)
	}
}

// Create appends a CREATE message.
func (e *Encoder) Create(id, kind uint32, descriptor []byte) {
	if id == 0 {
		e.fail("object id 0")
		return
	}
	if !e.begin(MsgCreate, 12+pad4(uint64(len(descriptor)))) {
		return
	}
	e.buf = le.AppendUint32(e.buf, id)
	e.buf = le.AppendUint32(e.buf, kind)
	e.buf = le.AppendUint32(e.buf, uint32(len(descriptor)))
	e.buf = append(e.buf, descriptor...)
	e.pad()
}

// Destroy appends a DESTROY message.
func (e *Encoder) Destroy(ids ...uint32) {
	if len(ids) == 0 {
		e.fail("DESTROY of no ids")
		return
	}
	for _, id := range ids {
		if id == 0 {
			e.fail("object id 0")
			return
		}
	}
	if !e.begin(MsgDestroy, 4*uint64(len(ids))) {
		return
	}
	for _, id := range ids {
		e.buf = le.AppendUint32(e.buf, id)
	}
}

// Stream appends a STREAM message carrying one Recorder submit.
func (e *Encoder) Stream(words []uint32, data []byte) {
	if !e.begin(MsgStream, 8+4*uint64(len(words))+pad4(uint64(len(data)))) {
		return
	}
	e.buf = le.AppendUint32(e.buf, uint32(len(words)))
	e.buf = le.AppendUint32(e.buf, uint32(len(data)))
	for _, w := range words {
		e.buf = le.AppendUint32(e.buf, w)
	}
	e.buf = append(e.buf, data...)
	e.pad()
}

// BindResource appends a BIND_RESOURCE message.
func (e *Encoder) BindResource(r BindResource) {
	switch {
	case r.ID == 0:
		e.fail("object id 0")
		return
	case r.ResourceID == 0:
		e.fail("resource id 0")
		return
	case r.Size == 0 || r.Offset > math.MaxUint64-r.Size:
		e.fail("window [%d, +%d)", r.Offset, r.Size)
		return
	}
	if !e.begin(MsgBindResource, 24) {
		return
	}
	e.buf = le.AppendUint32(e.buf, r.ID)
	e.buf = le.AppendUint32(e.buf, r.ResourceID)
	e.buf = le.AppendUint64(e.buf, r.Offset)
	e.buf = le.AppendUint64(e.buf, r.Size)
}

// Frame completes the header and returns the frame, or the first error.
// The slice is the Encoder's memory until the next Reset.
func (e *Encoder) Frame() ([]byte, error) {
	if e.err != nil {
		return nil, e.err
	}
	le.PutUint32(e.buf[0:], Magic)
	le.PutUint16(e.buf[4:], Version)
	le.PutUint16(e.buf[6:], 0)
	le.PutUint32(e.buf[8:], uint32(len(e.buf)))
	le.PutUint32(e.buf[12:], e.count)
	return e.buf, nil
}
