package gpu

// Throwaway buffer-capture harness for the virglrenderer vtest validation.
// Run with: GOWORK=off VIRGL_DUMP_DIR=/abs/path go test -run TestDumpVirglBuffers -count=1
// It writes the exact bytes of the three virgl command buffers (clear/draw/tex)
// plus a manifest of the resource setup the host must have in place.
// NOT part of the package's real test suite.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestDumpVirglBuffers(t *testing.T) {
	dir := os.Getenv("VIRGL_DUMP_DIR")
	if dir == "" {
		t.Skip("VIRGL_DUMP_DIR not set")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Small render target keeps the TransferGet readback tiny.
	const W, H = uint32(16), uint32(16)

	g := &VirtioGPU{}

	clear := buildClearVirglBuffer(1.0, 0.0, 0.0, 1.0) // RED
	draw := g.buildDrawVirglBuffer(W, H, triColor)
	tex := g.buildTexDrawVirglBuffer(W, H)

	// Vertex buffer contents for draw (9 floats) and tex (15 floats).
	drawVB := make([]byte, len(triVerts)*4)
	for i, f := range triVerts {
		binary.LittleEndian.PutUint32(drawVB[i*4:], float32bits(f))
	}
	texVB := make([]byte, len(texTriVerts)*4)
	for i, f := range texTriVerts {
		binary.LittleEndian.PutUint32(texVB[i*4:], float32bits(f))
	}

	write := func(name string, b []byte) {
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s: %d bytes", name, len(b))
	}
	write("clear.bin", clear)
	write("draw.bin", draw)
	write("tex.bin", tex)
	write("draw_vb.bin", drawVB)
	write("tex_vb.bin", texVB)
	write("tex_texels.bin", texData2)

	type manifest struct {
		Width, Height  uint32
		RTResourceID   uint32
		RTFormat       uint32 // virgl_formats
		VBufResourceID uint32
		VBufByteSize   uint32
		TexResourceID  uint32
		TexW, TexH     uint32
		TexFormat      uint32
		ClearLenDwords int
		DrawLenDwords  int
		TexLenDwords   int
	}
	m := manifest{
		Width: W, Height: H,
		RTResourceID: rtResourceID, RTFormat: virglFormatB8G8R8A8Unorm,
		VBufResourceID: vbufResourceID, VBufByteSize: uint32(len(drawVB)),
		TexResourceID: texResourceID, TexW: texW2, TexH: texH2,
		TexFormat:      virglFormatR8G8B8A8Unorm,
		ClearLenDwords: len(clear) / 4,
		DrawLenDwords:  len(draw) / 4,
		TexLenDwords:   len(tex) / 4,
	}
	mb, _ := json.MarshalIndent(m, "", "  ")
	write("manifest.json", mb)
}

func float32bits(f float32) uint32 { return math.Float32bits(f) }
