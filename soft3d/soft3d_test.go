package soft3d

import (
	"math"
	"testing"
)

const eps = 1e-9

func approx(a, b float64) bool { return math.Abs(a-b) < eps }

// --- Vec3 ---

func TestVec3Ops(t *testing.T) {
	a := Vec3{1, 2, 3}
	b := Vec3{4, 5, 6}

	if got := a.Add(b); got != (Vec3{5, 7, 9}) {
		t.Fatalf("Add: %v", got)
	}
	if got := b.Sub(a); got != (Vec3{3, 3, 3}) {
		t.Fatalf("Sub: %v", got)
	}
	if got := a.Scale(2); got != (Vec3{2, 4, 6}) {
		t.Fatalf("Scale: %v", got)
	}
	if got := a.Dot(b); got != 32 {
		t.Fatalf("Dot: %v", got)
	}
	// Cross of standard basis: x × y = z.
	if got := (Vec3{1, 0, 0}).Cross(Vec3{0, 1, 0}); got != (Vec3{0, 0, 1}) {
		t.Fatalf("Cross: %v", got)
	}
	if got := (Vec3{3, 4, 0}).Length(); !approx(got, 5) {
		t.Fatalf("Length: %v", got)
	}
}

func TestVec3Normalize(t *testing.T) {
	n := (Vec3{0, 0, 5}).Normalize()
	if !approx(n.Length(), 1) || n != (Vec3{0, 0, 1}) {
		t.Fatalf("Normalize unit: %v", n)
	}
	// Zero-length branch.
	if z := (Vec3{}).Normalize(); z != (Vec3{}) {
		t.Fatalf("Normalize zero: %v", z)
	}
}

// --- Mat4 ---

func TestIdentityMul(t *testing.T) {
	tr := Translate(3, 4, 5)
	if got := Identity().Mul(tr); got != tr {
		t.Fatalf("Identity.Mul(Translate) != Translate: %v", got)
	}
	if got := tr.Mul(Identity()); got != tr {
		t.Fatalf("Translate.Mul(Identity) != Translate: %v", got)
	}
}

func TestMulVec4Translate(t *testing.T) {
	tr := Translate(1, 2, 3)
	x, y, z, w := tr.MulVec4(10, 20, 30, 1)
	if !approx(x, 11) || !approx(y, 22) || !approx(z, 33) || !approx(w, 1) {
		t.Fatalf("Translate MulVec4: %v %v %v %v", x, y, z, w)
	}
	// A direction vector (w=0) is unaffected by translation.
	x, y, z, w = tr.MulVec4(10, 20, 30, 0)
	if !approx(x, 10) || !approx(y, 20) || !approx(z, 30) || !approx(w, 0) {
		t.Fatalf("Translate dir: %v %v %v %v", x, y, z, w)
	}
}

func TestRotateY90(t *testing.T) {
	// RotateY(90°): +X axis maps to -Z.
	x, y, z, _ := RotateY(math.Pi/2).MulVec4(1, 0, 0, 1)
	if !approx(x, 0) || !approx(y, 0) || !approx(z, -1) {
		t.Fatalf("RotateY 90 on +X: %v %v %v", x, y, z)
	}
}

func TestRotateX90(t *testing.T) {
	// RotateX(90°): +Y axis maps to +Z.
	x, y, z, _ := RotateX(math.Pi/2).MulVec4(0, 1, 0, 1)
	if !approx(x, 0) || !approx(y, 0) || !approx(z, 1) {
		t.Fatalf("RotateX 90 on +Y: %v %v %v", x, y, z)
	}
}

func TestPerspectiveFinite(t *testing.T) {
	p := Perspective(math.Pi/3, 4.0/3.0, 0.1, 100)
	for i, v := range p {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("Perspective[%d] not finite: %v", i, v)
		}
	}
	// A point on the near plane (z=-near) maps to NDC z = -1.
	near := 0.1
	cx, cy, cz, cw := p.MulVec4(0, 0, -near, 1)
	_ = cx
	_ = cy
	if cw <= 0 {
		t.Fatalf("near plane w should be positive, got %v", cw)
	}
	if !approx(cz/cw, -1) {
		t.Fatalf("near plane NDC z = %v, want -1", cz/cw)
	}
	// Far plane → NDC z = +1.
	far := 100.0
	_, _, fz, fw := p.MulVec4(0, 0, -far, 1)
	if !approx(fz/fw, 1) {
		t.Fatalf("far plane NDC z = %v, want 1", fz/fw)
	}
}

func TestMatMulComposition(t *testing.T) {
	// Translate then rotate must equal the composed matrix applied to a point.
	m := RotateY(0.7).Mul(Translate(1, 0, 0))
	ax, ay, az, _ := m.MulVec4(0, 0, 0, 1)
	// Direct: translate (0,0,0)->(1,0,0), then RotateY(0.7).
	bx, by, bz, _ := RotateY(0.7).MulVec4(1, 0, 0, 1)
	if !approx(ax, bx) || !approx(ay, by) || !approx(az, bz) {
		t.Fatalf("composition mismatch: %v,%v,%v vs %v,%v,%v", ax, ay, az, bx, by, bz)
	}
}

// --- Clear ---

func TestClear(t *testing.T) {
	w, h := 4, 3
	pix := make([]byte, w*h*4)
	c := Color{R: 0x11, G: 0x22, B: 0x33}
	Clear(pix, w, h, c)
	check := func(x, y int) {
		off := (y*w + x) * 4
		if pix[off+0] != 0x33 || pix[off+1] != 0x22 || pix[off+2] != 0x11 || pix[off+3] != 0xFF {
			t.Fatalf("pixel (%d,%d) BGRA = %v", x, y, pix[off:off+4])
		}
	}
	check(0, 0)
	check(3, 2)
}

// --- rasterizer ---

func countNonBG(pix []byte, w, h int, bg Color) int {
	n := 0
	for i := 0; i < w*h; i++ {
		off := i * 4
		if pix[off+0] != bg.B || pix[off+1] != bg.G || pix[off+2] != bg.R {
			n++
		}
	}
	return n
}

func centerColor(pix []byte, w, h int) Color {
	x, y := w/2, h/2
	off := (y*w + x) * 4
	return Color{R: pix[off+2], G: pix[off+1], B: pix[off+0]}
}

func newBuf(w, h int) ([]byte, []float64) {
	pix := make([]byte, w*h*4)
	zb := make([]float64, w*h)
	for i := range zb {
		zb[i] = math.Inf(1)
	}
	return pix, zb
}

func TestRasterCenterCCW(t *testing.T) {
	w, h := 20, 20
	pix, zb := newBuf(w, h)
	col := Color{R: 200, G: 100, B: 50}
	// CCW-ish big triangle covering the center.
	rasterTriangle(pix, zb, w, h, 1, 1, 0, 18, 1, 0, 10, 18, 0, col)
	if got := centerColor(pix, w, h); got != col {
		t.Fatalf("center color CCW = %v want %v", got, col)
	}
}

func TestRasterBothWindings(t *testing.T) {
	w, h := 20, 20
	col := Color{R: 1, G: 2, B: 3}

	// Winding A.
	pix, zb := newBuf(w, h)
	rasterTriangle(pix, zb, w, h, 1, 1, 0, 18, 1, 0, 10, 18, 0, col)
	if centerColor(pix, w, h) != col {
		t.Fatal("winding A did not color center")
	}

	// Winding B (reversed order → opposite sign edge functions).
	pix2, zb2 := newBuf(w, h)
	rasterTriangle(pix2, zb2, w, h, 10, 18, 0, 18, 1, 0, 1, 1, 0, col)
	if centerColor(pix2, w, h) != col {
		t.Fatal("winding B did not color center")
	}
}

func TestRasterZTest(t *testing.T) {
	w, h := 20, 20
	pix, zb := newBuf(w, h)
	far := Color{R: 255, G: 0, B: 0}
	near := Color{R: 0, G: 255, B: 0}

	full := func(c Color, z float64) {
		rasterTriangle(pix, zb, w, h, -5, -5, z, 30, -5, z, 12, 40, z, c)
	}

	// Draw far, then near over it → near wins.
	full(far, 0.9)
	full(near, 0.1)
	if got := centerColor(pix, w, h); got != near {
		t.Fatalf("near should win, got %v", got)
	}

	// Draw far again on top (z greater) → near stays.
	full(far, 0.9)
	if got := centerColor(pix, w, h); got != near {
		t.Fatalf("far after near should not overwrite, got %v", got)
	}
}

func TestRasterDegenerate(t *testing.T) {
	w, h := 10, 10
	pix, zb := newBuf(w, h)
	bg := Color{}
	// Collinear points → zero area → skipped.
	rasterTriangle(pix, zb, w, h, 0, 0, 0, 5, 5, 0, 9, 9, 0, Color{R: 255})
	if n := countNonBG(pix, w, h, bg); n != 0 {
		t.Fatalf("degenerate triangle painted %d pixels", n)
	}
}

func TestRasterPartlyOffscreen(t *testing.T) {
	w, h := 16, 16
	pix, zb := newBuf(w, h)
	bg := Color{}
	// Large triangle spilling off all edges; bbox must clamp on every side.
	rasterTriangle(pix, zb, w, h, -50, -50, 0, 50, -50, 0, 0, 50, 0, Color{R: 255})
	if n := countNonBG(pix, w, h, bg); n == 0 {
		t.Fatal("partly-offscreen triangle painted nothing")
	}
}

func TestRasterFullyOffscreen(t *testing.T) {
	w, h := 16, 16
	pix, zb := newBuf(w, h)
	bg := Color{}
	// Entirely to the right of the screen.
	rasterTriangle(pix, zb, w, h, 100, 100, 0, 120, 100, 0, 110, 120, 0, Color{R: 255})
	if n := countNonBG(pix, w, h, bg); n != 0 {
		t.Fatalf("fully-offscreen triangle painted %d pixels", n)
	}
}

// --- project helper ---

func TestProjectBehindCamera(t *testing.T) {
	w, h := 64, 64
	mvp := Perspective(math.Pi/3, 1, 0.1, 100).Mul(Translate(0, 0, -4))
	// A point well in front maps fine.
	if _, _, _, ok := project(mvp, Vec3{0, 0, 0}, w, h); !ok {
		t.Fatal("front point should project ok")
	}
	// A point far behind the camera (large +Z) → clip w <= 0 → ok=false.
	sx, sy, sz, ok := project(mvp, Vec3{0, 0, 10}, w, h)
	if ok {
		t.Fatalf("point behind camera projected ok (%v,%v,%v)", sx, sy, sz)
	}
}

// --- scale8 clamps ---

func TestScale8(t *testing.T) {
	if got := scale8(100, 0.5); got != 50 {
		t.Fatalf("scale8 mid = %d", got)
	}
	if got := scale8(200, 0); got != 0 {
		t.Fatalf("scale8 zero = %d", got)
	}
	if got := scale8(200, 2.0); got != 255 { // overflow clamp
		t.Fatalf("scale8 high = %d", got)
	}
	if got := scale8(200, -1.0); got != 0 { // negative clamp
		t.Fatalf("scale8 neg = %d", got)
	}
}

// --- RenderCube ---

func TestRenderCubeNormal(t *testing.T) {
	w, h := 64, 64
	pix := make([]byte, w*h*4)
	bg := Color{R: 10, G: 10, B: 20}
	for _, ang := range []float64{0, 0.3, 0.9, 1.7, 2.5, math.Pi} {
		RenderCube(pix, w, h, ang)
		n := countNonBG(pix, w, h, bg)
		if n < 50 {
			t.Fatalf("angle %v: only %d non-bg pixels", ang, n)
		}
	}
}

func TestRenderCubeTiny(t *testing.T) {
	// Must not panic for tiny buffers.
	w, h := 8, 8
	pix := make([]byte, w*h*4)
	RenderCube(pix, w, h, 0.5)
}

func TestRenderCubeManyAngles(t *testing.T) {
	w, h := 32, 24
	pix := make([]byte, w*h*4)
	for i := 0; i < 360; i += 7 {
		RenderCube(pix, w, h, float64(i)*math.Pi/180)
	}
}

// TestRenderFaceCornerBehind drives the project-fail path in renderFace by
// placing the cube corners at/behind the camera (no view translation).
func TestRenderFaceCornerBehind(t *testing.T) {
	w, h := 32, 32
	pix, zb := newBuf(w, h)
	verts := [8]Vec3{
		{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
		{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
	}
	// Project at the origin (no -4 translate): the +Z corners have clip w<=0,
	// so the front face's projection fails and the face is dropped.
	mvp := Perspective(math.Pi/3, 1, 0.1, 100)
	f := face{4, 5, 6, 7, Color{R: 200}}
	renderFace(pix, zb, w, h, &verts, f, mvp, Identity(), Vec3{0, 0, 1})
	if n := countNonBG(pix, w, h, Color{}); n != 0 {
		t.Fatalf("face with corner behind camera painted %d pixels", n)
	}
}

// TestRenderFaceLambertClamp drives the negative-Lambert branch: a visible
// (front-facing after cull) face lit from behind clamps to ambient only.
func TestRenderFaceLambertClamp(t *testing.T) {
	w, h := 32, 32
	pix, zb := newBuf(w, h)
	verts := [8]Vec3{
		{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
		{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
	}
	mvp := Perspective(math.Pi/3, 1, 0.1, 100).Mul(Translate(0, 0, -4))
	// Back face (-Z) faces the camera, so it survives the cull. Its outward
	// normal is ~ -Z in world space; a light pointing toward +Z gives a
	// negative Lambert term → clamped to 0 → ambient-only shade.
	f := face{0, 3, 2, 1, Color{R: 200, G: 200, B: 200}}
	renderFace(pix, zb, w, h, &verts, f, mvp, Identity(), Vec3{0, 0, 1})
	c := centerColor(pix, w, h)
	// Ambient shade is 0.25 → 200*0.25 = 50.
	if c.R != 50 {
		t.Fatalf("ambient-only red = %d, want 50", c.R)
	}
}
