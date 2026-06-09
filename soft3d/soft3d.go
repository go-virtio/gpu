// Package soft3d is a minimal, dependency-free software 3D rasterizer.
//
// It implements just enough linear algebra and triangle rasterization to
// render a rotating, flat-shaded, perspective-projected, z-buffered cube
// directly into a caller-supplied BGRA byte buffer. The buffer layout
// matches go-virtio/gpu's VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM: for a pixel
// (x, y) in a w×h image the byte offset is (y*w+x)*4, with
//
//	pix[off+0] = B
//	pix[off+1] = G
//	pix[off+2] = R
//	pix[off+3] = A  (always 0xFF, fully opaque)
//
// soft3d imports nothing outside the Go standard library (in fact only
// math): it is pure math plus rasterization. It does NOT depend on
// go-virtio/common or go-virtio/gpu, so it can be reused anywhere a BGRA
// framebuffer is available.
//
// Typical wiring against go-virtio/gpu (illustrative, not executed here):
//
//	vg, _ := gpu.OpenVirtioGPU(t); fb, _ := vg.SetupFramebuffer(0, 640, 480)
//	soft3d.RenderCube(fb.Pix, 640, 480, angle); _ = fb.Flush()
package soft3d

import "math"

// Vec3 is a 3-component vector of float64.
type Vec3 struct {
	X, Y, Z float64
}

// Add returns a + b.
func (a Vec3) Add(b Vec3) Vec3 { return Vec3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }

// Sub returns a - b.
func (a Vec3) Sub(b Vec3) Vec3 { return Vec3{a.X - b.X, a.Y - b.Y, a.Z - b.Z} }

// Scale returns a scaled by f.
func (a Vec3) Scale(f float64) Vec3 { return Vec3{a.X * f, a.Y * f, a.Z * f} }

// Dot returns the dot product a · b.
func (a Vec3) Dot(b Vec3) float64 { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }

// Cross returns the cross product a × b.
func (a Vec3) Cross(b Vec3) Vec3 {
	return Vec3{
		a.Y*b.Z - a.Z*b.Y,
		a.Z*b.X - a.X*b.Z,
		a.X*b.Y - a.Y*b.X,
	}
}

// Length returns the Euclidean length of a.
func (a Vec3) Length() float64 { return math.Sqrt(a.Dot(a)) }

// Normalize returns a unit vector in the direction of a. The zero vector
// normalizes to the zero vector (rather than producing NaNs).
func (a Vec3) Normalize() Vec3 {
	l := a.Length()
	if l == 0 {
		return Vec3{}
	}
	return a.Scale(1 / l)
}

// Mat4 is a 4×4 matrix stored row-major: element (row r, column c) lives at
// index r*4+c. A vector is treated as a column and transformed as M·v, so
// Translate puts the translation in the last column (indices 3, 7, 11).
type Mat4 [16]float64

// Identity returns the 4×4 identity matrix.
func Identity() Mat4 {
	return Mat4{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
}

// Mul returns the matrix product m·n (row-major).
func (m Mat4) Mul(n Mat4) Mat4 {
	var out Mat4
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			var s float64
			for k := 0; k < 4; k++ {
				s += m[r*4+k] * n[k*4+c]
			}
			out[r*4+c] = s
		}
	}
	return out
}

// MulVec4 transforms the homogeneous column vector (x, y, z, w) by m and
// returns the result (ox, oy, oz, ow).
func (m Mat4) MulVec4(x, y, z, w float64) (ox, oy, oz, ow float64) {
	ox = m[0]*x + m[1]*y + m[2]*z + m[3]*w
	oy = m[4]*x + m[5]*y + m[6]*z + m[7]*w
	oz = m[8]*x + m[9]*y + m[10]*z + m[11]*w
	ow = m[12]*x + m[13]*y + m[14]*z + m[15]*w
	return
}

// RotateX returns a rotation of rad radians about the X axis.
func RotateX(rad float64) Mat4 {
	s, c := math.Sincos(rad)
	return Mat4{
		1, 0, 0, 0,
		0, c, -s, 0,
		0, s, c, 0,
		0, 0, 0, 1,
	}
}

// RotateY returns a rotation of rad radians about the Y axis.
func RotateY(rad float64) Mat4 {
	s, c := math.Sincos(rad)
	return Mat4{
		c, 0, s, 0,
		0, 1, 0, 0,
		-s, 0, c, 0,
		0, 0, 0, 1,
	}
}

// Translate returns a translation by (x, y, z).
func Translate(x, y, z float64) Mat4 {
	return Mat4{
		1, 0, 0, x,
		0, 1, 0, y,
		0, 0, 1, z,
		0, 0, 0, 1,
	}
}

// Perspective returns a right-handed perspective projection matrix.
//
// fovYRad is the vertical field of view in radians, aspect is width/height,
// and near/far are positive distances along -Z. Points in front of the
// camera have negative Z (the camera looks down -Z); after projection and
// the perspective divide by w (= -Z), such points map into the canonical
// view volume with -1 at the near plane and +1 at the far plane.
func Perspective(fovYRad, aspect, near, far float64) Mat4 {
	f := 1 / math.Tan(fovYRad/2)
	nf := 1 / (near - far)
	return Mat4{
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, (far + near) * nf, 2 * far * near * nf,
		0, 0, -1, 0,
	}
}

// Color is an opaque RGB color; the alpha channel is always written as 0xFF.
type Color struct {
	R, G, B uint8
}

// Clear fills the whole w×h BGRA buffer with color c (opaque).
func Clear(pix []byte, w, h int, c Color) {
	for i := 0; i < w*h; i++ {
		off := i * 4
		pix[off+0] = c.B
		pix[off+1] = c.G
		pix[off+2] = c.R
		pix[off+3] = 0xFF
	}
}

// setPixel writes color c at (x, y) in BGRA. The caller guarantees bounds.
func setPixel(pix []byte, w, x, y int, c Color) {
	off := (y*w + x) * 4
	pix[off+0] = c.B
	pix[off+1] = c.G
	pix[off+2] = c.R
	pix[off+3] = 0xFF
}

// edge returns the signed area (times two) of the triangle (a, b, p) in
// screen space; its sign tells which side of edge a→b the point p lies on.
func edge(ax, ay, bx, by, px, py float64) float64 {
	return (px-ax)*(by-ay) - (py-ay)*(bx-ax)
}

// rasterTriangle fills the triangle (x0,y0,z0)-(x1,y1,z1)-(x2,y2,z2) given
// in screen space (x, y in pixels; z is depth, smaller = nearer) into pix,
// performing a per-pixel depth test against zbuf. It is winding-agnostic: a
// pixel is inside when the three edge functions all share a sign. Degenerate
// (zero-area) triangles are skipped. The bounding box is clamped to the
// screen, so partly or fully off-screen triangles are handled safely.
func rasterTriangle(pix []byte, zbuf []float64, w, h int,
	x0, y0, z0, x1, y1, z1, x2, y2, z2 float64, c Color) {

	area := edge(x0, y0, x1, y1, x2, y2)
	if area == 0 {
		return // degenerate triangle
	}

	minX := int(math.Floor(min3(x0, x1, x2)))
	maxX := int(math.Ceil(max3(x0, x1, x2)))
	minY := int(math.Floor(min3(y0, y1, y2)))
	maxY := int(math.Ceil(max3(y0, y1, y2)))

	if minX < 0 {
		minX = 0
	}
	if minY < 0 {
		minY = 0
	}
	if maxX > w-1 {
		maxX = w - 1
	}
	if maxY > h-1 {
		maxY = h - 1
	}

	for y := minY; y <= maxY; y++ {
		for x := minX; x <= maxX; x++ {
			px := float64(x) + 0.5
			py := float64(y) + 0.5
			w0 := edge(x1, y1, x2, y2, px, py)
			w1 := edge(x2, y2, x0, y0, px, py)
			w2 := edge(x0, y0, x1, y1, px, py)

			// Inside when all edge functions share a sign (winding
			// agnostic). Allow zero (on-edge) for either orientation.
			posInside := w0 >= 0 && w1 >= 0 && w2 >= 0
			negInside := w0 <= 0 && w1 <= 0 && w2 <= 0
			if !posInside && !negInside {
				continue
			}

			b0 := w0 / area
			b1 := w1 / area
			b2 := w2 / area
			z := b0*z0 + b1*z1 + b2*z2

			idx := y*w + x
			if z < zbuf[idx] {
				zbuf[idx] = z
				setPixel(pix, w, x, y, c)
			}
		}
	}
}

func min3(a, b, c float64) float64 { return math.Min(a, math.Min(b, c)) }
func max3(a, b, c float64) float64 { return math.Max(a, math.Max(b, c)) }

// project transforms an object-space point by the model-view-projection
// matrix and maps it to screen space. It returns the screen x, y (pixels),
// the depth z (in NDC, smaller = nearer), and ok=false when the point is at
// or behind the camera (clip w <= 0), so callers can reject such triangles
// without dividing by a non-positive w.
func project(mvp Mat4, p Vec3, w, h int) (sx, sy, sz float64, ok bool) {
	cx, cy, cz, cw := mvp.MulVec4(p.X, p.Y, p.Z, 1)
	if cw <= 0 {
		return 0, 0, 0, false
	}
	ndcX := cx / cw
	ndcY := cy / cw
	ndcZ := cz / cw
	sx = (ndcX + 1) * 0.5 * float64(w)
	sy = (1 - ndcY) * 0.5 * float64(h) // flip Y: NDC up → screen down
	sz = ndcZ
	return sx, sy, sz, true
}

// face is one cube face: four corner indices (as a quad) plus a flat color.
type face struct {
	a, b, c, d int
	col        Color
}

// RenderCube renders a rotating unit cube ([-1,1]^3) into the w×h BGRA
// buffer pix. The cube is rotated by RotateY(angleRad)·RotateX(angleRad*0.5),
// translated to z = -4 (camera at the origin looking down -Z), and
// perspective-projected. Visible faces are backface-culled (faces whose
// screen-space signed area indicates they point away from the camera are
// skipped) and flat-shaded by (face-normal · light) with a fixed light
// direction. The z-buffer is reset on every call.
func RenderCube(pix []byte, w, h int, angleRad float64) {
	bg := Color{R: 10, G: 10, B: 20}
	Clear(pix, w, h, bg)

	// Per-call z-buffer initialised to +∞ (anything is nearer).
	zbuf := make([]float64, w*h)
	for i := range zbuf {
		zbuf[i] = math.Inf(1)
	}

	verts := [8]Vec3{
		{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
		{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
	}

	// Each face's corners are wound counter-clockwise when viewed from
	// outside the cube, so the outward normal follows the right-hand rule.
	faces := [6]face{
		{0, 3, 2, 1, Color{R: 220, G: 60, B: 60}},  // back  (-Z)
		{4, 5, 6, 7, Color{R: 60, G: 220, B: 60}},  // front (+Z)
		{0, 4, 7, 3, Color{R: 60, G: 60, B: 220}},  // left  (-X)
		{1, 2, 6, 5, Color{R: 220, G: 220, B: 60}}, // right (+X)
		{0, 1, 5, 4, Color{R: 220, G: 60, B: 220}}, // bottom(-Y)
		{3, 7, 6, 2, Color{R: 60, G: 220, B: 220}}, // top   (+Y)
	}

	model := RotateY(angleRad).Mul(RotateX(angleRad * 0.5))
	view := Translate(0, 0, -4)
	proj := Perspective(math.Pi/3, float64(w)/float64(h), 0.1, 100)
	mvp := proj.Mul(view.Mul(model))

	lightDir := Vec3{X: 0.4, Y: 0.6, Z: 1}.Normalize()

	for _, f := range faces {
		renderFace(pix, zbuf, w, h, &verts, f, mvp, model, lightDir)
	}
}

// renderFace projects, backface-culls, flat-shades, and rasterizes a single
// cube face. It is split out of RenderCube so each guard (corner at/behind the
// camera, backface cull, negative-Lambert clamp) is independently testable.
func renderFace(pix []byte, zbuf []float64, w, h int, verts *[8]Vec3,
	f face, mvp, model Mat4, lightDir Vec3) {

	idx := [4]int{f.a, f.b, f.c, f.d}
	var s [4]struct{ x, y, z float64 }
	for i, vi := range idx {
		sx, sy, sz, ok := project(mvp, verts[vi], w, h)
		if !ok {
			return // a corner is at/behind the camera: drop the face
		}
		s[i].x, s[i].y, s[i].z = sx, sy, sz
	}

	// Backface cull via screen-space signed area of the quad's first
	// triangle. With the Y-flip in project, a front-facing (CCW outside)
	// face yields a negative signed area here; skip non-negative.
	sa := edge(s[0].x, s[0].y, s[1].x, s[1].y, s[2].x, s[2].y)
	if sa >= 0 {
		return
	}

	// Flat shade by object-space face normal · light.
	e1 := verts[f.b].Sub(verts[f.a])
	e2 := verts[f.c].Sub(verts[f.a])
	nrm := e1.Cross(e2).Normalize()
	// Rotate the normal into world space (model has no scale/shear).
	nx, ny, nz, _ := model.MulVec4(nrm.X, nrm.Y, nrm.Z, 0)
	wn := Vec3{X: nx, Y: ny, Z: nz}.Normalize()
	lambert := wn.Dot(lightDir)
	if lambert < 0 {
		lambert = 0
	}
	shade := 0.25 + 0.75*lambert // ambient + diffuse
	col := Color{
		R: scale8(f.col.R, shade),
		G: scale8(f.col.G, shade),
		B: scale8(f.col.B, shade),
	}

	// Two triangles per quad: (0,1,2) and (0,2,3).
	rasterTriangle(pix, zbuf, w, h,
		s[0].x, s[0].y, s[0].z,
		s[1].x, s[1].y, s[1].z,
		s[2].x, s[2].y, s[2].z, col)
	rasterTriangle(pix, zbuf, w, h,
		s[0].x, s[0].y, s[0].z,
		s[2].x, s[2].y, s[2].z,
		s[3].x, s[3].y, s[3].z, col)
}

// scale8 multiplies an 8-bit channel by f in [0,1], clamping to [0,255].
func scale8(v uint8, f float64) uint8 {
	r := float64(v) * f
	if r < 0 {
		r = 0
	}
	if r > 255 {
		r = 255
	}
	return uint8(r)
}
