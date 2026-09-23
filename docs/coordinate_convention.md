# CauPano ERP alignment pilot

ERP longitude/latitude use pixel centres as specified by executable detailed v3:
lambda=2*pi*((u+.5)/W-.5), phi=pi*(.5-(v+.5)/H).
Rays are (cos(phi)*sin(lambda), sin(phi), cos(phi)*cos(lambda)).
The centre faces +z, positive longitude turns toward +x, and north is +y.
This reuses qwen_pano.geometry; horizontal sampling wraps at the seam.

The original DiT360 evaluation preparation uses the bundled PanFusion converter
with an endpoint-inclusive lattice. Pilot rgb_erp.png reproduces its face order
(source faces 0..5 = U,L,F,R,B,D), flips and rotations unchanged. For geometry
diagnostics ONLY, this RGB is resampled onto pixel centres. The saved legacy RGB
must not later be interpreted as pixel-centred without this conversion. This
difference must be handled explicitly when creating the final canonical RGB.

Source world coordinates and house P positions are preserved. Undistorted conf
cameras use x-right/y-up/z-backward and bottom-left intrinsics; to project an
image with top-left pixels we set cy=H-1-cy_conf and use
B_world_from_image = R_conf @ diag(1,-1,-1).
Observation translations differ slightly within a panorama; the image-bearing
orientation fit ignores that baseline. Close objects can therefore show parallax.
No claim of a common optical centre or exact translation validation is made.

ERP x-right/y-up/z-forward has different handedness from the source camera
frame. The fitted ERP-to-world basis is orthogonal and is allowed determinant
-1. It is a basis change, not necessarily an SO(3) physical rotation; do not pass
it directly to quaternion/Euler APIs. world_direction=B @ erp_direction.
Inverse direction mapping is erp_direction=B.T @ world_direction.

This pilot estimates B from SIFT bearing matches to known perspective cameras.
Even yaw-index observations fit the candidate, odd indices are held out.
The 2.5-degree threshold and match counts are diagnostic heuristics, not proof
of metric alignment. Depth, visibility, support and mesh projection remain
unvalidated. No ERP pose is written back into the canonical index.

Depth convention for the eventual ERP is radial distance along the unit ERP ray.
This pilot does not decode or transform depth/normal data.
