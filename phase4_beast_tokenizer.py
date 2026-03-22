import numpy as np
from scipy.interpolate import make_lsq_spline

class BeastTokenizer:
    """
    Minimal + Correct BEAST Tokenizer (Research-ready)

    Features:
    - Arc-length parameterization (approx)
    - Relative motion (anchor at first frame)
    - Global scaling
    - B-spline fitting (N=4 control points)
    - Quantization to [0,255]
    - Dequantization for reconstruction
    """

    def __init__(self, frames_per_chunk=8):
        self.frames_per_chunk = frames_per_chunk

        # Cubic B-spline (degree=3), 4 control points → 8 knots
        self.knots = np.array([0., 0., 0., 0., 1., 1., 1., 1.])

    # ---------------------------
    # 1. Arc-length parameterization
    # ---------------------------
    def compute_arc_length_param(self, chunk_positions):
        """
        Compute approximate arc-length parameter.
        We average motion across joints to avoid one joint dominating.
        """

        # distance per joint: (T-1, 17)
        joint_diffs = np.linalg.norm(
            chunk_positions[1:] - chunk_positions[:-1], axis=2
        )

        # average over joints: (T-1,)
        mean_diffs = np.mean(joint_diffs, axis=1)

        # cumulative distance
        s = np.concatenate([[0], np.cumsum(mean_diffs)])

        # normalize to [0,1]
        if s[-1] > 0:
            s = s / s[-1]

        return s

    # ---------------------------
    # 2. Fit B-spline
    # ---------------------------
    def fit_chunk_to_spline(self, chunk_positions, t_eval):
        """
        Fit cubic B-spline to each joint and axis independently.
        Output shape: (4 control points, 17 joints, 3 dims)
        """

        control_points = np.zeros((4, 17, 3))

        for j in range(17):
            for d in range(3):
                y = chunk_positions[:, j, d]

                spline = make_lsq_spline(t_eval, y, self.knots, k=3)

                control_points[:, j, d] = spline.c

        return control_points

    # ---------------------------
    # 3. Quantization
    # ---------------------------
    def quantize(self, x):
        """
        Float [-1,1] → Int [0,255]
        """
        x = np.clip(x, -1.0, 1.0)
        return ((x + 1.0) * 127.5).astype(np.uint8)

    def dequantize(self, x):
        """
        Int [0,255] → Float [-1,1]
        """
        return (x.astype(np.float32) / 127.5) - 1.0

    # ---------------------------
    # 4. Encode (MAIN FUNCTION)
    # ---------------------------
    def encode_chunk(self, chunk_positions, time_delta=0.26):
        """
        Input:
            chunk_positions: (8,17,3)

        Output:
            dict with tokens + metadata
        """

        # ---- (1) Relative motion ----
        anchor = chunk_positions[0].copy()
        rel = chunk_positions - anchor

        # ---- (2) Global scaling ----
        scale = np.max(np.abs(rel)) + 1e-6
        norm = rel / scale

        # ---- (3) Arc-length ----
        t_eval = self.compute_arc_length_param(norm)

        # ---- (4) B-spline fitting ----
        cp = self.fit_chunk_to_spline(norm, t_eval)

        # ---- (5) Flatten ----
        cp_flat = cp.reshape(-1)

        # ---- (6) Quantize ----
        tokens = self.quantize(cp_flat)

        return {
            "time_delta": float(time_delta),     # speed
            "anchor": anchor.tolist(),           # for reconstruction
            "scale": float(scale),               # for reconstruction
            "tokens": tokens.tolist()            # 204 integers
        }

    # ---------------------------
    # 5. Decode (for simulation)
    # ---------------------------
    def decode_chunk(self, package):
        """
        Recover control points in real coordinates
        """

        tokens = np.array(package["tokens"])
        scale = package["scale"]
        anchor = np.array(package["anchor"])

        cp_norm = self.dequantize(tokens).reshape(4, 17, 3)

        cp = cp_norm * scale + anchor

        return cp


# ================= TEST =================
if __name__ == "__main__":

    # Fake input (like MotionBERT output)
    x = np.random.uniform(-1, 1, (8, 17, 3))

    tokenizer = BeastTokenizer()

    # ========================
    # 1. ENCODE
    # ========================
    encoded = tokenizer.encode_chunk(x)

    print("Tokens:", encoded["tokens"][:10])
    print("Length:", len(encoded["tokens"]))

    # ========================
    # 2. DECODE (control points)
    # ========================
    decoded_cp = tokenizer.decode_chunk(encoded)

    print("Recovered CP shape:", decoded_cp.shape)

    # ========================
    # 3. CHECK 1: Arc-length sanity
    # ========================
    print("\n--- ARC LENGTH CHECK ---")
    t_eval = tokenizer.compute_arc_length_param(x)
    print("t_eval:", t_eval)

    # ========================
    # 4. CHECK 2: Reconstruction error (control points)
    # ========================
    print("\n--- CONTROL POINT ERROR ---")

    # recompute original cp (before quantization)
    anchor = x[0].copy()
    rel = x - anchor
    scale = np.max(np.abs(rel)) + 1e-6
    norm = rel / scale

    t_eval = tokenizer.compute_arc_length_param(norm)
    cp_original = tokenizer.fit_chunk_to_spline(norm, t_eval)

    # recovered cp (normalized space)
    tokens = np.array(encoded["tokens"])
    cp_recovered = tokenizer.dequantize(tokens).reshape(4, 17, 3)

    error_cp = np.mean((cp_original - cp_recovered) ** 2)
    print("Control Point MSE:", error_cp)

    # ========================
    # 5. CHECK 3: Trajectory reconstruction
    # ========================
    print("\n--- TRAJECTORY RECONSTRUCTION ---")

    from scipy.interpolate import BSpline

    def reconstruct(cp):
        t = np.linspace(0, 1, 8)
        recon = np.zeros((8, 17, 3))

        for j in range(17):
            for d in range(3):
                spline = BSpline(tokenizer.knots, cp[:, j, d], 3)
                recon[:, j, d] = spline(t)

        return recon

    # reconstruct trajectory
    recon = reconstruct(decoded_cp)

    traj_error = np.mean((recon - x) ** 2)
    print("Trajectory MSE:", traj_error)