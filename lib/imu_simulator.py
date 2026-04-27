import numpy as np


class IMUSimulator:
    """
    Synthesizes realistic IMU (accelerometer + gyroscope) readings
    from ground-truth 4x4 transformation matrices.

    On-the-fly augmentation: each call produces different noise so the
    model learns to handle real sensor variance.

    Accelerometer: specific force in sensor frame (includes gravity).
    Gyroscope:     angular velocity in sensor frame.
    """

    def __init__(self, dt=1 / 30.0,
                 accel_noise_std=0.1,
                 gyro_noise_std=0.02,
                 accel_bias_instability=0.02,
                 gyro_bias_instability=0.005,
                 gravity=9.81,
                 noise_scale_range=(0.5, 2.0)):
        """
        Args:
            dt: time between consecutive frames (seconds)
            accel_noise_std: white-noise σ for accelerometer (m/s²)
            gyro_noise_std: white-noise σ for gyroscope (rad/s)
            accel_bias_instability: random-walk σ for accel bias drift
            gyro_bias_instability: random-walk σ for gyro bias drift
            gravity: gravity magnitude; set 0 if positions are in mm
                     and you don't want a gravity component
            noise_scale_range: (lo, hi) uniform multiplier applied to
                               noise each call — acts as augmentation
        """
        self.dt = dt
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_instability = accel_bias_instability
        self.gyro_bias_instability = gyro_bias_instability
        self.gravity_vec = np.array([0.0, 0.0, gravity], dtype=np.float64)
        self.noise_scale_range = noise_scale_range

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rotation_log(R):
        """SO(3) logarithmic map: rotation matrix → rotation vector."""
        cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < 1e-6:
            return np.zeros(3)
        k = angle / (2.0 * np.sin(angle))
        return k * np.array([R[2, 1] - R[1, 2],
                             R[0, 2] - R[2, 0],
                             R[1, 0] - R[0, 1]])

    # ------------------------------------------------------------------
    # clean signal derivation
    # ------------------------------------------------------------------
    def _compute_accel(self, positions, rotations):
        """World-frame acceleration via central differences → sensor frame."""
        N = len(positions)
        accel_world = np.zeros((N, 3))
        dt2 = self.dt ** 2

        for i in range(1, N - 1):
            accel_world[i] = (positions[i + 1] - 2 * positions[i] + positions[i - 1]) / dt2
        accel_world[0] = accel_world[1]
        accel_world[-1] = accel_world[-2]

        # a_sensor = Rᵀ (a_world + g)
        accel_sensor = np.zeros((N, 3))
        for i in range(N):
            accel_sensor[i] = rotations[i].T @ (accel_world[i] + self.gravity_vec)
        return accel_sensor

    def _compute_gyro(self, rotations):
        """Angular velocity in sensor (body) frame from consecutive rotations.

        A real gyroscope returns ω expressed in the sensor's own frame, i.e.
        the right-multiplicative increment R[i+1] = R[i] · exp([ω_body·dt]×).
        Using the world-frame increment (R[i+1] · R[i]^T) was wrong and made
        any downstream filter integrate rotation as if R[i] = I.
        """
        N = len(rotations)
        gyro = np.zeros((N, 3))
        for i in range(N - 1):
            dR_body = rotations[i].T @ rotations[i + 1]
            gyro[i] = self._rotation_log(dR_body) / self.dt
        if N > 1:
            gyro[-1] = gyro[-2]
        return gyro

    # ------------------------------------------------------------------
    # noise model
    # ------------------------------------------------------------------
    def _add_noise(self, signal, white_std, bias_std):
        """White noise + random-walk bias drift + per-axis scale error."""
        N, D = signal.shape
        scale = np.random.uniform(*self.noise_scale_range)

        white = np.random.randn(N, D) * (white_std * scale)
        bias = np.cumsum(np.random.randn(N, D) * (bias_std * scale), axis=0)
        scale_err = 1.0 + np.random.randn(D) * 0.005

        return signal * scale_err + white + bias

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def generate(self, tforms, add_noise=True):
        """
        Args:
            tforms:    [N, 4, 4] homogeneous transformation matrices
            add_noise: False → clean signal (useful for debugging)
        Returns:
            [N, 6] float32 — [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        """
        positions = tforms[:, :3, 3].astype(np.float64)
        rotations = tforms[:, :3, :3].astype(np.float64)

        accel = self._compute_accel(positions, rotations)
        gyro = self._compute_gyro(rotations)

        if add_noise:
            accel = self._add_noise(accel, self.accel_noise_std,
                                    self.accel_bias_instability)
            gyro = self._add_noise(gyro, self.gyro_noise_std,
                                   self.gyro_bias_instability)

        return np.concatenate([accel, gyro], axis=-1).astype(np.float32)
