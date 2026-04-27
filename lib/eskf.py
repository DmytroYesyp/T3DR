"""
Loosely-coupled 15-state Error-State Kalman Filter for trackerless
ultrasound probe tracking.

The ESKF runs IMU dead-reckoning between visual frames and accepts
|Δz| measurements from the visual model at frame rate. Online
estimation of accelerometer and gyroscope biases keeps long-term
drift bounded — the previous open-loop double-integration cannot do
this and is what made the linear-α verifier marginal.

State (16 nominal):
    p ∈ R³        position
    v ∈ R³        velocity
    R ∈ SO(3)     sensor→world orientation (3×3 matrix)
    b_a ∈ R³      accelerometer bias
    b_g ∈ R³      gyroscope bias

Error state (15):
    [δp, δv, δθ, δb_a, δb_g]
δθ ∈ R³ is a small rotation in the so(3) tangent space; avoids the
redundancy of carrying a quaternion inside the covariance.
"""

import numpy as np


def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _exp_so3(omega):
    """Rodrigues: rotation vector → rotation matrix."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-9:
        return np.eye(3) + _skew(omega)
    K = _skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class ESKF:
    def __init__(self, dt, gravity=9.81,
                 accel_noise=0.1, gyro_noise=0.02,
                 accel_bias_walk=0.02, gyro_bias_walk=0.005,
                 init_pos_std=0.0, init_vel_std=0.5,
                 init_rot_std=0.02, init_ba_std=0.05, init_bg_std=0.01):
        self.dt = float(dt)
        self.g_world = np.array([0.0, 0.0, gravity], dtype=np.float64)

        self.Q = np.zeros((15, 15))
        self.Q[3:6, 3:6]     = (accel_noise ** 2)     * np.eye(3) * self.dt
        self.Q[6:9, 6:9]     = (gyro_noise ** 2)      * np.eye(3) * self.dt
        self.Q[9:12, 9:12]   = (accel_bias_walk ** 2) * np.eye(3) * self.dt
        self.Q[12:15, 12:15] = (gyro_bias_walk ** 2)  * np.eye(3) * self.dt

        self._P0_diag = np.concatenate([
            np.full(3, init_pos_std ** 2),
            np.full(3, init_vel_std ** 2),
            np.full(3, init_rot_std ** 2),
            np.full(3, init_ba_std ** 2),
            np.full(3, init_bg_std ** 2),
        ])
        self.reset()

    def reset(self, position=None, rotation=None):
        self.p = np.zeros(3) if position is None else np.asarray(position, dtype=np.float64).copy()
        self.v = np.zeros(3)
        self.R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64).copy()
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)
        self.P = np.diag(self._P0_diag)
        self.p_anchor = self.p.copy()

    def predict(self, accel_meas, gyro_meas):
        dt = self.dt
        a = np.asarray(accel_meas, dtype=np.float64) - self.b_a
        w = np.asarray(gyro_meas, dtype=np.float64) - self.b_g
        a_world = self.R @ a - self.g_world

        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.R = self.R @ _exp_so3(w * dt)

        F = np.eye(15)
        F[0:3,   3:6]   = np.eye(3) * dt
        F[3:6,   6:9]   = -self.R @ _skew(a) * dt
        F[3:6,   9:12]  = -self.R * dt
        F[6:9,   6:9]   = _exp_so3(-w * dt)
        F[6:9,   12:15] = -np.eye(3) * dt

        self.P = F @ self.P @ F.T + self.Q

    def update_dz_magnitude(self, dz_visual_magnitude, sigma):
        """
        |Δz| measurement update against the last committed anchor.
        The IMU-integrated state already carries a sign, so we cast the
        visual magnitude into the IMU-implied direction. With proper σ
        the bias states absorb persistent IMU drift over time.
        """
        dz_pred = float(self.p[2] - self.p_anchor[2])
        sign = np.sign(dz_pred) if abs(dz_pred) > 1e-9 else 1.0
        z_signed = sign * float(dz_visual_magnitude)

        H = np.zeros((1, 15))
        H[0, 2] = 1.0
        innovation = z_signed - dz_pred

        R = np.array([[max(float(sigma) ** 2, 1e-6)]])
        S = float(H @ self.P @ H.T + R)
        K = (self.P @ H.T) / S

        delta = (K * innovation).flatten()
        self.p   += delta[0:3]
        self.v   += delta[3:6]
        self.R    = self.R @ _exp_so3(delta[6:9])
        self.b_a += delta[9:12]
        self.b_g += delta[12:15]

        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def commit_anchor(self):
        self.p_anchor = self.p.copy()
