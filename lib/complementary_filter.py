"""Position-level complementary filter: a simple classical baseline for the ESKF.

Blends an IMU and a visual velocity estimate each frame:
    v_imu = v_prev + (accel_z_world - gravity) * dt
    v_vis = z_visual / dt
    v     = α·v_imu + (1-α)·v_vis,   p = p_prev + v·dt

No bias estimation, no rotation tracking (accel_z taken in the sensor frame ≈
world-z only when the probe is near-vertical), no covariance. α=0 is pure
visual, α=1 is pure IMU; typical α ∈ [0.05, 0.3], lower trusts vision more.
Drop-in API match with lib.imu_verifier.IMUVerifier.
"""

import numpy as np


class ComplementaryFilter:
    def __init__(self, imu_simulator, dt=1 / 30.0, gravity=9.81, alpha=0.2):
        self.imu_sim = imu_simulator
        self.dt = float(dt)
        self.gravity = float(gravity)
        self.alpha = float(alpha)
        self.p = 0.0
        self.v = 0.0

    def reset(self, init_position, init_rotation=None, init_velocity=None):
        # init_rotation ignored (no rotation state); kept for IMUVerifier API parity.
        if hasattr(init_position, '__len__'):
            self.p = float(init_position[2])
        else:
            self.p = float(init_position)
        if init_velocity is None:
            self.v = 0.0
        elif hasattr(init_velocity, '__len__'):
            self.v = float(init_velocity[2])
        else:
            self.v = float(init_velocity)

    def precompute_imu(self, tforms):
        return self.imu_sim.generate(tforms)

    def step(self, accel, gyro, z_visual, sigma_visual=None):
        """Advance one frame; return signed Δz.

        accel: 3-vector accelerometer reading in sensor frame, INCLUDES gravity.
        gyro:  ignored (the filter doesn't track rotation).
        z_visual: signed Δz prediction from the visual model.
        sigma_visual: ignored (filter uses fixed α blending).
        """
        # Assume sensor z ≈ world z and subtract gravity directly; inaccurate under tilt.
        accel_z_world = float(accel[2]) - self.gravity

        z_before = self.p
        v_imu = self.v + accel_z_world * self.dt
        v_vis = float(z_visual) / self.dt
        self.v = self.alpha * v_imu + (1.0 - self.alpha) * v_vis
        self.p = self.p + self.v * self.dt
        return self.p - z_before
