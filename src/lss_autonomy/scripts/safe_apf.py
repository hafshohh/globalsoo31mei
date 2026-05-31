#!/usr/bin/env python3
"""
Safe Artificial Potential Field (SAPF)
Local obstacle avoidance using attractive and rotated repulsive forces.
Reference: Algorithm 2 — eq. 4, 5, 6.

Parameter notation matches the paper:
    zeta (ζ) — attractive coefficient
    eta  (η) — repulsive coefficient
"""

import numpy as np


class SafeAPF:
    """Safe Artificial Potential Field for local obstacle avoidance."""

    def __init__(self, zeta=2.0, eta=10.0, d_rep=3.0, d_vort=2.4, d_safe=1.5):
        """
        Args:
            zeta (ζ): Attractive force coefficient
            eta  (η): Repulsive force coefficient
            d_rep   : Outer influence radius Q* (metres)
            d_vort  : Distance where vortex rotation begins  (d_safe < d_vort < d_rep)
            d_safe  : Safety distance — triggers emergency avoidance  (< d_vort)
        """
        self.zeta   = zeta
        self.eta    = eta
        self.d_rep  = d_rep
        self.d_vort = d_vort
        self.d_safe = d_safe

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute_force(self, position, goal, obstacles):
        """
        Compute total SAPF force: F = F_att + R(γ)·F_rep

        Returns:
            (force, emergency):
                force     — total force vector [fx, fy]
                emergency — True if any obstacle is within d_safe
        """
        pos      = np.array(position, dtype=float)
        goal_arr = np.array(goal,     dtype=float)

        f_att            = self._compute_attractive_force(pos, goal_arr)
        f_rep, emergency = self._compute_repulsive_force(pos, goal_arr, obstacles)

        return f_att + f_rep, emergency

    def set_gains(self, zeta=None, eta=None, d_rep=None,
                  d_vort=None, d_safe=None):
        if zeta   is not None: self.zeta   = zeta
        if eta    is not None: self.eta    = eta
        if d_rep  is not None: self.d_rep  = d_rep
        if d_vort is not None: self.d_vort = d_vort
        if d_safe is not None: self.d_safe = d_safe

    # ------------------------------------------------------------------ #
    #  Internal computations                                              #
    # ------------------------------------------------------------------ #

    def _compute_attractive_force(self, pos, goal):
        """F_att = ζ · (q_goal − q) / ‖q_goal − q‖"""
        direction = goal - pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            return self.zeta * direction / dist
        return np.zeros(2)

    def _compute_repulsive_force(self, pos, goal, obstacles):
        """
        For each obstacle within d_rep:
            grad_rep  = η · (1/d − 1/Q*) / d² · (pos−obs)/d
            γ         = π/2 · (1 − d_rel),  d_rel = (d−d_safe)/(d_vort−d_safe)
            grad_obst = R(γ) · grad_rep   — eq. 9 of Algorithm 2
        """
        f_rep     = np.zeros(2)
        emergency = False

        for obs in obstacles:
            obs_arr = np.array(obs, dtype=float)
            diff    = pos - obs_arr
            dist    = np.linalg.norm(diff)

            if dist < 1e-6 or dist >= self.d_rep:
                continue

            if dist <= self.d_safe:
                emergency = True

            # ∇U_rep
            magnitude = self.eta * (1.0 / dist - 1.0 / self.d_rep) / dist ** 2
            grad_rep  = magnitude * (diff / dist)

            # γ — vortex rotation angle (eq. 4 & 5)
            gamma = self._compute_gamma(dist)

            # rotation direction: steer around obstacle toward goal
            sign  = self._rotation_sign(pos, goal, obs_arr)

            # R(γ) · grad_rep
            f_rep += self._rotation_matrix(sign * gamma) @ grad_rep

        return f_rep, emergency

    def _compute_gamma(self, dist):
        """
        γ = π/2 · (1 − d_rel)
        d_rel = (d − d_safe) / (d_vort − d_safe)  ∈ [0, 1]
        """
        if dist <= self.d_safe:
            return np.pi / 2.0
        if dist >= self.d_vort:
            return 0.0
        d_rel = (dist - self.d_safe) / (self.d_vort - self.d_safe)
        return (np.pi / 2.0) * (1.0 - d_rel)

    @staticmethod
    def _rotation_sign(pos, goal, obs):
        """
        Obstacle kanan jalur → USV belok kiri (+1, CCW).
        Obstacle kiri jalur  → USV belok kanan (−1, CW).
        """
        to_goal = goal - pos
        to_obs  = obs  - pos
        cross   = to_goal[0] * to_obs[1] - to_goal[1] * to_obs[0]
        return 1.0 if cross <= 0.0 else -1.0

    @staticmethod
    def _rotation_matrix(gamma):
        """R(γ) = [[cos γ, −sin γ], [sin γ, cos γ]]  — eq. 6"""
        c, s = np.cos(gamma), np.sin(gamma)
        return np.array([[c, -s],
                         [s,  c]])
