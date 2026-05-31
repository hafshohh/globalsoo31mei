#!/usr/bin/env python3
import os
import sys
import math
import rospy
import numpy as np
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_apf import SafeAPF


class ObstacleAvoidance:
    def __init__(self):
        rospy.init_node('obstacle_avoidance', anonymous=False)

        # State
        self.obstacle_detected = False
        self.enabled     = rospy.get_param('~enabled', False)
        self.raw_yaw     = 1500        # PWM center
        self.pos         = np.array([0.0, 0.0])
        self.heading     = 0.0         # rad, dari odom
        self.smooth_path = []          # [(x,y), ...] dari /planned_path_smooth
        self._avoiding   = False       # state hysteresis d_rejoin

        # Parameters
        self.base_speed  = rospy.get_param('~base_speed',       0.333)
        self.obs_dist_m  = rospy.get_param('~obs_assumed_dist', 3.0)
        self.yaw_max_deg = rospy.get_param('~yaw_max_deg',      60.0)
        self.lookahead   = rospy.get_param('~lookahead',         3.0)

        # d_rejoin: keluar dari avoidance mode saat jarak obstacle > d_rejoin
        # harus: d_safe < d_rejoin < d_rep
        self.d_rejoin = rospy.get_param('~d_rejoin', 2.2)

        # SAPF — default dihitung dari dinamika 4-DOF LSS-01:
        #   τ_sway=6.83s, lateral_drift=0.77m → d_safe=1.5m
        #   dist_turn_90°=0.99m + buffer → d_rep=3.0m
        # Turunkan d_safe bertahap (1.5→1.2→1.0) setelah simulasi stabil.
        # Variasi MATLAB:
        #   Var 2: ζ=2.5 η=8.0  d_safe=1.0 R_inf=2.5 d_rejoin=1.5
        #   Var 3: ζ=3.0 η=6.0  d_safe=0.6 R_inf=1.6 d_rejoin=1.0
        self.apf = SafeAPF(
            zeta   = rospy.get_param('~zeta',   2.0),
            eta    = rospy.get_param('~eta',   10.0),
            d_rep  = rospy.get_param('~d_rep',  3.0),
            d_vort = rospy.get_param('~d_vort', 2.4),
            d_safe = rospy.get_param('~d_safe', 1.5),
        )

        # Subscribers
        rospy.Subscriber('/detect/object/bool',                  Bool,     self.cb_obstacle)
        rospy.Subscriber('/robot/vision/raw_yaw',                Int32,    self.cb_yaw)
        rospy.Subscriber('/asv/odom',                            Odometry, self.cb_odom)
        rospy.Subscriber('/planned_path_smooth',                 Path,     self.cb_path)
        rospy.Subscriber('/autonomy/obstacle_avoidance/enabled', Bool,     self.cb_enabled)

        # Publishers
        self.pub_cmd    = rospy.Publisher('/autonomy/pathfollowing', Twist, queue_size=1)
        self.pub_status = rospy.Publisher('/autonomy/obstacle_status', Bool, queue_size=1)

        rospy.loginfo("[ObstacleAvoidance] Node started — SAPF active "
                      "(ζ=%.1f η=%.1f d_safe=%.1f d_rejoin=%.1f d_rep=%.1f)",
                      self.apf.zeta, self.apf.eta,
                      self.apf.d_safe, self.d_rejoin, self.apf.d_rep)

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def cb_enabled(self, msg):
        self.enabled = msg.data
        if not self.enabled:
            if self._avoiding:
                self.pub_cmd.publish(Twist())   # batalkan perintah avoidance terakhir
            self._avoiding = False
            self.pub_status.publish(Bool(data=False))
        rospy.loginfo("[ObstacleAvoidance] %s",
                      "AKTIF" if self.enabled else "NONAKTIF — dimatikan dari web UI")

    def cb_obstacle(self, msg):
        self.obstacle_detected = msg.data
        if self.enabled:
            self.pub_status.publish(Bool(data=self.obstacle_detected))

    def cb_yaw(self, msg):
        self.raw_yaw = msg.data

    def cb_odom(self, msg):
        self.pos = np.array([msg.pose.pose.position.x,
                             msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.heading = math.atan2(siny, cosy)

    def cb_path(self, msg):
        self.smooth_path = [(p.pose.position.x, p.pose.position.y)
                            for p in msg.poses]

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _get_goal(self):
        """Ambil titik lookahead di smooth_path."""
        if not self.smooth_path:
            return (self.pos[0] + self.lookahead * math.cos(self.heading),
                    self.pos[1] + self.lookahead * math.sin(self.heading))

        dists   = [np.linalg.norm(self.pos - np.array(p)) for p in self.smooth_path]
        closest = int(np.argmin(dists))

        for i in range(closest, len(self.smooth_path)):
            if np.linalg.norm(self.pos - np.array(self.smooth_path[i])) >= self.lookahead:
                return self.smooth_path[i]

        return self.smooth_path[-1]

    def _estimate_obstacle_pos(self):
        """Estimasi posisi obstacle dari raw_yaw kamera.
        PWM 1500=center, 1000=kiri penuh, 2000=kanan penuh."""
        normalized = (self.raw_yaw - 1500) / 500.0
        yaw_offset = normalized * math.radians(self.yaw_max_deg)
        obs_dir    = self.heading + yaw_offset
        return (self.pos[0] + self.obs_dist_m * math.cos(obs_dir),
                self.pos[1] + self.obs_dist_m * math.sin(obs_dir))

    def _obs_distance(self, obs_pos):
        return float(np.linalg.norm(self.pos - np.array(obs_pos)))

    @staticmethod
    def _wrap(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _emergency_avoid(self, obs_pos):
        """Berhenti dan putar menjauhi obstacle (d ≤ d_safe)."""
        away_dir = math.atan2(self.pos[1] - obs_pos[1],
                              self.pos[0] - obs_pos[0])
        e_psi = self._wrap(away_dir - self.heading)
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = float(np.clip(e_psi,
                                      -math.radians(40.0),
                                       math.radians(40.0)))
        self.pub_cmd.publish(cmd)
        rospy.logwarn_throttle(0.5,
            "[SAPF] EMERGENCY AVOID — obstacle dalam d_safe (%.2f m), USV berhenti",
            self._obs_distance(obs_pos))

    # ------------------------------------------------------------------ #
    #  Avoidance activation — hysteresis berbasis d_rejoin               #
    # ------------------------------------------------------------------ #

    def _update_avoiding_state(self, obs_dist):
        """
        Masuk avoidance : obstacle terdeteksi DAN obs_dist < d_rep
        Keluar avoidance: obstacle tidak terdeteksi ATAU obs_dist > d_rejoin
        Hysteresis mencegah chattering saat USV berada di tepi zona.
        """
        if not self.obstacle_detected:
            self._avoiding = False
            return

        if not self._avoiding and obs_dist < self.apf.d_rep:
            self._avoiding = True
            rospy.loginfo("[SAPF] Avoidance AKTIF — obs_dist=%.2f m", obs_dist)
        elif self._avoiding and obs_dist > self.d_rejoin:
            self._avoiding = False
            rospy.loginfo("[SAPF] Avoidance SELESAI — obs_dist=%.2f m > d_rejoin=%.2f m",
                          obs_dist, self.d_rejoin)

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run(self):
        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
            if self.enabled:
                obs_pos  = self._estimate_obstacle_pos()
                obs_dist = self._obs_distance(obs_pos)

                self._update_avoiding_state(obs_dist)

                if self._avoiding:
                    goal             = self._get_goal()
                    force, emergency = self.apf.compute_force(
                        self.pos, goal, [obs_pos])

                    if emergency:
                        self._emergency_avoid(obs_pos)
                    else:
                        mag = np.linalg.norm(force)
                        if mag > 1e-6:
                            desired_hdg = math.atan2(force[1], force[0])
                            e_psi       = self._wrap(desired_hdg - self.heading)
                        else:
                            e_psi = 0.0

                        _RUDDER_MAX   = math.radians(40.0)
                        cmd           = Twist()
                        cmd.linear.x  = self.base_speed
                        cmd.angular.z = float(np.clip(e_psi,
                                                      -_RUDDER_MAX,
                                                       _RUDDER_MAX))
                        self.pub_cmd.publish(cmd)

                        rospy.loginfo_throttle(
                            1.0,
                            "[SAPF] obs=%.2fm  obs_dir=%.1f°  goal_dir=%.1f°  e_psi=%.1f°",
                            obs_dist,
                            math.degrees(math.atan2(obs_pos[1] - self.pos[1],
                                                    obs_pos[0] - self.pos[0])),
                            math.degrees(math.atan2(goal[1] - self.pos[1],
                                                    goal[0] - self.pos[0])),
                            math.degrees(e_psi),
                        )

            rate.sleep()


if __name__ == '__main__':
    try:
        node = ObstacleAvoidance()
        node.run()
    except rospy.ROSInterruptException:
        pass
