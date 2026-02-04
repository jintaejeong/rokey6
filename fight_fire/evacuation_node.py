#!/usr/bin/env python3
"""
evacuation_node.py (state machine)

- start_evacuation=True 받으면 대피 안내 시작
- exit_goal 파라미터가 설정되면 Nav2 goal을 걸어 출구로 이동
- 이동 중 "check_interval_sec"마다 멈춰서 뒤돌아(180도) 사람이 따라오는지 확인
- 확인 대기(confirm_wait_sec) 동안 stand_detected가 다시 들어오면 OK
- 확인 후 다시 정면으로 돌아 출구 이동 재개
"""

import math
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from turtlebot4_navigation.turtlebot4_navigator import (
    TurtleBot4Directions,
    TurtleBot4Navigator,
)
from nav2_simple_commander.robot_navigator import TaskResult


class EvacuationState(Enum):
    IDLE = 0
    WAIT_PERSON = 1
    WAIT_EXIT = 2
    NAVIGATING = 3
    CHECK_SPIN_BACK = 4
    CHECK_WAIT_CONFIRM = 5
    CHECK_SPIN_FRONT = 6
    COMPLETED = 7
    FAILED = 8


class EvacuationNode(Node):
    def __init__(self):
        super().__init__("evacuation_node")

        # =========================
        # 1) Parameters
        # =========================
        self.declare_parameter("tb4_ns", "robot6")
        self.declare_parameter("visible_timeout_sec", 1.0)
        self.declare_parameter("check_interval_sec", 2.0)
        self.declare_parameter("confirm_wait_sec", 2.0)
        self.declare_parameter("turn_back_rad", math.pi)
        self.declare_parameter("exit_goal", [])

        self.tb4_ns = str(self.get_parameter("tb4_ns").value).strip("/")
        self.visible_timeout_sec = float(self.get_parameter("visible_timeout_sec").value)
        self.check_interval_sec = float(self.get_parameter("check_interval_sec").value)
        self.confirm_wait_sec = float(self.get_parameter("confirm_wait_sec").value)
        self.turn_back_rad = float(self.get_parameter("turn_back_rad").value)

        # =========================
        # 2) State
        # =========================
        self.state = EvacuationState.IDLE
        self.active = False
        self.has_person = False
        self.last_true_time = 0.0
        self.last_check_time = 0.0
        self.confirm_deadline = 0.0
        self.check_reference_time = 0.0

        # =========================
        # 3) Subscriptions
        # =========================
        self.create_subscription(Bool, "start_evacuation", self.cb_start, 10)
        self.create_subscription(Bool, f"/{self.tb4_ns}/start_evacuation", self.cb_start, 10)

        self.create_subscription(Bool, "stand_detected", self.cb_stand, 10)
        self.create_subscription(Bool, f"/{self.tb4_ns}/stand_detected", self.cb_stand, 10)

        # =========================
        # 4) Navigator
        # =========================
        node_ns = self.get_namespace().strip("/")
        nav_ns = "" if node_ns == self.tb4_ns else self.tb4_ns

        self.nav = TurtleBot4Navigator(namespace=nav_ns)
        self.nav.waitUntilNav2Active()

        # =========================
        # 5) Timer
        # =========================
        self.create_timer(0.1, self.tick)
        self.get_logger().info("EvacuationNode(state machine) ready.")

    # ======================================================
    # Callback: start_evacuation
    # ======================================================
    def cb_start(self, msg: Bool):
        if not msg.data:
            return

        if self.active:
            return

        self.active = True
        self.state = EvacuationState.WAIT_PERSON
        self.get_logger().warn("[EVAC] start_evacuation=True -> mission start")

    # ======================================================
    # Callback: stand_detected
    # ======================================================
    def cb_stand(self, msg: Bool):
        if msg.data:
            self.last_true_time = time.time()
            self.has_person = True

    # ======================================================
    # Helper: stand visible? (recent True)
    # ======================================================
    def stand_visible(self) -> bool:
        return (time.time() - self.last_true_time) < self.visible_timeout_sec

    # ======================================================
    # Helper: stand visible after check started?
    # ======================================================
    def stand_visible_since_check(self) -> bool:
        return self.last_true_time > self.check_reference_time and self.stand_visible()

    # ======================================================
    # Helper: parse exit goal
    # ======================================================
    def parse_exit_goal(self):
        raw = self.get_parameter("exit_goal").value
        if not raw or len(raw) != 3:
            return None

        x, y, d = raw
        d = str(d).upper()

        if d == "NORTH":
            direction = TurtleBot4Directions.NORTH
        elif d == "EAST":
            direction = TurtleBot4Directions.EAST
        elif d == "SOUTH":
            direction = TurtleBot4Directions.SOUTH
        elif d == "WEST":
            direction = TurtleBot4Directions.WEST
        else:
            direction = TurtleBot4Directions.NORTH

        return (float(x), float(y), direction)

    # ======================================================
    # Helper: start navigation to exit
    # ======================================================
    def start_to_exit(self) -> bool:
        goal = self.parse_exit_goal()
        if goal is None:
            self.get_logger().warn("[EXIT] exit_goal not set yet -> waiting")
            return False

        x, y, d = goal
        pose = self.nav.getPoseStamped([x, y], d)
        self.nav.startToPose(pose)

        self.last_check_time = time.time()
        self.get_logger().info(f"[EXIT] navigating to ({x:.2f},{y:.2f})")
        return True

    # ======================================================
    # Helper: start spin
    # ======================================================
    def start_spin(self, rad: float, allowance: float = 8.0):
        self.nav.spin(spin_dist=float(rad), time_allowance=float(allowance))

    # ======================================================
    # Helper: start follower check
    # ======================================================
    def begin_follower_check(self):
        self.nav.cancelTask()
        self.check_reference_time = time.time()
        self.confirm_deadline = time.time() + self.confirm_wait_sec
        self.start_spin(self.turn_back_rad)
        self.state = EvacuationState.CHECK_SPIN_BACK
        self.get_logger().info("[CHECK] pause + spin back")

    # ======================================================
    # Main tick (state machine)
    # ======================================================
    def tick(self):
        if not self.active:
            return

        if self.state == EvacuationState.WAIT_PERSON:
            if self.has_person:
                self.state = EvacuationState.WAIT_EXIT
            return

        if self.state == EvacuationState.WAIT_EXIT:
            if self.start_to_exit():
                self.state = EvacuationState.NAVIGATING
            return

        if self.state == EvacuationState.NAVIGATING:
            if self.nav.isTaskComplete():
                result = self.nav.getResult()
                if result == TaskResult.SUCCEEDED:
                    self.get_logger().warn("[EXIT] arrived!")
                    self.state = EvacuationState.COMPLETED
                    self.active = False
                else:
                    self.get_logger().warn(f"[EXIT] finished but not success (result={result})")
                    self.state = EvacuationState.FAILED
                    self.active = False
                return

            now = time.time()
            if (now - self.last_check_time) >= self.check_interval_sec:
                self.last_check_time = now
                self.begin_follower_check()
            return

        if self.state == EvacuationState.CHECK_SPIN_BACK:
            if self.nav.isTaskComplete():
                self.state = EvacuationState.CHECK_WAIT_CONFIRM
                self.get_logger().info("[CHECK] waiting for stand detection")
            return

        if self.state == EvacuationState.CHECK_WAIT_CONFIRM:
            if self.stand_visible_since_check():
                self.get_logger().info("[CHECK] stand detected -> resume")
                self.start_spin(-self.turn_back_rad)
                self.state = EvacuationState.CHECK_SPIN_FRONT
                return

            if time.time() >= self.confirm_deadline:
                self.get_logger().warn("[CHECK] stand not detected -> resume anyway")
                self.start_spin(-self.turn_back_rad)
                self.state = EvacuationState.CHECK_SPIN_FRONT
            return

        if self.state == EvacuationState.CHECK_SPIN_FRONT:
            if self.nav.isTaskComplete():
                if self.start_to_exit():
                    self.state = EvacuationState.NAVIGATING
                else:
                    self.state = EvacuationState.WAIT_EXIT
            return


def main(args=None):
    rclpy.init(args=args)
    node = EvacuationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
