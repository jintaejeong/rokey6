#!/usr/bin/env python3
"""
evacuation_node.py (state machine)

목표(한 줄 요약):
- start_evacuation=True가 오면 “대피 안내 미션”을 시작한다.
- stand_detected(True)가 들어오면 “사람이 확보(따라오는 대상)”됐다고 판단한다.
- exit_goal([x, y, direction])이 설정되면 Nav2로 출구까지 주행한다.
- 주행 중 주기적으로 멈춰 뒤돌아(180도) 사람이 여전히 따라오는지 확인한다.
- 확인 시간(confirm_wait_sec) 동안 stand_detected가 다시 오면 OK → 다시 정면 보고 출구 주행 재개
- 안 오면: 마지막으로 사람을 봤던 pose로 되돌아가고, 한 바퀴 회전(recover_spin_rad)으로 탐색 후 출구 주행 재개

주의:
- 이 코드는 "타이머 기반 상태머신"으로 동작한다.
- tick()이 0.1초마다 호출되며, 상태(state)에 따라 Navigator에 명령을 내린다.
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


# ==========================================================
# 상태(State) 정의
# - tick()에서 self.state 값에 따라 분기한다.
# - "지금 로봇이 무엇을 하는 중인지"를 명확하게 표시하기 위한 enum.
# ==========================================================
class EvacuationState(Enum):
    IDLE = 0                # 미션 시작 전(비활성)
    WAIT_PERSON = 1         # 사람(stand_detected) 확보 기다림
    WAIT_EXIT = 2           # exit_goal 파라미터 세팅 기다림
    NAVIGATING = 3          # 출구로 Nav2 주행 중
    CHECK_SPIN_BACK = 4     # 주행 잠깐 멈추고 180도 뒤돌아보기(회전 수행 중)
    CHECK_WAIT_CONFIRM = 5  # 뒤돌아본 상태로 사람이 다시 감지되는지 대기
    CHECK_SPIN_FRONT = 6    # 다시 정면으로 돌아오기(원래 방향 회전 수행 중)
    RECOVER_GO_TO_LAST = 7  # 사람이 안 보이면 마지막으로 본 위치로 이동(복구)
    RECOVER_SPIN = 8        # 마지막 위치 도착 후 주변을 회전 탐색
    COMPLETED = 9           # 미션 성공 종료
    FAILED = 10             # 미션 실패 종료


class EvacuationNode(Node):
    def __init__(self):
        super().__init__("evacuation_node")

        # =========================
        # 1) Parameters (런타임 설정값)
        # =========================
        # tb4_ns: 로봇 네임스페이스 (예: robot6)  -> 토픽 prefix 및 Navigator namespace에 사용
        self.declare_parameter("tb4_ns", "robot6")

        # visible_timeout_sec:
        # - stand_detected(True)가 "최근에" 들어왔는지 판단하는 시간창(window)
        # - 예: 1.0이면 1초 이내에 True가 들어왔으면 "사람 보인다"로 판정
        self.declare_parameter("visible_timeout_sec", 1.0)

        # check_interval_sec:
        # - 출구로 이동(NAVIGATING) 중에 몇 초마다 뒤돌아 확인할지
        self.declare_parameter("check_interval_sec", 2.0)

        # confirm_wait_sec:
        # - 뒤돌아서 사람이 보이는지 기다리는 시간
        self.declare_parameter("confirm_wait_sec", 2.0)

        # turn_back_rad:
        # - 뒤돌아보기 각도(기본 pi = 180도)
        self.declare_parameter("turn_back_rad", math.pi)

        # exit_goal:
        # - 출구 goal을 파라미터로 받는 리스트
        # - 형태: [x, y, "NORTH/EAST/SOUTH/WEST"] (길이 3)
        self.declare_parameter("exit_goal", [])

        # recover_retry_limit:
        # - Nav2 주행이 실패했을 때 재시도 횟수 제한
        self.declare_parameter("recover_retry_limit", 2)

        # recover_spin_rad:
        # - 사람이 안 보일 때 복구 단계에서 회전 탐색할 총 회전량(기본 2pi = 360도)
        self.declare_parameter("recover_spin_rad", math.tau)

        # ---- 파라미터 값을 변수로 캐싱(읽기 쉽게) ----
        self.tb4_ns = str(self.get_parameter("tb4_ns").value).strip("/")
        self.visible_timeout_sec = float(self.get_parameter("visible_timeout_sec").value)
        self.check_interval_sec = float(self.get_parameter("check_interval_sec").value)
        self.confirm_wait_sec = float(self.get_parameter("confirm_wait_sec").value)
        self.turn_back_rad = float(self.get_parameter("turn_back_rad").value)
        self.recover_retry_limit = int(self.get_parameter("recover_retry_limit").value)
        self.recover_spin_rad = float(self.get_parameter("recover_spin_rad").value)

        # =========================
        # 2) State (상태/변수들)
        # =========================
        self.state = EvacuationState.IDLE  # 현재 상태
        self.active = False                # 미션이 활성화됐는지(시작됐는지)

        # 사람 관련 플래그/시간
        self.has_person = False            # stand_detected(True)를 한 번이라도 받았는지
        self.last_true_time = 0.0          # 마지막 stand_detected(True)가 들어온 시각(epoch time)

        # 주행 중 확인(check) 타이밍 제어용
        self.last_check_time = 0.0         # 마지막으로 "뒤돌아 확인"을 시작한 시각

        # 확인 대기(confirm) 제어용
        self.confirm_deadline = 0.0        # 확인 대기 끝나는 시각(이 시각 넘으면 사람 못 본 것으로 처리)
        self.check_reference_time = 0.0    # "이번 확인(check) 시작" 기준 시각(이후에 들어온 True만 인정)

        # 마지막으로 사람을 봤던 위치(복구용)
        # - cb_stand(True) 들어올 때 현재 pose를 저장
        self.last_person_pose = None

        # Nav2 실패 재시도 카운터
        self.recover_retries = 0

        # 복구 스핀을 이미 시작했는지 (RECOVER_SPIN 상태 내부의 2단계 진행을 위해)
        self.recover_spin_started = False

        # =========================
        # 3) Subscriptions (토픽 구독)
        # =========================
        # start_evacuation:
        # - 미션 시작 신호
        # - 로컬 토픽("start_evacuation")과 네임스페이스 토픽("/robot6/start_evacuation") 둘 다 구독
        self.create_subscription(Bool, "start_evacuation", self.cb_start, 10)
        self.create_subscription(Bool, f"/{self.tb4_ns}/start_evacuation", self.cb_start, 10)

        # stand_detected:
        # - 사람이 감지됐다는 신호(외부 비전/YOLO 노드 등에서 publish한다고 가정)
        # - 로컬/네임스페이스 토픽 둘 다 구독
        self.create_subscription(Bool, "stand_detected", self.cb_stand, 10)
        self.create_subscription(Bool, f"/{self.tb4_ns}/stand_detected", self.cb_stand, 10)

        # =========================
        # 4) Navigator (Nav2 제어 래퍼)
        # =========================
        # node_ns: 이 노드 자체가 어떤 namespace에서 실행되는지
        # nav_ns: TurtleBot4Navigator에게 넘길 namespace
        #
        # 의도:
        # - 노드가 이미 robot6 네임스페이스 안에서 돌고 있으면 nav_ns는 "" (중복 방지)
        # - 노드가 루트에서 돌고 있으면 nav_ns="robot6" (해당 로봇을 조종)
        node_ns = self.get_namespace().strip("/")
        nav_ns = "" if node_ns == self.tb4_ns else self.tb4_ns

        self.nav = TurtleBot4Navigator(namespace=nav_ns)

        # Nav2 stack이 활성화될 때까지 대기(AMCL/BT navigator 등 준비)
        self.nav.waitUntilNav2Active()

        # =========================
        # 5) Timer (상태머신 주기 실행)
        # =========================
        # 0.1초마다 tick()을 호출하여 상태머신을 굴린다.
        self.create_timer(0.1, self.tick)
        self.get_logger().info("EvacuationNode(state machine) ready.")

    # ======================================================
    # Callback: start_evacuation
    # - True가 들어오면 미션을 시작한다.
    # - 이미 active면 무시한다(중복 시작 방지)
    # ======================================================
    def cb_start(self, msg: Bool):
        if not msg.data:
            return  # False는 무시

        if self.active:
            return  # 이미 미션 수행 중이면 무시

        self.active = True
        self.state = EvacuationState.WAIT_PERSON
        self.get_logger().warn("[EVAC] start_evacuation=True -> mission start")

    # ======================================================
    # Callback: stand_detected
    # - True가 들어오면 "사람을 봤다"로 기록한다.
    # - last_true_time 갱신 + has_person True + 마지막 pose 저장
    # - recover_retries=0으로 초기화(사람 다시 보였으니 복구카운트 리셋 느낌)
    # ======================================================
    def cb_stand(self, msg: Bool):
        if msg.data:
            self.last_true_time = time.time()
            self.has_person = True
            self.last_person_pose = self.safe_get_current_pose()
            self.recover_retries = 0

    # ======================================================
    # Helper: stand_visible?
    # - "최근 visible_timeout_sec 안에 True를 받았는가?"
    # - 즉, 사람을 '지금도' 보고 있다고 볼지 판단
    # ======================================================
    def stand_visible(self) -> bool:
        return (time.time() - self.last_true_time) < self.visible_timeout_sec

    # ======================================================
    # Helper: stand_visible_since_check?
    # - "이번 확인(check) 시작 이후에 들어온 True인가?"
    # - check_reference_time 이후에 True가 들어왔고(=이번 확인 도중 감지),
    #   동시에 stand_visible() 조건도 만족해야 한다.
    # ======================================================
    def stand_visible_since_check(self) -> bool:
        return self.last_true_time > self.check_reference_time and self.stand_visible()

    # ======================================================
    # Helper: parse_exit_goal
    # - 파라미터 exit_goal을 읽어 Nav2 목표 pose를 만들기 위한 값으로 변환
    # - exit_goal은 [x, y, direction_string] 형태여야 함
    # - direction_string을 TurtleBot4Directions enum으로 변환
    # ======================================================
    def parse_exit_goal(self):
        raw = self.get_parameter("exit_goal").value
        if not raw or len(raw) != 3:
            return None  # 아직 설정 안 됐거나 형태가 다르면 None

        x, y, d = raw
        d = str(d).upper()

        # 문자열 방향 -> TurtleBot4Directions로 매핑
        if d == "NORTH":
            direction = TurtleBot4Directions.NORTH
        elif d == "EAST":
            direction = TurtleBot4Directions.EAST
        elif d == "SOUTH":
            direction = TurtleBot4Directions.SOUTH
        elif d == "WEST":
            direction = TurtleBot4Directions.WEST
        else:
            direction = TurtleBot4Directions.NORTH  # 디폴트

        return (float(x), float(y), direction)

    # ======================================================
    # Helper: start navigation to exit
    # - exit_goal이 유효하면 startToPose()로 주행 시작
    # - 유효하지 않으면 WAIT_EXIT 상태에서 계속 기다리게 됨
    # ======================================================
    def start_to_exit(self) -> bool:
        goal = self.parse_exit_goal()
        if goal is None:
            self.get_logger().warn("[EXIT] exit_goal not set yet -> waiting")
            return False

        x, y, d = goal

        # TurtleBot4Navigator 유틸을 사용해 PoseStamped 생성
        pose = self.nav.getPoseStamped([x, y], d)

        # 비동기 주행 시작(블로킹 아님)
        self.nav.startToPose(pose)

        # "뒤돌아 확인" 타이머 기준 시각 갱신
        self.last_check_time = time.time()

        self.get_logger().info(f"[EXIT] navigating to ({x:.2f},{y:.2f})")
        return True

    # ======================================================
    # Helper: start spin
    # - 지정한 rad 만큼 회전(spin)
    # - time_allowance는 spin 동작에 허용되는 최대 시간(초)
    # ======================================================
    def start_spin(self, rad: float, allowance: float = 8.0):
        self.nav.spin(spin_dist=float(rad), time_allowance=float(allowance))

    # ======================================================
    # Helper: safe_get_current_pose
    # - navigator가 getCurrentPose()를 제공하면 현재 pose를 얻는다.
    # - 일부 버전/환경에서 없을 수 있으니 방어적으로 처리
    # ======================================================
    def safe_get_current_pose(self):
        if hasattr(self.nav, "getCurrentPose"):
            return self.nav.getCurrentPose()
        self.get_logger().warn("[RECOVER] Navigator has no getCurrentPose(); skip pose capture")
        return None

    # ======================================================
    # Helper: begin_follower_check
    # - 출구 주행 중 "사람 따라오는지 확인" 절차 시작
    #
    # 순서:
    # 1) 현재 Nav2 task 취소(잠깐 멈춰야 하니까)
    # 2) check_reference_time 설정(이 시각 이후의 stand_detected True만 '이번 확인'으로 인정)
    # 3) confirm_deadline 설정(기다릴 시간 제한)
    # 4) 뒤로 180도 회전 시작
    # 5) 상태를 CHECK_SPIN_BACK으로 전환
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
    # - 0.1초마다 호출
    # - active=False면 아무것도 안 함
    # - 상태별로 Navigator 명령을 시작/완료 체크하며 다음 상태로 넘어감
    # ======================================================
    def tick(self):
        if not self.active:
            return  # 미션 비활성

        # --------------------------
        # WAIT_PERSON: 사람 확보 대기
        # --------------------------
        if self.state == EvacuationState.WAIT_PERSON:
            # stand_detected(True)를 한 번이라도 받으면 다음 단계
            if self.has_person:
                self.state = EvacuationState.WAIT_EXIT
            return

        # --------------------------
        # WAIT_EXIT: 출구 목표 파라미터 대기
        # --------------------------
        if self.state == EvacuationState.WAIT_EXIT:
            # exit_goal이 설정되면 출구 주행 시작
            if self.start_to_exit():
                self.state = EvacuationState.NAVIGATING
            return

        # --------------------------
        # NAVIGATING: 출구로 주행 중
        # --------------------------
        if self.state == EvacuationState.NAVIGATING:
            # 1) Nav2 task 종료 여부 체크
            if self.nav.isTaskComplete():
                result = self.nav.getResult()

                # 성공: 미션 종료
                if result == TaskResult.SUCCEEDED:
                    self.get_logger().warn("[EXIT] arrived!")
                    self.state = EvacuationState.COMPLETED
                    self.active = False

                # 실패/중단: 재시도 가능하면 WAIT_EXIT로 돌아가 다시 걸기
                elif self.recover_retries < self.recover_retry_limit:
                    self.recover_retries += 1
                    self.get_logger().warn(
                        "[EXIT] navigation failed -> retrying "
                        f"({self.recover_retries}/{self.recover_retry_limit})"
                    )
                    self.state = EvacuationState.WAIT_EXIT

                # 재시도 한도 초과: 실패 종료
                else:
                    self.get_logger().warn(f"[EXIT] finished but not success (result={result})")
                    self.state = EvacuationState.FAILED
                    self.active = False
                return

            # 2) 아직 주행 중이면 일정 주기마다 "뒤돌아 확인" 트리거
            now = time.time()
            if (now - self.last_check_time) >= self.check_interval_sec:
                self.last_check_time = now
                self.begin_follower_check()
            return

        # --------------------------
        # CHECK_SPIN_BACK: 뒤돌아보기 회전 수행 중
        # --------------------------
        if self.state == EvacuationState.CHECK_SPIN_BACK:
            # 회전이 끝나면 확인 대기 상태로
            if self.nav.isTaskComplete():
                self.state = EvacuationState.CHECK_WAIT_CONFIRM
                self.get_logger().info("[CHECK] waiting for stand detection")
            return

        # --------------------------
        # CHECK_WAIT_CONFIRM: 뒤돌아본 채로 사람 감지 대기
        # --------------------------
        if self.state == EvacuationState.CHECK_WAIT_CONFIRM:
            # 이번 확인 시작 이후에 stand_detected(True)가 들어오면 성공으로 간주
            if self.stand_visible_since_check():
                self.get_logger().info("[CHECK] stand detected -> resume")
                # 다시 정면으로 돌아가기(뒤돌았던 만큼 반대로 회전)
                self.start_spin(-self.turn_back_rad)
                self.state = EvacuationState.CHECK_SPIN_FRONT
                return

            # 시간 초과면 사람 못 본 것으로 간주 -> 복구 루틴
            if time.time() >= self.confirm_deadline:
                self.get_logger().warn("[CHECK] stand not detected -> recover")
                # 일단 정면으로 복귀(회전 반대로)
                self.start_spin(-self.turn_back_rad)
                self.state = EvacuationState.RECOVER_GO_TO_LAST
            return

        # --------------------------
        # CHECK_SPIN_FRONT: 정면 복귀 회전 수행 중
        # --------------------------
        if self.state == EvacuationState.CHECK_SPIN_FRONT:
            if self.nav.isTaskComplete():
                # 정면 복귀 완료 -> 다시 출구 주행 시작
                if self.start_to_exit():
                    self.state = EvacuationState.NAVIGATING
                else:
                    self.state = EvacuationState.WAIT_EXIT
            return

        # --------------------------
        # RECOVER_GO_TO_LAST:
        # - (정면 복귀 회전이 끝났다고 가정하고)
        # - 마지막으로 사람을 봤던 pose로 이동을 걸어준다.
        # --------------------------
        if self.state == EvacuationState.RECOVER_GO_TO_LAST:
            # 현재 진행 중인 task가 있으면(예: 정면 복귀 스핀) 끝날 때까지 기다림
            if not self.nav.isTaskComplete():
                return

            # last_person_pose가 없으면 복구 이동 불가 -> 그냥 출구 주행 재개
            if self.last_person_pose is None:
                self.get_logger().warn("[RECOVER] no last person pose -> resume exit")
                self.state = EvacuationState.WAIT_EXIT
                return

            # 마지막 사람 위치로 이동 시작
            self.nav.startToPose(self.last_person_pose)
            self.get_logger().warn("[RECOVER] moving to last person pose")
            self.recover_spin_started = False
            self.state = EvacuationState.RECOVER_SPIN
            return

        # --------------------------
        # RECOVER_SPIN:
        # - last_person_pose로 이동이 끝나면 주변 회전 탐색
        # - 탐색 중/후 stand_detected가 들어오면 사람 찾은 것으로 보고 출구 주행 재개
        # --------------------------
        if self.state == EvacuationState.RECOVER_SPIN:
            # 이동/회전 중이면 완료될 때까지 대기
            if not self.nav.isTaskComplete():
                return

            # 아직 복구 스핀을 시작하지 않았다면:
            # - 이동이 끝났고 지금 주변을 둘러볼 차례
            if not self.recover_spin_started:
                # 이동 직후 이미 사람이 보이면 바로 출구 재개
                if self.stand_visible():
                    self.get_logger().info("[RECOVER] stand detected -> resume exit")
                    self.state = EvacuationState.WAIT_EXIT
                    return

                # 한 바퀴 회전 탐색 시작
                self.start_spin(self.recover_spin_rad)
                self.recover_spin_started = True
                return

            # 스핀까지 끝난 상태:
            # - 그래도 사람 보이면/안 보이면 로그만 다르게 찍고 출구 재개
            if self.stand_visible():
                self.get_logger().info("[RECOVER] stand detected after spin -> resume exit")
            else:
                self.get_logger().warn("[RECOVER] stand not found -> resume exit")
            self.state = EvacuationState.WAIT_EXIT


def main(args=None):
    # ROS2 초기화
    rclpy.init(args=args)

    # 노드 생성
    node = EvacuationNode()

    # spin: 콜백/타이머(tick)를 계속 돌린다.
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 정리
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
