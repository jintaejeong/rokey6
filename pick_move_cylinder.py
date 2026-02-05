import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.franka import Franka

import omni.usd
from pxr import UsdGeom, Gf, Sdf, Usd, UsdPhysics


class RMPFlowController(mg.MotionPolicyController):
    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
    ) -> None:
        cfg = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        rmp = mg.lula.motion_policies.RmpFlow(**cfg)
        amp = mg.ArticulationMotionPolicy(robot_articulation, rmp, physics_dt)
        super().__init__(name=name, articulation_motion_policy=amp)

        pos, ori = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(robot_position=pos, robot_orientation=ori)
        self._default_position, self._default_orientation = pos, ori

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )


class PickMoveCylinder:
    def __init__(self) -> None:
        self._world: World | None = None
        self._franka: Franka | None = None
        self.ctrl: RMPFlowController | None = None

        # ===== 스샷 원기둥과 동일하게 만들기 =====
        self.cyl_path = "/World/Cylinder"
        self.cyl_translate = np.array([0.0, 0.5, 0.0])  # X=0.0, Y=0.5, Z=0.0
        self.cyl_scale = np.array([0.05, 0.05, 0.3])  # X=0.05, Y=0.05, Z=0.3
        self.target_y = -0.7

        # ===== pick&place 튜닝 =====
        self.approach_z = 0.25
        self.grasp_z = 0.08
        self.lift_z = 0.25
        self.drop_height = 0.2

        self.ee_euler = np.array([0.0, np.pi, 0.0])
        self.phase = 0
        self.attached = False
        self.joint_path: str | None = None

    @property
    def world(self) -> World:
        if self._world is None:
            raise RuntimeError("World is not initialized. Call setup_scene() first.")
        return self._world

    @property
    def franka(self) -> Franka:
        if self._franka is None:
            raise RuntimeError("Franka is not initialized. Call setup_scene() first.")
        return self._franka

    def setup_scene(self) -> None:
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()
        self._franka = self._world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))

    # ✅ world.reset() 이후에 생성(안전)
    def create_cylinder_like_screenshot(self) -> None:
        stage = omni.usd.get_context().get_stage()
        path = Sdf.Path(self.cyl_path)

        # 없으면 새로 생성
        if not stage.GetPrimAtPath(path).IsValid():
            cyl = UsdGeom.Cylinder.Define(stage, path)
            # UI 기본 크기랑 비슷하게 맞추기: (radius=0.5, height=1.0) * scale
            cyl.CreateRadiusAttr(0.5)
            cyl.CreateHeightAttr(1.0)
            carb.log_warn(f"[Cylinder] created: {self.cyl_path}")
        else:
            carb.log_warn(f"[Cylinder] already exists: {self.cyl_path}")

        prim = stage.GetPrimAtPath(path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()

        xform.AddTranslateOp().Set(Gf.Vec3d(*self.cyl_translate.tolist()))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        xform.AddScaleOp().Set(Gf.Vec3f(*self.cyl_scale.tolist()))

        # 물리 설정 (RigidBody + Collision)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        if not prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(0.2)

    def setup_post_load(self) -> None:
        self.world.reset()

        # ✅ reset 후 몇 프레임 워밍업(컨트롤러 안정)
        for _ in range(5):
            self.world.step(render=True)

        # ✅ 원기둥 생성
        self.create_cylinder_like_screenshot()

        self.ctrl = RMPFlowController(name="rmpflow_controller", robot_articulation=self.franka)

        # 그리퍼 오픈
        self.franka.gripper.set_joint_positions(self.franka.gripper.joint_opened_positions)

        self.world.add_physics_callback("sim_step", self.physics_step)
        self.world.play()

    # ✅ 버전 영향 적은 월드좌표 읽기
    def get_world_position(self, prim_path: str) -> np.ndarray:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        m = cache.GetLocalToWorldTransform(prim)
        p = m.ExtractTranslation()
        return np.array([p[0], p[1], p[2]], dtype=float)

    def move_ee_to(self, goal_position: np.ndarray, ee_euler: np.ndarray | None = None) -> bool:
        if ee_euler is None:
            ee_euler = self.ee_euler
        q = euler_angles_to_quat(ee_euler)

        if self.ctrl is None:
            raise RuntimeError("Controller not initialized. Call setup_post_load() first.")

        action = self.ctrl.forward(
            target_end_effector_position=goal_position,
            target_end_effector_orientation=q,
        )
        self.franka.apply_action(action)

        # 도착 판정(간단)
        curr = self.franka.get_joint_positions()
        return bool(np.all(np.abs(curr[:7] - action.joint_positions) < 0.003))

    def attach_to_hand(self) -> None:
        if self.attached:
            return
        stage = omni.usd.get_context().get_stage()
        hand_path = "/World/Fancy_Franka/panda_hand"

        if not stage.GetPrimAtPath(hand_path).IsValid():
            raise RuntimeError(f"panda_hand not found: {hand_path}")
        if not stage.GetPrimAtPath(self.cyl_path).IsValid():
            raise RuntimeError(f"cylinder not found: {self.cyl_path}")

        joint_path = "/World/HandCylinderJoint"
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self.cyl_path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        self.joint_path = joint_path
        self.attached = True
        carb.log_warn(f"[Attach] {self.cyl_path} via {joint_path}")

    def detach_to_world(self) -> None:
        if not self.attached:
            return
        stage = omni.usd.get_context().get_stage()
        if self.joint_path and stage.GetPrimAtPath(self.joint_path).IsValid():
            stage.RemovePrim(self.joint_path)
        self.joint_path = None
        self.attached = False
        carb.log_warn(f"[Detach] {self.cyl_path}")

    def physics_step(self, step_size: float) -> None:
        # ✅ 콜백 죽는 것 방지 (안 움직이면 여기서 예외 터진 거임)
        try:
            cyl_pos = self.get_world_position(self.cyl_path)

            # 0) 위에서 접근
            if self.phase == 0:
                goal = np.array([cyl_pos[0], cyl_pos[1], cyl_pos[2] + self.approach_z])
                if self.move_ee_to(goal):
                    if self.ctrl is not None:
                        self.ctrl.reset()
                    self.phase = 1

            # 1) 내려가서 집기
            elif self.phase == 1:
                goal = np.array([cyl_pos[0], cyl_pos[1], cyl_pos[2] + self.grasp_z])
                if self.move_ee_to(goal):
                    self.franka.gripper.set_joint_positions(self.franka.gripper.joint_closed_positions)
                    self.attach_to_hand()
                    if self.ctrl is not None:
                        self.ctrl.reset()
                    self.phase = 2

            # 2) 들어올리기
            elif self.phase == 2:
                cyl_pos = self.get_world_position(self.cyl_path)
                goal = np.array([cyl_pos[0], cyl_pos[1], cyl_pos[2] + self.lift_z])
                if self.move_ee_to(goal):
                    if self.ctrl is not None:
                        self.ctrl.reset()
                    self.phase = 3

            # 3) ✅ y=-0.7 이동 (세워서 유지)
            elif self.phase == 3:
                cyl_pos = self.get_world_position(self.cyl_path)
                goal = np.array([cyl_pos[0], self.target_y, cyl_pos[2]])
                if self.move_ee_to(goal, ee_euler=self.ee_euler):
                    if self.ctrl is not None:
                        self.ctrl.reset()
                    self.phase = 4

            # 4) 0.2 높이에서 세워서 놓기 + detach
            elif self.phase == 4:
                cyl_pos = self.get_world_position(self.cyl_path)
                goal = np.array([cyl_pos[0], cyl_pos[1], max(self.drop_height, cyl_pos[2])])
                if self.move_ee_to(goal, ee_euler=self.ee_euler):
                    self.franka.gripper.set_joint_positions(self.franka.gripper.joint_opened_positions)
                    self.detach_to_world()
                    if self.ctrl is not None:
                        self.ctrl.reset()
                    self.phase = 5

            elif self.phase == 5:
                pass

        except Exception as e:
            carb.log_error(f"[physics_step] {e}")


def main() -> None:
    app = PickMoveCylinder()
    app.setup_scene()
    app.setup_post_load()

    try:
        while simulation_app.is_running():
            app.world.step(render=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
