''' 강화학습 환경 세팅 '''
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import os
# PyRep말고 CoppeliaSim ZMQ 내부 라이브러리 사용
class Barista_2D_Env(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self):
        super(Barista_2D_Env, self).__init__()

        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        self.save_dir = "images"  # 이미지를 저장할 폴더 이름
        self.save_interval = 100  # 몇 스텝마다 저장할지
        self.step_count = 0
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"폴더 생성 완료: {self.save_dir}")
        else:
            print(f"이미지 저장 경로: {self.save_dir}")


        # zmq의 경우 coppeliaSim을 미리 켜놔야 함. (load_scene.py로 열거나 직접 열거나)
        try:
            self.sim.getSimulationState()
            print("CoppeliaSim 연결 성공")
        except Exception as e:
            print("CoppeliaSim 연결 실패.")
            raise e

        self.client.setStepping(True)

        # CoppeliaSim 객체 불러오기
        try:
            self.joint_1 = self.sim.getObject('/Base/joint1')
            self.joint_2 = self.sim.getObject('/Base/joint1/link1/joint2')
            self.camera = self.sim.getObject('/Vision_Sensor')
            self.end_effector = self.sim.getObject('/Base/joint1/link1/joint2/link2/End_Effector')
            self.target = self.sim.getObject('/Target')


            # =======================================================================
            # 객체 디버깅 코드
            # obj_type = self.sim.getObjectType(self.camera)
            # obj_name = self.sim.getObjectAlias(self.camera)
            # print(f" 카메라 핸들 값: {self.camera}")
            # print(f" 가져온 객체 이름: {obj_name}")
            # print(f" 객체 타입 번호: {obj_type}")

            # if obj_type != self.sim.object_visionsensor_type:
            #     print(f" 가져온 객체는 비전 센서가 아닙니다 (타입: {obj_type})")
            # ======================================================================
        except Exception as e:
            print(f"객체를 찾을 수 없습니다. 이름이 정확한지 확인하세요: {e}")
            raise e

        # Action Space  (Reacher Document 참고함)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation Space
        self.img_width = 84
        self.img_height = 84
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.img_height, self.img_width, 1),
            dtype=np.uint8
        )

    def _get_vision_obs(self):
        # 이미지 데이터 받기
        img_data, res = self.sim.getVisionSensorImg(self.camera)

        # 바이트 스트링을 uint8 Numpy 배열로 변환
        img = np.frombuffer(img_data, dtype=np.uint8)

        # Shape 변환 (Resolution Y, Resolution X, 3)
        # ZMQ API는 보통 1차원 배열로 주므로 3차원으로 복원해야 함
        img = img.reshape((res[1], res[0], 3))

        # 상하 반전 (CoppeliaSim 좌표계 보정)
        img = cv2.flip(img, 0)

        # Grayscale 변환 및 Resize
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)

        # 이미지가 어떻게 보이는 지 저장 (interval step 마다)
        if self.step_count % self.save_interval == 0:
            file_name = f"{self.save_dir}/step_{self.step_count:07d}.png"
            cv2.imwrite(file_name, img)
            #print(f"Saved: {file_name}")
        self.step_count += 1

        # 차원 추가 (H, W, 1) - CNN 입력용
        img = np.expand_dims(img, axis=-1)

        return img
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print("리셋")
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            pass

        self.sim.startSimulation()
        self.client.step()  # 첫 프레임 진행

        obs = self._get_vision_obs()
        info = {}
        return obs, info

    def step(self, action):
        # print("action 값 :", action)

        # ---------------------------------------------------------
        # 1. Action (Torque) 적용 [Reacher-v5 방식]
        # ---------------------------------------------------------
        # action 범위: -1.0 ~ 1.0
        MAX_TORQUE = 1.0  # Torque 힘 조절할 때 사용

        action = np.clip(action, -1.0, 1.0)  # action 최대, 최솟값 제한
        target_torques = action * MAX_TORQUE

        # 각 관절에 토크 적용
        joints = [self.joint_1, self.joint_2]

        for i, joint_handle in enumerate(joints):
            torque = float(target_torques[i])


            self.sim.setJointTargetForce(joint_handle, torque)


        # 물리 엔진 진행
        self.client.step()

        # ---------------------------------------------------------
        # 2. Reward 계산 (Reacher-v5 공식)
        # ---------------------------------------------------------

        # 거리 보상
        end_pos = np.array(self.sim.getObjectPosition(self.end_effector, -1))
        target_pos = np.array(self.sim.getObjectPosition(self.target, -1))
        distance = np.linalg.norm(end_pos - target_pos)

        reward_dist = -distance

        # 제어(Torque) 패널티
        # 힘을 많이 쓸수록 감점 (에너지 효율)
        reward_control = -0.1 * np.sum(np.square(action))

        # reward = 보상 + 패널티
        reward = reward_dist + reward_control

        # target에 도착하면 reset
        terminated = False
        if distance < 0.5:
            print(f"목표 도착! (Dist: {distance:.3f})")
            reward += 1.0
            terminated = True

        obs = self._get_vision_obs()
        info = {
            "distance": distance,
            "reward_dist": reward_dist,
            "reward_ctrl": reward_control
        }

        return obs, reward, terminated, False, info

    def close(self):
        self.sim.stopSimulation()