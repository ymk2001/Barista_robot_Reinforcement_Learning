''' 강화학습 환경 세팅 '''
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import os
import time

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
        self.current_step = 0
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
            print("CoppeliaSim 연결 실패. 시뮬레이터를 켜주세요.")
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

        if len(img_data) ==0:
            return np.zeros((self.img_height, self.img_width,1), dtype=np.unit8)

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
        # if self.step_count % self.save_interval == 0:
        #     file_name = f"{self.save_dir}/step_{self.step_count:07d}.png"
        #     cv2.imwrite(file_name, img)
            #print(f"Saved: {file_name}")
        
        self.step_count += 1

        # 차원 추가 (H, W, 1) - CNN 입력용
        img = np.expand_dims(img, axis=-1)  #(H, W, 1)

        return img
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print("환경 리셋")
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)

            # [핵심 수정 2] 리셋할 때 동기화가 풀릴 수 있으니 다시 한번 겁니다.
        self.client.setStepping(True)

        self.sim.startSimulation()
        """
        # 관절을 랜덤 위치로 초기화 (Position Loss 시뮬레이션)
        if options and options.get('random_start', False):
            for joint in [self.joint_1, self.joint_2]:
                random_pos = np.random.uniform(-np.pi, np.pi)
                self.sim.setJointPosition(joint, random_pos)
        """

        # 1. 관절 각도를 랜덤하게 설정
        # 범위: -1 ~ 1 라디안 (-180도 ~ 180도)
        # random_j1 = np.random.uniform(-1.0, 1.0)

        # 30도를 라디안으로 변환 (약 0.52 rad)
        limit_angle = 30 * (np.pi / 180) 
        
        # 50% 확률로 왼쪽 구간(-180 ~ -30) 또는 오른쪽 구간(30 ~ 180) 선택
        if np.random.rand() < 0.5:
            # 왼쪽 구간: -pi(-180도) ~ -limit_angle(-30도)
            random_j1 = np.random.uniform(-np.pi, -limit_angle)
        else:

            # 오른쪽 구간: limit_angle(30도) ~ pi(180도)
            random_j1 = np.random.uniform(limit_angle, np.pi)

        random_j2 = np.random.uniform(-np.pi/2, np.pi/2)

        self.sim.setJointPosition(self.joint_1, random_j1)
        self.sim.setJointPosition(self.joint_2, random_j2)

        self.current_step = 0   # [추가] 에피소드 시작 시 스텝 0으로 초기화
        # python과 coppeliasim 동기화
        self.sim.setBoolParam(self.sim.boolparam_realtime_simulation, 0)

        self.client.step()  # 첫 프레임 진행

        obs = self._get_vision_obs()
        info = {}
        return obs, info

    def step(self, action):
        # print("action 값 :", action)

        # ---------------------------------------------------------
        # 1. Action 적용 (속도 기반 제어로 변경)
        # ---------------------------------------------------------
        MAX_VELOCITY = 0.8  # 최대 각속도 (rad/s)
        
        action = np.clip(action, -1.0, 1.0)
        target_velocities = action * MAX_VELOCITY

        joints = [self.joint_1, self.joint_2]

        for i, joint_handle in enumerate(joints):
            self.sim.setJointTargetVelocity(joint_handle, float(target_velocities[i]))
            self.sim.setJointMaxForce(joint_handle, 50.0)

        """
                joints = [self.joint_1, self.joint_2]

        for i, joint_handle in enumerate(joints):
            velocity = float(target_velocities[i])
            
            # 속도 기반 제어 (더 안정적)
            self.sim.setJointTargetVelocity(joint_handle, velocity)
            
            # 충분한 토크 제공
            self.sim.setJointMaxForce(joint_handle, 100.0)


        """


        """
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

        """

        # 물리 엔진 진행
        self.client.step()

        # ---------------------------------------------------------
        # 2. Reward 계산 (Reacher-v5 공식)
        # ---------------------------------------------------------

        # 거리 보상
        end_pos = np.array(self.sim.getObjectPosition(self.end_effector, -1))
        target_pos = np.array(self.sim.getObjectPosition(self.target, -1))
        distance = np.linalg.norm(end_pos - target_pos)

        reward_dist = -distance * 0.5

        # 제어(Torque) 패널티
        # 힘을 많이 쓸수록 감점 (에너지 효율)
        reward_control = -0.01 * np.sum(np.square(action))

        # reward = 보상 + 패널티
        reward = reward_dist + reward_control

        # target에 도착하면 reset
        terminated = False
        if distance < 0.1:
            print(f"목표 도착! (Dist: {distance:.3f})")
            reward += 10.0
            terminated = True

        obs = self._get_vision_obs()
        info = {
            "distance": distance,
            "reward_dist": reward_dist,
            "reward_ctrl": reward_control
        }

        self.current_step += 1
        if self.current_step >= 3000: # 500 스텝 넘으면 강제 종료
            truncated = True # gymnasium 최신 버전은 terminated 대신 truncated 사용 권장
        else:
            truncated = False

        return obs, reward, terminated, truncated, info

    def close(self):
        self.sim.stopSimulation()
        print("환경 종료")