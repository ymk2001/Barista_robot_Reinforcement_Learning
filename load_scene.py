''' 열고자 하는 ttt파일 coppeliaSim으로 열게 하는 코드 '''

import os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ttt파일이랑 이 코드가 같은 디렉토리에 있어야 함
current_dir = os.path.dirname(os.path.abspath(__file__))
scene_path = os.path.join(current_dir, 'safety_rl_2dof.ttt')

print(f"불러올 장면 경로: {scene_path}")

client = RemoteAPIClient()
sim = client.require('sim')

try:
    # 기존에 실행 중인 시뮬레이션이 있으면 멈춤
    sim.stopSimulation()

    sim.loadScene(scene_path)

    sim.startSimulation()
    print("CoppeliaSim 실행 완료")

except Exception as e:
    print(f"❌ 오류 발생: {e}")