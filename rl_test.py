import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from barista_2D_env import Barista_2D_Env


def main():
    env = None

    try:
        # 환경 생성
        env = Barista_2D_Env()

        # 환경 검증
        check_env(env)

        # 모델 정의 (Vision Sensor 이므로 CnnPolicy 사용, PPO 내장 policy)
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_robot_tensorboard/")

        # 학습 시작
        model.learn(total_timesteps=10000)

        # 모델 저장
        model.save("ppo_coppelia_robot")

    except KeyboardInterrupt:
        print("\n실험 종료")

    finally:
        if env is not None:
            print("CoppeliaSim 시뮬레이션 종료")
            env.close()


if __name__ == "__main__":
    main()