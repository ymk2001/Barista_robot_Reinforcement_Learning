import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import get_linear_fn
from barista_2D_env import Barista_2D_Env
import wandb
from wandb.integration.sb3 import WandbCallback
from datetime import datetime
import os


def main():
    # Terminal로 실행 할 경우 미리 wandb login 필요
    # API Key 입력까지

    # Pycharm과 같은 IDE에서 실행 할 경우 IDE terminal에서 wandb login 필요
    wandb.login()
    exp_path = "exp"
    os.makedirs(exp_path, exist_ok=True)

    current_time = datetime.now().strftime("%m%d_%H%M") # 실험 구분을 위해
    run_name = f"PPO_Run_{current_time}"
    env = None
    run = wandb.init(
        project="Barista_Robot_RL",  # 이 부분 이름 바꾸고 실험하는 게 좋음
        name=run_name,
        config={
            "policy_type": "CnnPolicy",
            "total_timesteps": 100000,
            "learning_rate_start": 3e-4,
            "learning_rate_end": 1e-5,
            "env_id": "Barista_2D_Env",
        },
        sync_tensorboard=True,  # SB3의 로그를 WandB 대시보드로 자동 동기화
        monitor_gym=True,  # (선택) 환경 비디오 기록
        save_code=True,  # (선택) 현재 코드 저장
    )
    try:
        # 환경 생성
        env = Barista_2D_Env()

        # 환경 검증
        check_env(env)

        # 학습률이 0.0003에서 시작해 0으로 줄어들게 설정
        lr_schedule = get_linear_fn(start=3e-4, end=1e-5, end_fraction=1.0)

        # 모델 정의 (Vision Sensor 이므로 CnnPolicy 사용, PPO 내장 policy)
        model = PPO("CnnPolicy", env, learning_rate=lr_schedule, verbose=1, tensorboard_log="./ppo_robot_tensorboard/")
        print(f"--> 학습 시작 (WandB Project: Barista_Robot_RL / Run ID: {run.id})")
        # 학습 시작
        model.learn(
            total_timesteps=100000,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            )
        )
        save_file_name = f"ppo_barista_robot_{current_time}"
        final_path = os.path.join(exp_path, save_file_name)
        # 모델 저장
        model.save(final_path)
        print(f"✅ 모델 저장 완료: {final_path}.zip")
    except KeyboardInterrupt:
        print("\n실험 종료")

    finally:
        if env is not None:
            print("CoppeliaSim 시뮬레이션 종료")
            env.close()
        wandb.finish()

if __name__ == "__main__":
    main()