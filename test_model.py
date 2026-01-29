import gymnasium as gym
from stable_baselines3 import PPO
from barista_2D_env import Barista_2D_Env
import time
import numpy as np


def evaluate_model(model_path, num_episodes=10):
    #episode 기본값은 10. 아래 main에서 설정 가능
    env = None
    try:

        env = Barista_2D_Env()

        print(f"모델 로딩 중: {model_path}...")
        model = PPO.load(model_path, env=env)
        print("모델 로드 완료!")

        success_count = 0
        total_steps_list = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            step = 0

            print(f"Episode {episode + 1}/{num_episodes}", end="")

            while not done:
                action, _states = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                step += 1


                if terminated:
                    print(f" 성공! (Steps: {step})")
                    success_count += 1
                    total_steps_list.append(step)
                    done = True


                elif truncated:
                    print(f" ❌ 실패 (시간 초과).)")
                    done = True


        success_rate = (success_count / num_episodes) * 100
        avg_steps = np.mean(total_steps_list) if total_steps_list else 0

        print("테스트 결과")
        print(f"총 시도: {num_episodes}회")
        print(f"성공 횟수: {success_count}회")
        print(f"성공률: {success_rate:.1f}%")
        if success_count > 0:
            print(f"평균 소요 스텝: {avg_steps:.1f} steps")
        else:
            print("   - 성공한 에피소드가 없습니다.")

    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    # 저장했던 모델 이름 입력 (확장자 .zip은 생략)
    evaluate_model("./exp/ppo_barista_robot_0129_1241", num_episodes=100)