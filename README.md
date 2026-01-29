# Barista Robot 2D Reinforcement Learning 기본 셋팅


## < Version >	

Ubuntu : 20.04

CoppeliaSim : 4.9.0_Edu

Python: 3.10.19	

PyTorch: 2.6.0+cu126	

CUDA Available: True (Version : 12.6)	

Device: NVIDIA GeForce RTX 3060	

## < 환경 설치 >

```conda create -n name python=3.10 -y```

```conda activate name ```

``` pip install -r requirements.txt ```


``` pip install wandb ```

## < Code >

#### load_scene.py
: 특정 ttt파일 coppeliaSim으로 열게 해주는 코드, 

ZMQ library를 사용할 경우 코드 실행 전에 coppeliasim이 먼저 실행되어있어야 하므로 coppelia Sim 먼저 키고, 이 코드 실행해서 파일 불러온 뒤, 강화학습 코드 실행시키면 된다.

#### barista_2D_env.py
: 2D 강화학습 환경. safety_rl_2dof.ttt 파일 기준의 환경

#### train.py
: 강화학습 main 실행 코드

#### test_model.py
: model test 코드

참고한 Document : Reacher in Mujoco Document --> https://gymnasium.farama.org/environments/mujoco/reacher/

