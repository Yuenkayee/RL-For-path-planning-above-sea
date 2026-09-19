# RL-For-path-planning-above-sea

面向动态雷雨区和移动护卫舰的直升机返场路径规划仿真项目。主方案为全局时空规划引导的 Masked PPO，对照组包括 Masked Dueling Double DQN、离散最大熵 SAC、滚动时域 SIPP 和时空 A*。

## 工程边界

- `src/model/`：天气、直升机和护卫舰的物理数据模型。
- `src/env/`：统一强化学习环境、观测、奖励、动作掩码和终止条件。
- `src/planner/`：截获预测、引导图以及 SIPP/时空 A* 传统规划器。
- `src/algorithm/`：PPO、DQN、SAC 算法和共享网络组件。
- `src/training/`：样本收集、训练、课程学习、评估和检查点。
- `src/inference/`：训练后在线使用和传统规划基线执行。
- `src/experiment/`：实验组的唯一注册表。
- `config/`：天气、环境、算法和规划器配置。
- `scripts/`：供命令行调用的薄入口。
- `build/`：动图、日志、检查点和评估结果，不纳入版本控制。

算法定义和训练流程刻意分离：算法模块只实现策略、网络和参数更新；训练模块负责创建环境、收集样本、评估和保存模型。

## 已实现功能

- 高低分辨率共享二值天气地图；
- 随时间移动、成长和消散的雷雨区仿真；
- JSON 天气配置读取；
- 天气仿真 GIF 演示；
- 直升机两档速度和护卫舰匀速直线运动的数据模型。
- 6秒控制步与60秒天气步的统一返场环境；
- 等待加16航向动作、扫掠碰撞检测、动作掩码和0.1海里进场判定；
- 截获预测、引导图、SIPP和时空A*；
- 基于PyTorch双分辨率CNN的Masked PPO、Dueling Double DQN和离散SAC；
- Gymnasium标准动作空间、观测空间和环境接口；
- 课程学习、PyTorch检查点、TensorBoard日志、确定性推理和统一评估；
- NumPy批量观测、Matplotlib评估图和pytest自动化测试。

## 运行已有天气演示

从仓库根目录执行：

```bash
python3 src/test/weatherDemo.py
```

默认读取 `config/weatherConfig.json`，输出到 `build/weatherSimu/weather_simulation.gif`。

## 训练与评估

```bash
uv run python scripts/train_ppo.py --episodes 10 --max-steps 1200
uv run python scripts/train_dqn.py --episodes 10 --max-steps 1200
uv run python scripts/train_sac.py --episodes 10 --max-steps 1200

uv run python scripts/evaluate.py ppo --checkpoint build/checkpoints/ppo.pt
uv run python scripts/evaluate.py sipp --max-steps 1200
uv run python scripts/evaluate.py astar --max-steps 1200
```

查看训练曲线：

```bash
tensorboard --logdir build/logs --port 6006
```

在固定 `seed` 场景下回放路径规划过程（输出 GIF 动画和最终路径 PNG）：

```bash
python scripts/visualize_episode.py ppo \
  --checkpoint build/checkpoints/ppo.pt \
  --seed 0 \
  --max-steps 1200

# 不需要 checkpoint 的传统规划器
python scripts/visualize_episode.py sipp --seed 0 --max-steps 1200
```

默认输出到 `build/evaluation/<method>_seed<seed>.gif` 和同名 `.png`。

运行全部测试：

```bash
uv run pytest
uv run ruff check .
```

首次使用前执行 `uv sync --all-groups --frozen`。依赖声明位于 `pyproject.toml`，完整传递依赖由 `uv.lock` 锁定，详细说明见 [requirements.md](requirements.md)。

Ubuntu 22.04训练服务器可以直接配置已有的 Python 3.11–3.13 环境；该脚本使用
所选 Python 的 `pip`，不安装 `uv`、不安装 Python，也不创建 `.venv`：

```bash
./shell/check_training_server.sh --python /path/to/python3.11 --require-cuda
```

若已激活 Conda 或其他环境，可执行
`./shell/check_training_server.sh --python "$(command -v python)" --require-cuda`。仅检查而不安装任何内容时加 `--check-only`。

PPO默认并行运行4个环境，自动选择CUDA、Apple MPS或CPU，并每100步输出各episode进度：

```bash
python scripts/train_ppo.py --episodes 1000 --max-steps 1200 --num-envs 8 --device auto
```

依赖与建议安装方式见 [requirements.md](requirements.md)。
