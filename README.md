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
- 等待加16航向动作、扫掠碰撞检测、动作掩码和1.0海里进场判定；
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
uv run python scripts/train_ppo.py --episodes 5000 --max-steps 1200
uv run python scripts/train_dqn.py --episodes 10 --max-steps 1200
uv run python scripts/train_sac.py --episodes 10 --max-steps 1200

uv run python scripts/evaluate.py ppo --checkpoint build/checkpoints/ppo_residual.pt
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
  --checkpoint build/checkpoints/ppo_residual.pt \
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
python scripts/train_ppo.py --episodes 5000 --max-steps 1200 --num-envs 8 --device auto
```

也可以在已激活的服务器 Python 环境中直接使用启动脚本。脚本默认训练
5000 个 episode、8 个并行环境，同时保存终端输出、TensorBoard 日志和模型检查点：

```bash
./shell/start_ppo_training.sh
```

如需在退出 SSH 后继续训练：

```bash
mkdir -p build/logs/ppo_residual
nohup ./shell/start_ppo_training.sh > build/logs/ppo_residual/nohup.log 2>&1 &
```

可通过环境变量修改默认参数，例如
`NUM_ENVS=4 DEVICE=cuda ./shell/start_ppo_training.sh`。命令行追加参数也会传给
`scripts/train_ppo.py`，例如 `./shell/start_ppo_training.sh --quiet`。

启动脚本默认将模型保存为 `build/checkpoints/ppo_residual.pt`，TensorBoard 和终端
日志写入 `build/logs/ppo_residual`。训练成功结束后脚本会自动关闭服务器；若训练
失败则不会关机。启动前脚本会检查当前用户是否为 root，或是否具有免密 `sudo`
关机权限，不满足条件时会在训练开始前退出。

PPO 默认启用四阶段课程学习，按全局 episode 编号依次使用无雷雨、静态雷雨、
移动雷雨和完整动态雷雨；最后 40% 的 episode 使用 8 个初始、最多 12 个雷雨
单体以及 1.0--1.5 的面积倍率。消融实验可添加 `--no-curriculum` 关闭。直升机
固定在西南角 `(5, 5)` 海里；驱逐舰航向从西北、北、东北、东、东南五个方向
抽样，起点按 seed 随机化，并保证与直升机至少相距 20 海里、按当前航向航行
一小时不越界且可在地图内截获。

动态天气阶段中至少 60% 的 episode 会强制无天气名义截获航线与移动雷雨
冲突，其中至少 20% 包含两段独立雷雨区。碰撞场景的 seed 会被自动加入各并行
worker 的困难 seed 池，每个池最多保留 128 个 seed。PPO 学习率为 `1e-4`，
每批更新 5 轮。

训练每 250 个 episode 使用 10 个固定困难 seed 进行确定性评估，每个场景
均经启动校验确认至少包含两段航路雷雨区。最终模型保存为
`build/checkpoints/ppo_residual.pt`，固定集上表现最好的模型另存为
`build/checkpoints/ppo_residual.best.pt`。可使用下面命令重新验证并生成每个
固定场景的 GIF/PNG：

```bash
./shell/test_ppo.sh
```

TensorBoard 除奖励和成功率外，还会记录碰撞率、超时率、航路冲突段数、
最小雷雨净空、等待/改向/避障决策步数、困难 seed 池大小，以及 actor loss、
value loss、熵和固定集评估指标。固定场景清单保存在日志目录下的
`evaluation_scenarios.json`。

学习算法使用残差动作空间：动作 0 表示等待，其余动作在时空 A* 给出的前方
3 海里引导航向上叠加 `-45°、-22.5°、0°、22.5°、45°`。规划器每 12 秒重规划，
使用 90 分钟的雷雨单体匀速外推预测；奖励包含规划剩余航程的势函数差分。
`config/ppoConfig.json` 以 0.2 的概率从初始及训练期间动态收集的困难 seed 中重放。

上述残差动作与旧版 17 动作绝对航向策略不兼容，因此需要从零训练新的 PPO
检查点；加载旧检查点时程序会报告动作空间不兼容。

依赖与建议安装方式见 [requirements.md](requirements.md)。
