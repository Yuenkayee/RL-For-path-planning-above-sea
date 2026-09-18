# 环境与依赖建议

项目使用 `pyproject.toml` 声明依赖，并使用 `uv.lock` 锁定完整依赖图。不要手工编辑 `uv.lock`。

## Python

- Python 3.11 或更高版本

## 运行依赖

| 包 | 建议版本 | 用途 |
|---|---:|---|
| `numpy` | `>=1.26` | 栅格、观测和批量数值计算 |
| `gymnasium` | `>=0.29,<2` | 标准环境、动作空间和观测空间 |
| `torch` | `>=2.2,<3` | 双分辨率CNN和PPO、DQN、SAC训练 |
| `tensorboard` | `>=2.16` | 训练指标记录 |
| `matplotlib` | `>=3.8` | 路径、天气和评估图表 |

## 开发与测试依赖

| 包 | 建议版本 | 用途 |
|---|---:|---|
| `pytest` | `>=8.0,<10` | 单元测试和端到端集成测试 |
| `ruff` | `>=0.6` | 静态检查和格式检查 |

## 创建完全一致的环境

```bash
uv sync --all-groups --frozen
```

Ubuntu 22.04训练服务器可以直接运行预检脚本。脚本会检查并安装缺失的uv、Python 3.11和锁定依赖：

```bash
./shell/check_training_server.sh
```

若服务器必须使用NVIDIA GPU训练：

```bash
./shell/check_training_server.sh --require-cuda
```

`--frozen`确保安装严格使用现有锁文件，不在本地隐式更新版本。更新依赖时修改 `pyproject.toml`，然后执行：

```bash
uv lock --upgrade
uv sync --all-groups
```

当前锁文件由uv基于Python 3.11解析，为macOS、Linux和Windows保留环境标记。测试和静态检查命令：

```bash
uv run pytest
uv run ruff check .
```
