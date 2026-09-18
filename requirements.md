# 环境与依赖建议

当前天气地图、环境、传统规划器、RL训练、推理、GIF演示和自动化测试均只使用 Python 标准库，无强制第三方依赖。

## Python

- Python 3.11 或更高版本

## 可选增强依赖

以下依赖不是当前代码运行所必需，仅在将线性函数逼近器升级为CNN、增加批量张量训练或绘制科研图表时建议安装。

| 包 | 建议版本 | 用途 |
|---|---:|---|
| `numpy` | `>=1.26` | 栅格、观测和批量数值计算 |
| `gymnasium` | `>=0.29` | 接入第三方RL生态；当前环境接口已经兼容其核心调用方式 |
| `torch` | `>=2.2` | 将现有线性网络替换为双分辨率CNN |
| `tensorboard` | `>=2.16` | 训练指标记录 |
| `matplotlib` | `>=3.8` | 路径、天气和评估图表 |

## 可选开发工具

| 包 | 建议版本 | 用途 |
|---|---:|---|
| `pytest` | `>=8.0` | 可运行现有unittest测试，但不是必需 |
| `ruff` | `>=0.6` | 静态检查和格式检查 |

可在虚拟环境中按需安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy gymnasium torch tensorboard matplotlib pytest ruff
```

项目不使用传统的 `requirements.txt` 锁定直接依赖；本文件用于记录可选增强依赖及其用途。进入基于PyTorch的可复现实验阶段后，建议额外生成锁文件，但保留本说明作为人工可读入口。
