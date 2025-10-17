## 新的配置系统（使用hydra-core包）

使用方式：在脚本的main函数中，加入装饰器

```python
@hydra.main(version_base=None, config_path="configs", config_name="config")
```

Hydra 会自动加载 configs/config.yaml，并根据其中的 defaults 设置组合出完整的配置。

最大的好处是你可以从命令行轻松覆盖任何配置，而无需修改文件。而且支持分层的配置，如环境配置和agent超参数配置分开，可以自由组合。

例如，要将设备更改为 cpu：

```python
python train.py device=cpu
```

要更改 steps_per_train 参数：

```python
python train.py env.steps_per_train=512
```

如果你未来在 configs/agent/ 目录下添加了另一个 agent 的配置，例如 my_awesome_agent.yaml，你可以通过以下方式轻松切换：

```python
python train.py agent=my_awesome_agent
```
