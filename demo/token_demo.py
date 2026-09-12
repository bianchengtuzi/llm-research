## 演示 token 包的版本
## 用于检查 token 包的版本
from importlib.metadata import version

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))