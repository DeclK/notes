# FullQuantTransformer

🤗 **FullQuantTransformer (FQT) 是一个简洁、高效、强大的量化工具箱**

## HightLight

1. Easy to hack：少抽象，易理解，轻量化（~1000行核心量化代码）
2. Easy to use：只需要~4行核心代码即可完成量化
3. Flex DataRecorder：快速适配任何项目以构建 calibration set
4. 支持多种量化方法以及多种精度实现：GPTQ, AWQ, SmoothQuant, W4/W8/A4/A8/A16
5. 支持各种不同模型的量化：Qwen Dense, MoE, ViT
6. 支持 w4a16 模型压缩，兼容云端 serving 框架（vllm & sglang）
7. 支持量化加速算子：w8a8 scaled mm (int8 & fp8)
8. 工具齐全：激活可视化、日志系统、io 接口、timer benchmark

## Code Structure Overview

TODO：思维导图

## Code Design

接下来将更具体地介绍一些核心模块的功能以及其使用方式，如果有比较特别的设计思路也将进行描述

### Data

#### DataRecorder

在量化框架中，校准集的收集是必经之路。在量化算法中，校准集通常就是模型前向过程中的输入 or 输出。要获得这些输入 or 输出，可以利用 pytorch 提供的 `register_forward_hook` 接口，该接口会在 `nn.Module `运行其 `forward` 方法之前，调用所注册的 hook 函数，并且给 hook 函数传入其 `forward` 中的输入参数。

那么 `DataRecorder` 的设计思路就很简单了：只需要把 hook 函数设置为保存这些输入参数即可，hook 函数和保存的内容交由 `DataRecorder` 进行管理

`DataRecorder` 的**最典型**用法如下：

```python
model = Qwen2ForCausalLM(config)

# create data recorder, set target module & saving device
dc = DataRecorder(model=model.model.layers[0],
                  device="cpu")

dc.register_input_hooks()		# register input hooks
dc.register_pos_embed_hooks()	# register pos emb hooks

for i, prompt in enumerate(dataset): # model evaluation loop
	model.generate(prompts)
    dc.save_inputs(f"{i}.pkl")	# save inputs
    dc.clear_inputs()			# clear inputs for next round conversation

dc.remove_input_hooks()			# remove hooks
```

在上述例子中，使用了 `register_input_hooks` 方法，该方法会保存所指定模型的第一个 positional argument，该 argument 对于 transformer layers 来说就是所输入的 hidden states

```python
class Qwen2DecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ...
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
```

保存的 `i.pkl` 是一个 dict of list，其保存了一个 generate 过程中每次 forward 所保存所 input hidden states & position embeddings

```python
{
    "model_input": [hs_prefill, hs_decode_0, hs_decode_1, ...],
    "model_pos_embed": [pos_prefill, pos_decode_0, pos_decode_1, ...]
}
```

除此之外，还有其他的 hooks 可用，可按需取用：

1. `register_linear_input_hooks`，会记录该模型中所有 linear 层的 input，保存关键字为 linear 层的名字
2. `register_input_ids_hooks`，会记录模型输入中的 `input_ids`，保存关键字为 `model_input_ids`
3. `register_output_hooks`，会记录模型输出中的所有 output，保存关键字为 `model_output`

#### DefaultDataset

`DefaultDataset` 用于配合 `DataRecorder`，对保存的数据做 concat 处理：因为一次 generate 数据会保存 prefill + decode 的所有激活值，需要进行 concat 处理以方便计算。`DefaultDataset` 中每一个 item 为一个 dict：

1. `input_tensor`，即为 `hidden_states (B, N, C)` 
2. `kwargs`：一个 dict，包含 `position_ids (B, N)`  & `position_embeddings (B, N, C)`

在 `dataset.py` 中还有一个 `build_calib_datset_inputs` 函数，该函数就是将 dataset 中的随机 `num_samples` 个数据进行打包，构建符合量化算法的 inputs

### Quantizer

为了实现多种量化算法，FQT 实现了一个灵活的 quantizer，其能够实现以下**对称 int 量化**

1. per-channel，针对于二维张量 `(N, K)`，计算出一个 `(1, K)` 的 scale
2. per-token，针对于多维张量 `(.., M, K)`，计算出一个 `(..., M, 1)` 的 scale
3. per-tensor，针对于多维张量，计算其最大值为 scale
4. per-block，针对于多维张量 `(..., M, K)`，计算出一个 `(..., M, G)` 的 scale，其中 `G = K // group size`

构建 `Quantizer` 需要确定以下参数

```python
class Quantizer(nn.Module):
    def __init__(self, 
                 bits: int, 
                 method: str, 
                 group_size: int = None):
```

1. bits，量化比特，无任何限制
2. method，量化方法，四选一 `["per-tensor", "per-token", "per-channel", "per-block"]`
3. group size，专门为 `per-block` 量化方法所提供的参数，可选

`Quantizer` 最重要的两个方法：

1. `find_scale(x)`，计算输入 `x` 的 scale，保存到 `self.scale` 当中
2. `quantize(x)`，计算输入 `x` 的量化结果，包含了 quantize & dequantize 过程，故返回值并非整数，而是浮点

#### FakeQuantLinear

有了灵活的 `Quantizer`，FQT 构建一个 `FakeQuantLinear` 就变得简单起来了，一下是构建 `FakeQuantLinear` 所需要的参数

```python
class FakeQuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        w_bits=8,
        a_bits=8,
        w_method="per-token",
        a_method="per-token",
        w_group_size=None,
        a_group_size=None,
    ):
```

可以看到，除了普通 linear 所需要的参数外，还有和 weight & activation quantizer 相关的参数。所以该 `FakeQuantLinear` 支持任意精度、任意量化方法的伪量化线性层。通常，构建 `FakeQuantLinear` 的权重通常来自于原模型的 linear，所以 `FakeQuantLinear` 还提供了一个 `from_module` 的方法，用于从已存在的 linear 创建

```python
FakeQuantLinear.from_module(linear, w_bits=8, a_bits=8)
```

同时 FQT 还预设了一些常用的伪量化线性层，可直接通过 `from_module` 快速生成对应伪量化线性层

```python
class W8A8FakeLinear(FakeQuantLinear):
    @classmethod
    def from_module(cls, module, w_method="per-token", a_method="per-token"):
        return FakeQuantLinear.from_module(module, 8, 8, w_method=w_method, a_method=a_method)


class W4A16FakeLinear(FakeQuantLinear):
    @classmethod
    def from_module(cls, module, w_method="per-token", a_method="per-token"):
        return FakeQuantLinear.from_module(module, 4, 16, w_method=w_method, a_method=a_method)
```

#### QuantLinear

`FakeQuantLinear` 是作为伪量化线性层，其权重仍然是浮点，而真正的量化线性层其权重为 int，在 FQT 中对 `QuantLinear` 的支持比较有限，其作用仅为对权重进行压缩，无法使用压缩权重进行 forward 计算

构建一个 `QuantLinear` 的方式也推荐使用 `from_module` 方法

```python
QuantLinear.from_module(linear, bits=4, method="per-block", group_size=64)
```

在 FQT 中实现了对 w4a16 的量化压缩模型的导出，其导出格式适配于 vLLM & SGLang 框架

### LayerForwardManager

在量化算法中，通常需要对各种各样的 module 运行其 forward 方法。这些 forward 方法有不同的输入参数，其输出也可能五花八门。FQT 针对于量化场景，实现了通用的 `LayerForwardManager`，其使用同一接口，可运行不同 module 的前向过程

该 `LayerForwardManager` 对这些 module 的输入参数有如下假设：**有且只有一个 positional arguments，通常为 input hidden states，剩余的参数均为 kwargs 参数**。该假设对于 linear 和 transformer decoder layer & attention & mlp 均成立

`LayerForwardManager` 的使用方法如下

```python
# import global forward_manager
from fqt.quantization.utils import forward_manager

def qwen_forward(layer: Qwen2DecoderLayer, input_tensor, **kwargs):
    output = layer(input_tensor, **kwargs)
    return output[0]

forward_manager.register("Qwen2DecoderLayer", qwen_forward)
```

此时就类名为 `Qwen2DecoderLayer` 的 module，注册到 `forward_manager` 当中，通过如下方法运行前向：

```python
from fqt.quantization.utils import forward_manager
def forward_layer(layer, input_tensor, **kwargs):
    return forward_manager.forward(layer, input_tensor, **kwargs)
```

如果检测到该 layer 是 `Qwen2DecoderLayer`，那么就会调用 `qwen_forward`

如果传入了某个未注册的 layer，日志系统将会发出 warning，然后采用 `default_forward` 方法，该方法通常不会成功

```python
    def default_forward(self, layer, input_tensor, **kwargs):
        return layer(input_tensor, **kwargs)
```

### GPTQ

GPTQ 算法是目前最常用的量化算法，FQT 构建了 `run_gptq_for_sequential_layers` API 函数，用于快速量化 sequential 模型，其输入参数为：

```python
def run_gptq_for_sequential_layers(layers, 
                                   inputs, 
                                   cfg: GPTQConfig):
```

1. layers，通常为 transformer blocks (ModuleList or Sequential)

2. inputs，由 `build_calib_datset_inputs` 所构建的 inputs，包含 input hidden states & position embeddings

3. cfg，`GPTQConfig` 是一个简单的 dataclass，以配置 GPTQ 算法参数

   ```python
   @dataclass
   class GPTQConfig:
       bits: int = 4
       group_size: int = 64
       method: str = "per-block"
       damp: float = 0.01
       blocksize: int = 128
   ```

### Unify AWQ & SmoothQuant

在 FQT 中，将 SmoothQuant 和 AWQ 统一起来，二者本质上原理是相同的：计算 scale 对 activation 进行 smooth，同时将 scale 融入到 weight 当中以保证计算结果不变。通过不断地计算不同 scale 的量化损失，寻找到一个最优的 scale。二者的区别在于：在计算量化损失时，SmoothQuant 会将权重和激活都进行量化，而 AWQ 仅会将权重进行量化。该区别可以通过配置 `Quantizer` 方便完成

FQT 中使用 `run_smooth_quant_for_sequential_layers` API 函数，用于量化 sequential 模型，其输入参数和 GPTQ 类似

```python
def run_smooth_quant_for_sequential_layers(layers, 
                                           inputs, 
                                           cfg: SmoothQuantConfig):
```

1. layers，通常为 transformer blocks (ModuleList or Sequential)

2. inputs，由 `build_calib_datset_inputs` 所构建的 inputs，包含 input hidden states & position embeddings

3. cfg，`SmoothQuantConfig` 是一个简单的 dataclass，以配置 SmoothQuant/AWQ 算法参数

   ```python
   @dataclass
   class SmoothQuantConfig:
       w_method: str = "per-tensor"
       a_method: str = "per-tensor"
       q_linear_type: FakeQuantLinear = W8A8FakeLinear
       group_size: int = None
       duo_scale: bool = False
   ```

   其中 `w_mehtod & a_method` 作为参数，用于实例化一个具体的 `FakeQuantLinear` 类或者 fp8 linear，可选的类有：

   1. `W8A8FakeLinear`
   2. `W4A16FakeLinear`
   3. `FP8Linear`

### Other support

1. Custom op support

   为了让量化真正发挥加速作用，需要定制的量化算子。在 FQT 中集成了 int8 & fp8 的 scaled mm 可用于加速 linear 计算。其中 fp8 scaled mm 算子已经使用 `FP8Linear` 进行集成，`Int8Linear` 待开发

2. Logging

   FQT 构建了一个简单的 logger 配置，会将量化过程中的量化误差进行记录。只需要再代码运行前调用 `setup_logging` 即可

   ```python
   from fqt.utils.logger import setup_logging
   setup_logging("path/to/quant.log")
   ```

3. IO

   FQT 构建了简单的 io 系统，可以通过统一的 `load & dump` 对 pickle & json & bin & pth 文件进行读取

   ```python
   from fqt.utils.io import load, dump
   ```

4. Visualization

   提供了对二维张量可视化的脚本，可以通过 3D surface 直观查看激活分布

   ```python
   from fqt.utils.visualization import plot_3D_tensor
   plot_3D_tensor("layer_name", tensor, "save_name.png")
   ```

5. benchmark

   提供了一些测量延时和精度的小工具

   ```python
   from fqt.utils.benchmark import benchmark, calculate_errors, Timer
   # a simple benchmark tool
   benchmark(func, inputs, num_warmup=10, num_repeats=100, desc="function")
   # check errors
   calculate_errors(actuals, predictions , desc="")
   # a context timer
   with Timer("name"):
       run()
   Timer.stats()
   ```

## User Guide

### Install

如果不需要 custom op support，即不需要编译算子，那么可以直接用以下命令进行安装

```shell
ENABLE_NO_CSRC=1 pip install -e . --no-build-isolation --config-settings editable-mode=compat
```

其中 `--config-settings` 是为了让 pylance 能够在其他库也解析到 FQT

如果要编译算子则直接去掉前面的环境变量即可

```shell
pip install -e . --no-build-isolation --config-settings editable-mode=compat
```

### Build calibration

构建校准集的方式可以直接使用 DataRecorder 小节中的最典型用法。该用法能够以较小的代价，适配不同的项目和模型：大多数项目都会提供模型的评测/推理代码，通过在评测代码中插入 `DataRecorder` 就可以获得校准集

```python
model = Qwen2ForCausalLM(config)

# create data recorder, set target module & saving device
dc = DataRecorder(model=model.model.layers[0],
                  device="cpu")

dc.register_input_hooks()		# register input hooks
dc.register_pos_embed_hooks()	# register pos emb hooks

for i, prompt in enumerate(dataset): # model evaluation loop
	model.generate(prompts)
    dc.save_inputs(f"{i}.pkl")	# save inputs
    dc.clear_inputs()			# clear inputs for next round conversation

dc.remove_input_hooks()			# remove hooks
```

### Quantize

在 FQT 项目中提供了量化 `Qwen2MoeForCausalLM` 的脚本示例 `projects/quantize_moe.py`，整个代码~30行，核心代码只有下方4行

```python
# create dataset
dataset = DefaultDataset(root_dir=args.dataset_path)
# inputs for quantization algorithm
inputs = build_calib_datset_inputs(dataset, num_samples=num_samples, seed=seed, device=device)
# create model
model: Qwen2MoeForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
# quantize and save
run_gptq_for_sequential_layers(model.model.layers, inputs, GPTQConfig())
```

### Compress Model

在 FQT 项目中提供了压缩 `Qwen2ForCausalLM` 为 int4 模型的脚本示例 `projects/compress_qwen.py`，整个代码 ~20行。其核心在于将模型中的线性层替换为 `QuantLinear`，然后进行保存。并且为了让 vLLM or SGLang 这样的框架能够使用，还需要保存符合 transformers 库规范的量化 config

```python
config = GPTQConfig(bits=4, group_size=128, method="per-block")

# replace with QuantLinear
replace_linear_with_custom(
    model.model.layers,
    QuantLinear,
    bits=config.bits,
    method=config.method,
    group_size=config.group_size
)

# add `quantization_config` key 
save_config = build_vllm_compat_gptq_config(config)
model.config.quantization_config = save_config
model.config.save_pretrained(qmodel_path)
```

## Future Work

1. w4a4 method integration
2. w4a16 & other fused kernel
3. int8/fp8 model compression
4. attention quantization
5. GPTQ static weight re-order