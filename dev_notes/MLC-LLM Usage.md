# MLC-LLM Usage

如何使用 MLC-LLM 工具链，需要回答以下问题：

1. MLC-LLM 的安装如何运行
2. 如何获得模型
3. 如何将模型进行编译并运行

## Concept

1. 安装 mlc-llm，[installation page](https://llm.mlc.ai/docs/install/mlc_llm.html)

2. 安装 tvm-unity compiler，如果需要自己编译模型的话 [install tvm unity compiler](https://llm.mlc.ai/docs/install/tvm.html)

3. Models and model lib

   想要使用 mlc-llm 运行一个 chat model 需要两件事情：符合 mlc 要求的模型权重和模型库（model weights and model library）

   获取途径有两个

   1. 使用 mlc-llm 已经准备好的模型权重 [model cards hf](https://huggingface.co/mlc-ai)，模型库 [binary-mlc-llm-libs](https://github.com/mlc-ai/binary-mlc-llm-libs)

      ```python
      # Download pre-conveted weights
      git lfs install && mkdir dist/
      git clone https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC \
                                         dist/Llama-2-7b-chat-hf-q4f16_1-MLC
      
      # Download pre-compiled model library
      git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt_libs
      ```

      

   2. 自己编译模型权重和模型库 [convert model weights via mlc](https://llm.mlc.ai/docs/compilation/convert_weights.html)，[compile model libraries](https://llm.mlc.ai/docs/compilation/compile_models.html)

4. 使用 Python API 运行 chat model

5. 配置 MLCChat，mlc-llm 提供两个 dataclass 来设定配置

6. **Convert Model Weights** and **Compile Model Library**

   pre-request: tvm unity compiler & mlc_chat

   直接从 huggingface 上拉取模型，然后使用命令行工具转换

   ```python
   # Create directory
   mkdir -p dist/models && cd dist/models
   # Clone HF weights
   git lfs install
   git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1
   cd ../..
   # Convert weight
   mlc_chat convert_weight ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ \
       --quantization q4f16_1 \
       -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC
   ```

   教程还让我们生成 MLC Chat Config，为之后生成 model libraries 提供一些信息

   ```shell
   mlc_chat compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
       --device cuda -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so
   ```

## Question

