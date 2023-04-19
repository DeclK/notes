# Tennis.ai

- Usage of MMPose

  1. 预编译的 SDK (官方推荐)
  2. MMPose Demo Scripts

- Install MMDeploy (onnx runtime version)

  1. 下载 [MMDeploy](https://mmdeploy.readthedocs.io/en/latest/get_started.html)，似乎这个项目有一些预编译包。我询问了一下 GPT，预编译包包含了已经编译好的框架和库，主要用于解决在不同硬件和操作系统中运行的问题

     直接从 release 里面下载对应的版本，例如我希望使用 linux 版本，并且我不想用 cuda，我就下载 [这个](https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-linux-x86_64.tar.gz)

     除此之外还要下载一下软件

     ```shell
     # pytorch, cpu only
     pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
     
     # mmengine & mmcv
     pip install -U openmim
     mim install mmengine
     mim install "mmcv>=2.0.0rc2"
     
     # mmdeploy model converter
     pip install mmdeploy==1.0.0
     # mmdeploy sdk inference
     pip install mmdeploy-runtime==1.0.0
     ```

  2. 下载 onnx runtime

     ```shell
     wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
     tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
     export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
     export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
     ```

     教程里没有写 

     ```python
     pip install onnxruntime
     ```

- **Convert Models**

  1. 安装 MMDetection & MMPose，并且下载预训练权重 [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth) & [pose](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth)，然后整理项目的目录如下

     ```txt
     |----mmdeploy
     |----mmdetection
     |----mmpose
     |----rtmdet_nano
     |    |----rtmdet_nano.pth
     |----rtmpose_m
          |----rtmpose_m.pth
     ```

     安装这两个项目我都使用源码安装，即

     ```
     git clone ...
     pip install -r requirements.txt
     pip install -v -e .
     ```

  2. Convert

     ```shell
     # 前往 mmdeploy 目录
     cd mmdeploy
     
     # 转换 RTMDet, 4个必须参数
     # 1. onnx config
     # 2. model config
     # 3. checkpoint
     # 4. inference image
     python tools/deploy.py \
         configs/mmdet/detection/detection_onnxruntime_static.py \
         /github/Tennis.ai/mmpose/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py\
         /github/Tennis.ai/rtmdet_m/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
         demo/resources/human-pose.jpg \
         --work-dir mmdeploy_models/mmdet/ort \
         --device cpu \
         --show
         
     # 转换 RTMPose
     python tools/deploy.py \
         configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
         /github/Tennis.ai/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py\
         /github/Tennis.ai/rtmpose_m/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
         demo/resources/human-pose.jpg \
         --work-dir mmdeploy_models/mmpose/ort \
         --device cpu \
         --show
     ```

     可以看到在 `work_dir` 下出现了 `end2end.onnx`

- Inference by SDK

  
