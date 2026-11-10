# 树莓派5 CSI人脸检测与Flask视频流

运行入口是 `face_stream.py`。直接使用原生 `ncnn.Net` 加载同目录中的
`model.ncnn.param` 和 `model.ncnn.bin`；依赖只有 NCNN、NumPy、OpenCV 和 Flask。
不需要安装 PyTorch、Ultralytics，也不需要运行导出时生成的 `model_ncnn.py`
（那个文件是含 torch 的导出测试脚本）。

## 1. 上传文件

本目录完整上传到树莓派：

```text
/home/yyx/face_recognition/yolov8n-face_ncnn_model/
  model.ncnn.param
  model.ncnn.bin
  metadata.yaml
  face_detector.py
  face_stream.py
  requirements-face.txt
```

Windows PowerShell上传示例（替换IP）：

```powershell
scp -r "E:\face_recognition\yolov8n-face_ncnn_model" yyx@树莓派IP:/home/yyx/face_recognition/
```

## 2. 安装依赖

在树莓派64位系统的终端执行：

```bash
sudo apt update
sudo apt install -y rpicam-apps python3-opencv python3-venv
cd /home/yyx/face_recognition/yolov8n-face_ncnn_model
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-face.txt
```

已有可用的 NCNN/OpenCV 环境时，直接使用原环境即可。

## 3. 检查并启动

```bash
cd /home/yyx/face_recognition/yolov8n-face_ncnn_model
source .venv/bin/activate
python face_stream.py --check-model
rpicam-hello --list-cameras
python face_stream.py
```

电脑与树莓派在同一局域网时，在电脑浏览器打开：

```text
http://树莓派IP:5000/
```

`/video_feed` 是标注后的 MJPEG 流，`/healthz` 返回检测帧率、人数、推理耗时、
模型输出形状、最近一帧年龄。无帧或过期时healthz返回503。CSI断开或长时无帧会
自动重启采集进程；推理失败则打印明确异常并退出。终端按 Ctrl+C 停止。
网页仅在局域网提供画面，没有用户认证，不要直接映射到公网。

默认CSI采集640×480、30FPS、无翻转；原生推理4线程。实际推理/网页FPS由树莓派
算力与散热决定，相机30FPS并不代表推理能达到30FPS。

## 4. 参数

```bash
python face_stream.py --conf 0.5 --threads 4 --port 5000
python face_stream.py --width 960 --height 540 --camera-fps 30
python face_stream.py --model-dir /home/yyx/face_recognition/yolov8n-face_ncnn_model
```

当前权重的metadata表明它是单类别`face`、带5个关键点的pose模型，固定输入
640×640，输出20×8400。程序只使用前5项框坐标/置信度，不显示关键点，也不识别身份。
采集图像等比例缩放并以114填充成640×640，BGR转RGB、归一化，经过NCNN后执行NMS，
最终将坐标还原到原始相机画面。不要只修改输入尺寸；如果重新导出其他尺寸的权重，
必须连同对应metadata一起替换。

置信度高可减少误检，但会增加漏检；`--conf`默认0.4，`--iou`默认0.45。
本地不会创建显示窗口，JPEG每帧只编码一次，各浏览器共享最新编码帧，不堆积历史帧。

## 5. 无摄像头测试

```bash
python -m unittest test_face_stream -v
python face_stream.py --check-model
```

模型预热测试会真实加载现有bin/param，检查输出解析，不验证真实人脸的检测精度。
真实摄像头、树莓派速度和网络访问需要在设备上验证。
