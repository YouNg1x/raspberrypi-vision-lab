# 树莓派5：CSI动物检测 + Flask实时视频流

使用 `animal_stream.py` 启动。直接用原生NCNN读取同目录
`model.ncnn.param` / `model.ncnn.bin`，无需安装PyTorch或Ultralytics。
导出自带的 `model_ncnn.py` 是含torch的测试脚本，本程序不使用它。

## 上传与安装

上传整个目录到 `/home/yyx/animals_recognition/yolov8n_animals_ncnn_model`。
Windows PowerShell示例（先在树莓派创建父目录）：

```powershell
ssh yyx@树莓派IP "mkdir -p /home/yyx/animals_recognition"
scp -r "E:\animals_recognition\yolov8n_animals_ncnn_model" yyx@树莓派IP:/home/yyx/animals_recognition/
```

树莓派64位系统执行：

```bash
sudo apt update
sudo apt install -y rpicam-apps python3-opencv python3-venv
cd /home/yyx/animals_recognition/yolov8n_animals_ncnn_model
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-animals.txt
```

也可在之前已有NCNN/OpenCV的环境中安装缺少的Flask、PyYAML，直接运行。

## 启动

```bash
python animal_stream.py --check-model
rpicam-hello --list-cameras
python animal_stream.py
```

电脑与树莓派在同一局域网，用浏览器访问 `http://树莓派IP:5000/`。
只显示标注后的画面，包含动物英文名、置信度、数量、FPS和推理耗时。
没有本地显示窗口。`/video_feed`提供MJPEG流，`/healthz`返回实时状态，
包含`animals`总数、`counts`各类别数量、`fps`、`inference_ms`、`output_shape`。
无画面或画面过期时healthz返回503。

CSI通过rpicam-vid采集，默认640×480、30FPS、不翻转；4线程CPU推理。
实际检测帧率取决于算力与散热，不保证等于摄像头30FPS。最新帧覆盖旧帧以避免排队，
JPEG编码结果共享给所有浏览器，摄像头异常会自动重试。按Ctrl+C关闭程序。

同一树莓派同时运行人脸程序时会争用相机，先停止它；端口5000也不能重复占用。

## 模型与类别

当前实际metadata为480×480输入、COCO80类的YOLOv8n，读取metadata自动确定
输入尺寸及类别。默认只保留bird/cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe。
仅更改文件夹名字不会改变模型能识别的动物。兔子、鸡、鸭等不在当前模型类别中。

程序先取每个候选的最高分类分数，再筛动物，最后分类别NMS，重叠的不同动物类别
不会互相抑制。保留全部80类计算，仅过滤显示并不会显著减少推理耗时。

```bash
# 可选：显示模型中全部类别
python animal_stream.py --all-classes
# 调整阈值、线程数、端口、画面质量
python animal_stream.py --conf 0.5 --threads 4 --port 5001 --jpeg-quality 75
# 指定模型目录
python animal_stream.py --model-dir /home/yyx/animals_recognition/yolov8n_animals_ncnn_model
```

模型固定输入480，不能直接运行时改成320。需要提速时在电脑重新导出320模型，
将匹配的bin、param和metadata一起替换，程序会读取新的metadata尺寸。

## 验证

```bash
python -m unittest test_animals -v
python animal_stream.py --check-model
```

模型检查会对空白图实际推理，验证加载和输出格式；摄像头实拍精度、树莓派速度
以及局域网访问需要在设备端验证。
