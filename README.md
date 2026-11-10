# Raspberry Pi Vision Lab

基于树莓派5和第三方IMX219 CSI摄像头的实时视觉实验，包含OpenCV Haar人脸检测、YOLOv8n人脸检测和YOLOv8n动物检测。三个工程都通过Flask提供标注后的MJPEG视频流，可在同一局域网的电脑浏览器查看。

## 功能与边界

- 通过`rpicam-vid`采集CSI画面，默认640×480、30 FPS，不翻转，不创建本地窗口。
- OpenCV使用官方正脸Haar级联；YOLO直接使用原生`ncnn.Net`加载`.param/.bin`。
- 树莓派推理环境不需要PyTorch或Ultralytics。电脑端训练和重新导出PT权重时才需要它们。
- 采集、检测、推流分离，只保留最新帧，摄像头异常自动重试。
- `/`查看画面，`/video_feed`为MJPEG，`/healthz`查看状态和耗时。
- 人脸任务只定位人脸，不识别身份。动物模型支持COCO中的10类动物，不是任意物种识别器。

## 目录结构

```text
raspberrypi-vision-lab/
├── README.md                  # GitHub首页说明
├── read.md                    # 本文完整项目说明
├── requirements.txt           # 树莓派运行依赖
├── SHA256SUMS.txt             # 模型文件校验
├── opencv/                    # 独立OpenCV人脸方案及官方XML
├── yolo_face/                 # YOLO人脸程序和NCNN模型
├── yolo_animals/              # YOLO动物程序和NCNN模型
└── weights/                   # 桌面端导出用原始PT权重
```

`model_ncnn.py`是导出工具生成的含torch测试脚本，不是树莓派运行入口。请使用下面的三个入口。

## 硬件和系统

树莓派5、64位Raspberry Pi OS、CSI排线、稳定电源和局域网。我们使用的第三方IMX219模块与课程配发摄像头不同，在本组实验中更符合成像及测试需求。IMX219是传感器型号，不代表模组厂商，也不意味着所有同型号模组都需要手动配置。

## IMX219无法自动识别时

先关机断电检查排线和接口，再执行：

```bash
rpicam-hello --list-cameras
```

本组模块曾无法在默认自动检测配置下列出。查阅CSDN排障经验并核对官方配置后，我们修改`/boot/firmware/config.txt`（旧系统可能为`/boot/config.txt`）：

```bash
sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.before-imx219
sudo nano /boot/firmware/config.txt
```

将已有`camera_auto_detect=1`改为`camera_auto_detect=0`，不要另加同名冲突项。确认配置不受其他型号条件段限制，在合适的`[all]`段添加摄像头实际接口对应的一行：

```ini
[all]
camera_auto_detect=0
dtoverlay=imx219,cam0
```

上例用于CAM/DISP0。若接CAM/DISP1，改用`dtoverlay=imx219`，不要同时照抄两条。配置名称是`camera_auto_detect`，不是`camera_detect`。保存重启后验证：

```bash
sudo reboot
# 重新SSH登录后
rpicam-hello --list-cameras
rpicam-hello --nopreview --timeout 3000
```

驱动接口`cam0`与程序参数`--camera 0`分别是物理接口选择和枚举设备编号，不应直接混为一谈。先确保`--list-cameras`成功再排查Python程序。

## 安装

仓库地址：https://github.com/YouNg1x/raspberrypi-vision-lab

当前仓库为私有，需要使用有访问权限的GitHub账号认证后克隆。若用于课程成员协作，请仓库所有者添加相应协作者。

```bash
sudo apt update
sudo apt install -y git
git clone https://github.com/YouNg1x/raspberrypi-vision-lab.git
cd raspberrypi-vision-lab
```

取得仓库后，在项目根目录执行：

```bash
sudo apt update
sudo apt install -y git rpicam-apps python3-opencv opencv-data python3-venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
sha256sum -c SHA256SUMS.txt
```

仅运行OpenCV方案时，可只安装`numpy Flask`，不需要`ncnn`。系统OpenCV通过`--system-site-packages`对虚拟环境可见。用`python -m pip --version`确认pip属于当前解释器。

## 运行

同一时间只运行一个方案，否则会争用摄像头和默认5000端口。

```bash
# OpenCV正脸检测
python opencv/face_opencv.py --check-detector
python opencv/face_opencv.py

# YOLO人脸检测
python yolo_face/face_stream.py --check-model
python yolo_face/face_stream.py

# YOLO动物检测
python yolo_animals/animal_stream.py --check-model
python yolo_animals/animal_stream.py
```

浏览器打开`http://树莓派IP:5000/`。树莓派IP可通过`hostname -I`查询。原生NCNN权重默认从入口脚本所在目录加载，与启动时工作目录无关。Ctrl+C退出。

```bash
curl --max-time 5 http://127.0.0.1:5000/healthz
python opencv/face_opencv.py --detect-width 480 --min-neighbors 5
python yolo_animals/animal_stream.py --conf 0.5 --threads 4
```

## 模型和结果

两个NCNN模型的metadata固定输入为480×480。人脸模型带5个关键点但程序只使用框和分数；动物模型仍计算80类输出，默认显示鸟、猫、狗、马、羊、牛、大象、熊、斑马、长颈鹿。要新增兔子等类别需要对应权重，不能只修改显示名称。

本组保存截图中，YOLO人脸瞬时约13.4～21.7 FPS，Haar正脸多数约28～30 FPS；动物约12.4～20.2 FPS。这不是严格同条件连续性能基准。约85%的动物识别正确情况是人工操作估计，不是mAP，也不是检测框置信度。Haar正脸速度较快，大角度侧脸检出较弱；YOLO姿态适应性更好但计算开销更高。

## 测试与故障排查

```bash
(cd opencv && python -m unittest test_face_opencv -v)
(cd yolo_face && python -m unittest test_face_stream -v)
(cd yolo_animals && python -m unittest test_animals -v)
vcgencmd measure_temp
vcgencmd measure_clock arm
vcgencmd get_throttled
```

| 现象 | 排查方法 |
|---|---|
| 无法导入cv2、Flask、ncnn | 核对当前Python与pip；检查虚拟环境是否可访问系统OpenCV |
| 找不到级联XML | 使用仓库附带XML或安装opencv-data，必要时用`--cascade`指定 |
| 摄像头日志停下不刷屏 | 无窗口程序正常时不逐帧打印；检查网页和`/healthz` |
| 网页无法打开 | 确认树莓派IP、局域网连接和5000端口；不要在电脑用127.0.0.1代替树莓派IP |
| 框偏移或NCNN形状异常 | 检查RGB、归一化、letterbox及模型和metadata是否配套 |
| 温升接近80 ℃或帧率下降 | 查看温度、时钟、降频状态，改善散热并比较输入尺寸和线程数 |

## 电脑端训练与导出

原始PT文件在`weights/`，供桌面端验证或重新导出。树莓派运行不需要打开PT。

```bash
# 在装好Ultralytics及PyTorch的电脑环境执行
yolo export model=weights/yolov8n-face.pt format=ncnn imgsz=480 device=cpu
yolo export model=weights/yolov8n_animals.pt format=ncnn imgsz=480 device=cpu
```

输入尺寸调整必须重新导出，并配套更新`.param`、`.bin`和`metadata.yaml`。本仓库提供部署工程和已有权重，不包含可复现训练曲线的原始训练数据集及训练日志。

## 分工

组长承担全部代码编写和调试、全部模型训练和部署，以及实验报告主笔与统稿。组员各自完成环境搭建后的复现、运行测试和结果检查，向组长反馈问题；组长据此完善程序与报告。成员姓名、学号和个人测试记录见课程报告填写项。

## 第三方来源与许可

YOLO及模型metadata声明采用AGPL-3.0，使用或再分发时应遵守其许可；OpenCV级联XML保留文件内的原始许可。详见`THIRD_PARTY.md`。项目演示仅在局域网开放，无鉴权，请勿直接暴露公网。

## 参考

- https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
- https://docs.ultralytics.com/integrations/ncnn/
- https://www.raspberrypi.com/documentation/computers/config_txt.html
- https://docs.waveshare.com/IMX219-X-Camera/Guide
