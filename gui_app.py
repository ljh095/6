"""
FoundationPose GUI可视化界面
美观简约的参数配置和算法控制界面
"""

import sys
import os
import cv2
import numpy as np
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit,
        QFileDialog, QSplitter, QFrame, QScrollArea, QComboBox,
        QDoubleSpinBox, QSpinBox, QGridLayout, QTabWidget,
        QSizePolicy
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
    from PyQt6.QtWidgets import QStyleFactory
except ImportError:
    print("PyQt6 is not installed. Please install it with: pip install PyQt6")
    sys.exit(1)


class AlgorithmSignals(QObject):
    """算法线程信号"""
    update_result = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # 更新结果图像
    log_message = pyqtSignal(str)  # 日志消息
    finished = pyqtSignal()  # 算法完成
    error = pyqtSignal(str)  # 错误信息


class PoseRequestHandler(BaseHTTPRequestHandler):
    """处理位姿估计的HTTP请求"""

    def __init__(self, algorithm_thread, *args, **kwargs):
        self.algorithm_thread = algorithm_thread
        super().__init__(*args, **kwargs)

    def do_POST(self):
        """处理POST请求"""
        if self.path == '/estimate_pose':
            try:
                # 读取请求体
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8')) if content_length > 0 else {}

                # 执行位姿估计
                result = self.algorithm_thread.execute_pose_estimation(request_data)

                if result['success']:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    response = json.dumps(result, ensure_ascii=False, indent=2)
                    self.wfile.write(response.encode('utf-8'))
                else:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    response = json.dumps(result, ensure_ascii=False)
                    self.wfile.write(response.encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                error_response = json.dumps({
                    'success': False,
                    'error': f'服务器错误: {str(e)}'
                }, ensure_ascii=False)
                self.wfile.write(error_response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        """处理GET请求 - 健康检查"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            response = json.dumps({
                'status': 'running',
                'algorithm': 'FoundationPose',
                'message': '服务运行中，发送POST请求到 /estimate_pose 进行位姿估计'
            }, ensure_ascii=False)
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """禁用默认的日志输出"""
        pass


class AlgorithmThread(threading.Thread):
    """算法执行线程 - 监听HTTP请求并执行位姿估计"""

    def __init__(self, params, signals, port=8888):
        super().__init__()
        self.params = params
        self.signals = signals
        self.running = True
        self.paused = False
        self.port = port

        # 存储初始化的估计器
        self.est = None
        self.mesh = None
        self.request_count = 0

    def run(self):
        """初始化并启动HTTP服务器监听请求"""
        try:
            self.signals.log_message.emit("开始初始化 FoundationPose...")

            # 导入 FoundationPose 相关模块
            from estimater import FoundationPose
            from learning.training.predict_score import ScorePredictor
            from learning.training.predict_pose_refine import PoseRefinePredictor
            import trimesh
            import nvdiffrast.torch as dr
            import Utils

            self.signals.log_message.emit("加载 CAD 模型...")
            self.mesh = trimesh.load(self.params['mesh_path'])
            self.signals.log_message.emit(f"模型加载成功: {len(self.mesh.vertices)} 个顶点")

            self.signals.log_message.emit("初始化估计器...")
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx = dr.RasterizeCudaContext()

            self.est = FoundationPose(
                model_pts=self.mesh.vertices,
                model_normals=self.mesh.vertex_normals,
                mesh=self.mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir='./debug_gui',
                debug=0,
                glctx=glctx
            )
            self.signals.log_message.emit("估计器初始化完成")

            # 创建HTTP服务器
            def handler(*args, **kwargs):
                return PoseRequestHandler(self, *args, **kwargs)

            self.server = HTTPServer(('0.0.0.0', self.port), handler)
            self.signals.log_message.emit(f"HTTP服务器已启动，监听端口: {self.port}")
            self.signals.log_message.emit(f"访问 http://localhost:{self.port}/health 检查服务状态")
            self.signals.log_message.emit(f"发送POST请求到 http://localhost:{self.port}/estimate_pose 进行位姿估计")

            # 持续监听请求
            self.server.serve_forever()

        except Exception as e:
            self.signals.error.emit(f"算法初始化错误: {str(e)}")
            self.signals.finished.emit()

    def execute_pose_estimation(self, request_data):
        """执行单次位姿估计"""
        self.request_count += 1
        request_id = self.request_count

        try:
            self.signals.log_message.emit(f"[请求 #{request_id}] 收到位姿估计请求")

            # 从 RealSense D435 获取 RGB 和深度图
            self.signals.log_message.emit(f"[请求 #{request_id}] 正在获取相机数据...")
            rgb, depth, K = self.get_realsense_data()

            if rgb is None or depth is None:
                return {
                    'success': False,
                    'error': '无法从 RealSense 相机获取数据',
                    'request_id': request_id
                }

            self.signals.log_message.emit(f"[请求 #{request_id}] RGB 图像尺寸: {rgb.shape[1]}x{rgb.shape[0]}")
            self.signals.log_message.emit(f"[请求 #{request_id}] 深度图尺寸: {depth.shape[1]}x{depth.shape[0]}")

            # 从分割算法获取掩码
            self.signals.log_message.emit(f"[请求 #{request_id}] 正在运行分割算法...")
            mask = self.get_segmentation_mask(rgb)

            if mask is None:
                return {
                    'success': False,
                    'error': '分割算法返回的掩码为空',
                    'request_id': request_id
                }

            self.signals.log_message.emit(f"[请求 #{request_id}] 掩码生成完成，有效像素: {mask.sum()}")

            # 使用配置的相机内参
            K = self.params['camera_K']
            iteration = self.params['refine_iter']

            # 执行姿态估计
            self.signals.log_message.emit(f"[请求 #{request_id}] 开始姿态估计...")
            pose = self.est.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=iteration
            )

            self.signals.log_message.emit(f"[请求 #{request_id}] 姿态估计完成")

            # 可视化结果
            import Utils
            to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            center_pose = pose @ np.linalg.inv(to_origin)

            vis = Utils.draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
            vis = Utils.draw_xyz_axis(
                rgb, ob_in_cam=center_pose, scale=0.1, K=K,
                thickness=3, transparency=0, is_input_rgb=True
            )

            # 更新GUI显示
            self.signals.update_result.emit(rgb, depth, vis)

            # 将姿态矩阵转换为列表以便JSON序列化
            pose_list = pose.tolist()

            result = {
                'success': True,
                'request_id': request_id,
                'pose': pose_list,
                'image_shape': {'height': int(rgb.shape[0]), 'width': int(rgb.shape[1])},
                'mask_pixels': int(mask.sum()),
                'message': f'位姿估计成功 (请求 #{request_id})'
            }

            self.signals.log_message.emit(f"[请求 #{request_id}] 姿态矩阵:\n{pose}")
            self.signals.log_message.emit(f"[请求 #{request_id}] ✅ 估计完成")

            return result

        except Exception as e:
            error_msg = f"位姿估计错误: {str(e)}"
            self.signals.log_message.emit(f"[请求 #{request_id}] ❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'request_id': request_id
            }

    def get_realsense_data(self):
        """
        从 RealSense D435 获取 RGB 和深度图
        TODO: 实现实际的相机数据获取逻辑
        """
        try:
            # 这里是预留的接口，需要根据实际硬件实现
            # 示例代码结构：
            # import pyrealsense2 as rs
            # pipeline = rs.pipeline()
            # config = rs.config()
            # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # profile = pipeline.start(config)
            # frames = pipeline.wait_for_frames()
            # color_frame = frames.get_color_frame()
            # depth_frame = frames.get_depth_frame()
            # rgb = np.asanyarray(color_frame.get_data())
            # depth = np.asanyarray(depth_frame.get_data())
            # pipeline.stop()

            # 临时：使用占位数据，实际使用时需要替换为真实的相机获取代码
            # 如果有测试图像，可以临时加载测试数据
            self.signals.log_message.emit("⚠️ 使用测试数据模式（RealSense接口预留）")

            # 尝试加载测试数据（如果存在）
            test_rgb_path = 'demo_data/mustard0/rgb/000000.png'
            test_depth_path = 'demo_data/mustard0/depth/000000.png'

            if os.path.exists(test_rgb_path) and os.path.exists(test_depth_path):
                rgb = cv2.imread(test_rgb_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                depth = cv2.imread(test_depth_path, cv2.IMREAD_UNCHANGED)
                depth = depth.astype(np.float32) / 1000.0

                # 使用默认相机内参
                K = np.array([
                    [577.5,  0.0,  319.5],
                    [0.0,   577.5, 239.5],
                    [0.0,    0.0,   1.0]
                ], dtype=np.float32)

                return rgb, depth, K
            else:
                return None, None, None

        except Exception as e:
            self.signals.log_message.emit(f"RealSense 数据获取错误: {str(e)}")
            return None, None, None

    def get_segmentation_mask(self, rgb):
        """
        从分割算法获取掩码
        TODO: 实现实际的分割算法调用
        """
        try:
            # 这里是预留的接口，需要根据实际使用的分割算法实现
            # 示例代码结构：
            # from your_segmentation_module import segment_object
            # mask = segment_object(rgb)
            # return mask

            # 临时：使用测试数据的掩码（如果存在）
            self.signals.log_message.emit("⚠️ 使用测试掩码数据（分割算法接口预留）")

            test_mask_path = 'demo_data/mustard0/masks/000000.png'

            if os.path.exists(test_mask_path):
                mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(bool)
                return mask
            else:
                # 创建一个全False的掩码作为占位
                return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=bool)

        except Exception as e:
            self.signals.log_message.emit(f"分割算法错误: {str(e)}")
            return None

    def pause(self):
        """暂停算法"""
        self.paused = True
        self.signals.log_message.emit("算法已暂停")

    def resume(self):
        """恢复算法"""
        self.paused = False
        self.signals.log_message.emit("算法继续执行")

    def stop(self):
        """停止算法"""
        self.running = False
        if hasattr(self, 'server'):
            self.signals.log_message.emit("正在关闭HTTP服务器...")
            self.server.shutdown()
        self.signals.log_message.emit("算法已停止")


class FoundationPoseGUI(QMainWindow):
    """FoundationPose GUI 主窗口"""

    def __init__(self):
        super().__init__()

        # 初始化路径输入框（必须在 init_ui 之前）
        self.mesh_path_edit = QLineEdit()
        self.rgb_path_edit = QLineEdit()
        self.depth_path_edit = QLineEdit()
        self.mask_path_edit = QLineEdit()

        # 初始化UI
        self.init_ui()

        # 初始化其他变量
        self.algorithm_thread = None
        self.algorithm_signals = AlgorithmSignals()
        self.algorithm_running = False
        self.algorithm_paused = False

        # 连接信号
        self.algorithm_signals.update_result.connect(self.update_result_display)
        self.algorithm_signals.log_message.connect(self.append_log)
        self.algorithm_signals.finished.connect(self.algorithm_finished)
        self.algorithm_signals.error.connect(self.algorithm_error)

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('FoundationPose GUI')
        # 设置最小窗口尺寸，确保所有内容可见
        self.setMinimumSize(1000, 700)
        # 设置默认窗口尺寸
        self.setGeometry(100, 100, 1400, 800)

        # 设置现代主题
        self.set_modern_style()

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧：参数配置面板
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # 右侧：结果显示面板
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # 设置分割器比例 (左侧:右侧 = 45:55)
        splitter.setSizes([500, 750])

    def create_left_panel(self):
        """创建左侧参数配置面板"""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(15, 15, 15, 8)
        title = QLabel('参数配置')
        title.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50;")
        title_layout.addWidget(title)
        layout.addWidget(title_widget)

        # 创建标签页
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: #ecf0f1;
                color: #7f8c8d;
                padding: 10px 20px;
                font-size: 11px;
                font-weight: bold;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: white;
                color: #3498db;
                border-bottom: 3px solid #3498db;
            }
            QTabBar::tab:hover:!selected {
                background: #d5dbdb;
            }
        """)

        # 标签页1: 数据路径
        data_tab = self.create_data_path_tab()
        tab_widget.addTab(data_tab, '📁 数据路径')

        # 标签页2: 相机参数
        camera_tab = self.create_camera_param_tab()
        tab_widget.addTab(camera_tab, '📷 相机参数')

        # 标签页3: 算法设置
        algo_tab = self.create_algo_setting_tab()
        tab_widget.addTab(algo_tab, '⚙️ 算法设置')

        layout.addWidget(tab_widget)

        # 日志输出
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(15, 8, 15, 8)

        log_title = QLabel('📋 运行日志')
        log_title.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        log_title.setStyleSheet("color: #2c3e50;")
        log_layout.addWidget(log_title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9px;
                border: 1px solid #34495e;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_widget)

        # 添加弹性空间
        layout.addStretch()

        # 控制按钮
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(15, 8, 15, 15)

        # 操作按钮（开始/停止切换）
        self.action_btn = QPushButton('▶ 开始算法')
        self.action_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #27ae60, stop:1 #2ecc71);
                color: white;
                padding: 12px 25px;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #229954, stop:1 #27ae60);
            }
            QPushButton:pressed {
                background: #1e8449;
            }
            QPushButton:disabled {
                background: #95a5a6;
                color: #bdc3c7;
            }
        """)
        self.action_btn.clicked.connect(self.toggle_algorithm)
        control_layout.addWidget(self.action_btn)

        layout.addWidget(control_widget)

        return panel

    def create_data_path_tab(self):
        """创建数据路径标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # CAD模型路径
        mesh_group = self.create_path_group(
            '🧊 CAD 模型',
            '选择 .obj 或 .ply 格式的模型文件',
            'model_placeholder.png'
        )
        layout.addWidget(mesh_group)

        # 添加说明信息
        info_group = QGroupBox()
        info_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #d5f5e3;
                border-radius: 10px;
                padding: 12px;
                background: #e8f8f5;
            }
        """)
        info_layout = QVBoxLayout(info_group)

        info_label = QLabel('ℹ️ 数据来源说明')
        info_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        info_label.setStyleSheet("color: #1e8449;")
        info_layout.addWidget(info_label)

        info_text = QLabel(
            '• RGB 和深度图数据将自动从 RealSense D435 相机获取\n'
            '• 掩码图像将由独立的分割算法提供\n'
            '• 无需手动配置上述数据路径'
        )
        info_text.setStyleSheet("color: #27ae60; font-size: 10px; line-height: 1.4;")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        layout.addWidget(info_group)

        layout.addStretch()
        return widget

    def create_path_group(self, title, placeholder, icon):
        """创建路径选择组"""
        group = QGroupBox()
        group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ecf0f1;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
                font-size: 11px;
                font-weight: bold;
                color: #34495e;
            }
        """)

        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 15, 12, 12)

        # 标题
        title_label = QLabel(title)
        title_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title_label)

        # 路径输入框和预览按钮
        row_layout = QHBoxLayout()

        if icon == 'model_placeholder.png':
            path_edit = self.mesh_path_edit
            path_edit.setPlaceholderText(placeholder)
        elif icon == 'rgb_placeholder.png':
            path_edit = self.rgb_path_edit
            path_edit.setPlaceholderText(placeholder)
        elif icon == 'depth_placeholder.png':
            path_edit = self.depth_path_edit
            path_edit.setPlaceholderText(placeholder)
        else:
            path_edit = self.mask_path_edit
            path_edit.setPlaceholderText(placeholder)

        path_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px 10px;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                background: #fafafa;
                font-size: 10px;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background: white;
            }
        """)
        row_layout.addWidget(path_edit)

        # 浏览按钮
        browse_btn = QPushButton('📂 浏览')
        browse_btn.setStyleSheet("""
            QPushButton {
                background: #95a5a6;
                color: white;
                padding: 8px 15px;
                font-size: 10px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #7f8c8d;
            }
            QPushButton:pressed {
                background: #6c7a7b;
            }
        """)

        if icon == 'model_placeholder.png':
            browse_btn.clicked.connect(self.browse_mesh_file)
        elif icon == 'rgb_placeholder.png':
            browse_btn.clicked.connect(self.browse_rgb_file)
        elif icon == 'depth_placeholder.png':
            browse_btn.clicked.connect(self.browse_depth_file)
        else:
            browse_btn.clicked.connect(self.browse_mask_file)

        row_layout.addWidget(browse_btn)
        layout.addLayout(row_layout)

        return group

    def create_camera_param_tab(self):
        """创建相机参数标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        info_label = QLabel('相机内参')
        info_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        info_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(info_label)

        # 参数输入卡片
        param_widget = QWidget()
        param_widget.setStyleSheet("""
            QWidget {
                background: white;
                border: 2px solid #ecf0f1;
                border-radius: 10px;
                padding: 12px;
            }
        """)
        param_layout = QVBoxLayout(param_widget)
        param_layout.setSpacing(8)
        param_layout.setContentsMargins(0, 0, 0, 0)

        # fx - 水平焦距
        fx_group = self.create_camera_param_input('fx', '水平焦距', '像素', 577.5)
        param_layout.addWidget(fx_group)

        # fy - 垂直焦距
        fy_group = self.create_camera_param_input('fy', '垂直焦距', '像素', 577.5)
        param_layout.addWidget(fy_group)

        # ppx - 主点X坐标
        ppx_group = self.create_camera_param_input('ppx', '主点 X 坐标', '像素', 319.5)
        param_layout.addWidget(ppx_group)

        # ppy - 主点Y坐标
        ppy_group = self.create_camera_param_input('ppy', '主点 Y 坐标', '像素', 239.5)
        param_layout.addWidget(ppy_group)

        layout.addWidget(param_widget)
        layout.addStretch()
        return widget

    def create_camera_param_input(self, param_name, label_text, unit, default_value):
        """创建单个相机参数输入"""
        group = QGroupBox()
        group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #e8e8e8;
                border-radius: 6px;
                padding: 8px 12px;
                background: #fafafa;
                margin-top: 0px;
            }
        """)

        layout = QHBoxLayout(group)
        layout.setSpacing(8)

        # 标签
        label = QLabel(f'{label_text}:')
        label.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        label.setStyleSheet("color: #2c3e50;")
        label.setMinimumWidth(120)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        layout.addWidget(label)

        # 参数名显示
        param_name_label = QLabel(f'<b>{param_name}</b>')
        param_name_label.setStyleSheet("color: #7f8c8d; font-size: 9px;")
        param_name_label.setMinimumWidth(45)
        param_name_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        layout.addWidget(param_name_label)

        # 输入框
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 10000.0)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setValue(default_value)
        spin.setMinimumWidth(80)
        spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        spin.setStyleSheet("""
            QDoubleSpinBox {
                padding: 5px 8px;
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                background: white;
                font-size: 10px;
                font-weight: bold;
            }
            QDoubleSpinBox:focus {
                border: 2px solid #3498db;
                background: white;
            }
        """)
        layout.addWidget(spin)

        # 单位
        unit_label = QLabel(unit)
        unit_label.setStyleSheet("color: #95a5a6; font-size: 9px;")
        unit_label.setMinimumWidth(35)
        unit_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        layout.addWidget(unit_label)

        # 存储引用
        if not hasattr(self, 'camera_params'):
            self.camera_params = {}
        self.camera_params[param_name] = spin

        return group

    def create_algo_setting_tab(self):
        """创建算法设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 15, 15)

        # 迭代次数
        iter_group = QGroupBox()
        iter_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ecf0f1;
                border-radius: 10px;
                padding: 12px;
                background: white;
            }
        """)
        iter_layout = QVBoxLayout(iter_group)

        iter_label = QLabel('🔄 迭代次数')
        iter_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        iter_label.setStyleSheet("color: #2c3e50;")
        iter_layout.addWidget(iter_label)

        iter_desc = QLabel('姿态优化的迭代次数，数值越大越精确但耗时越长')
        iter_desc.setStyleSheet("color: #7f8c8d; font-size: 9px;")
        iter_layout.addWidget(iter_desc)

        self.refine_iter_spin = QSpinBox()
        self.refine_iter_spin.setRange(1, 100)
        self.refine_iter_spin.setValue(5)
        self.refine_iter_spin.setStyleSheet("""
            QSpinBox {
                padding: 8px 10px;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                background: #fafafa;
                font-size: 11px;
                font-weight: bold;
            }
            QSpinBox:focus {
                border: 2px solid #3498db;
                background: white;
            }
        """)
        iter_layout.addWidget(self.refine_iter_spin)

        layout.addWidget(iter_group)

        # 调试级别
        debug_group = QGroupBox()
        debug_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ecf0f1;
                border-radius: 10px;
                padding: 12px;
                background: white;
            }
        """)
        debug_layout = QVBoxLayout(debug_group)

        debug_label = QLabel('🔍 调试级别')
        debug_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        debug_label.setStyleSheet("color: #2c3e50;")
        debug_layout.addWidget(debug_label)

        debug_desc = QLabel('控制输出日志的详细程度')
        debug_desc.setStyleSheet("color: #7f8c8d; font-size: 9px;")
        debug_layout.addWidget(debug_desc)

        self.debug_combo = QComboBox()
        self.debug_combo.addItems(['0 - 无输出', '1 - 基础信息', '2 - 详细信息', '3 - 完整调试'])
        self.debug_combo.setCurrentIndex(1)
        self.debug_combo.setStyleSheet("""
            QComboBox {
                padding: 8px 10px;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                background: #fafafa;
                font-size: 11px;
            }
            QComboBox:hover {
                border: 2px solid #3498db;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTMgNEw2IDdMOSA0IiBzdHJva2U9IiMzNDk4ZGIiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+Cjwvc3ZnPg==);
            }
        """)
        debug_layout.addWidget(self.debug_combo)

        layout.addWidget(debug_group)
        layout.addStretch()
        return widget

    def preview_file(self, path_edit):
        """预览文件"""
        file_path = path_edit.text()
        if not file_path or not os.path.exists(file_path):
            return

        # 检查文件类型并预览
        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            self.show_image_preview(file_path)
        elif file_path.endswith(('.obj', '.ply')):
            self.show_model_info(file_path)
        else:
            self.append_log(f"不支持预览的文件类型: {file_path}")

    def show_image_preview(self, image_path):
        """显示图像预览"""
        try:
            img = cv2.imread(image_path)
            if img is not None:
                self.append_log(f"预览图像: {image_path} ({img.shape[1]}x{img.shape[0]})")
                # 可以在这里添加更复杂的预览功能
            else:
                self.append_log(f"无法加载图像: {image_path}")
        except Exception as e:
            self.append_log(f"预览图像时出错: {str(e)}")

    def show_model_info(self, model_path):
        """显示模型信息"""
        try:
            import trimesh
            mesh = trimesh.load(model_path)
            self.append_log(f"模型信息: {len(mesh.vertices)} 个顶点, {len(mesh.faces)} 个面")
        except Exception as e:
            self.append_log(f"加载模型时出错: {str(e)}")

    def create_right_panel(self):
        """创建右侧结果显示面板"""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # 标题
        title = QLabel('实时结果')
        title.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # 创建标签页
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #f0f0f0;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 3px solid #2196F3;
            }
        """)

        # 1. RGB显示
        rgb_tab = QWidget()
        rgb_layout = QVBoxLayout(rgb_tab)
        rgb_layout.setContentsMargins(0, 0, 0, 0)
        self.rgb_label = QLabel('未运行算法')
        self.rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rgb_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #cccccc;
                border-radius: 8px;
                color: #666666;
                font-size: 16px;
            }
        """)
        self.rgb_label.setMinimumSize(400, 300)
        rgb_layout.addWidget(self.rgb_label)

        # 2. 深度图显示
        depth_tab = QWidget()
        depth_layout = QVBoxLayout(depth_tab)
        depth_layout.setContentsMargins(0, 0, 0, 0)
        self.depth_label = QLabel('未运行算法')
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #cccccc;
                border-radius: 8px;
                color: #666666;
                font-size: 16px;
            }
        """)
        self.depth_label.setMinimumSize(400, 300)
        depth_layout.addWidget(self.depth_label)

        # 3. 结果显示
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        result_layout.setContentsMargins(0, 0, 0, 0)
        self.result_label = QLabel('未运行算法')
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #cccccc;
                border-radius: 8px;
                color: #666666;
                font-size: 16px;
            }
        """)
        self.result_label.setMinimumSize(400, 300)
        result_layout.addWidget(self.result_label)

        tab_widget.addTab(rgb_tab, '原始 RGB')
        tab_widget.addTab(depth_tab, '深度图')
        tab_widget.addTab(result_tab, '姿态估计结果')

        layout.addWidget(tab_widget)

        # 信息显示区域
        info_group = QGroupBox('姿态信息')
        info_layout = QVBoxLayout()
        self.pose_info = QLabel('等待运行算法...')
        self.pose_info.setStyleSheet("""
            QLabel {
                background-color: #f9f9f9;
                padding: 10px;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        self.pose_info.setWordWrap(True)
        info_layout.addWidget(self.pose_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        return panel

    def set_modern_style(self):
        """设置现代风格"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #333333;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                font-size: 12px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
            QPushButton {
                padding: 8px 16px;
                border: none;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QLabel {
                color: #333333;
                font-size: 12px;
            }
            QComboBox {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: white;
            }
            QDoubleSpinBox, QSpinBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: white;
            }
        """)

    def browse_mesh_file(self):
        """浏览CAD模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择 CAD 模型文件', '',
            '模型文件 (*.obj *.ply);;所有文件 (*.*)'
        )
        if file_path:
            self.mesh_path_edit.setText(file_path)

    def get_camera_K(self):
        """获取相机内参矩阵"""
        # 从 fx, fy, ppx, ppy 参数构建相机内参矩阵
        fx = self.camera_params['fx'].value()
        fy = self.camera_params['fy'].value()
        ppx = self.camera_params['ppx'].value()
        ppy = self.camera_params['ppy'].value()

        K = np.array([
            [fx,   0.0,  ppx],
            [0.0,  fy,   ppy],
            [0.0,  0.0,  1.0]
        ], dtype=np.float32)

        return K

    def lock_parameters(self, lock):
        """锁定/解锁参数输入"""
        self.mesh_path_edit.setEnabled(not lock)
        self.refine_iter_spin.setEnabled(not lock)
        self.debug_combo.setEnabled(not lock)

        # 锁定/解锁相机参数
        if hasattr(self, 'camera_params'):
            for param_name, spin in self.camera_params.items():
                spin.setEnabled(not lock)

    def validate_parameters(self):
        """验证参数"""
        if not self.mesh_path_edit.text():
            return False, "请选择 CAD 模型文件"

        # RGB、Depth、Mask 数据将从 RealSense 和分割算法自动获取
        # 不需要在此处验证路径

        return True, "参数验证通过"

    def toggle_algorithm(self):
        """切换算法状态（开始/停止）"""
        # 如果算法正在运行，则停止
        if self.algorithm_running:
            self.stop_algorithm()
            return

        # 否则，开始算法
        # 验证参数
        valid, message = self.validate_parameters()
        if not valid:
            self.append_log(f"错误: {message}")
            return

        # 准备参数
        params = {
            'mesh_path': self.mesh_path_edit.text(),
            'camera_K': self.get_camera_K(),
            'refine_iter': self.refine_iter_spin.value(),
            'debug_level': self.debug_combo.currentIndex()
        }

        # 锁定参数
        self.lock_parameters(True)

        # 更新按钮状态为停止
        self.action_btn.setText('⏹ 停止算法')
        self.action_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                padding: 12px 25px;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #c0392b, stop:1 #a93226);
            }
            QPushButton:pressed {
                background: #922b21;
            }
        """)

        # 启动算法线程
        self.algorithm_running = True
        self.algorithm_paused = False
        self.append_log("启动算法线程...")
        self.algorithm_thread = AlgorithmThread(params, self.algorithm_signals)
        self.algorithm_thread.start()

    def stop_algorithm(self):
        """停止算法"""
        if self.algorithm_thread is not None:
            self.algorithm_thread.stop()
            self.algorithm_running = False
            self.algorithm_paused = False

        # 恢复按钮状态为开始
        self.action_btn.setText('▶ 开始算法')
        self.action_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #27ae60, stop:1 #2ecc71);
                color: white;
                padding: 12px 25px;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #229954, stop:1 #27ae60);
            }
            QPushButton:pressed {
                background: #1e8449;
            }
            QPushButton:disabled {
                background: #95a5a6;
                color: #bdc3c7;
            }
        """)

        # 解锁参数
        self.lock_parameters(False)

        self.append_log("算法已停止")

    def algorithm_finished(self):
        """算法完成回调"""
        self.algorithm_running = False
        self.algorithm_paused = False

        # 恢复按钮状态为开始
        self.action_btn.setText('▶ 开始算法')
        self.action_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #27ae60, stop:1 #2ecc71);
                color: white;
                padding: 12px 25px;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #229954, stop:1 #27ae60);
            }
            QPushButton:pressed {
                background: #1e8449;
            }
            QPushButton:disabled {
                background: #95a5a6;
                color: #bdc3c7;
            }
        """)

        # 解锁参数
        self.lock_parameters(False)

        self.append_log("算法执行完成")

    def algorithm_error(self, error_msg):
        """算法错误回调"""
        self.append_log(f"错误: {error_msg}")
        self.algorithm_finished()

    def update_result_display(self, rgb, depth, result):
        """更新结果显示"""
        # 更新RGB显示
        if rgb is not None:
            self.display_image(rgb, self.rgb_label)

        # 更新深度图显示
        if depth is not None:
            # 归一化深度图用于显示
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
            self.display_image(depth_vis, self.depth_label)

        # 更新结果显示
        if result is not None:
            self.display_image(result, self.result_label)

    def display_image(self, img_array, label):
        """在标签上显示图像"""
        if img_array is None:
            return

        # 转换为QImage
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = (img_array * 255).astype(np.uint8)

        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        h, w, c = img_array.shape
        bytes_per_line = c * w
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # 缩放以适应标签
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def append_log(self, message):
        """添加日志消息"""
        import datetime
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    window = FoundationPoseGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
