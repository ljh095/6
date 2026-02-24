#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense D435 + FoundationPose 角点姿态估计
针对 2x2m 泡沫托盘的单次拍摄、高精度6D姿态估计
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import trimesh
import torch
import nvdiffrast.torch as dr
import time
from estimater import FoundationPose
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
from Utils import *
import json
import logging
from datetime import datetime


class RealSenseCornerPoseEstimator:
    """
    使用 RealSense D435 和 FoundationPose 进行角点6D姿态估计
    """

    def __init__(self,
                 mesh_file,
                 weights_dir='./weights',
                 camera_width=1280,
                 camera_height=720,
                 camera_fps=30,
                 debug=False,
                 debug_dir='./debug_corner'):
        """
        初始化姿态估计器

        参数:
            mesh_file: CAD模型文件路径 (.obj/.ply)
            weights_dir: 预训练权重目录
            camera_width: 图像宽度
            camera_height: 图像高度
            camera_fps: 帧率
            debug: 是否启用调试模式
            debug_dir: 调试输出目录
        """
        self.mesh_file = mesh_file
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.debug = debug
        self.debug_dir = debug_dir

        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

        # 初始化日志
        logging.basicConfig(
            level=logging.INFO if not debug else logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 加载CAD模型
        self.logger.info(f"Loading CAD model from: {mesh_file}")
        self.mesh = trimesh.load(mesh_file)
        self.model_pts = self.mesh.vertices
        self.model_normals = self.mesh.vertex_normals

        # 初始化评分器和细化器
        self.logger.info("Initializing scorer and refiner...")
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()

        # 创建渲染上下文
        self.glctx = dr.RasterizeCudaContext()

        # 初始化 FoundationPose
        self.logger.info("Initializing FoundationPose...")
        self.est = FoundationPose(
            model_pts=self.model_pts,
            model_normals=self.model_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=debug_dir,
            debug=1 if debug else 0,
            glctx=self.glctx
        )

        # RealSense 管道
        self.pipeline = None
        self.config = None
        self.align = None

        self.logger.info("CornerPoseEstimator initialized successfully")

    def init_camera(self):
        """
        初始化 RealSense D435 相机
        """
        self.logger.info("Initializing RealSense D435 camera...")

        # 创建管道
        self.pipeline = rs.pipeline()

        # 创建配置
        self.config = rs.config()

        # 配置流
        self.config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, self.camera_fps)
        self.config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, self.camera_fps)

        # 启动管道
        profile = self.pipeline.start(self.config)

        # 获取深度传感器
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        self.logger.info(f"Depth scale: {depth_scale}")

        # 创建对齐对象（将深度对齐到彩色）
        self.align = rs.align(rs.stream.color)

        # 等待相机稳定
        time.sleep(2)

        # 获取相机内参
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.logger.info(f"Camera matrix:\n{self.camera_matrix}")
        self.logger.info(f"Camera intrinsics: fx={intrinsics.fx}, fy={intrinsics.fy}, "
                        f"cx={intrinsics.ppx}, cy={intrinsics.ppy}")

        return True

    def capture_frame(self):
        """
        捕获一帧RGB-D数据

        返回:
            rgb: RGB图像 (H, W, 3), BGR格式
            depth: 深度图 (H, W), 单位: 米
            timestamp: 捕获时间戳
        """
        # 等待有效帧
        frames = self.pipeline.wait_for_frames()

        # 对齐深度帧到彩色帧
        aligned_frames = self.align.process(frames)

        # 获取彩色和深度帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get frames from camera")

        # 转换为numpy数组
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # 转换深度为米
        depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        depth = depth.astype(np.float32) * depth_scale

        # 将深度小于0.001m的值设为无效
        depth[depth < 0.001] = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return rgb, depth, timestamp

    def generate_mask_from_depth(self, depth, threshold_method='adaptive'):
        """
        从深度图生成物体掩码
        针对角点场景优化：检测深度异常区域

        参数:
            depth: 深度图 (H, W), 单位: 米
            threshold_method: 'adaptive' (自适应) 或 'fixed' (固定阈值)

        返回:
            mask: 二值掩码 (H, W), 0或1
        """
        H, W = depth.shape

        if threshold_method == 'adaptive':
            # 自适应方法：基于深度直方图分析
            valid_depth = depth[depth > 0.001]

            if len(valid_depth) == 0:
                return np.zeros((H, W), dtype=np.float32)

            # 计算深度统计量
            depth_median = np.median(valid_depth)
            depth_std = np.std(valid_depth)

            # 找到比中值近的深度（可能是托盘角点）
            # 托盘角点应该比背景近
            threshold = depth_median - 1.5 * depth_std

            # 创建掩码
            mask = ((depth > 0.001) & (depth < threshold)).astype(np.float32)

        else:
            # 固定阈值方法
            # 假设托盘距离相机在0.3-2.0米之间
            mask = ((depth > 0.3) & (depth < 2.0)).astype(np.float32)

        # 形态学操作：去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 保留最大的连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

        if num_labels > 1:
            # 找到最大的区域（不包括背景）
            max_area = 0
            max_label = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    max_label = i

            # 创建最终掩码
            mask = (labels == max_label).astype(np.float32)

        return mask

    def preprocess_depth(self, depth, method='bilateral'):
        """
        预处理深度图以提高精度

        参数:
            depth: 原始深度图
            method: 预处理方法 ('bilateral', 'gaussian')

        返回:
            processed_depth: 处理后的深度图
        """
        # 转换为torch tensor
        depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float32)

        # 侵蚀深度图（去除离群点）
        depth_tensor = erode_depth(depth_tensor, radius=2, device='cuda')

        if method == 'bilateral':
            # 双边滤波
            depth_tensor = bilateral_filter_depth(depth_tensor, radius=2, device='cuda')
        elif method == 'gaussian':
            # 高斯滤波
            depth_np = depth_tensor.cpu().numpy()
            depth_np = cv2.GaussianBlur(depth_np, (5, 5), 1.0)
            depth_tensor = torch.as_tensor(depth_np, device='cuda', dtype=torch.float32)

        return depth_tensor.cpu().numpy()

    def estimate_corner_pose(self, rgb, depth, mask=None, iteration=10):
        """
        估计角点的6D姿态

        参数:
            rgb: RGB图像 (H, W, 3), BGR格式
            depth: 深度图 (H, W), 单位: 米
            mask: 物体掩码 (H, W), 如果为None则自动生成
            iteration: 细化迭代次数

        返回:
            pose: 4x4变换矩阵，物体在相机坐标系中的姿态
            score: 姿态得分
        """
        # 预处理深度图
        depth = self.preprocess_depth(depth)

        # 如果没有提供掩码，自动生成
        if mask is None:
            mask = self.generate_mask_from_depth(depth)
        else:
            mask = mask.astype(np.float32)

        # 确保RGB是RGB格式
        rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 估计姿态
        self.logger.info("Estimating corner pose...")
        pose = self.est.register(
            K=self.camera_matrix,
            rgb=rgb_input,
            depth=depth,
            ob_mask=mask,
            iteration=iteration
        )

        # 获取得分
        if hasattr(self.est, 'scores') and len(self.est.scores) > 0:
            score = self.est.scores[0].item()
        else:
            score = 0.0

        self.logger.info(f"Pose estimation completed. Score: {score:.4f}")

        return pose, score

    def capture_and_estimate(self, save_debug=True):
        """
        捕获一帧并估计姿态（一次性拍摄）

        返回:
            pose: 4x4变换矩阵
            score: 姿态得分
            rgb: RGB图像
            depth: 深度图
            mask: 使用掩码
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting single-frame capture and pose estimation")

        # 捕获帧
        rgb, depth, timestamp = self.capture_frame()
        self.logger.info(f"Captured frame at {timestamp}")

        # 生成掩码
        mask = self.generate_mask_from_depth(depth)

        # 估计姿态
        pose, score = self.estimate_corner_pose(rgb, depth, mask, iteration=10)

        # 保存调试信息
        if save_debug and self.debug:
            self._save_debug_images(rgb, depth, mask, pose, timestamp)

        self.logger.info("=" * 60)

        return pose, score, rgb, depth, mask

    def _save_debug_images(self, rgb, depth, mask, pose, timestamp):
        """
        保存调试图像
        """
        cv2.imwrite(f'{self.debug_dir}/rgb_{timestamp}.png', rgb)
        cv2.imwrite(f'{self.debug_dir}/depth_{timestamp}.png', (depth * 1000).astype(np.uint16))
        cv2.imwrite(f'{self.debug_dir}/mask_{timestamp}.png', (mask * 255).astype(np.uint8))

        # 可视化姿态
        vis = self.visualize_pose(rgb, pose)
        cv2.imwrite(f'{self.debug_dir}/vis_{timestamp}.png', vis)

    def visualize_pose(self, rgb, pose, line_thickness=3):
        """
        可视化姿态估计结果

        返回:
            vis: 可视化图像
        """
        # 计算边界框
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        # 绘制3D边界框
        vis = draw_posed_3d_box(
            self.camera_matrix,
            img=rgb,
            ob_in_cam=pose @ np.linalg.inv(to_origin),
            bbox=bbox,
            line_color=(0, 255, 0),
            linewidth=line_thickness
        )

        # 绘制坐标轴
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=pose @ np.linalg.inv(to_origin),
            scale=0.1,
            K=self.camera_matrix,
            thickness=line_thickness,
            transparency=0,
            is_input_rgb=False
        )

        return vis

    def save_pose_result(self, pose, score, timestamp, save_dir='./results'):
        """
        保存姿态估计结果为JSON文件

        参数:
            pose: 4x4变换矩阵
            score: 姿态得分
            timestamp: 时间戳
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        result = {
            'timestamp': timestamp,
            'pose_matrix': pose.tolist(),
            'score': float(score),
            'camera_matrix': self.camera_matrix.tolist()
        }

        filename = f'{save_dir}/pose_result_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)

        self.logger.info(f"Pose result saved to: {filename}")

        return filename

    def stop_camera(self):
        """
        停止相机
        """
        if self.pipeline:
            self.pipeline.stop()
            self.logger.info("Camera stopped")


def main():
    """
    主函数：演示如何使用CornerPoseEstimator
    """
    # 参数配置
    MESH_FILE = './assets/demo_data/mustard/mustard_model/mustard.obj'  # 修改为你的托盘CAD模型
    DEBUG = True
    DEBUG_DIR = './debug_corner'
    RESULTS_DIR = './results'

    # 创建姿态估计器
    print("=" * 60)
    print("Initializing CornerPoseEstimator...")
    print("=" * 60)

    estimator = RealSenseCornerPoseEstimator(
        mesh_file=MESH_FILE,
        camera_width=1280,
        camera_height=720,
        camera_fps=30,
        debug=DEBUG,
        debug_dir=DEBUG_DIR
    )

    try:
        # 初始化相机
        estimator.init_camera()

        print("\n" + "=" * 60)
        print("Camera initialized. Press Enter to capture and estimate...")
        print("Or type 'q' to quit")
        print("=" * 60)

        while True:
            user_input = input()

            if user_input.lower() == 'q':
                print("Exiting...")
                break

            print("\nCapturing frame and estimating pose...")
            start_time = time.time()

            # 捕获并估计姿态
            pose, score, rgb, depth, mask = estimator.capture_and_estimate(save_debug=True)

            elapsed_time = time.time() - start_time

            # 保存结果
            _, _, timestamp = estimator.capture_frame()  # Get timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            estimator.save_pose_result(pose, score, timestamp, save_dir=RESULTS_DIR)

            # 打印结果
            print("\n" + "=" * 60)
            print("POSE ESTIMATION RESULT")
            print("=" * 60)
            print(f"Score: {score:.4f}")
            print(f"Processing time: {elapsed_time:.3f} seconds")
            print(f"\nPose Matrix:")
            for row in pose:
                print(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}, {row[3]:8.4f}]")
            print("=" * 60)
            print("\nPress Enter to capture again, or 'q' to quit:")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 停止相机
        estimator.stop_camera()
        print("\nDone.")


if __name__ == '__main__':
    main()
