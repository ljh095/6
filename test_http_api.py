"""
测试 FoundationPose HTTP 接口
向运行中的 GUI 发送 POST 请求进行位姿估计
"""

import requests
import json
import sys


def test_pose_estimation(host='http://localhost:8888'):
    """
    测试位姿估计接口

    Args:
        host: HTTP服务器地址，默认为 http://localhost:8888
    """

    # 首先检查服务是否运行
    try:
        response = requests.get(f'{host}/health', timeout=5)
        if response.status_code == 200:
            print(f"✅ 服务运行中")
            print(f"   响应: {response.json()}")
        else:
            print(f"❌ 服务状态异常: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器 {host}")
        print(f"   请确保 GUI 应用已启动并点击了'开始算法'按钮")
        return
    except Exception as e:
        print(f"❌ 连接错误: {str(e)}")
        return

    print("\n" + "="*60)
    print("发送位姿估计请求...")
    print("="*60)

    # 发送位姿估计请求
    try:
        # 可以在请求体中传递额外的参数（可选）
        request_data = {
            # 可以在这里添加自定义参数
            # "timeout": 30,
            # "debug": True
        }

        response = requests.post(
            f'{host}/estimate_pose',
            json=request_data,
            timeout=60  # 设置较长的超时时间，因为位姿估计可能需要一些时间
        )

        print(f"\n状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            if result.get('success'):
                print("\n✅ 位姿估计成功!")
                print(f"   请求ID: #{result.get('request_id')}")
                print(f"   消息: {result.get('message')}")
                print(f"   图像尺寸: {result.get('image_shape')}")
                print(f"   掩码像素数: {result.get('mask_pixels')}")

                # 打印姿态矩阵
                pose = result.get('pose')
                if pose:
                    print(f"\n   姿态矩阵 (4x4):")
                    for i, row in enumerate(pose):
                        print(f"   [{i}] {row}")

                print("\n请在 GUI 右侧面板查看可视化结果")
            else:
                print(f"\n❌ 位姿估计失败")
                print(f"   错误: {result.get('error')}")
                print(f"   请求ID: #{result.get('request_id')}")
        else:
            print(f"\n❌ 请求失败")
            try:
                error_info = response.json()
                print(f"   错误信息: {error_info}")
            except:
                print(f"   响应内容: {response.text}")

    except requests.exceptions.Timeout:
        print(f"\n❌ 请求超时")
        print(f"   位姿估计耗时过长，请检查算法是否正常运行")
    except Exception as e:
        print(f"\n❌ 请求错误: {str(e)}")


def continuous_test(host='http://localhost:8888', interval=5, count=5):
    """
    连续发送多次位姿估计请求

    Args:
        host: HTTP服务器地址
        interval: 每次请求之间的间隔时间（秒）
        count: 发送请求的总次数
    """
    import time

    print(f"连续测试模式: 将发送 {count} 次请求，间隔 {interval} 秒")
    print("="*60)

    for i in range(count):
        print(f"\n【第 {i+1}/{count} 次请求】")
        test_pose_estimation(host)

        if i < count - 1:
            print(f"\n等待 {interval} 秒...")
            time.sleep(interval)

    print("\n" + "="*60)
    print("连续测试完成")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试 FoundationPose HTTP 接口')
    parser.add_argument('--host', default='http://localhost:8888',
                        help='HTTP服务器地址 (默认: http://localhost:8888)')
    parser.add_argument('--continuous', action='store_true',
                        help='连续测试模式')
    parser.add_argument('--interval', type=int, default=5,
                        help='连续测试的间隔时间（秒）')
    parser.add_argument('--count', type=int, default=5,
                        help='连续测试的次数')

    args = parser.parse_args()

    if args.continuous:
        continuous_test(args.host, args.interval, args.count)
    else:
        test_pose_estimation(args.host)
