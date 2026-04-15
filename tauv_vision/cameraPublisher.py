import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
import cv2
import depthai as dai
from datetime import timedelta, datetime
from rclpy.qos import qos_profile_sensor_data
from pathlib import Path
import concurrent.futures
from ament_index_python.packages import get_package_share_directory
import os
import json
import time
import collections

BENCHMARKING = False

class VideoSaver(dai.node.HostNode):
    def __init__(self, *args, **kwargs):
        dai.node.HostNode.__init__(self, *args, **kwargs)
        self.file_handle = open(f"/tauv-mono/videos/video-{datetime.now()}.encoded", 'wb')

    def build(self, *args):
        self.link_args(*args)
        return self

    def process(self, frame):
        frame.getData().tofile(self.file_handle)

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_res', False), # True = Stream rgb at 4k, False = stream rgb at 1080p
                ('send_monos', False), # True = Stream and publish raw mono camera outputs, False = don't
                ('save_rgb_video', True), # True = Save rgb camera output to a video file
                ('target_fps', 10) # FPS to try to target
            ]
        )

        # publishers for each stream
        self.pub_rgb = self.create_publisher(Image, 'vision/camera/rgb', qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, 'vision/camera/depth', qos_profile_sensor_data)
        self.pub_rgb_compressed = self.create_publisher(CompressedImage, 'vision/camera/rgb/compressed', qos_profile_sensor_data)

        if self.get_parameter('send_monos').value:
            self.pub_left = self.create_publisher(Image, 'vision/camera/left', qos_profile_sensor_data)
            self.pub_right = self.create_publisher(Image, 'vision/camera/right', qos_profile_sensor_data)
        
        # info publisher
        self.pub_info = self.create_publisher(CameraInfo, 'vision/camera/camera_info', qos_profile_sensor_data)
        (rgbStreamWidth, rgbStreamHeight) = (3840, 2160) if self.get_parameter('max_res').value else (1920, 1080)
        self.info = CameraInfo()
        self.setup_camera_info(rgbStreamWidth, rgbStreamHeight)

        self.bridge = CvBridge()

        # benchmarking init
        self.last_time = time.perf_counter()
        self.frame_deltas = collections.deque(maxlen=30)
        self.pub_durations = collections.deque(maxlen=30)

    def setup_camera_info(self, rgbStreamWidth, rgbStreamHeight):
        with open(os.path.join(
            get_package_share_directory('tauv_vision'),
            'configs',
            'real_factory_calibration_backup.json'
        )) as f:
            jsonParse = json.load(f)

        RGB_CAMERA_INDEX = 2
        cameraData = jsonParse["cameraData"][RGB_CAMERA_INDEX][1]

        xIntrinsicScale = rgbStreamWidth/cameraData["width"]
        yIntrinsicScale = rgbStreamHeight/cameraData["height"]

        self.info.width = rgbStreamWidth
        self.info.height = rgbStreamHeight

        configIntrinsics = cameraData["intrinsicMatrix"]
        self.info.k = [float(configIntrinsics[0][0]*xIntrinsicScale), 0.0, float(configIntrinsics[0][2]*xIntrinsicScale), 
                       0.0, float(configIntrinsics[1][1]*yIntrinsicScale), float(configIntrinsics[1][2]*yIntrinsicScale), 
                       0.0, 0.0, 1.0
                       ]
        
        if cameraData["cameraType"] == 0:
            self.info.distortion_model = "plumb_bob"
        elif cameraData["cameraType"] == 1:
            self.info.distortion_model = "equidistant"

        self.info.d = cameraData["distortionCoeff"]

        # Since depth is already aligned to the RGB, the RGB 'R' is identity.
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        self.info.p = [
            self.info.k[0], self.info.k[1], self.info.k[2], 0.0,
            self.info.k[3], self.info.k[4], self.info.k[5], 0.0,
            self.info.k[6], self.info.k[7], self.info.k[8], 0.0
        ]


    def publish_frames(self, rgb, bytes_rgb, depth, left=None, right=None):
        # Sync timestamps with the current ROS time
        timestamp = self.get_clock().now().to_msg()

        # Convert numpy arrays to ROS 2 Image messages
        # 'bgr8' for color, 'mono8' for grayscale
        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        msg_rgb.header.stamp = timestamp
        msg_rgb.header.frame_id = "camera_link"

        # depthmap is unsigned 16 bit ints
        msg_depth = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        msg_depth.header.stamp = timestamp
        msg_depth.header.frame_id = "camera_link"

        send_monos = self.get_parameter('send_monos').value

        send_monos = self.get_parameter('send_monos').value

        if send_monos:
            msg_left = self.bridge.cv2_to_imgmsg(left, encoding="mono8")
            msg_right = self.bridge.cv2_to_imgmsg(right, encoding="mono8")

        # Sync timestamps with the current ROS time
        timestamp = self.get_clock().now().to_msg()
        msg_rgb.header.stamp = timestamp
        if send_monos:
            msg_left.header.stamp = timestamp
            msg_right.header.stamp = timestamp

        msg_rgb_comp = CompressedImage()
        msg_rgb_comp.header.stamp = timestamp
        msg_rgb_comp.header.frame_id = "camera_link"
        msg_rgb_comp.format = "jpeg"
        
        # DepthAI getData() returns a numpy array. We convert it to a flat list of bytes for ROS 2.
        msg_rgb_comp.data = bytes_rgb.tolist() 

        # benchmarking
        now = time.perf_counter()
        delta = now - self.last_time
        self.frame_deltas.append(delta)
        self.last_time = now

        pub_start = time.perf_counter()
        pub_end = time.perf_counter()
        self.pub_durations.append(pub_end - pub_start)

        self.pub_rgb.publish(msg_rgb)
        self.pub_depth.publish(msg_depth)
        self.pub_rgb_compressed.publish(msg_rgb_comp)

        if send_monos:
            self.pub_left.publish(msg_left)
            self.pub_right.publish(msg_right)
        
        #publish camera info
        self.pub_info.publish(self.info)

        if BENCHMARKING:
            self.report_metrics()

    def report_metrics(self):
        if not self.frame_deltas: return
        
        avg_hz = 1.0 / (sum(self.frame_deltas) / len(self.frame_deltas))
        avg_pub_ms = (sum(self.pub_durations) / len(self.pub_durations)) * 1000
        
        self.get_logger().info(
            f"Internal Rate: {avg_hz:.2f} Hz | "
            f"Publish Call Latency: {avg_pub_ms:.2f} ms",
            throttle_duration_sec=1.0
        )

def make_rgb_camera_node(pipeline, socket : dai.CameraBoardSocket, target_fps, max_res) -> dai.node.ColorCamera:
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setFps(target_fps)
    cam_rgb.setBoardSocket(socket)
    if max_res:
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    else:
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    
    return cam_rgb

def make_mono_camera_node(pipeline, socket, target_fps) -> dai.node.MonoCamera:
    cam_mono = pipeline.create(dai.node.MonoCamera)
    cam_mono.setFps(target_fps)
    cam_mono.setBoardSocket(socket)
    cam_mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    cam_mono.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    
    return cam_mono

def make_output_encoder_node(pipeline, cameraNodeOut, format : dai.VideoEncoderProperties.Profile, quality, target_fps) -> dai.node.VideoEncoder:
    enc_cam = pipeline.create(dai.node.VideoEncoder)
    enc_cam.setDefaultProfilePreset(target_fps, format)
    enc_cam.setQuality(quality)
    cameraNodeOut.link(enc_cam.input)
    return enc_cam

def main(args=None):
    rclpy.init(args=args)
    cam_node = CameraPublisher()

    with dai.Pipeline() as pipeline:
        target_fps = cam_node.get_parameter('target_fps').value
        max_res = cam_node.get_parameter('max_res').value
        send_monos = cam_node.get_parameter('send_monos').value
        save_rgb_video = cam_node.get_parameter('save_rgb_video').value

        # Define Camera Nodes
        cam_rgb = make_rgb_camera_node(pipeline, dai.CameraBoardSocket.CAM_A, target_fps, max_res)
        enc_rgb = make_output_encoder_node(pipeline, cam_rgb.video, dai.VideoEncoderProperties.Profile.MJPEG, 95, target_fps)

        if save_rgb_video:
            enc_h264 = make_output_encoder_node(pipeline, cam_rgb.video, dai.VideoEncoderProperties.Profile.H264_MAIN, 95, target_fps)
            saver = pipeline.create(VideoSaver).build(enc_h264.out)
        
        cam_left = make_mono_camera_node(pipeline, dai.CameraBoardSocket.CAM_B, target_fps)
        cam_right = make_mono_camera_node(pipeline, dai.CameraBoardSocket.CAM_C, target_fps)
        if send_monos:
            enc_left = make_output_encoder_node(pipeline, cam_left.out, dai.VideoEncoderProperties.Profile.MJPEG, 95, target_fps)
            enc_right = make_output_encoder_node(pipeline, cam_right.out, dai.VideoEncoderProperties.Profile.MJPEG, 95, target_fps)

        # Define Sync Node
        # We request the output from the cameras (internal) and generated depthmap and link to Sync
        sync = pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=15))

        # Make stereo dpeth node
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
        stereo.initialConfig.setConfidenceThreshold(0)
        stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 10000
        stereo.setRectifyEdgeFillColor(0)
        stereo.enableDistortionCorrection(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) # Align depth to RGB
        enc_rgb.bitstream.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth"])
        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        if send_monos:
            enc_left.bitstream.link(sync.inputs["left"])
            enc_right.bitstream.link(sync.inputs["right"])

        # We call createOutputQueue directly from the sync node's output port
        syncQueue = sync.out.createOutputQueue()

        ctrl = dai.CameraControl()
        ctrl.setManualFocus(130)
        ctrl.setManualExposure(17000, 1400)
        ctrl.setManualWhiteBalance(7000)

        # Set manual focus once the pipeline is running
        controlQueueRGB = cam_rgb.inputControl.createInputQueue()
        controlQueueRGB.send(ctrl)

        if send_monos:
            controlQueueLeft = cam_left.inputControl.createInputQueue()
            controlQueueLeft.send(ctrl)
            controlQueueRight = cam_right.inputControl.createInputQueue()
            controlQueueRight.send(ctrl)

        # and, go!
        pipeline.start()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        cam_node.get_logger().info(f"Starting! Target FPS: {cam_node.get_parameter('target_fps').value}")

        try:
            while pipeline.isRunning() and rclpy.ok():
                # Get the synchronized group
                msg_group = syncQueue.get()
                
                # Extract raw JPEG bytes
                bytes_rgb = msg_group["rgb"].getData()

                # get raw depthmap
                f_depth = msg_group["depth"].getFrame()

                if send_monos:
                    bytes_left = msg_group["left"].getData()
                    bytes_right = msg_group["right"].getData()

                # Publish!
                if send_monos:
                    # Decode all 3 images simultaneously 
                    future_rgb = executor.submit(cv2.imdecode, bytes_rgb, cv2.IMREAD_COLOR)
                    future_left = executor.submit(cv2.imdecode, bytes_left, cv2.IMREAD_GRAYSCALE)
                    future_right = executor.submit(cv2.imdecode, bytes_right, cv2.IMREAD_GRAYSCALE)

                    f_rgb = future_rgb.result()
                    f_left = future_left.result()
                    f_right = future_right.result()
                    
                    cam_node.publish_frames(f_rgb, bytes_rgb, f_depth, f_left, f_right)
                else:
                    f_rgb = cv2.imdecode(bytes_rgb, cv2.IMREAD_COLOR) 
                    cam_node.publish_frames(f_rgb, bytes_rgb, f_depth)

                # Spin once to handle any callbacks if necessary
                rclpy.spin_once(cam_node, timeout_sec=0)
        finally:
            if save_rgb_video:
                cam_node.get_logger().info("Saving video")
                saver.file_handle.close()
                cam_node.get_logger().info("video saved")
            

    cam_node.destroy_node()
    rclpy.shutdown()
