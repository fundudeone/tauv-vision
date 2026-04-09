import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import depthai as dai
from datetime import timedelta, datetime
from rclpy.qos import qos_profile_sensor_data
from pathlib import Path

MAX_RES = False
SEND_MONOS = False
SAVE_RGB = True
target_fps = 10


class VideoSaver(dai.node.HostNode):
    def __init__(self, *args, **kwargs):
        dai.node.HostNode.__init__(self, *args, **kwargs)
        self.file_handle = open(f"{Path.home()}/video-{datetime.now()}.encoded", 'wb')

    def build(self, *args):
        self.link_args(*args)
        return self

    def process(self, frame):
        frame.getData().tofile(self.file_handle)

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # publishers for each stream
        self.pub_rgb = self.create_publisher(Image, 'vision/camera/rgb', qos_profile_sensor_data)
        self.pub_rgb_compressed = self.create_publisher(CompressedImage, 'vision/camera/rgb/compressed', qos_profile_sensor_data)

        if SEND_MONOS:
            self.pub_left = self.create_publisher(Image, 'vision/camera/left', qos_profile_sensor_data)
            self.pub_right = self.create_publisher(Image, 'vision/camera/right', qos_profile_sensor_data)
        
        self.bridge = CvBridge()

    def publish_frames(self, rgb, bytes_rgb, left=None, right=None):
        # Convert numpy arrays to ROS 2 Image messages
        # 'bgr8' for color, 'mono8' for grayscale
        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")

        if SEND_MONOS:
            msg_left = self.bridge.cv2_to_imgmsg(left, encoding="mono8")
            msg_right = self.bridge.cv2_to_imgmsg(right, encoding="mono8")

        # Sync timestamps with the current ROS time
        timestamp = self.get_clock().now().to_msg()
        msg_rgb.header.stamp = timestamp
        if SEND_MONOS:
            msg_left.header.stamp = timestamp
            msg_right.header.stamp = timestamp

        # Add frame IDs for TF2 compatibility
        # TODO: Implement this with CAD or even camera IMU data
        # msg_rgb.header.frame_id = "cam_rgb_optical_frame"
        # msg_left.header.frame_id = "cam_left_optical_frame"
        # msg_right.header.frame_id = "cam_right_optical_frame"

        msg_rgb_comp = CompressedImage()
        msg_rgb_comp.header.stamp = timestamp
        msg_rgb_comp.header.frame_id = "cam_rgb_optical_frame"
        msg_rgb_comp.format = "jpeg"
        # DepthAI getData() returns a numpy array. We convert it to a flat list of bytes for ROS 2.
        msg_rgb_comp.data = bytes_rgb.tolist() 
        self.pub_rgb_compressed.publish(msg_rgb_comp)
        self.pub_rgb.publish(msg_rgb)

        if SEND_MONOS:
            self.pub_left.publish(msg_left)
            self.pub_right.publish(msg_right)

def main(args=None):
    rclpy.init(args=args)
    cam_node = CameraPublisher()

    with dai.Pipeline() as pipeline:
        # Define Camera Nodes
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setFps(target_fps)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        if MAX_RES:
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        else:
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

        enc_rgb = pipeline.create(dai.node.VideoEncoder)
        enc_rgb.setDefaultProfilePreset(target_fps, dai.VideoEncoderProperties.Profile.MJPEG)
        enc_rgb.setQuality(95)
        cam_rgb.video.link(enc_rgb.input)

        if SAVE_RGB:
            enc_h264 = pipeline.create(dai.node.VideoEncoder)
            enc_h264.setDefaultProfilePreset(target_fps, dai.VideoEncoderProperties.Profile.H264_MAIN)
            cam_rgb.video.link(enc_h264.input)
            saver = pipeline.create(VideoSaver).build(enc_h264.out)
        
        if SEND_MONOS:
            cam_left = pipeline.create(dai.node.MonoCamera)
            cam_left.setFps(target_fps)
            cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            cam_left.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

            enc_left = pipeline.create(dai.node.VideoEncoder)
            enc_left.setDefaultProfilePreset(target_fps, dai.VideoEncoderProperties.Profile.MJPEG)
            enc_left.setQuality(95)
            cam_left.out.link(enc_left.input)

            cam_right = pipeline.create(dai.node.MonoCamera)
            cam_right.setFps(target_fps)
            cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            cam_right.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

            enc_right = pipeline.create(dai.node.VideoEncoder)
            enc_right.setDefaultProfilePreset(target_fps, dai.VideoEncoderProperties.Profile.MJPEG)
            enc_right.setQuality(95)
            cam_right.out.link(enc_right.input)

        # Define Sync Node
        sync = pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=15))

        # We request the output from the cameras (internal) and link to Sync
        enc_rgb.bitstream.link(sync.inputs["rgb"])
        
        if SEND_MONOS:
            enc_left.bitstream.link(sync.inputs["left"])
            enc_right.bitstream.link(sync.inputs["right"])

        # We call createOutputQueue directly from the sync node's output port
        syncQueue = sync.out.createOutputQueue()

        # Set manual focus once the pipeline is running
        controlQueueRGB = cam_rgb.inputControl.createInputQueue()

        if SEND_MONOS:
            controlQueueLeft = cam_left.inputControl.createInputQueue()
            controlQueueRight = cam_right.inputControl.createInputQueue()

        ctrl = dai.CameraControl()
        ctrl.setManualFocus(130)
        ctrl.setManualExposure(17000, 1400)
        ctrl.setManualWhiteBalance(7000)

        controlQueueRGB.send(ctrl)

        if SEND_MONOS:
            controlQueueLeft.send(ctrl)
            controlQueueRight.send(ctrl)

        # and, go!
        pipeline.start()
        cam_node.get_logger().info(f"Starting! Target FPS: {target_fps}")

        try:
            while pipeline.isRunning() and rclpy.ok():
                # Get the synchronized group
                msg_group = syncQueue.get()
                
                # Extract raw JPEG bytes
                bytes_rgb = msg_group["rgb"].getData()

                if SEND_MONOS:
                    bytes_left = msg_group["left"].getData()
                    bytes_right = msg_group["right"].getData()

                # Decode JPEG bytes back to OpenCV numpy arrays on the host CPU
                # RGB will now decode back into a full 1080p (or 4K) array
                f_rgb = cv2.imdecode(bytes_rgb, cv2.IMREAD_COLOR)

                if SEND_MONOS:
                    f_left = cv2.imdecode(bytes_left, cv2.IMREAD_GRAYSCALE)
                    f_right = cv2.imdecode(bytes_right, cv2.IMREAD_GRAYSCALE)

                # Publish!
                if SEND_MONOS:
                    cam_node.publish_frames(f_rgb, bytes_rgb, f_left, f_right)
                else:
                    cam_node.publish_frames(f_rgb, bytes_rgb)

                # Spin once to handle any callbacks if necessary
                rclpy.spin_once(cam_node, timeout_sec=0)
        finally:
            if SAVE_RGB:
                cam_node.get_logger().info("Saving video")
                saver.file_handle.close()
                cam_node.get_logger().info("video saved")
            

    cam_node.destroy_node()
    rclpy.shutdown()
