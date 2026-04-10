import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import depthai as dai
from datetime import timedelta, datetime
from rclpy.qos import qos_profile_sensor_data
from pathlib import Path

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
        self.pub_rgb_compressed = self.create_publisher(CompressedImage, 'vision/camera/rgb/compressed', qos_profile_sensor_data)

        if self.get_parameter('send_monos'):
            self.pub_left = self.create_publisher(Image, 'vision/camera/left', qos_profile_sensor_data)
            self.pub_right = self.create_publisher(Image, 'vision/camera/right', qos_profile_sensor_data)
        
        self.bridge = CvBridge()

    def publish_frames(self, rgb, bytes_rgb, left=None, right=None):
        # Convert numpy arrays to ROS 2 Image messages
        # 'bgr8' for color, 'mono8' for grayscale
        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")

        if self.get_parameter('send_monos'):
            msg_left = self.bridge.cv2_to_imgmsg(left, encoding="mono8")
            msg_right = self.bridge.cv2_to_imgmsg(right, encoding="mono8")

        # Sync timestamps with the current ROS time
        timestamp = self.get_clock().now().to_msg()
        msg_rgb.header.stamp = timestamp
        if self.get_parameter('send_monos'):
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

        if self.get_parameter('send_monos'):
            self.pub_left.publish(msg_left)
            self.pub_right.publish(msg_right)

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

def make_output_encoder_node(pipeline, cameraNodeOut, format : dai.VideoEncoderProperties.Profile, quality, target_fps) -> dai.node.VideoEncoder:
    enc_cam = pipeline.create(dai.node.VideoEncoder)
    enc_cam.setDefaultProfilePreset(target_fps, format)
    enc_cam.setQuality(quality)
    cameraNodeOut.video.link(enc_cam.input)
    return enc_cam

def main(args=None):
    rclpy.init(args=args)
    cam_node = CameraPublisher()

    with dai.Pipeline() as pipeline:
        target_fps = cam_node.get_parameter('target_fps')
        max_res = cam_node.get_parameter('max_res')

        # Define Camera Nodes
        cam_rgb = make_rgb_camera_node(pipeline, dai.CameraBoardSocket.CAM_A, target_fps, max_res)
        enc_rgb = make_output_encoder_node(pipeline, cam_rgb.video, dai.VideoEncoderProperties.Profile.MJPEG, 95, target_fps)

        if cam_node.get_parameter('save_rgb_video'):
            enc_h264 = make_output_encoder_node(pipeline, cam_rgb.video, dai.VideoEncoderProperties.profile.H264_MAIN, 95, target_fps)
            saver = pipeline.create(VideoSaver).build(enc_h264.out)
        
        if cam_node.get_parameter('send_monos'):
            cam_left = make_mono_camera_node(pipeline, dai.CameraBoardSocket.CAM_B, target_fps)
            enc_left = make_output_encoder_node(pipeline, cam_left.out, dai.VideoEncoderProperties.Profile.MJPEG, 95, target_fps)

            cam_right = make_mono_camera_node(pipeline, dai.CameraBoardSocket.CAM_C, target_fps)
            enc_right = make_output_encoder_node(pipeline, cam_right.out, dai.VideoEncoderProperties.Profile.MJPEG, 95, target_fps)

        # Define Sync Node
        sync = pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=15))

        # We request the output from the cameras (internal) and link to Sync
        enc_rgb.bitstream.link(sync.inputs["rgb"])
        
        if cam_node.get_parameter('send_monos'):
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

        if cam_node.get_parameter('send_monos'):
            controlQueueLeft = cam_left.inputControl.createInputQueue()
            controlQueueLeft.send(ctrl)
            controlQueueRight = cam_right.inputControl.createInputQueue()
            controlQueueRight.send(ctrl)

        # and, go!
        pipeline.start()
        cam_node.get_logger().info(f"Starting! Target FPS: {cam_node.get_parameter('target_fps')}")

        try:
            while pipeline.isRunning() and rclpy.ok():
                # Get the synchronized group
                msg_group = syncQueue.get()
                
                # Extract raw JPEG bytes
                bytes_rgb = msg_group["rgb"].getData()

                if cam_node.get_parameter('send_monos'):
                    bytes_left = msg_group["left"].getData()
                    bytes_right = msg_group["right"].getData()

                # Decode JPEG bytes back to OpenCV numpy arrays on the host CPU
                # RGB will now decode back into a full 1080p (or 4K) array
                f_rgb = cv2.imdecode(bytes_rgb, cv2.IMREAD_COLOR)

                # Publish!
                if cam_node.get_parameter('send_monos'):
                    f_left = cv2.imdecode(bytes_left, cv2.IMREAD_GRAYSCALE)
                    f_right = cv2.imdecode(bytes_right, cv2.IMREAD_GRAYSCALE)
                    cam_node.publish_frames(f_rgb, bytes_rgb, f_left, f_right)
                else:
                    cam_node.publish_frames(f_rgb, bytes_rgb)

                # Spin once to handle any callbacks if necessary
                rclpy.spin_once(cam_node, timeout_sec=0)
        finally:
            if cam_node.get_parameter('save_rgb_video'):
                cam_node.get_logger().info("Saving video")
                saver.file_handle.close()
                cam_node.get_logger().info("video saved")
            

    cam_node.destroy_node()
    rclpy.shutdown()
