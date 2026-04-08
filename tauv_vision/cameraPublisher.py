import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import depthai as dai
from datetime import timedelta

MAX_RES = False
target_fps = 10

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # publishers for each stream
        self.pub_rgb = self.create_publisher(Image, 'vision/camera/rgb', 10)
        self.pub_left = self.create_publisher(Image, 'vision/camera/left', 10)
        self.pub_right = self.create_publisher(Image, 'vision/camera/right', 10)
        
        self.bridge = CvBridge()

    def publish_frames(self, rgb, left, right):
        # Convert numpy arrays to ROS 2 Image messages
        # 'bgr8' for color, 'mono8' for grayscale
        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        msg_left = self.bridge.cv2_to_imgmsg(left, encoding="mono8")
        msg_right = self.bridge.cv2_to_imgmsg(right, encoding="mono8")

        # Sync timestamps with the current ROS time
        timestamp = self.get_clock().now().to_msg()
        msg_rgb.header.stamp = timestamp
        msg_left.header.stamp = timestamp
        msg_right.header.stamp = timestamp

        # Add frame IDs for TF2 compatibility
        # TODO: Implement this with CAD or even camera IMU data
        # msg_rgb.header.frame_id = "cam_rgb_optical_frame"
        # msg_left.header.frame_id = "cam_left_optical_frame"
        # msg_right.header.frame_id = "cam_right_optical_frame"

        self.pub_rgb.publish(msg_rgb)
        self.pub_left.publish(msg_left)
        self.pub_right.publish(msg_right)

def main(args=None):
    rclpy.init(args=args)
    cam_node = CameraPublisher()

    with dai.Pipeline() as pipeline:
        # Define Camera Nodes
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        # Define Sync Node
        sync = pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=15))

        # We request the output from the cameras (internal) and link to Sync
        if MAX_RES:
            cam_rgb.requestOutput((4056,3040)).link(sync.inputs["rgb"])
        else:
            cam_rgb.requestOutput((1920, 1080)).link(sync.inputs["rgb"])
        cam_left.requestOutput((1280, 800)).link(sync.inputs["left"])
        cam_right.requestOutput((1280, 800)).link(sync.inputs["right"])

        # 5. Create the Synced Output Queue
        # We call createOutputQueue directly from the sync node's output port
        syncQueue = sync.out.createOutputQueue()

        controlQueueRGB = cam_rgb.inputControl.createInputQueue()

        pipeline.start()

        # Set manual focus once the pipeline is running
        controlQueueRGB = cam_rgb.inputControl.createInputQueue()
        controlQueueLeft = cam_left.inputControl.createInputQueue()
        controlQueueRight = cam_right.inputControl.createInputQueue()

        ctrl = dai.CameraControl()
        ctrl.setManualFocus(130)
        ctrl.setManualExposure(20000, 1400)
        ctrl.setManualWhiteBalance(7000)

        controlQueueRGB.send(ctrl)
        controlQueueLeft.send(ctrl)
        controlQueueRight.send(ctrl)

        # and, go!
        pipeline.start()


        while pipeline.isRunning():
            # Get the synchronized group
            msg_group = syncQueue.get()
            
            # Access frames by the keys used in the links
            f_rgb = msg_group["rgb"].getCvFrame()
            f_left = msg_group["left"].getCvFrame()
            f_right = msg_group["right"].getCvFrame()

            # Publish!
            cam_node.publish_frames(f_rgb, f_left, f_right)
            
            # Spin once to handle any callbacks if necessary
            rclpy.spin_once(cam_node, timeout_sec=0)

    cam_node.destroy_node()
    rclpy.shutdown()
