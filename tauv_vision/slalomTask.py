import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class SlalomTask(Node):
    def __init__(self):
        super().__init__('slalom_task')

        # ---- parameters ----
        self.declare_parameters(
            namespace='',
            parameters=[
                ('input_rgb', '/oak/rgb/image_raw'),
                ('input_stereo', '/oak/stereo/image_raw'),
                ('output_topic', '/vision/slalom/red_location_on_image'),
                ('debug_mask_topic', '/vision/slalom/debug_mask'),
                ('debug', False),
                # Percent of the darkest pixels kept. The slalom pipes are red,
                # and red is the first wavelength the water eats, so underwater
                # they read as the darkest thing in frame rather than the
                # reddest -- hence a darkness percentile, not a hue window.
                ('threshold', 10.0),
                # Millimetres, matching the 16UC1 stereo image.
                ('depth_min', 2000.0),
                ('depth_max', 4000.0),
                # A pipe is 1 wide to 4 tall, i.e. width/height = 0.25.
                ('aspect_ratio', 0.25),
                ('aspect_tolerance', 0.10),
                ('min_blob_area', 100.0),
                ('sync_slop', 0.1),
            ]
        )

        self.debug = self.get_parameter('debug').value
        self.threshold = self.get_parameter('threshold').value
        self.depth_min = self.get_parameter('depth_min').value
        self.depth_max = self.get_parameter('depth_max').value
        self.aspect_ratio = self.get_parameter('aspect_ratio').value
        self.aspect_tolerance = self.get_parameter('aspect_tolerance').value
        self.min_blob_area = self.get_parameter('min_blob_area').value

        # ---- state ----
        self.bridge = CvBridge()

        # ---- pubs / subs ----
        self.pub_location = self.create_publisher(
            Float32, self.get_parameter('output_topic').value, 10)
        self.pub_mask = self.create_publisher(
            Image, self.get_parameter('debug_mask_topic').value, 10) \
            if self.debug else None

        sub_rgb = message_filters.Subscriber(
            self, Image, self.get_parameter('input_rgb').value,
            qos_profile=qos_profile_sensor_data)
        sub_depth = message_filters.Subscriber(
            self, Image, self.get_parameter('input_stereo').value,
            qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth], queue_size=5,
            slop=self.get_parameter('sync_slop').value)
        self.sync.registerCallback(self._image_cb)

        self.get_logger().info('slalom_task node ready.')

    # ------------------------------------------------------------------
    # Main update step. Any early `return` just skips this frame.
    # ------------------------------------------------------------------
    def _image_cb(self, rgb_msg, depth_msg):
        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg)  # 16UC1, mm

        # The camera is mounted upside down.
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        depth = cv2.rotate(depth, cv2.ROTATE_180)

        mask = self._build_mask(bgr, depth)

        if self.pub_mask is not None:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            mask_msg.header = rgb_msg.header
            self.pub_mask.publish(mask_msg)

        center = self._find_pipe(mask)
        if center is None:
            return

        msg = Float32()
        msg.data = float(center[0] / bgr.shape[1])
        self.pub_location.publish(msg)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_mask(self, bgr, depth):
        """Darkest `threshold` percent of the rgb image, intersected with the
        depth band. Returned at rgb resolution."""
        # Darkness is the HSV value channel: low V == dark.
        v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        dark = v <= np.percentile(v, self.threshold)

        # Depth may be a different resolution than rgb even when aligned.
        h, w = v.shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        in_band = (depth >= self.depth_min) & (depth <= self.depth_max)

        return (dark & in_band).astype(np.uint8) * 255

    def _find_pipe(self, mask):
        """Largest blob whose bounding box is about 1 wide to 4 tall, as an
        (x, y) center in pixels. None if nothing qualifies."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area, best_center = self.min_blob_area, None
        for c in contours:
            area = cv2.contourArea(c)
            if area < best_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if h < 1:
                continue
            if abs(w / h - self.aspect_ratio) > self.aspect_tolerance:
                continue

            best_area = area
            best_center = (x + w / 2.0, y + h / 2.0)

        return best_center


def main(args=None):
    rclpy.init(args=args)
    node = SlalomTask()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
