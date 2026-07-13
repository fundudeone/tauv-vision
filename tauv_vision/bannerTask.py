import os

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import tf2_ros

from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# Fixed keypoint order the banner2026 model was trained with. This is the
# model's index -> symbol mapping and is not recoverable from the .pt (which
# carries kpt_shape but no keypoint names), so it is asserted here.
KEYPOINT_NAMES = ['blood', 'ambulance', 'truck', 'fire']


def transform_stamped_to_Rt(tf_stamped):
    """Convert a geometry_msgs/TransformStamped into a (3x3 rotation, 3 translation)."""
    q = tf_stamped.transform.rotation
    t = tf_stamped.transform.translation

    x, y, z, w = q.x, q.y, q.z, q.w
    # Quaternion -> rotation matrix
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    translation = np.array([t.x, t.y, t.z], dtype=np.float64)
    return R, translation


class BannerTask(Node):
    def __init__(self):
        super().__init__('banner_task')

        # ---- parameters ----
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rgb_topic', 'oak/rgb/image_raw'),
                ('depth_topic', 'oak/stereo/image_raw'),
                ('camera_info_topic', 'oak/rgb/camera_info'),
                ('targets_topic', 'bannerTargets'),
                ('keypoints_topic', 'bannerKeypoints'),
                ('filter_frame', 'odom'),
                ('output_frame', 'os/base_link'),
                ('confidence_threshold', 0.65),
                ('kp_conf_threshold', 0.5),
                ('mask_threshold', 0.10),
                ('min_ellipse_area', 20.0),
                # Bridging folds by closing the mask sounds appealing but made
                # things worse in practice -- it merges a ring with neighbouring
                # ink and manufactures false positives. Off by default.
                ('close_kernel', 0),
                # A ring is judged by its *fitted ellipse* against the mask, not
                # by any property of the contour: the banner is fabric, and a
                # folded ring yields a thin arc whose fill is ~1.0, which no
                # contour-fill threshold can tell apart from a solid blob.
                #
                # Hole emptiness is the whole test. A ring has an empty middle
                # however broken its stroke is; a blob does not. Both rings'
                # holes are wider than hole_scale (290/385 = 0.75, 190/290 =
                # 0.66), so sampling at 0.6 lands inside the hole of either.
                # Measured on bag frames this separates perfectly: rings score
                # 0.000, everything else scores 0.41 or above.
                ('hole_scale', 0.6),
                ('max_hole_ink', 0.10),
                # Two arcs of the same folded ring fit two near-identical
                # ellipses; merge candidates whose centers are within this
                # fraction of their mean axis length.
                ('dedup_center_dist', 0.35),
                # Rings foreshorten to ellipses when viewed off-axis; this bounds
                # how oblique before we call it a sliver. Higher = closer to circle required
                ('min_aspect_ratio', 0.7),
                # The four keypoint symbols are dark ink too, so they fit
                # ellipses just like the rings do. Once a candidate's center is
                # reprojected onto the plane we can tell them apart by identity
                # rather than by shape: anything landing on a filtered keypoint
                # is that symbol, not a target. Metres.
                ('min_keypoint_clearance', 0.1),
                ('banner_class_name', 'banner'),
                ('model_filename', 'banner2026_best.pt'),
                ('sync_slop', 0.1),
                ('tf_timeout', 0.2),
                ('measurement_variance', 0.01),
                ('process_variance', 1e-5),
            ]
        )

        self.filter_frame = self.get_parameter('filter_frame').value
        self.output_frame = self.get_parameter('output_frame').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.kp_conf_threshold = self.get_parameter('kp_conf_threshold').value
        self.mask_threshold = self.get_parameter('mask_threshold').value
        self.min_ellipse_area = self.get_parameter('min_ellipse_area').value
        self.close_kernel = self.get_parameter('close_kernel').value
        self.hole_scale = self.get_parameter('hole_scale').value
        self.max_hole_ink = self.get_parameter('max_hole_ink').value
        self.dedup_center_dist = self.get_parameter('dedup_center_dist').value
        self.min_aspect_ratio = self.get_parameter('min_aspect_ratio').value
        self.min_keypoint_clearance = self.get_parameter(
            'min_keypoint_clearance').value
        self.banner_class_name = self.get_parameter('banner_class_name').value
        self.measurement_variance = self.get_parameter('measurement_variance').value
        self.process_variance = self.get_parameter('process_variance').value

        # ---- model ----
        model_path = os.path.join(
            get_package_share_directory('tauv_vision'),
            'models',
            self.get_parameter('model_filename').value,
        )
        self.get_logger().info(f'Loading YOLO keypoint model: {model_path}')
        self.model = YOLO(model_path)

        # ---- TF ----
        # spin_thread=True gives the listener its own executor, so the blocking
        # lookup below can wait for a transform without deadlocking this node's
        # callback thread.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)
        self.tf_timeout = Duration(
            seconds=self.get_parameter('tf_timeout').value)

        # ---- state ----
        self.bridge = CvBridge()
        self.camera_info = None
        # One constant-position Kalman filter per tracked point, built lazily,
        # keyed 'kp:<name>' for the four keypoints and 'target:<name>' for the
        # ring beside each of them.
        self.kalman_filters = {}
        # Latest filtered ring position, keyed by keypoint slot. Persists across
        # frames so a momentarily lost ring keeps reporting its estimate.
        self.target_estimates = {}

        # ---- pubs / subs ----
        self.pub_targets = self.create_publisher(
            PoseArray, self.get_parameter('targets_topic').value, 10)
        self.pub_keypoints = self.create_publisher(
            PoseArray, self.get_parameter('keypoints_topic').value, 10)

        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self._camera_info_cb, qos_profile_sensor_data)

        sub_rgb = message_filters.Subscriber(
            self, Image, self.get_parameter('rgb_topic').value,
            qos_profile=qos_profile_sensor_data)
        sub_depth = message_filters.Subscriber(
            self, Image, self.get_parameter('depth_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth], queue_size=5,
            slop=self.get_parameter('sync_slop').value)
        self.sync.registerCallback(self._image_cb)

        self.get_logger().info('banner_task node ready.')

    def _camera_info_cb(self, msg):
        self.camera_info = msg

    # ------------------------------------------------------------------
    # Main update step. Any early `return` just skips this frame.
    # ------------------------------------------------------------------
    def _image_cb(self, rgb_msg, depth_msg):
        if self.camera_info is None:
            self.get_logger().warn('No camera_info yet; skipping frame.', throttle_duration_sec=5.0)
            return

        camera_frame = self.camera_info.header.frame_id
        stamp = rgb_msg.header.stamp
        tf_time = Time.from_msg(stamp)

        # Cache capture-time transforms BEFORE running YOLO so that drift during
        # inference does not corrupt the geometry.
        # Odometry lags the image by a few tens of ms, so wait briefly rather
        # than dropping the frame outright.
        try:
            tf_filter_cam = self.tf_buffer.lookup_transform(
                self.filter_frame, camera_frame, tf_time, timeout=self.tf_timeout)
            tf_output_filter = self.tf_buffer.lookup_transform(
                self.output_frame, self.filter_frame, tf_time, timeout=self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF unavailable ({e}); skipping frame.',
                                   throttle_duration_sec=5.0)
            return

        R_fc, t_fc = transform_stamped_to_Rt(tf_filter_cam)      # filter <- camera
        R_of, t_of = transform_stamped_to_Rt(tf_output_filter)   # output <- filter

        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg)  # 16UC1 (mm) aligned to rgb
        depth_m = depth.astype(np.float32) / 1000.0

        # --- YOLO inference ---
        results = self.model.predict(bgr, verbose=False)
        if not results:
            return
        result = results[0]

        box_idx = self._best_banner_box(result)
        if box_idx is None:
            return  # no banner above confidence threshold

        keypoints_px = self._extract_keypoints(result, box_idx)
        if keypoints_px is None:
            return  # missing one or more of the four keypoints

        # --- deproject + filter keypoints in filter_frame ---
        fx, fy, cx, cy = self._intrinsics()
        filtered_points = {}
        for name, (u, v) in keypoints_px.items():
            p_cam = self._deproject(u, v, depth_m, fx, fy, cx, cy)
            if p_cam is None:
                return  # invalid depth for a required keypoint
            p_filter = R_fc @ p_cam + t_fc
            filtered_points[name] = self._update_kalman(f'kp:{name}', p_filter)

        pts = np.array([filtered_points[n] for n in KEYPOINT_NAMES], dtype=np.float64)

        # --- plane of best fit to the 4 filtered keypoints ---
        plane_point, plane_normal = self._fit_plane(pts)

        # --- target detection inside the banner bbox ---
        x1, y1, x2, y2 = self._bbox(result, box_idx, bgr.shape)
        centers_px = self._find_target_ellipses(bgr, x1, y1, x2, y2)

        # --- project ellipse centers onto the plane, then identify them ---
        # Each ring is named by the keypoint it is nearest to. That slot is its
        # identity: it is what the ring's Kalman filter is keyed on, and what
        # fixes its index in the published PoseArray from frame to frame.
        candidates = []   # (slot, distance, point)
        for (u, v) in centers_px:
            p_filter = self._ray_plane_intersect(
                u, v, fx, fy, cx, cy, R_fc, t_fc, plane_point, plane_normal)
            if p_filter is None:
                continue

            distances = np.linalg.norm(pts - p_filter, axis=1)
            slot = int(distances.argmin())

            # The keypoint symbols are dark ink and fit ellipses too. In 3D they
            # are distinguishable by identity: a candidate sitting *on* a
            # filtered keypoint IS that symbol, not a ring.
            if distances[slot] < self.min_keypoint_clearance:
                continue

            candidates.append((slot, float(distances[slot]), p_filter))

        # One ring per keypoint. If two candidates claim the same keypoint, the
        # closer one wins -- which also caps us at four targets by construction,
        # with no need to guess that the biggest blobs are the real ones.
        claimed = set()
        for slot, _dist, point in sorted(candidates, key=lambda c: c[1]):
            if slot in claimed:
                continue
            claimed.add(slot)
            name = KEYPOINT_NAMES[slot]
            self.target_estimates[slot] = self._update_kalman(
                f'target:{name}', point)

        # Publish every ring we have an estimate for, ordered by keypoint id, so
        # bannerTargets[i] is always the ring beside KEYPOINT_NAMES[i]. A ring
        # missed this frame (folded away, occluded) still reports its filtered
        # position rather than shifting everything after it.
        target_slots = sorted(self.target_estimates)
        targets = np.array([self.target_estimates[s] for s in target_slots],
                           dtype=np.float64)

        # --- publish (transform filter_frame -> output_frame) ---
        self._publish(self.pub_keypoints, pts, R_of, t_of, stamp)
        self._publish(self.pub_targets, targets, R_of, t_of, stamp)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _intrinsics(self):
        k = self.camera_info.k
        return k[0], k[4], k[2], k[5]  # fx, fy, cx, cy

    def _best_banner_box(self, result):
        """Index of the highest-confidence banner box above threshold, or None."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        names = result.names
        banner_ids = [cid for cid, nm in names.items() if nm == self.banner_class_name]
        if not banner_ids:
            self.get_logger().warn(
                f"Class '{self.banner_class_name}' not in model classes {names}.",
                throttle_duration_sec=10.0)
            return None
        banner_ids = set(banner_ids)

        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        best_idx, best_conf = None, self.confidence_threshold
        for i in range(len(confs)):
            if clss[i] in banner_ids and confs[i] >= best_conf:
                best_idx, best_conf = i, confs[i]
        return best_idx

    def _extract_keypoints(self, result, box_idx):
        """Return {name: (u, v)} if all four keypoints are visible, else None."""
        kpts = result.keypoints
        if kpts is None or kpts.data is None:
            return None

        data = kpts.data.cpu().numpy()  # [N, K, 3] = (x, y, conf)
        if box_idx >= data.shape[0] or data.shape[1] < len(KEYPOINT_NAMES):
            return None

        instance = data[box_idx]
        out = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            u, v, conf = instance[i]
            if conf < self.kp_conf_threshold:
                return None
            out[name] = (float(u), float(v))
        return out

    def _bbox(self, result, box_idx, shape):
        h, w = shape[:2]
        x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[box_idx]
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h))
        return x1, y1, x2, y2

    def _deproject(self, u, v, depth_m, fx, fy, cx, cy):
        """Pixel + depth -> 3D point in the camera optical frame (or None)."""
        # depth may be a different resolution than rgb even when aligned; scale.
        dh, dw = depth_m.shape[:2]
        su = int(round(u * dw / self.camera_info.width))
        sv = int(round(v * dh / self.camera_info.height))
        if not (0 <= su < dw and 0 <= sv < dh):
            return None
        z = float(depth_m[sv, su])
        if not np.isfinite(z) or z <= 0.0:
            return None
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)

    def _update_kalman(self, name, measurement):
        kf = self.kalman_filters.get(name)
        if kf is None:
            kf = KalmanFilter(dim_x=3, dim_z=3)
            kf.F = np.eye(3)
            kf.H = np.eye(3)
            kf.R = np.eye(3) * self.measurement_variance
            kf.Q = np.eye(3) * self.process_variance
            kf.P = np.eye(3) * 1.0
            kf.x = measurement.reshape(3, 1)  # initialise on first measurement
            self.kalman_filters[name] = kf
            return measurement

        kf.predict()
        kf.update(measurement.reshape(3, 1))
        return kf.x.reshape(3).copy()

    @staticmethod
    def _fit_plane(points):
        """Plane of best fit via SVD -> (point_on_plane, unit_normal)."""
        centroid = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - centroid)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        return centroid, normal

    def _find_target_ellipses(self, bgr, x1, y1, x2, y2):
        """Mask the darkest `mask_threshold` fraction of pixels in the bbox and
        find the ring targets in it.

        Candidates are judged by fitting an ellipse and then scoring that ellipse
        against the mask (hole emptiness), rather than by any property of the
        contour itself. That is what makes a folded ring -- whose stroke breaks
        into thin arcs -- still recoverable: an arc is solid ink, so any
        contour-fill test would call it a blob, but the ellipse it fits still has
        an empty middle.

        Returns the ellipse centers in full-image pixel coordinates, largest
        first.
        """
        if x2 - x1 < 1 or y2 - y1 < 1:
            return []
        crop = bgr[y1:y2, x1:x2]

        # Darkness is the HSV value channel: low V == dark.
        v = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]

        thresh = np.percentile(v, 100.0 * self.mask_threshold)
        mask = (v <= thresh).astype(np.uint8) * 255

        # Close small gaps first: a crease in the fabric can sever a ring's
        # stroke, and bridging it here recovers the ring as one contour.
        if self.close_kernel > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scored = []   # (major axis, ellipse)
        for c in contours:
            # A conic has 5 degrees of freedom, so fitEllipse needs >= 5 points.
            if len(c) < 5 or cv2.contourArea(c) < self.min_ellipse_area:
                continue

            ellipse = cv2.fitEllipse(c)
            _center, (axis_a, axis_b), _angle = ellipse
            major = max(axis_a, axis_b)
            aspect = min(axis_a, axis_b) / max(major, 1e-6)

            if (aspect >= self.min_aspect_ratio
                    and self._hole_ink(ellipse, mask) <= self.max_hole_ink):
                scored.append((major, ellipse))

        # Deliberately not capped at four here: a keypoint symbol can still be in
        # this list, and it is the 3D stage -- nearest-keypoint assignment -- that
        # tells rings and symbols apart and enforces one ring per keypoint.
        accepted = self._dedup(scored)
        accepted.sort(key=lambda item: item[0], reverse=True)

        return [(ex + x1, ey + y1)
                for _major, ((ex, ey), _axes, _angle) in accepted]

    def _hole_ink(self, ellipse, mask):
        """Ink fraction inside the fitted ellipse shrunk to `hole_scale`.

        This is the ring test. A ring is empty in the middle however broken its
        stroke is, so a fold cannot defeat it; a solid blob is all ink there.
        """
        (cx, cy), (axis_a, axis_b), angle = ellipse
        hole = np.zeros(mask.shape, dtype=np.uint8)
        cv2.ellipse(hole, ((cx, cy),
                           (axis_a * self.hole_scale, axis_b * self.hole_scale),
                           angle), 255, thickness=cv2.FILLED)
        hole_px = cv2.countNonZero(hole)
        if not hole_px:
            return 1.0
        return cv2.countNonZero(cv2.bitwise_and(hole, mask)) / hole_px

    def _dedup(self, scored):
        """A folded ring can survive as two arcs that fit near-identical
        ellipses. Keep the largest of each cluster."""
        kept = []
        for cand in sorted(scored, key=lambda item: item[0], reverse=True):
            major, ((cx, cy), _axes, _angle) = cand
            duplicate = False
            for kept_major, ((kx, ky), _ka, _kang) in kept:
                limit = self.dedup_center_dist * 0.5 * (major + kept_major)
                if np.hypot(cx - kx, cy - ky) <= limit:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

    def _ray_plane_intersect(self, u, v, fx, fy, cx, cy, R_fc, t_fc,
                             plane_point, plane_normal):
        """Back-project pixel to a camera ray, move it into filter_frame, and
        intersect with the fitted plane. Returns a 3D point or None."""
        d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        d_filter = R_fc @ d_cam
        d_filter = d_filter / (np.linalg.norm(d_filter) + 1e-12)
        origin = t_fc  # camera center in filter_frame

        denom = float(plane_normal @ d_filter)
        if abs(denom) < 1e-6:
            return None  # ray parallel to plane
        t = float(plane_normal @ (plane_point - origin)) / denom
        if t <= 0.0:
            return None  # intersection behind the camera
        return origin + t * d_filter

    def _publish(self, publisher, points_filter, R_of, t_of, stamp):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.output_frame
        if points_filter is not None and len(points_filter) > 0:
            for p in points_filter:
                p_out = R_of @ p + t_of
                pose = Pose()
                pose.position.x = float(p_out[0])
                pose.position.y = float(p_out[1])
                pose.position.z = float(p_out[2])
                pose.orientation.w = 1.0
                msg.poses.append(pose)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BannerTask()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
