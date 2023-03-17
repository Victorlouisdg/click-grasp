import cv2
import numpy as np
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.utils import ImageConverter
from airo_dataset_tools.pose import Pose
from airo_spatial_algebra import SE3Container


zed = Zed2i(resolution=Zed2i.RESOLUTION_720, fps=30)

window_name = "Camera feed"
cv2.namedWindow(window_name)

pose_saved = Pose.parse_file("camera_pose.json")
position = pose_saved.position_in_meters
euler_angles = pose_saved.rotation_euler_xyz_in_radians

position_array = np.array([position.x, position.y, position.z])
euler_angles_array = np.array([euler_angles.roll, euler_angles.pitch, euler_angles.yaw])

camera_in_base = SE3Container.from_euler_angles_and_translation(euler_angles_array, position_array).homogeneous_matrix

print(camera_in_base)

while(True):
    _, h, w = zed.get_rgb_image().shape
    image = zed.get_rgb_image()
    image = ImageConverter(image).image_in_opencv_format

    base_in_camera = np.linalg.inv(camera_in_base)
    rvec = base_in_camera[:3, :3]
    tvec = base_in_camera[:3, -1]
    image = cv2.drawFrameAxes(image, zed.intrinsics_matrix(), np.zeros(4), rvec, tvec, 0.5)


    cv2.imshow(window_name, image)  # refresh image
    key = cv2.waitKey(10)

    if key == ord("q"):
        cv2.destroyAllWindows()
        break