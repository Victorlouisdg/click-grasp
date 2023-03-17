import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.reprojection import reproject_to_frame_z_plane
from airo_camera_toolkit.utils import ImageConverter
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_dataset_tools.pose import Pose
from airo_spatial_algebra import SE3Container


def draw_clicked_grasp(image, clicked_image_points, current_mouse_point):
    """If we don't have tow clicks yet, draw a line between the first point and the current cursor position."""
    for point in clicked_image_points:
        image = cv2.circle(image, point, 5, (0, 255, 0), thickness=2)

    if len(clicked_image_points) >= 1:
        first_point = clicked_image_points[0]
        second_point = current_mouse_point[0]

        if len(clicked_image_points) >= 2:
            second_point = clicked_image_points[1]

        image = cv2.line(image, first_point, second_point, color=(0, 255, 0), thickness=1)
        middle = (np.array(first_point) + np.array(second_point)) // 2
        image = cv2.circle(image, middle.T, 2, (0, 255, 0), thickness=2)

    return image


def draw_pose(image, pose_in_base, camera_in_base, intrinsics):
    pose_in_camera = np.linalg.inv(camera_in_base) @ pose_in_base
    rvec = pose_in_camera[:3, :3]
    tvec = pose_in_camera[:3, -1]
    image = cv2.drawFrameAxes(image, intrinsics, np.zeros(4), rvec, tvec, 0.1)
    return image



def make_grasp_pose(clicked_points):
    grasp_location = (clicked_points[1] + clicked_points[0]) / 2

    # Build the orientation matrix so that the gripper opens along the line between the clicked points.
    gripper_open_direction = clicked_points[1] - clicked_points[0]
    X = gripper_open_direction / np.linalg.norm(gripper_open_direction)
    Z = np.array([0, 0, -1])  # topdown
    Y = np.cross(Z, X)
    grasp_orientation = np.column_stack([X, Y, Z])

    # Assemble the 4x4 pose matrix
    grasp_pose = np.identity(4)
    grasp_pose[:3, -1] = grasp_location
    grasp_pose[:3, :3] = grasp_orientation
    return grasp_pose


if __name__ == "__main__":  # noqa: C901
    if not os.path.exists(Path(__file__).parent / "camera_pose.json"):
        print("Please run camera_calibration.py first.")
        sys.exit(0)

    pose_saved = Pose.parse_file("camera_pose.json")
    position = pose_saved.position_in_meters
    euler_angles = pose_saved.rotation_euler_xyz_in_radians

    position_array = np.array([position.x, position.y, position.z])
    euler_angles_array = np.array([euler_angles.roll, euler_angles.pitch, euler_angles.yaw])

    camera_in_base = SE3Container.from_euler_angles_and_translation(euler_angles_array, position_array).homogeneous_matrix
    
    ip_victor = "10.42.0.162"
    ur3e = URrtde(ip_victor, URrtde.UR3E_CONFIG)

    gripper = Robotiq2F85(ip_victor)

    home_joints = np.deg2rad([0, -60, -90, -120, 90, 90])
    ur3e.move_to_joint_configuration(home_joints).wait()
    gripper.open().wait()

    zed = Zed2i(resolution=Zed2i.RESOLUTION_720, fps=30)
    intrinsics_matrix = zed.intrinsics_matrix()

    current_mouse_point = [(0, 0)]  # has to be a list so that the callback can edit it
    clicked_image_points = []

    def mouse_callback(event, x, y, flags, parm):
        if len(clicked_image_points) >= 2:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_image_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            current_mouse_point[0] = x, y

    window_name = "Camera feed"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    grasp_executed = False

    while True:
        _, h, w = zed.get_rgb_image().shape
        image = zed.get_rgb_image()
        image = ImageConverter(image).image_in_opencv_format
        image = draw_clicked_grasp(image, clicked_image_points, current_mouse_point)

        image = draw_pose(image, np.identity(4), camera_in_base, intrinsics_matrix)

        if len(clicked_image_points) == 2:
            points_in_image = np.array(clicked_image_points)
            points_in_world = reproject_to_frame_z_plane(
                points_in_image, intrinsics_matrix, camera_in_base
            )

            grasp_pose = make_grasp_pose(points_in_world)
            grasp_pose[2, -1] -= 0.005 # grasp at height of base

            pregrasp_pose = np.copy(grasp_pose)
            pregrasp_pose[2, -1] += 0.12

            drop_pose = np.copy(pregrasp_pose)
            drop_pose[:3, -1] = np.array([0.3, 0.25, 0.1])

            image = draw_pose(image, grasp_pose, camera_in_base, intrinsics_matrix)

            if not grasp_executed:
                cv2.imshow(window_name, image)  # refresh image

                ur3e.move_linear_to_tcp_pose(pregrasp_pose).wait()
                ur3e.move_linear_to_tcp_pose(grasp_pose).wait()
                gripper.close().wait()
                ur3e.move_linear_to_tcp_pose(pregrasp_pose).wait()
                ur3e.move_linear_to_tcp_pose(drop_pose).wait()
                gripper.open().wait()
                ur3e.move_to_joint_configuration(home_joints)

                clicked_image_points = []

        cv2.imshow(window_name, image)
        key = cv2.waitKey(10)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break
