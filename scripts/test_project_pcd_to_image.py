# test_project_pcd_to_image.py

import os
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# ---------- USER PATHS ----------
PCD_PATH = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\\April Scan (alex)\\deco2_backup_20260527_154759\\PCD\\all_raw_points.pcd"
IMAGE_PATH = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\\April Scan (alex)\\deco2_backup_20260527_154759\\Colmap\\images\\00190.png"
CAMERAS_TXT = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\\April Scan (alex)\\deco2_backup_20260527_154759\\Colmap\\sparse\\0\\cameras.txt"
IMAGES_TXT = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\\April Scan (alex)\\deco2_backup_20260527_154759\\Colmap\\sparse\\0\\images.txt"

OUTPUT_DIR = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\\April Scan (alex)\\test00"

OUTPUT_OVERLAY = os.path.join(OUTPUT_DIR, "projection_overlay_00190.png")
OUTPUT_VISIBLE_PCD = os.path.join(OUTPUT_DIR, "visible_points_00190.pcd")
OUTPUT_CORRESPONDENCE_DIR = os.path.join(OUTPUT_DIR, "correspondence_00190")


# ---------- LOAD CAMERA ----------
def read_camera(cameras_txt):
    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue

            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])

            fx, fy, cx, cy = map(float, parts[4:8])

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)

            return cam_id, model, width, height, K

    raise RuntimeError("No camera found")


# ---------- LOAD IMAGE POSE ----------
def read_image_pose(images_txt, target_name):
    """
    COLMAP images.txt convention:
    X_cam = R * X_world + t
    quaternion is qw qx qy qz
    """
    with open(images_txt, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("#") or line == "":
            i += 1
            continue

        parts = line.split()

        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]

        if name == target_name:
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz], dtype=np.float64)
            return image_id, cam_id, rot, t

        i += 2

    raise RuntimeError(f"Image {target_name} not found")


# ---------- MAIN ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_name = Path(IMAGE_PATH).name

    cam_id, model, width, height, K = read_camera(CAMERAS_TXT)
    image_id, pose_cam_id, R_wc_to_cam, t_wc_to_cam = read_image_pose(
        IMAGES_TXT,
        image_name
    )

    print("Camera K:\n", K)
    print("Image:", image_name)
    print("R world→cam:\n", R_wc_to_cam)
    print("t world→cam:", t_wc_to_cam)

    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points_world = np.asarray(pcd.points)

    if len(points_world) == 0:
        raise RuntimeError("PCD has no points")

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points_world) * 0.8

    point_ids = np.arange(len(points_world), dtype=np.int64)

    points_cam = (R_wc_to_cam @ points_world.T).T + t_wc_to_cam

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    valid = z > 0.1

    x = x[valid]
    y = y[valid]
    z = z[valid]
    valid_ids = point_ids[valid]
    valid_points_world = points_world[valid]
    valid_colors = colors[valid]

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]

    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    inside = (
        (u_int >= 0) & (u_int < width) &
        (v_int >= 0) & (v_int < height)
    )

    u_int = u_int[inside]
    v_int = v_int[inside]
    z_inside = z[inside]
    visible_ids = valid_ids[inside]
    visible_points_world = valid_points_world[inside]
    visible_colors = valid_colors[inside]

    print(f"Total PCD points: {len(points_world)}")
    print(f"Points in front of camera: {len(valid_ids)}")
    print(f"Projected inside image: {len(visible_ids)}")

    pixel_key = v_int.astype(np.int64) * width + u_int.astype(np.int64)
    order = np.argsort(z_inside)

    seen = set()
    keep_indices = []

    for idx in order:
        key = int(pixel_key[idx])
        if key not in seen:
            seen.add(key)
            keep_indices.append(idx)

    keep_indices = np.array(keep_indices, dtype=np.int64)

    u_keep = u_int[keep_indices]
    v_keep = v_int[keep_indices]
    depth_keep = z_inside[keep_indices].astype(np.float32)

    selected_ids = visible_ids[keep_indices]
    selected_points = visible_points_world[keep_indices]
    selected_colors = visible_colors[keep_indices]

    print(f"After z-buffer visible points: {len(selected_ids)}")

    overlay = img.copy()

    overlay[v_keep, u_keep] = [0, 0, 255]

    cv2.imwrite(OUTPUT_OVERLAY, overlay)
    print(f"Saved overlay: {OUTPUT_OVERLAY}")

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(selected_points)
    out_pcd.colors = o3d.utility.Vector3dVector(selected_colors)

    o3d.io.write_point_cloud(OUTPUT_VISIBLE_PCD, out_pcd)
    print(f"Saved visible point subset: {OUTPUT_VISIBLE_PCD}")

    os.makedirs(OUTPUT_CORRESPONDENCE_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "point_id.npy"), selected_ids.astype(np.int64))
    np.save(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "u.npy"), u_keep.astype(np.int32))
    np.save(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "v.npy"), v_keep.astype(np.int32))
    np.save(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "depth.npy"), depth_keep.astype(np.float32))
    np.save(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "xyz.npy"), selected_points.astype(np.float32))
    np.save(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "rgb.npy"), selected_colors.astype(np.float32))

    np.save(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "K.npy"), K.astype(np.float64))
    np.save(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "R_world_to_cam.npy"), R_wc_to_cam.astype(np.float64))
    np.save(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "t_world_to_cam.npy"), t_wc_to_cam.astype(np.float64))

    with open(os.path.join(OUTPUT_CORRESPONDENCE_DIR, "metadata.txt"), "w") as f:
        f.write(f"image_name={image_name}\n")
        f.write(f"image_id={image_id}\n")
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")
        f.write(f"num_points={len(selected_ids)}\n")

    print(f"Saved correspondence folder: {OUTPUT_CORRESPONDENCE_DIR}")


if __name__ == "__main__":
    main()