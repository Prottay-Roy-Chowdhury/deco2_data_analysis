# select_polygon_extract_pcd.py

import os
import cv2
import numpy as np
import open3d as o3d


# ---------- USER PATHS ----------
IMAGE_PATH = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\\April Scan (alex)\\deco2_backup_20260527_154759\\Colmap\\images\\00190.png"

CORRESPONDENCE_DIR = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\\April Scan (alex)\\test00\\correspondence_00190"

OUTPUT_DIR = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\\April Scan (alex)\\test00\\selected_object_00190"

OUTPUT_SELECTED_PCD = os.path.join(OUTPUT_DIR, "selected_object_00190.pcd")
OUTPUT_SELECTED_IDS = os.path.join(OUTPUT_DIR, "selected_object_00190_ids.npy")
OUTPUT_MASK = os.path.join(OUTPUT_DIR, "selected_object_00190_mask.png")
OUTPUT_PREVIEW = os.path.join(OUTPUT_DIR, "selected_object_00190_preview.png")


# ---------- GLOBAL DRAWING STATE ----------
polygon_points = []
mouse_position = None
is_polygon_closed = False


def load_metadata(metadata_path):
    metadata = {}

    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key] = value

    return metadata


def mouse_callback(event, x, y, flags, param):
    global polygon_points, mouse_position, is_polygon_closed

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_position = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        if not is_polygon_closed:
            polygon_points.append((x, y))
            print(f"Added point: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(polygon_points) >= 3:
            is_polygon_closed = True
            print("Polygon closed")


def draw_interface(image):
    display = image.copy()

    # Draw polygon vertices and lines
    if len(polygon_points) > 0:
        for pt in polygon_points:
            cv2.circle(display, pt, 4, (0, 255, 255), -1)

        for i in range(len(polygon_points) - 1):
            cv2.line(
                display,
                polygon_points[i],
                polygon_points[i + 1],
                (0, 255, 255),
                2
            )

        # Draw temporary line to mouse
        if not is_polygon_closed and mouse_position is not None:
            cv2.line(
                display,
                polygon_points[-1],
                mouse_position,
                (0, 255, 255),
                1
            )

        # Draw closing line
        if is_polygon_closed and len(polygon_points) >= 3:
            cv2.line(
                display,
                polygon_points[-1],
                polygon_points[0],
                (0, 255, 255),
                2
            )

            overlay = display.copy()
            polygon_np = np.array(polygon_points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon_np], (0, 255, 255))
            display = cv2.addWeighted(overlay, 0.25, display, 0.75, 0)

    help_text = [
        "Left click: add polygon point",
        "Right click: close polygon",
        "s: save/extract selected object",
        "r: reset polygon",
        "esc/q: quit"
    ]

    y0 = 30
    for text in help_text:
        cv2.putText(
            display,
            text,
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y0 += 28

    return display


def extract_polygon_points(image, point_id, u, v, xyz, rgb):
    height, width = image.shape[:2]

    if len(polygon_points) < 3:
        raise RuntimeError("Need at least 3 polygon points")

    polygon_np = np.array(polygon_points, dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_np], 255)

    selected_mask = mask[v, u] > 0

    selected_point_ids = point_id[selected_mask]
    selected_xyz = xyz[selected_mask]
    selected_rgb = rgb[selected_mask]

    print(f"Selected points: {len(selected_point_ids)}")

    if len(selected_point_ids) == 0:
        print("No points found inside selected polygon")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save selected point cloud
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(selected_xyz.astype(np.float64))

    # Open3D expects colors in 0-1 range
    selected_rgb = selected_rgb.astype(np.float64)

    if selected_rgb.max() > 1.0:
        selected_rgb = selected_rgb / 255.0

    object_pcd.colors = o3d.utility.Vector3dVector(selected_rgb)

    o3d.io.write_point_cloud(OUTPUT_SELECTED_PCD, object_pcd)

    # Save selected point IDs
    np.save(OUTPUT_SELECTED_IDS, selected_point_ids.astype(np.int64))

    # Save mask
    cv2.imwrite(OUTPUT_MASK, mask)

    # Save preview
    preview = image.copy()
    overlay = preview.copy()
    cv2.fillPoly(overlay, [polygon_np], (0, 255, 255))
    preview = cv2.addWeighted(overlay, 0.3, preview, 0.7, 0)

    # Draw selected projected pixels in red
    selected_u = u[selected_mask]
    selected_v = v[selected_mask]
    preview[selected_v, selected_u] = [0, 0, 255]

    cv2.imwrite(OUTPUT_PREVIEW, preview)

    print(f"Saved selected PCD: {OUTPUT_SELECTED_PCD}")
    print(f"Saved selected IDs: {OUTPUT_SELECTED_IDS}")
    print(f"Saved mask: {OUTPUT_MASK}")
    print(f"Saved preview: {OUTPUT_PREVIEW}")


def main():
    global polygon_points, is_polygon_closed

    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    metadata_path = os.path.join(CORRESPONDENCE_DIR, "metadata.txt")
    metadata = load_metadata(metadata_path)

    print("Loaded metadata:")
    for k, val in metadata.items():
        print(f"  {k}: {val}")

    point_id = np.load(os.path.join(CORRESPONDENCE_DIR, "point_id.npy"))
    u = np.load(os.path.join(CORRESPONDENCE_DIR, "u.npy"))
    v = np.load(os.path.join(CORRESPONDENCE_DIR, "v.npy"))
    xyz = np.load(os.path.join(CORRESPONDENCE_DIR, "xyz.npy"))
    rgb = np.load(os.path.join(CORRESPONDENCE_DIR, "rgb.npy"))

    print(f"Loaded correspondences: {len(point_id)} points")
    print(f"u range: {u.min()} to {u.max()}")
    print(f"v range: {v.min()} to {v.max()}")

    if len(point_id) != len(u) or len(point_id) != len(v) or len(point_id) != len(xyz):
        raise RuntimeError("Correspondence arrays have inconsistent lengths")

    cv2.namedWindow("Polygon selector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Polygon selector", 1280, 1024)
    cv2.setMouseCallback("Polygon selector", mouse_callback)

    print("\nControls:")
    print("  Left click  : add polygon point")
    print("  Right click : close polygon")
    print("  s           : save/extract")
    print("  r           : reset")
    print("  esc/q       : quit\n")

    while True:
        display = draw_interface(image)
        cv2.imshow("Polygon selector", display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            if len(polygon_points) >= 3:
                is_polygon_closed = True
                extract_polygon_points(image, point_id, u, v, xyz, rgb)
            else:
                print("Need at least 3 polygon points before saving")

        elif key == ord("r"):
            polygon_points = []
            is_polygon_closed = False
            print("Polygon reset")

        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()