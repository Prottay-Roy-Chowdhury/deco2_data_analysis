# select_polygon_extract_pcd_multi.py

import os
import cv2
import numpy as np
import open3d as o3d


# ---------- USER PATHS ----------
IMAGE_PATH = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\April Scan (alex)\\deco2_backup_20260527_154759\\Colmap\\images\\00190.png"

CORRESPONDENCE_DIR = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\April Scan (alex)\\test00\\correspondence_00190"

OUTPUT_DIR = r"F:\\07. IAAC_Internship & Works_2025-2026\\15. Deco2_scanner\April Scan (alex)\\test00\\selected_object_00190_multi"

OUTPUT_SELECTED_PCD = os.path.join(OUTPUT_DIR, "selected_object_merged.pcd")
OUTPUT_SELECTED_IDS = os.path.join(OUTPUT_DIR, "selected_object_merged_ids.npy")
OUTPUT_MASK = os.path.join(OUTPUT_DIR, "selected_object_merged_mask.png")
OUTPUT_PREVIEW = os.path.join(OUTPUT_DIR, "selected_object_merged_preview.png")


# ---------- GLOBAL STATE ----------
finished_polygons = []
current_polygon = []
mouse_position = None


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
    global current_polygon, mouse_position, finished_polygons

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_position = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        print(f"Added point: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) >= 3:
            finished_polygons.append(current_polygon.copy())
            print(
                f"Closed polygon {len(finished_polygons)} "
                f"with {len(current_polygon)} points"
            )
            current_polygon = []
        else:
            print("Need at least 3 points to close polygon")


def draw_interface(image):
    display = image.copy()

    # Draw finished polygons
    for i, poly in enumerate(finished_polygons):
        pts = np.array(poly, dtype=np.int32)

        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        display = cv2.addWeighted(overlay, 0.25, display, 0.75, 0)

        cv2.polylines(
            display,
            [pts],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2
        )

        for pt in poly:
            cv2.circle(display, pt, 4, (0, 255, 255), -1)

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(
            display,
            str(i + 1),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Draw current polygon
    if len(current_polygon) > 0:
        for i in range(len(current_polygon) - 1):
            cv2.line(
                display,
                current_polygon[i],
                current_polygon[i + 1],
                (0, 0, 255),
                2
            )

        if mouse_position is not None:
            cv2.line(
                display,
                current_polygon[-1],
                mouse_position,
                (0, 0, 255),
                1
            )

        for pt in current_polygon:
            cv2.circle(display, pt, 4, (0, 0, 255), -1)

    help_text = [
        "Left click: add point to current polygon",
        "Right click: close current polygon",
        "Backspace: undo last current point",
        "r: reset current polygon",
        "c: clear all polygons",
        "s: save/extract all polygons merged",
        "p: save/extract each polygon separately",
        "q / Esc: quit",
        f"Finished polygons: {len(finished_polygons)} | Current points: {len(current_polygon)}"
    ]

    y0 = 30
    for text in help_text:
        cv2.putText(
            display,
            text,
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y0 += 26

    return display


def build_mask(image_shape, polygons):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in polygons:
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    return mask


def save_selected_pcd(output_pcd, output_ids, selected_point_ids, selected_xyz, selected_rgb):
    if len(selected_point_ids) == 0:
        print("No points selected")
        return

    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(selected_xyz.astype(np.float64))

    selected_rgb = selected_rgb.astype(np.float64)
    if selected_rgb.max() > 1.0:
        selected_rgb = selected_rgb / 255.0

    object_pcd.colors = o3d.utility.Vector3dVector(selected_rgb)

    o3d.io.write_point_cloud(output_pcd, object_pcd)
    np.save(output_ids, selected_point_ids.astype(np.int64))

    print(f"Saved PCD: {output_pcd}")
    print(f"Saved IDs: {output_ids}")


def extract_all_polygons_merged(image, point_id, u, v, xyz, rgb):
    polygons = finished_polygons.copy()

    if len(current_polygon) >= 3:
        polygons.append(current_polygon.copy())

    if len(polygons) == 0:
        print("No polygons to extract")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mask = build_mask(image.shape, polygons)
    selected_mask = mask[v, u] > 0

    selected_point_ids = point_id[selected_mask]
    selected_xyz = xyz[selected_mask]
    selected_rgb = rgb[selected_mask]

    print(f"Selected merged points: {len(selected_point_ids)}")

    save_selected_pcd(
        OUTPUT_SELECTED_PCD,
        OUTPUT_SELECTED_IDS,
        selected_point_ids,
        selected_xyz,
        selected_rgb
    )

    cv2.imwrite(OUTPUT_MASK, mask)

    preview = image.copy()
    overlay = preview.copy()

    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.polylines(preview, [pts], True, (0, 255, 255), 2)

    preview = cv2.addWeighted(overlay, 0.3, preview, 0.7, 0)

    selected_u = u[selected_mask]
    selected_v = v[selected_mask]
    preview[selected_v, selected_u] = [0, 0, 255]

    cv2.imwrite(OUTPUT_PREVIEW, preview)

    print(f"Saved mask: {OUTPUT_MASK}")
    print(f"Saved preview: {OUTPUT_PREVIEW}")


def extract_each_polygon_separately(image, point_id, u, v, xyz, rgb):
    polygons = finished_polygons.copy()

    if len(current_polygon) >= 3:
        polygons.append(current_polygon.copy())

    if len(polygons) == 0:
        print("No polygons to extract")
        return

    separate_dir = os.path.join(OUTPUT_DIR, "separate_polygons")
    os.makedirs(separate_dir, exist_ok=True)

    for i, poly in enumerate(polygons):
        mask = build_mask(image.shape, [poly])
        selected_mask = mask[v, u] > 0

        selected_point_ids = point_id[selected_mask]
        selected_xyz = xyz[selected_mask]
        selected_rgb = rgb[selected_mask]

        print(f"Polygon {i + 1}: {len(selected_point_ids)} points")

        output_pcd = os.path.join(separate_dir, f"polygon_{i + 1:03d}.pcd")
        output_ids = os.path.join(separate_dir, f"polygon_{i + 1:03d}_ids.npy")
        output_mask = os.path.join(separate_dir, f"polygon_{i + 1:03d}_mask.png")
        output_preview = os.path.join(separate_dir, f"polygon_{i + 1:03d}_preview.png")

        save_selected_pcd(
            output_pcd,
            output_ids,
            selected_point_ids,
            selected_xyz,
            selected_rgb
        )

        cv2.imwrite(output_mask, mask)

        preview = image.copy()
        overlay = preview.copy()
        pts = np.array(poly, dtype=np.int32)

        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        preview = cv2.addWeighted(overlay, 0.3, preview, 0.7, 0)

        selected_u = u[selected_mask]
        selected_v = v[selected_mask]
        preview[selected_v, selected_u] = [0, 0, 255]

        cv2.imwrite(output_preview, preview)


def main():
    global current_polygon, finished_polygons

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

    cv2.namedWindow("Multi polygon selector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi polygon selector", 1280, 1024)
    cv2.setMouseCallback("Multi polygon selector", mouse_callback)

    while True:
        display = draw_interface(image)
        cv2.imshow("Multi polygon selector", display)

        key = cv2.waitKey(500) & 0xFF

        if key == ord("s"):
            extract_all_polygons_merged(image, point_id, u, v, xyz, rgb)

        elif key == ord("p"):
            extract_each_polygon_separately(image, point_id, u, v, xyz, rgb)

        elif key == ord("r"):
            current_polygon = []
            print("Current polygon reset")

        elif key == ord("c"):
            current_polygon = []
            finished_polygons = []
            print("All polygons cleared")

        elif key == 8:
            if len(current_polygon) > 0:
                removed = current_polygon.pop()
                print(f"Removed last point: {removed}")

        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()