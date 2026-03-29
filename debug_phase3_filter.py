import argparse
import os
import cv2
import numpy as np

from phase3_kinematics_processor import KinematicPreprocessor

# Human3.6M-style 17-joint connectivity matching phase3_kinematics_processor.py
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize detect_hallucinations() on original video frames."
    )
    parser.add_argument("--video", required=True, help="Path to input mp4 video")
    parser.add_argument("--npy", required=True, help="Path to input 3D pose npy file")
    parser.add_argument("--output", required=True, help="Path to output avi file")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview with cv2.imshow and allow pressing 'q' to quit early",
    )
    return parser.parse_args()



def compute_global_xy_projection(pose_3d, frame_width, frame_height, padding=40):
    """
    Project metric XY coordinates to image pixels using one global affine scale
    for the whole clip, so the skeleton does not jitter from frame to frame.
    """
    xy = pose_3d[:, :, :2].reshape(-1, 2)
    valid = np.isfinite(xy).all(axis=1)

    if not np.any(valid):
        scale = 1.0
        min_xy = np.array([0.0, 0.0], dtype=np.float64)
        max_xy = np.array([1.0, 1.0], dtype=np.float64)
    else:
        xy_valid = xy[valid]
        min_xy = xy_valid.min(axis=0)
        max_xy = xy_valid.max(axis=0)

    span = np.maximum(max_xy - min_xy, 1e-6)
    draw_w = max(frame_width - 2 * padding, 1)
    draw_h = max(frame_height - 2 * padding, 1)
    scale = min(draw_w / span[0], draw_h / span[1])

    return min_xy, scale, padding



def project_points_xy(points_3d, min_xy, scale, frame_height, padding):
    """Project a single frame of 3D joints to 2D pixel coordinates using X/Y only."""
    xy = points_3d[:, :2].astype(np.float64)
    pts_2d = np.full((xy.shape[0], 2), -1, dtype=np.int32)

    valid = np.isfinite(xy).all(axis=1)
    if not np.any(valid):
        return pts_2d, valid

    xy_norm = (xy[valid] - min_xy) * scale
    x_pix = np.round(xy_norm[:, 0] + padding)
    y_pix = np.round(frame_height - (xy_norm[:, 1] + padding))

    pts_2d[valid, 0] = x_pix.astype(np.int32)
    pts_2d[valid, 1] = y_pix.astype(np.int32)
    return pts_2d, valid



def draw_skeleton(frame, points_2d, valid_mask, color):
    for a, b in SKELETON_EDGES:
        if valid_mask[a] and valid_mask[b]:
            pt1 = tuple(points_2d[a])
            pt2 = tuple(points_2d[b])
            cv2.line(frame, pt1, pt2, color, 2, lineType=cv2.LINE_AA)

    for idx, pt in enumerate(points_2d):
        if valid_mask[idx]:
            cv2.circle(frame, tuple(pt), 4, color, -1, lineType=cv2.LINE_AA)



def put_status_text(frame, is_hallucination, frame_idx, total_frames):
    if is_hallucination:
        status = "STATUS: HALLUCINATION (DROPPED)"
        color = (0, 0, 255)
    else:
        status = "STATUS: HUMAN (KEPT)"
        color = (0, 255, 0)

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Frame: {frame_idx + 1}/{total_frames}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )



def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.isfile(args.npy):
        raise FileNotFoundError(f"NPY not found: {args.npy}")

    pose_3d = np.load(args.npy)
    if pose_3d.ndim != 3 or pose_3d.shape[1:] != (17, 3):
        raise ValueError(f"Expected pose shape (N, 17, 3), got {pose_3d.shape}")

    processor = KinematicPreprocessor()
    hallucination_mask = processor.detect_hallucinations(pose_3d)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {args.output}")

    min_xy, scale, padding = compute_global_xy_projection(
        pose_3d, frame_width=frame_width, frame_height=frame_height
    )

    total_pose_frames = pose_3d.shape[0]
    total_frames = min(video_frame_count if video_frame_count > 0 else total_pose_frames, total_pose_frames)

    print(f"[INFO] Video frames available: {video_frame_count}")
    print(f"[INFO] Pose frames available : {total_pose_frames}")
    print(f"[INFO] Frames to render      : {total_frames}")
    print(f"[INFO] Hallucinations        : {int(hallucination_mask[:total_frames].sum())}/{total_frames}")
    print(f"[INFO] Output                : {args.output}")

    frame_idx = 0
    try:
        while frame_idx < total_frames:
            ok, frame = cap.read()
            if not ok:
                break

            joints_3d = pose_3d[frame_idx]
            is_hallucination = bool(hallucination_mask[frame_idx])
            color = (0, 0, 255) if is_hallucination else (0, 255, 0)

            points_2d, valid_mask = project_points_xy(
                joints_3d,
                min_xy=min_xy,
                scale=scale,
                frame_height=frame_height,
                padding=padding,
            )
            draw_skeleton(frame, points_2d, valid_mask, color)
            put_status_text(frame, is_hallucination, frame_idx, total_frames)

            writer.write(frame)

            if args.show:
                cv2.imshow("debug_phase3_filter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Interrupted by user via 'q'.")
                    break

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"[DONE] Rendered {frame_idx} frames to: {args.output}")


if __name__ == "__main__":
    main()
