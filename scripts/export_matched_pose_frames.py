"""
Export 2 matched-pose frames from UCO_PhyRehab silhouettes and run RTMW3D 3D pose.

Match criterion: same exercise (01 = seated left knee flexion), different subjects,
frames chosen so OptiTrack ground-truth knee angle matches within ~0.1 deg.

Inputs : sils_cropped/<subject>/<exercise>/cam0.mp4  (binary silhouette videos)
Outputs: matched_pose_export/
    subj0_frame133.png          -- raw silhouette frame
    subj1_frame678.png          -- raw silhouette frame
    subj0_frame133_kp3d.npy     -- (133,3) RTMW3D 3D keypoints
    subj1_frame678_kp3d.npy
    subj0_frame133_annotated.png-- silhouette + 2D keypoint overlay
    subj1_frame678_annotated.png
    keypoints.json              -- both subjects' keypoints + angles
    match_report.txt            -- ground-truth angle + RTMW3D angle summary
"""

import os
import sys
import json

# --- paths ---
ADAPT_REHAB = "/home/haipd/ADAPT-Rehab"
MMPOSE_REPO = "/home/haipd/mmpose"  # has projects/rtmpose3d/
os.environ["MMPOSE_PATH"] = MMPOSE_REPO
sys.path.insert(0, ADAPT_REHAB)
sys.path.insert(0, os.path.join(ADAPT_REHAB, "core"))

DATA_ROOT = "/home/haipd/ADAPT-Rehab/data/UCO_PhyRehab"
OUT_DIR = "/home/haipd/ADAPT-Rehab/data/UCO_PhyRehab/matched_pose_export"
GT_JSON = os.path.join(DATA_ROOT, "dataset_3d_with_angles.json")

import numpy as np
import cv2

# Frame selection (precomputed from GT, see match_two_subjects step)
SELECTED = [
    {"subject": "0", "exercise": "01", "cam": "cam0", "frame_idx": 133,
     "gt_angle_deg": 139.59, "side": "left", "body_part": "lower"},
    {"subject": "1", "exercise": "01", "cam": "cam0", "frame_idx": 678,
     "gt_angle_deg": 139.43, "side": "left", "body_part": "lower"},
]


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def save_raw(frame, path):
    cv2.imwrite(path, frame)


def overlay_keypoints(frame, kps2d, conf, path, threshold=0.3):
    """Draw 2D keypoints + basic skeleton on a copy of the frame."""
    out = frame.copy()
    # COCO-17 skeleton edges (subset of RTMW3D 133)
    edges = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),
             (5,6),(11,12),(0,5),(0,6)]
    for i, j in edges:
        if i < len(kps2d) and j < len(kps2d):
            if conf[i] > threshold and conf[j] > threshold:
                a, b = kps2d[i], kps2d[j]
                cv2.line(out, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])),
                         (0,255,0), 2)
    for k in range(min(17, len(kps2d))):
        if conf[k] > threshold:
            x, y = int(kps2d[k,0]), int(kps2d[k,1])
            cv2.circle(out, (x,y), 4, (0,0,255), -1)
    cv2.imwrite(path, out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Verify ground-truth match
    with open(GT_JSON) as f:
        gt = json.load(f)["data"]
    print("=" * 60)
    print("GROUND-TRUTH MATCH VERIFICATION")
    print("=" * 60)
    for sel in SELECTED:
        e = next(e for e in gt if e["folder"] == sel["subject"]
                 and e["exercise"] == sel["exercise"] and e["side"] == sel["side"])
        # frame_idx in video == frame id in JSON (offset verified)
        fr = next(fr for fr in e["frames"] if fr["id"] == sel["frame_idx"])
        angle = float(fr["joints"]["angle"])
        sel["gt_angle_verified"] = angle
        print(f"  subj {sel['subject']} ex{sel['exercise']} cam{sel['cam'][-1]} "
              f"frame {sel['frame_idx']}: GT knee angle = {angle:.2f} deg")
    diff = abs(SELECTED[0]["gt_angle_verified"] - SELECTED[1]["gt_angle_verified"])
    print(f"  angle difference between two subjects: {diff:.2f} deg")

    # 2. Read raw frames
    print("\nREADING SILHOUETTE FRAMES")
    frames = []
    for sel in SELECTED:
        vpath = f"{DATA_ROOT}/sils_cropped/{sel['subject']}/{sel['exercise']}/{sel['cam']}.mp4"
        fr = read_frame(vpath, sel["frame_idx"])
        sel["frame"] = fr
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        sel["fg_ratio"] = float((gray > 127).mean())
        print(f"  subj {sel['subject']}: shape={fr.shape}, "
              f"fg_ratio={sel['fg_ratio']:.3f}")
        tag = f"subj{sel['subject']}_frame{sel['frame_idx']}"
        sel["tag"] = tag
        save_raw(fr, os.path.join(OUT_DIR, f"{tag}.png"))
        frames.append(sel)

    # 3. Initialize RTMW3D
    print("\nINITIALIZING RTMW3D")
    from core.pose3d.rtmw3d import RTMW3DEstimator
    estimator = RTMW3DEstimator()
    ok = estimator.initialize()
    if not ok:
        print("[FAIL] RTMW3D initialization failed")
        sys.exit(1)

    # 4. Run inference
    print("\nRUNNING RTMW3D INFERENCE")
    results = {}
    for sel in frames:
        res = estimator.estimate(sel["frame"])
        sel["result"] = res
        if res.is_valid:
            kp3d = res.keypoints_3d
            kp2d = res.keypoints_2d
            conf = res.confidence
            print(f"  subj {sel['subject']}: 3D kpts {kp3d.shape}, "
                  f"mean conf={conf[:17].mean():.3f}")
            print(f"    RTMW3D left_knee angle = "
                  f"{res.joint_angles.get('left_knee', float('nan')):.2f} deg "
                  f"(GT = {sel['gt_angle_verified']:.2f})")
            # save npy
            np.save(os.path.join(OUT_DIR, f"{sel['tag']}_kp3d.npy"), kp3d)
            # save annotated
            overlay_keypoints(sel["frame"], kp2d, conf,
                              os.path.join(OUT_DIR, f"{sel['tag']}_annotated.png"))
            results[sel["subject"]] = {
                "frame_idx": sel["frame_idx"],
                "gt_knee_angle_deg": sel["gt_angle_verified"],
                "rtmw3d_left_knee_deg": res.joint_angles.get("left_knee"),
                "rtmw3d_right_knee_deg": res.joint_angles.get("right_knee"),
                "mean_confidence_body17": float(conf[:17].mean()),
                "keypoints_3d": kp3d.tolist(),
                "keypoints_2d": kp2d.tolist(),
                "confidence": conf.tolist(),
            }
        else:
            print(f"  subj {sel['subject']}: INVALID -- {res.error_message}")
            results[sel["subject"]] = {"error": res.error_message}

    # 5. Save JSON
    with open(os.path.join(OUT_DIR, "keypoints.json"), "w") as f:
        json.dump({
            "model": "RTMW3D-L (133 kpts)",
            "inputs": "UCO_PhyRehab sils_cropped (binary silhouettes)",
            "caveat": "RTMW3D trained on natural RGB; silhouette input will "
                      "degrade keypoint quality.",
            "subjects": results,
        }, f, indent=2)

    # 6. Match report
    with open(os.path.join(OUT_DIR, "match_report.txt"), "w") as f:
        f.write("UCO_PhyRehab matched-pose export\n")
        f.write("=" * 50 + "\n\n")
        f.write("Match criterion: same exercise (01 = seated left knee\n")
        f.write("flexion), different subjects, OptiTrack GT knee angle\n")
        f.write("matched within ~0.2 deg.\n\n")
        for sel in frames:
            f.write(f"Subject {sel['subject']}:\n")
            f.write(f"  video: sils_cropped/{sel['subject']}/{sel['exercise']}/{sel['cam']}.mp4\n")
            f.write(f"  frame: {sel['frame_idx']}\n")
            f.write(f"  GT knee angle: {sel['gt_angle_verified']:.2f} deg\n")
            if sel["result"].is_valid:
                f.write(f"  RTMW3D left_knee: {sel['result'].joint_angles.get('left_knee', float('nan')):.2f} deg\n")
                f.write(f"  mean body-17 conf: {sel['result'].confidence[:17].mean():.3f}\n")
            f.write("\n")

    print(f"\nDONE. Output in {OUT_DIR}")
    for fn in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, fn))
        print(f"  {fn:42s} {sz:>10} bytes")


if __name__ == "__main__":
    main()
