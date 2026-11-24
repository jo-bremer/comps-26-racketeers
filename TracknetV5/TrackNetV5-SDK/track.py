# MODIFIED BY JO BREMER

# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
import argparse
import os
import math # Make sure math is imported
import csv
from pathlib import Path
from tqdm import tqdm
from collections import deque
import time

# Import the builder and models from your project!
from models_factory.builder import build_model
from datasets_factory.transforms.tracknet_transforms import (
    Resize, ConcatChannels
)

# --- 1. "Model config library" ---
MODEL_CONFIGS = {
    'v2': dict(
        type='TrackNetV2',
        backbone=dict(type='TrackNetV2Backbone', in_channels=9),
        neck=dict(type='TrackNetV2Neck'),
        head=dict(type='TrackNetV2Head', in_channels=64, out_channels=3)
    ),
    'v4': dict(
        type='TrackNetV4',
        backbone=dict(type='TrackNetV4Backbone', in_channels=9),
        neck=dict(type='TrackNetV4Neck'),
        head=dict(type='TrackNetV2Head', in_channels=64, out_channels=3)
    ),
    'v5': dict(
        type='TrackNetV5',
        backbone=dict(type='TrackNetV2Backbone', in_channels=13),
        neck=dict(type='TrackNetV2Neck'),
        head=dict(type='R_STRHead', in_channels=64, out_channels=3)
    )
}

# --- 2. Helper functions ---
def _heatmap_to_coords(heatmap: np.ndarray, threshold: int = 127):
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)

    _, binary_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Create a mask that only covers the region inside the largest contour
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Find the maximum value within that region of the original heatmap
            # minMaxLoc returns (minVal, maxVal, minLoc, maxLoc)
            _, max_val, _, _ = cv2.minMaxLoc(heatmap, mask=mask)
            
            # Normalize 0-255 to 0-1 as conf
            conf = round(max_val / 255.0, 4)
            
            return cx, cy, conf

    return None

def draw_comet_tail(frame, points_deque, head_radius=8):
    """
    Dot-trajectory visualization based on radius decay (no connecting lines)
    :param frame: Current image frame (BGR)
    :param points_deque: Queue storing coordinates, supports (x, y) or (x, y, vis)
    :param head_radius: Maximum radius of the frontmost dot
    """
    # No need to create an all-black overlay, since we draw filled circles directly on the original image
    # If a semi-transparent effect is needed, the overlay logic can be kept; here we use the direct drawing you requested
    
    q_len = len(points_deque)
    if q_len == 0:
        return frame

    for i, pt in enumerate(points_deque):
        # 1. Safety check: skip empty points or NaN
        if pt is None:
            continue
            
        # Compatibility handling: supports (x, y) or (x, y, vis/conf)
        if len(pt) >= 3:
            tx, ty, tvis = pt[:3]
            if tvis == 0 or tvis is None: continue 
        else:
            tx, ty = pt
            
        if tx is None or ty is None:
            continue

        # 2. Compute radius decay (core logic)
        # i=0 (oldest) -> smallest scale; i=q_len-1 (newest) -> scale=1.0
        scale = (i + 1) / q_len
        current_radius = int(head_radius * scale)

        # Ensure the radius is at least 1
        current_radius = max(1, current_radius)

        # 3. Draw the filled ball
        # Use LINE_AA to enable anti-aliasing so the dot edges are smoother
        cv2.circle(
            frame, 
            (int(tx), int(ty)), 
            current_radius, 
            (0, 0, 255), # Pure red
            -1,          # Filled
            lineType=cv2.LINE_AA
        )

    return frame


# --- 3. "Core workshop": process_video ---
def process_video(video_path: Path, model, device, args, output_root_dir: Path) -> dict:
    """
    Process a single video file and generate all required output files.
    New logic: read 3 frames at a time, run inference on 3 frames, write 3 frames, then move ahead 3 frames.
    """
    print(f"\n🏭 Processing video: {video_path.name}")
    
    video_output_dir = output_root_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution_str = f"{width}x{height}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    input_size = (288, 512)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    trajectory_video_path = video_output_dir / f"{video_path.stem}_trajectory.mp4"
    comparison_video_path = video_output_dir / f"{video_path.stem}_comparison.mp4"
    csv_path = video_output_dir / f"{video_path.stem}_data.csv"
    
    writer_traj = cv2.VideoWriter(str(trajectory_video_path), fourcc, fps, (input_size[1], input_size[0]))
    writer_comp = cv2.VideoWriter(str(comparison_video_path), fourcc, fps, (input_size[1] * 2, input_size[0]))

    trajectory_points = deque(maxlen=fps) 
    
    csv_data = []
    detected_frames_count = 0
    
    # Preprocessing transforms (unchanged)
    resizer = Resize(keys=['path_prev', 'path', 'path_next'], size=input_size)
    concatenator = ConcatChannels(
        keys=['path_prev', 'path', 'path_next'],
        output_key='image'
    )
    
    # --- New loop logic ---
    frame_idx_counter = 0
    iteration_count = 0
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.stem}")
    start_time = time.time() # Record the start time

    while cap.isOpened():
        # 1. Read 3 frames at once
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        ret3, frame3 = cap.read()

        # If any frame fails to read (end of video), terminate the loop
        if not ret1 or not ret2 or not ret3:
            break

        # 2. Prepare model input
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame3_rgb = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        
        data_dict = {'path_prev': frame1_rgb, 'path': frame2_rgb, 'path_next': frame3_rgb}
        data_dict = resizer(data_dict)
        data_dict = concatenator(data_dict)
        
        resized_frames = [data_dict['path_prev'], data_dict['path'], data_dict['path_next']]
        
        image_np = data_dict['image']
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)

        # 3. Batch inference
        with torch.no_grad():
            # heatmap_preds has shape [1, 3, H, W]
            heatmap_preds = model(image_tensor)
        
        # Remove the batch dimension to get a NumPy array of shape (3, H, W)
        heatmaps_np = heatmap_preds.squeeze(0).cpu().numpy()
        
        threshold_uint8 = int(args.threshold * 255) # The threshold is still controlled by the argument
            
        # 4. Loop over the results for these 3 frames
        for i in range(3):
            
            current_frame_idx = frame_idx_counter + i
            # Make sure the last few frames, which may not fill 3, don't cause errors
            if current_frame_idx >= total_frames:
                continue

            single_heatmap_np = heatmaps_np[i] # Shape (H, W)
            heatmap_uint8 = (single_heatmap_np * 255).astype(np.uint8)

            # (A) Extract coordinates
            coords = _heatmap_to_coords(
                heatmap_uint8, 
                threshold=threshold_uint8
            )
            
            # (B) Record CSV and trajectory
            if coords is not None:
                detected_frames_count += 1
                
                trajectory_points.append(coords)
                csv_data.append({'frame_number': current_frame_idx, 'detected': 1, 'x': coords[0], 'y': coords[1], 'conf': coords[2]})
            else:
                trajectory_points.append(None)
                csv_data.append({'frame_number': current_frame_idx, 'detected': 0, 'x': 0.0, 'y': 0.0, 'conf': 0.0})
            
            # (C) Draw and write video
            frame_to_draw = cv2.cvtColor(resized_frames[i], cv2.COLOR_RGB2BGR)
            
            # Draw trajectory video
            final_traj_frame = draw_comet_tail(frame_to_draw, trajectory_points)
            writer_traj.write(final_traj_frame)

            # Draw comparison video
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            combined_frame = np.hstack((final_traj_frame, heatmap_color))
            writer_comp.write(combined_frame)
            

        # 5. Update counters and progress bar (important!)
        frame_idx_counter += 3
        iteration_count += 1
        pbar.update(3)
    
    # --- Cleanup after the loop ends ---

    end_time = time.time()
    total_duration = end_time - start_time
    # Average number of iterations processed per second (each iteration handles 3 frames)
    avg_it_per_sec = iteration_count * 3 / total_duration if total_duration > 0 else 0
    print(f"⏱️  Processed {iteration_count * 3} frames of {resolution_str} in {total_duration:.2f} seconds. Avg: {avg_it_per_sec:.2f} frames/sec.")
    pbar.close() # Close the progress bar

    detection_ratio = (detected_frames_count / total_frames) if total_frames > 0 else 0
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_number', 'detected', 'x', 'y', 'conf'])
        writer.writeheader()
        writer.writerows(csv_data)
        f.write("\n")
        f.write(f"total_detected_frame,{detected_frames_count}\n")
        f.write(f"detection_ratio,{detection_ratio:.4f}\n")

    cap.release()
    writer_traj.release()
    writer_comp.release()
    print(f"✅ Finished processing. Results saved in: {video_output_dir}")
    
    # ✨ New: return statistics
    stats = {
        'video_name': video_path.name,
        'detected_frames': detected_frames_count,
        'total_frames': total_frames,
        'detection_ratio': round(detection_ratio, 4)
    }
    return stats


# --- 4. "Control room": ✨ main (✨ modified) ✨ ---
def main():
    parser = argparse.ArgumentParser(description="TrackNet Batch Inference Pipeline")
    parser.add_argument('input_dir', type=str, help='Path to the directory containing input videos.')
    parser.add_argument('weights_path', type=str, help='Path to the model weights (.pth file).')
    
    # ✨ New architecture selection argument
    parser.add_argument(
        '--arch', 
        type=str, 
        required=True, 
        choices=['v2', 'v4', 'v5'], 
        help='Model architecture to use (v2, v4, or v5).'
    )
    
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    
    # ✨ The only tunable post-processing parameter ✨
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detection (0-1).')
    # parser.add_argument('--max-threshold', type=float, default=0.95, help='Confidence threshold for detection (0-1).')


    # ✨✨✨ Removed --min-circularity and --min-area ✨✨✨
    
    args = parser.parse_args()

    # ✨ Dynamically get the model config
    model_cfg = MODEL_CONFIGS.get(args.arch)
    if model_cfg is None:
        print(f"❌ Error: Unknown architecture '{args.arch}'. Please choose from 'v2', 'v4', 'v5'.")
        return
        
    print(f"🚀 Starting Batch Inference Pipeline for [TrackNet {args.arch.upper()}]...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
    model.to(device).eval()
    print(f"✅ Model loaded from {args.weights_path} and sent to {device}.")

    input_dir = Path(args.input_dir)
    
    # ✨ Dynamically set the output directory
    output_root_dir = input_dir / args.arch 
    output_root_dir.mkdir(exist_ok=True)
    
    print("🔎 Searching for .mp4 and .mov files...")
    video_files = []
    supported_formats = ['*.mp4', '*.mov', '*.MOV', '*.MP4']
    for fmt in supported_formats:
        video_files.extend(input_dir.glob(fmt))
    
    if not video_files:
        print(f"❌ No supported video files (.mp4, .mov) found in {input_dir}. Exiting.")
        return
        
    video_files = sorted(list(set(video_files)))
    print(f"Found {len(video_files)} videos to process.")
    
    # ✨ 1. Initialize the summary list
    summary_data_list = [] 
    
    for video_path in video_files:
        # ✨ 2. Collect the returned result of each video
        try:
            video_stats = process_video(video_path, model, device, args, output_root_dir)
            if video_stats:
                summary_data_list.append(video_stats)
        except Exception as e:
            print(f"❌ ERROR processing {video_path.name}: {e}")
            print("Skipping this video and continuing...")

    # ✨ 3. After the loop ends, write the global summary CSV
    if summary_data_list:
        summary_csv_path = output_root_dir / f"_summary_report_{args.arch}.csv"
        print(f"\n📊 Writing summary report to {summary_csv_path}")
        
        fieldnames = ['video_name', 'detected_frames', 'total_frames', 'detection_ratio']
        # Define the header labels
        header_map = {
            'video_name': 'Video Name',
            'detected_frames': 'Detected Ball Frames',
            'total_frames': 'Total Video Frames',
            'detection_ratio': 'Detection Ratio'
        }
        
        try:
            with open(summary_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                # Write the UTF-8 BOM and the header row
                writer = csv.writer(f)
                writer.writerow([header_map[field] for field in fieldnames])
                
                # Use DictWriter to write the data rows
                dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
                dict_writer.writerows(summary_data_list)
        except Exception as e:
            print(f"❌ ERROR writing summary CSV: {e}")
            
    print(f"\n🎉🎉🎉 All videos processed! Check the results in: {output_root_dir} 🎉🎉🎉")


if __name__ == '__main__':
    main()