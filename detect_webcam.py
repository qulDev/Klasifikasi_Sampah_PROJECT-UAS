#!/usr/bin/env python3
"""
Real-time Webcam Detection Script for Klasifikasi Sampah

Run YOLO detection on webcam feed or static images/videos.

Usage:
    python detect_webcam.py --weights ./models/best.pt --source 0
    python detect_webcam.py --weights ./models/best.pt --source ./test_image.jpg
    python detect_webcam.py --weights ./models/best.pt --source ./test_video.mp4 --save-out ./output.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils.logger import setup_logger

logger = setup_logger(__name__)


def draw_detections(frame, results, conf_threshold=0.25):
    """
    Draw bounding boxes and labels on frame.

    Args:
        frame: Input frame (numpy array)
        results: YOLO detection results
        conf_threshold: Minimum confidence threshold

    Returns:
        Annotated frame
    """
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Filter by confidence
            if conf < conf_threshold:
                continue
            
            # Get class name
            class_name = result.names[cls]
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (int(x1), int(y1) - text_height - baseline - 5),
                (int(x1) + text_width, int(y1)),
                color,
                -1  # Filled
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - baseline - 5),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                thickness
            )
    
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Real-time waste detection using webcam or image/video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam detection (default camera)
  python detect_webcam.py --weights ./models/best.pt --source 0

  # Webcam detection (second camera)
  python detect_webcam.py --weights ./models/best.pt --source 1

  # Static image detection
  python detect_webcam.py --weights ./models/best.pt --source ./test.jpg

  # Video file detection
  python detect_webcam.py --weights ./models/best.pt --source ./video.mp4

  # Save output video
  python detect_webcam.py --weights ./models/best.pt --source 0 --save-out ./output.avi

  # Lower confidence threshold
  python detect_webcam.py --weights ./models/best.pt --source 0 --conf 0.15
        """
    )

    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained YOLO model weights (.pt file)')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: webcam index (0, 1, ...) or file path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, 0, 1, etc. (default: auto)')
    parser.add_argument('--save-out', type=str, default=None,
                        help='Path to save output video (optional)')
    parser.add_argument('--no-show', dest='show', action='store_false', default=True,
                        help='Disable display window (useful for headless servers)')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Klasifikasi Sampah - Real-time Detection")
    logger.info("=" * 70)

    # Load model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Model weights not found: {weights_path}")
        logger.error("Train a model first using train.py")
        return 1

    logger.info(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))
    logger.info(f"Model loaded successfully")

    # Determine source type
    try:
        source = int(args.source)  # Webcam index
        source_type = "webcam"
        logger.info(f"Using webcam: {source}")
    except ValueError:
        source = args.source  # File path
        source_path = Path(source)
        if not source_path.exists():
            logger.error(f"Source file not found: {source_path}")
            return 1
        
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            source_type = "image"
            logger.info(f"Using image: {source}")
        else:
            source_type = "video"
            logger.info(f"Using video: {source}")

    # Open video capture
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return 1

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if source_type != "webcam" else 30

    logger.info(f"Video properties: {width}x{height} @ {fps} FPS")

    # Setup video writer
    video_writer = None
    if args.save_out:
        output_path = Path(args.save_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        logger.info(f"Saving output to: {output_path}")

    logger.info("")
    logger.info("Controls:")
    logger.info("  'q' - Quit")
    logger.info("  's' - Save screenshot")
    logger.info("  'p' - Pause/Resume")
    logger.info("")
    logger.info("Starting detection...")
    logger.info("=" * 70)

    frame_count = 0
    start_time = time.time()
    paused = False
    screenshot_count = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    if source_type == "image":
                        logger.info("Image processed")
                    elif source_type == "video":
                        logger.info("Video ended")
                    else:
                        logger.error("Failed to read frame")
                    break

                # Run detection
                results = model(frame, conf=args.conf, device=args.device, verbose=False)

                # Draw detections
                annotated_frame = draw_detections(frame, results, args.conf)

                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

                # Draw FPS counter
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(
                    annotated_frame,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                # Save to video file
                if video_writer:
                    video_writer.write(annotated_frame)

                # Display frame
                if args.show:
                    cv2.imshow('Waste Detection', annotated_frame)
            
            # Handle keyboard input
            if args.show:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    screenshot_path = f'screenshot_{screenshot_count:04d}.jpg'
                    cv2.imwrite(screenshot_path, annotated_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
                    screenshot_count += 1
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"{'Paused' if paused else 'Resumed'}")
            else:
                # If not showing, just process all frames
                if source_type == "image":
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()

        # Final statistics
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        logger.info("")
        logger.info("=" * 70)
        logger.info("Detection Complete")
        logger.info("=" * 70)
        logger.info(f"Frames processed: {frame_count}")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info(f"Total time: {elapsed_time:.1f} seconds")
        logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
