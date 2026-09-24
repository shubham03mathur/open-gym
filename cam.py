import cv2
import mediapipe as mp
import time
import os

from typing import Any
from Ema import EMA
from calculate_angle import calculate
from state_machines.biceps_curls import BicepsCurls

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_pose_landmarks = None
HOLD_DURATION_SECONDS = 1.5
MIN_VISIBILITY = 0.45


def print_result(result: PoseLandmarkerResult, image: mp.Image, frame_timestamp_ms: int):
    global latest_pose_landmarks
    if result.pose_landmarks:
        latest_pose_landmarks = result.pose_landmarks
        #print(latest_pose_landmarks)
    else:
        latest_pose_landmarks = None


def display_x(normalized_x, width):
    """Convert an original-frame x coordinate to the mirrored display frame."""
    return int((1.0 - normalized_x) * width)

mlmodel = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mlmodel, delegate=BaseOptions.Delegate.GPU),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    start_time = time.perf_counter()
    snapshot = { 'left': [], 'right': []}
    progressive_snaphost = { 'left': [], 'right': [] }
    if not cap.isOpened():
        print('Cannot interact with your webcam.')
        exit()

    print("Press 'q' to exit video stream.")
    l_ema = None
    r_ema = None
    left_stationary_start = None
    right_stationary_start = None

    biceps_curl_sm_l = None
    biceps_curl_sm_r = None
    right_rep_count = 0
    left_rep_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame. Exiting...")
            break

        detection_frame = frame.copy()
        frame = cv2.flip(detection_frame, 1)

        elapsed_time = time.perf_counter() - start_time
        frame_timestamp_ms = int(elapsed_time * 1000)

        # Detect on the original frame; only the displayed preview is mirrored.
        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGBA)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgb_frame)
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        h, w, _ = frame.shape

        if latest_pose_landmarks is not None:
            # MediaPipe guarantees 33 landmarks per tracked body model
            # index 11 = Left Shoulder, index 13 = Left Elbow, index 15 = Left Wrist
            
            for pose in latest_pose_landmarks:
                h, w, _ = frame.shape

                # --- CALCULATE LEFT ARM ANGLE ---
                try:
                    # MediaPipe anatomical left arm landmarks.
                    l_shoulder = pose[11]
                    l_elbow = pose[13]
                    l_wrist = pose[15]

                    shoulder_visibility = pose[11].visibility
                    elbow_visibility = pose[13].visibility
                    wrist_visibility = pose[15].visibility
                    shoulder_presence = pose[11].presence
                    elbow_presence = pose[13].presence
                    wrist_presence = pose[15].presence

                    min_visibility = min(shoulder_visibility, elbow_visibility, wrist_visibility)
                    if min_visibility >= MIN_VISIBILITY:
                        # Standardize structural coordinate vectors
                        left_angle = calculate(
                            [l_shoulder.x, l_shoulder.y, l_shoulder.z],
                            [l_elbow.x, l_elbow.y, l_elbow.z],
                            [l_wrist.x, l_wrist.y, l_wrist.z]
                        )

                        if l_ema is None:
                            l_ema = EMA(left_angle, elapsed_time=elapsed_time)
                        else:
                            l_ema.update(left_angle, elapsed_time=elapsed_time)
                        ema_angle = l_ema.get_recent_ema()
                        delta = l_ema.get_delta()
                        if biceps_curl_sm_l is None:
                            biceps_curl_sm_l = BicepsCurls()
                        if delta is not None:
                            biceps_curl_sm_l.update(ema_angle, delta[0], delta[1])
                            left_rep_count = biceps_curl_sm_l.rep_count

                        if delta is not None:
                            delta_angle, rate = delta
                            is_stationary = (
                                abs(delta_angle) < 0.5
                                and abs(rate) < 15
                            )
                            if is_stationary:
                                if left_stationary_start is None:
                                    left_stationary_start = elapsed_time

                                hold_duration = elapsed_time - left_stationary_start
                                is_holding = hold_duration >= HOLD_DURATION_SECONDS
                            else:
                                left_stationary_start = None
                                hold_duration = 0.0
                                is_holding = False

                            progressive_snaphost["left"].append({
                                "time": elapsed_time,
                                "angle": ema_angle,
                                "delta": delta_angle,
                                "rate": rate,
                                "is_holding": is_holding,
                                "hold_duration": hold_duration,
                                "visibility": min_visibility,
                                "left_rep_count": left_rep_count,
                                "state": biceps_curl_sm_l.current_state,
                            })
                        l_measurement = {
                            "angle": int(left_angle),
                            "EMA": int(ema_angle),
                            "shoulder_visibility": shoulder_visibility,
                            "elbow_visibility": elbow_visibility,
                            "wrist_visibility": wrist_visibility,
                            "shoulder_presence": shoulder_presence,
                            "elbow_presence": elbow_presence,
                            "wrist_presence": wrist_presence
                        }
                        if (elapsed_time <= 5.0):
                            if 'left' in snapshot:
                                if(isinstance(snapshot['left'], list)):
                                    snapshot['left'].append(l_measurement)
                            else:
                                snapshot['left'] = [l_measurement]
                        # Draw text overlay next to left elbow joint area
                        cv2.putText(
                            frame, f"L: {int(left_angle)} deg EMA: {int(ema_angle)}",
                            (display_x(l_elbow.x, w) + 15, int(l_elbow.y * h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                        )
                except IndexError:
                    pass

                # --- CALCULATE RIGHT ARM ANGLE ---
                try:
                    # MediaPipe anatomical right arm landmarks.
                    r_shoulder = pose[12]
                    r_elbow = pose[14]
                    r_wrist = pose[16]

                    shoulder_visibility = pose[12].visibility
                    elbow_visibility = pose[14].visibility
                    wrist_visibility = pose[16].visibility
                    shoulder_presence = pose[12].presence
                    elbow_presence = pose[14].presence
                    wrist_presence = pose[16].presence
                    min_visibility = min(shoulder_visibility, elbow_visibility, wrist_visibility)
                    if min_visibility >= MIN_VISIBILITY:
                        # Standardize structural coordinate vectors 
                        right_angle = calculate(
                            [r_shoulder.x, r_shoulder.y, r_shoulder.z],
                            [r_elbow.x, r_elbow.y, r_elbow.z],
                            [r_wrist.x, r_wrist.y, r_wrist.z]
                        )

                        if r_ema is None:
                            r_ema = EMA(right_angle, elapsed_time=elapsed_time)
                        else:
                            r_ema.update(right_angle, elapsed_time=elapsed_time)

                        ema_angle = r_ema.get_recent_ema()
                        delta = r_ema.get_delta()
                        if biceps_curl_sm_r is None:
                            biceps_curl_sm_r = BicepsCurls()
                        if delta is not None:
                            biceps_curl_sm_r.update(ema_angle, delta[0], delta[1])
                            right_rep_count = biceps_curl_sm_r.rep_count
                        if delta is not None:
                            delta_angle, rate = delta
                            is_stationary = (
                                abs(delta_angle) < 0.5
                                and abs(rate) < 15
                            )
                            if is_stationary:
                                if right_stationary_start is None:
                                    right_stationary_start = elapsed_time

                                hold_duration = elapsed_time - right_stationary_start
                                is_holding = hold_duration >= HOLD_DURATION_SECONDS
                            else:
                                right_stationary_start = None
                                hold_duration = 0.0
                                is_holding = False
                            progressive_snaphost["right"].append({
                                "time": elapsed_time,
                                "angle": ema_angle,
                                "delta": delta_angle,
                                "rate": rate,
                                "is_holding": is_holding,
                                "hold_duration": hold_duration,
                                "visibility": min_visibility,
                                "right_rep_count": right_rep_count,
                                "state": biceps_curl_sm_r.current_state,
                            })
                        r_measurement = {
                            "angle": int(right_angle),
                            "EMA": int(ema_angle),
                            "shoulder_visibility": shoulder_visibility,
                            "elbow_visibility": elbow_visibility,
                            "wrist_visibility": wrist_visibility,
                            "shoulder_presence": shoulder_presence,
                            "elbow_presence": elbow_presence,
                            "wrist_presence": wrist_presence
                        }
                        if (elapsed_time <= 5.0):
                            if 'right' in snapshot:
                                if(isinstance(snapshot['right'], list)):
                                    snapshot['right'].append(r_measurement)
                            else:
                                snapshot['right'] = [r_measurement]
                        # Draw text overlay next to right elbow joint area
                        cv2.putText(
                            frame, f"R: {int(right_angle)} deg; EMA: {int(ema_angle)}", 
                            (display_x(r_elbow.x, w) + 15, int(r_elbow.y * h)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                        )
                except IndexError:
                    pass

                # Draw base tracker verification dots continuously
                for landmark in pose:
                    x, y = display_x(landmark.x, w), int(landmark.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (x, y), 4, (0, 255, 0))

        # Fixed, high-visibility repetition counter at the top of the frame.
        cv2.rectangle(frame, (0, 0), (w, 75), (0, 0, 0), -1)
        if biceps_curl_sm_l is not None:
            left_rep_count = biceps_curl_sm_l.rep_count
        if biceps_curl_sm_r is not None:
            right_rep_count = biceps_curl_sm_r.rep_count
        reps = max(left_rep_count, right_rep_count)
        cv2.putText(
            frame,
            f"REPS: {reps}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.imshow('Webcam feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print(progressive_snaphost)
cap.release()
cv2.destroyAllWindows()
