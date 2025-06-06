import os
import numpy as np
import cv2
import torch

def save_mesh_data(mesh_data, folder_path, file_name):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    np.save(file_path, mesh_data)
    
def save_audio_features(audio_features, folder_path, file_name):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    np.save(file_path, audio_features.cpu().numpy())

def visualize_and_save_mesh(video_chunk, mesh_data, output_folder):
    
    cap = cv2.VideoCapture(video_chunk)
    os.makedirs(output_folder, exist_ok=True)
    frame_idx = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if frame_idx < len(mesh_data):
            landmarks = mesh_data[frame_idx]
            for landmark in landmarks:
                x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            frame_path = os.path.join(output_folder, f"frame_{frame_idx}.png")
            cv2.imwrite(frame_path, image)
            frame_idx += 1

    cap.release()
    
def pad_or_truncate_sequence(sequence, target_len):
    sequence_len = len(sequence)
    if sequence_len < target_len:
        padding = [sequence[-1]] * (target_len - sequence_len)  # Repeat the last frame
        sequence.extend(padding)
    return sequence[:target_len]

