import os
import torch
import torchaudio
import moviepy.editor as mp
import mediapipe as mp_face_mesh
import numpy as np
import json
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from save_data import save_mesh_data, save_audio_features, visualize_and_save_mesh,pad_or_truncate_sequence

# Load the processors and models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wav2vec_model_name = 'facebook/wav2vec2-large-960h-lv60-self'
emotion_model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'

audio_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
emotion_processor = Wav2Vec2Processor.from_pretrained(emotion_model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name).to(device)
emotion_model = Wav2Vec2Model.from_pretrained(emotion_model_name).to(device)

#split the videos into chunks of 20 seconds
def split_video(input_video_path, chunk_length=20):
    video_output_dir = "video_chunks"
    os.makedirs(video_output_dir, exist_ok=True)
    
    video = mp.VideoFileClip(input_video_path)
    duration = int(video.duration)  # Get the duration in seconds
    video_chunk_paths = []

    for start_time in range(0, duration, chunk_length):
        end_time = min(start_time + chunk_length, duration)
        chunk_path = os.path.join(video_output_dir, f"chunk_{start_time}_{end_time}.mp4")
        ffmpeg_extract_subclip(input_video_path, start_time, end_time, targetname=chunk_path)
        video_chunk_paths.append(chunk_path)

    return video_chunk_paths


#extract only audio chunks into a seperate folder named audio_chunks
def extract_audio_from_video_chunks(video_chunk_paths):
    audio_output_dir = "audio_chunks"
    os.makedirs(audio_output_dir, exist_ok=True)
    audio_chunk_paths = []

    for video_chunk in video_chunk_paths:
        video = mp.VideoFileClip(video_chunk)
        audio_path = os.path.join(audio_output_dir, os.path.basename(video_chunk).replace(".mp4", ".wav"))
        video.audio.write_audiofile(audio_path)
        audio_chunk_paths.append(audio_path)

    return audio_chunk_paths

#Process audio to extract features
def load_and_process_audio(audio_path, processor):
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if necessary
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    
    # Ensure the waveform tensor has the correct shape
    waveform = waveform.squeeze(0)  # Remove the channel dimension if present
    inputs = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    
    # Debug: print shapes to ensure correctness
    print(f"waveform shape after squeeze: {waveform.shape}")
    print(f"inputs shape: {inputs.shape}")
    breakpoint
    return inputs

#concatenate extracted features
def extract_and_concatenate_features(audio_path):
    audio_inputs = load_and_process_audio(audio_path, audio_processor)
    emotion_inputs = load_and_process_audio(audio_path, emotion_processor)
    
    
    with torch.no_grad():
        audio_features = wav2vec_model(audio_inputs).last_hidden_state
        emotion_features = emotion_model(emotion_inputs).last_hidden_state
    
    concatenated_features = torch.cat((audio_features, emotion_features), dim=-1)
    return concatenated_features

def generate_3d_meshes_for_chunk(video_chunk,target_frame_count=600):
    cap = cv2.VideoCapture(video_chunk)
    face_mesh = mp_face_mesh.solutions.face_mesh.FaceMesh(
        static_image_mode=False,        # Indicates a video stream input
        max_num_faces=1,                # Detect at most one face
        min_detection_confidence=0.5    # Minimum confidence threshold for detections
    )
    frames_meshes = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if len(face_landmarks.landmark) == 468:
                    frames_meshes.append([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])

    cap.release()
    face_mesh.close()
    
    # Ensure the sequence has the target number of frames
    frames_meshes = pad_or_truncate_sequence(frames_meshes, target_frame_count)
    
    return frames_meshes

#define padding for audio which does not satisfy max length 
def pad_tensor(tensor, length):
    if tensor.size(1) < length:
        pad_size = length - tensor.size(1)
        padding = torch.zeros((tensor.size(0), pad_size, tensor.size(2)), device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=1)
    return tensor

#convert mesh landmork to numerical data
def landmarks_to_tensor(landmarks):
    landmark_array = []
    for landmark in landmarks:
        coords = [(lm.x, lm.y, lm.z) for lm in landmark]
        landmark_array.append(coords)
    return torch.tensor(landmark_array)

def process_video(input_video_path):
    video_chunks = split_video(input_video_path)
    audio_chunks = extract_audio_from_video_chunks(video_chunks)
    
    # Debug statement to check the content of audio_chunks
    print(f"Video Chunks: {video_chunks}")
    print(f"Audio Chunks: {audio_chunks}")
    
    audio_features_list = []
    video_meshes_list = []
    
    for idx, (audio_chunk, video_chunk) in enumerate(zip(audio_chunks, video_chunks)):
        print(f"Processing audio chunk: {audio_chunk}")  # Debug statement
        print(f"Processing video chunk: {video_chunk}")  # Debug statement

        audio_features = extract_and_concatenate_features(audio_chunk)
        video_meshes = generate_3d_meshes_for_chunk(video_chunk)
        
        audio_features_list.append(audio_features)
        video_meshes_list.append(torch.tensor(video_meshes))  # Convert to tensor
        
        # Save audio features and video meshes
        folder_path = f"output_chunk_{idx}"
        save_audio_features(audio_features, folder_path, "audio_features.npy")
        save_mesh_data(video_meshes, folder_path, "video_meshes.npy")
        
        # Convert video_meshes to a NumPy array for visualization
        visualize_and_save_mesh(video_chunk, np.array(video_meshes), folder_path)
        
    # Pad all audio feature tensors to the maximum length
    max_length = max([features.size(1) for features in audio_features_list])
    audio_features_list = [pad_tensor(features, max_length) for features in audio_features_list]
   
    
   
    
    # Convert lists to tensors
    audio_features_tensor = torch.stack(audio_features_list)
    video_meshes_tensor = torch.stack(video_meshes_list)
    
    return audio_features_tensor, video_meshes_tensor

    # Flatten audio features to align with the temporal frames of video meshes
    #audio_features_tensor = torch.cat([af.view(-1, af.size(-1)) for af in audio_features_list], dim=0)
    #video_meshes_tensor = torch.cat(video_meshes_list, dim=0)
    
    #return audio_features_tensor, video_meshes_tensor

# Example usage
input_video_path = "/home/poorvi/Workspace/Audio2Mesh/WIN_20240601_19_43_48_Pro.mp4"
audio_features, video_meshes = process_video(input_video_path)
print(f"Audio Features Tensor Shape: {audio_features.shape}")
print(f"Video Meshes Tensor Length: {video_meshes.shape}")

# Save tensors to disk for training
torch.save(audio_features, "audio_features.pt")
torch.save(video_meshes, "video_meshes.pt")