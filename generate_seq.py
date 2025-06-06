python
Copy code
import numpy as np
import torch
import cv2
import mediapipe as mp
import librosa
import matplotlib.pyplot as plt
import moviepy.editor as mpy

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def get_neutral_mesh(reference_image_path):
    # Load the reference image
    image = cv2.imread(reference_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform face mesh detection
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        raise ValueError("No face detected in the reference image.")
    
    # Extract the landmarks
    landmarks = results.multi_face_landmarks[0]
    neutral_mesh = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    
    return neutral_mesh

def apply_mesh_offsets(neutral_mesh, mesh_offsets):
    transformed_meshes = []
    for offset in mesh_offsets:
        transformed_mesh = neutral_mesh + offset
        transformed_meshes.append(transformed_mesh)
    return transformed_meshes

def project_to_2d(mesh, focal_length=1.0, image_width=500, image_height=500):
    # Perspective projection matrix
    aspect_ratio = image_width / image_height
    fov = np.radians(90)  # Field of view
    near = 0.1
    far = 1000

    f = 1 / np.tan(fov / 2)
    perspective_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

    # Transform mesh to homogeneous coordinates
    mesh_homogeneous = np.hstack((mesh, np.ones((mesh.shape[0], 1))))

    # Apply the projection matrix
    projected_mesh = mesh_homogeneous @ perspective_matrix.T

    # Normalize by the w component
    projected_mesh /= projected_mesh[:, 3].reshape(-1, 1)

    # Convert to image coordinates
    mesh_2d = projected_mesh[:, :2]
    mesh_2d[:, 0] = (mesh_2d[:, 0] + 1) * (image_width / 2)
    mesh_2d[:, 1] = (1 - mesh_2d[:, 1]) * (image_height / 2)

    return mesh_2d

def generate_2d_headpose_sequence(reference_image_path, audio_sequence, audiotomesh_model):
    # Step 1: Generate neutral mesh
    neutral_mesh = get_neutral_mesh(reference_image_path)

    # Step 2: Get mesh offsets from audio sequence using the Audiotomesh model
    audiotomesh_model.eval()
    with torch.no_grad():
        audio_tensor = torch.from_numpy(audio_sequence).float().unsqueeze(0)  # Add batch dimension
        mesh_offsets = audiotomesh_model(audio_tensor).numpy()

    # Step 3: Apply mesh offsets to the neutral mesh
    transformed_meshes = apply_mesh_offsets(neutral_mesh, mesh_offsets)

    # Step 4: Project 3D meshes to 2D to get head poses
    headpose_sequence = [project_to_2d(mesh) for mesh in transformed_meshes]

    return headpose_sequence

def create_animation(headpose_sequence, output_path='headpose_animation.mp4', fps=30):
    # Create a video from the sequence of 2D head poses
    def make_frame(t):
        idx = int(t * fps)
        if idx >= len(headpose_sequence):
            idx = len(headpose_sequence) - 1
        headpose = headpose_sequence[idx]
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        for (x, y) in headpose:
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)  # Use the 2D projected coordinates
        return img
    
    clip = mpy.VideoClip(make_frame, duration=len(headpose_sequence) / fps)
    clip.write_videofile(output_path, fps=fps)

# Example usage
reference_image_path = 'path/to/reference/image.jpg'
audio_path = 'path/to/audio/file.wav'

# Load the audio sequence
audio_sequence, _ = librosa.load(audio_path, sr=16000, mono=True)

# Import the pretrained model (assuming it is defined in another module)
from audiotomesh_model import AudioToMeshModel  # Adjust the import as per your project structure

# Load your trained Audiotomesh model
audiotomesh_model = AudioToMeshModel()
audiotomesh_model.load_state_dict(torch.load("/content/drive/MyDrive/Audio2Mesh/best_audio_to_mesh_model.pth"))

# Generate 2D head pose sequence
headpose_sequence = generate_2d_headpose_sequence(reference_image_path, audio_sequence, audiotomesh_model)

# Create animation
create_animation(headpose_sequence, output_path='headpose_animation.mp4', fps=30)

# Visualize the first frame
plt.imshow(headpose_sequence[0], cmap='gray')
plt.show()