import cv2
import os
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

# Example usage
output_folder = "visualized_meshes"
visualize_and_save_mesh("/home/poorvi/Workspace/Audio2Mesh/video_chunks", video_meshes_list[0].numpy(), output_folder)
