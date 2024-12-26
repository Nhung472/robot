import cv2
import os
import numpy as np

def ensure_directory_exists(directory):
    """Ensure the directory exists, and create it if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def record_direct_mode(p, step, img_dir='./images'):
    """Capture and save an image using the direct camera setup."""
    ensure_directory_exists(img_dir)  # Ensure the directory exists

    # Set up the camera
    width = 640
    height = 480
    fov = 60
    aspect = width / height
    near = 0.01
    far = 100
    camera_pos = [-2, 0, 2]
    target_pos = [0, 0, 1]
    up_vector = [1, 0, 0]
    view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get the camera image
    _, _, img, _, _ = p.getCameraImage(
        width, height, 
        viewMatrix=view_matrix, 
        projectionMatrix=projection_matrix, 
        shadow=False, 
        lightDirection=[1, 1, 1], 
        renderer=p.ER_TINY_RENDERER
    )
    
    # Ensure image has compatible format
    img = np.array(img, dtype=np.uint8)

    # Save the image
    img_path = os.path.join(img_dir, f"{step}.png")
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def record_gui_mode(p, step, img_dir='./images'):
    """Capture and save an image using the GUI camera setup."""
    ensure_directory_exists(img_dir)  # Ensure the directory exists

    # Compute the camera view matrix
    info = p.getDebugVisualizerCamera()
    viewMatrix = info[2]
    projectionMatrix = info[3]

    # Get the camera image
    _, _, img, _, _ = p.getCameraImage(
        width=640,
        height=480,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        shadow=False,
        lightDirection=[1, 1, 1],
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # Ensure image has compatible format
    img = np.array(img, dtype=np.uint8)

    # Save the image
    img_path = os.path.join(img_dir, f"{step}.png")
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def stitch_video_direct_mode(episode, img_dir='./images'):
    """Combine images into a playable video."""
    # Ensure the directory exists
    if not os.path.exists(img_dir):
        print(f"Error: Directory {img_dir} does not exist.")
        return

    # Prepare output video settings
    width = 640
    height = 480
    fps = 20
    video_path = f"./results/video_ep_{episode}.mp4"
    ensure_directory_exists('./results')  # Ensure the results directory exists
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Get and sort image files
    file_list = [file for file in os.listdir(img_dir) if file.endswith(".png")]
    file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))

    # Check if any images exist
    if not file_list:
        print("Error: No images found in the directory.")
        return

    # Write images to video
    for file in file_list:
        img_path = os.path.join(img_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Skipping invalid or missing file {img_path}")
            continue

        # Ensure the image has the correct resolution
        if img.shape[:2] != (height, width):
            print(f"Resizing frame {file} to {width}x{height}")
            img = cv2.resize(img, (width, height))

        # Ensure the frame has valid data type
        img = np.array(img, dtype=np.uint8)
        out.write(img)

    # Finalize the video
    out.release()

    # Verify if the video was created successfully
    if os.path.exists(video_path):
        print(f"Video saved successfully at {video_path}")
    else:
        print("Error: Video file was not created.")

    # Clean up the images
    for file in file_list:
        os.remove(os.path.join(img_dir, file))

# Example usage
if __name__ == "__main__":
    class FakePyBullet:
        """Mock PyBullet class for demonstration purposes."""
        def computeViewMatrix(self, camera_pos, target_pos, up_vector):
            return "fake_view_matrix"

        def computeProjectionMatrixFOV(self, fov, aspect, near, far):
            return "fake_projection_matrix"

        def getCameraImage(self, width, height, viewMatrix, projectionMatrix, shadow, lightDirection, renderer):
            return (0, 0, np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), None, None)

        def getDebugVisualizerCamera(self):
            return [None, None, "fake_view_matrix", "fake_projection_matrix"]

    # Simulate recording and video creation
    p = FakePyBullet()
    for step in range(10):  # Record 10 frames
        record_direct_mode(p, step)

    # Stitch the recorded frames into a video
    stitch_video_direct_mode(episode=1)
