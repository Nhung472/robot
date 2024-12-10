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
    cv2.imwrite(os.path.join(img_dir, str(step) + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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
    cv2.imwrite(os.path.join(img_dir, str(step) + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def stitch_video_direct_mode(episode, img_dir='./images'):
    """Combine images into a video file."""
    ensure_directory_exists(img_dir)  # Ensure the directory exists

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

    # Write images to video
    for file in file_list:
        img_path = os.path.join(img_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Skipping invalid or missing file {img_path}")
            continue

        # Ensure the image has the correct resolution
        img = cv2.resize(img, (width, height))
        out.write(img)

    out.release()

    # Verify if the video was created successfully
    if os.path.exists(video_path):
        print(f"Video saved successfully at {video_path}")
    else:
        print("Error: Video file was not created.")

    # Optionally clean up the images
    for file in file_list:
        os.remove(os.path.join(img_dir, file))
