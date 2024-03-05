import pyrealsense2 as rs
import numpy as np

# Start the pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)  # Enable the depth stream

# Start the pipeline and get the stream profile
pipeline.start(config)
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

# Now obtain the intrinsics
depth_intrinsics = depth_profile.get_intrinsics()

# Extracting the camera matrix and distortion coefficients
camera_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                          [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(depth_intrinsics.coeffs)  # Distortion coefficients

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Don't forget to stop the pipeline
pipeline.stop()
