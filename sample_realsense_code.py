"""
####### Princeton Robotics - Sample RealSense Code #########
 -> Uses PyGame to display a real-time feed of the camera
 -> Can use the clipping distance clip out parts of the background
 -> Yours to explore. Main data is the depth_image and color_image
    -> color_image is a 3D array with an array of length 3 representing the
        RGB color at pixel (x,y). color_image[0,0,0] would return the red channel at (0,0)
    -> depth_image is a 2D array with a float representing the depth at point (x,y)
        depth_image[0, 0] would return the depth (UNSCALED!!) at point (0,0)
 -> Known bug: I'm out of time at today's meeting, so the image's horizontal and
     vertical axes are flipped
"""

# Necessary Imports
import pyrealsense2 as rs
import numpy as np
import cv2
import pygame

# If clipping is enabled, this will crop out the background
CLIPPING_DIST_METERS = 10.0

# Enables background clipping
CLIPPING_ENABLED = False

# Initializing the real sense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Aligning is necessary for depth perception integration
align_to = rs.stream.color
align = rs.align(align_to)

# Starting the profile and getting the default depth sensor (needs scalability for multiple sensors)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

clipping_distance = CLIPPING_DIST_METERS / depth_scale

# Initializing pygame for visualization
pygame.init()
display = pygame.display.set_mode((1280, 720))

# PyGame logic
GAME_EXIT = False
while not GAME_EXIT:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            GAME_EXIT = True

    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()

    depth = frames.get_depth_frame() # Contains the
    color = frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data()) # a 2D array with the depth at pixel x, y
    color_image = np.asanyarray(color.get_data()) # NOTE: the color is slightly off
    if not depth or not color: continue # Skip missed frames

    background_color = 0 # Arbitrary color for the clipped background (if clipping is enabled)
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # Depth image is 1 channel, color is 3 channels
    # Sets all pixels with distance > clipping distance to background_color, otherwise remains the same.
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap)) # Combine the two images

    if CLIPPING_ENABLED:
        surf = pygame.surfarray.make_surface(images)
        display.blit(surf, (0, 0))
    else:
        surf = pygame.surfarray.make_surface(color_image)
        display.blit(surf, (0, 0))

    pygame.display.update()

pygame.quit()
pipeline.stop()
