# Target segmentation

This code keeps track of a target's segmentation mask, manipulating data of pushing actions. Initially mask-RCNN is implemented to segment the scene. The produced masks and the pushing actions are used as input for U-Net, that will output the final target's mask.

Data Format: 

S_{t}, S_{t+1}: segmentation masks of the t, t+1 step. (200x200 images)  (same format as the pybullet segmentation)

p12: pushing action scaled to [-0.25, 0.25] workspace (must be resized to [0,199])

target_t, target_t+1: target masks, at t, t+1 step
