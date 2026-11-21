# Keypoints for all detected persons in one frame.
std_msgs/Header header

# Flat list of keypoints.
KeyPoint2D[] keypoints

# Person index for each entry in keypoints (parallel array).
int32[] person_index
