def estimate_distance(box_height, frame_height):
    """
    Estimate distance using bounding box height.
    Assumes fixed camera.
    """
    focal_length = 800  # approx
    real_car_height = 1.5  # meters
    # protect against zero or extremely small box heights
    try:
        if box_height is None or box_height <= 1:
            return float('inf')
        return (real_car_height * focal_length) / box_height
    except Exception:
        return float('inf')
