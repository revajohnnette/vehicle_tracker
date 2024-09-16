# Vehicle Tracking and CSV Export using YOLOv7 and DeepSORT

This project uses **YOLOv7** for object detection and **DeepSORT** for object tracking to detect and track vehicles in a video. The user can click on a vehicle in the video to track it, and the position of the selected vehicle is saved to a CSV file.

## Features

- **Object Detection**: YOLOv7 detects all objects in each frame.
- **Object Tracking**: DeepSORT tracks objects between frames, maintaining unique IDs for each object.
- **User Interaction**: The user can click on a vehicle in the video to track it.
- **Data Logging**: The code logs the selected vehicle’s frame number, object ID, and position (X, Y) to a CSV file.
- **Video Output**: The output video highlights the tracked vehicle with a green bounding box.

## Installation

1. **Prerequisites**:
    - Python 3.8+
    - OpenCV (`cv2`)
    - Ikomia SDK for integrating YOLOv7 and DeepSORT tasks

2. **Install Dependencies**:
    ```bash
    pip install opencv-python ikomia-sdk
    ```

3. **Project Files**:
    - `trackonevehiclecsv.py`: Main code file for tracking vehicles and logging data to CSV.
    - `selected_object_tracking_data.csv`: Output CSV file storing the tracking data.
    - `deepsort_output_video.avi`: Output video showing the tracked vehicle with a bounding box.

## How to Run

1. Place your input video file (`test2.mp4`) in the same directory as the code or adjust the `input_video_path` to the correct file path.
2. Run the Python script:
    ```bash
    python trackonevehiclecsv.py
    ```
3. The video will open in a window. Click on the vehicle you want to track, and the system will automatically track it.
4. Press **'q'** to exit the video processing.

## Workflow Overview

### 1. **Object Detection with YOLOv7**
The code uses YOLOv7 as the object detection model. YOLOv7 detects all vehicles in each frame of the video and outputs their bounding boxes and class labels.

```python
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)
```

### 2. **Object Tracking with DeepSORT**
DeepSORT is used to track objects across frames. It assigns a unique ID to each detected object and updates the tracking information for each new frame.

```python
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)
tracking.set_parameters({"categories": "all", "conf_thres": "0.5"})
```

### 3. **User Click Selection**
The user can click on any vehicle in the video to select it for tracking. The coordinates of the click are used to determine which detected object to track.

```python
cv2.setMouseCallback("DeepSORT", click_event)
```

### 4. **Data Logging**
Once a vehicle is selected, its position (X, Y) is logged along with the frame number and object ID to a CSV file (`selected_object_tracking_data.csv`).

```python
csv_writer.writerow([frame_count, selected_object_id, x_min, y_min])
```

### 5. **Video Output**
The video output includes a green bounding box around the selected vehicle, showing the user which vehicle is being tracked.

```python
cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 3)
```

## CSV Output

The `selected_object_tracking_data.csv` file logs the following data for each frame where the vehicle is tracked:

- **Frame Number**: The frame number in the video.
- **Object ID**: The unique ID assigned to the tracked vehicle.
- **Position X**: The X coordinate of the top-left corner of the bounding box.
- **Position Y**: The Y coordinate of the top-left corner of the bounding box.

Sample CSV output:
```
Frame Number,Object ID,Position X,Position Y
1,4,120,85
2,4,125,88
3,4,130,90
```

## Troubleshooting

- **Error: Could not open video**: Ensure that the input video path is correct.
- **No object selected**: Ensure you are clicking directly on a vehicle. Adjust the threshold if necessary.
- **Video not displaying**: Ensure OpenCV is installed and functioning properly (`pip install opencv-python`).

## License

This project is licensed under the MIT License.

---

This README provides an overview of the code's functionality and gives instructions for installation, running the program, and understanding the output.
