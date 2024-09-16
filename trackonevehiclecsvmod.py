import csv
import cv2
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

input_video_path = "C:/Users/JOHNNETTE/Desktop/highway/Car-Tracking-Open-CV/test2.mp4"
output_video_path = 'deepsort_output_video.avi'

wf = Workflow()
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

tracking.set_parameters({
    "categories": "all",  
    "conf_thres": "0.5", 
})

clicked_point = None 
selected_object_id = None 

csv_file_path = 'selected_object_tracking_data.csv'
csv_file = open(csv_file_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Only logging Frame Number, Object ID, and Position (X, Y)
csv_writer.writerow(['Frame Number', 'Object ID', 'Position X', 'Position Y'])

def click_event(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")

stream = cv2.VideoCapture(input_video_path)
if not stream.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = stream.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

cv2.namedWindow("DeepSORT")
cv2.setMouseCallback("DeepSORT", click_event)

frame_count = 0
while True:
    ret, frame = stream.read()
    if not ret:
        print("End of video or error.")
        break

    frame_count += 1
    wf.run_on(array=frame)
    image_out = tracking.get_output(0) 
    obj_detect_out = tracking.get_output(1) 
    detections = obj_detect_out.get_objects()  

    if clicked_point is not None:
        for detection in detections:
            bbox = detection.box
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height

            if x_min <= clicked_point[0] <= x_max and y_min <= clicked_point[1] <= y_max:
                selected_object_id = detection.id  # Store selected object ID
                print(f"Selected object ID: {selected_object_id}")
                clicked_point = None  
                break  

    if selected_object_id is not None:
        selected_detection = next((d for d in detections if d.id == selected_object_id), None)

        if selected_detection:
            bbox = selected_detection.box
            x_min, y_min, width, height = bbox

            x_min = int(x_min)
            y_min = int(y_min)
            width = int(width)  # Ensure width is an integer
            height = int(height)  # Ensure height is an integer

            # Log only Frame Number, Object ID, and Position (X, Y) to CSV
            csv_writer.writerow([frame_count, selected_object_id, x_min, y_min])

            # Draw bounding box around the selected object
            cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 3)  # Green box for selected object

    out.write(frame) 
    display(frame, title="DeepSORT", viewer="opencv")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
out.release()
csv_file.close() 
cv2.destroyAllWindows()
