# from inference_sdk import InferenceHTTPClient
from inference import get_model
import cv2
import supervision as sv
import time


# cap = cv2.VideoCapture("Fire_Detection/Security_Camera_Video_of_Fire_at_WLNE.mp4")
cap = cv2.VideoCapture("Electrica_Fire_Caught_on_Surveillance_Video_with_Fire_Sprinkler_Activation.mp4")
model = get_model(model_id="fire-detection-jtuly/2", api_key="OjnJPIp598mu7NeirnOM")

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

while True:
    ret, frame = cap.read()
    # print(ret)
    if ret == False:
        print("[Failed] Can not load the video!!!!")
    else:

        img = frame.copy()
        results = model.infer(img)
        # print("=====")
        # print(results)
        detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
        print("--------------------------")
        print(detections)
        # for detecttion in detections:
            # create supervision annotators
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # annotate the image with our inference results
        img = bounding_box_annotator.annotate(
            scene=img, detections=detections)
        img = label_annotator.annotate(
            scene=img, detections=detections)
        # time when we finish processing for this frame 
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
    
        # converting the fps into integer 
        fps = int(fps) 
    
        # converting the fps to string so that we can display it on frame 
        # by using putText function 
        fps = str(fps) 
        
        cv2.putText(img, fps, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv2.imshow("annotated_image", img)
        cv2.waitKey(1)
