import cv2
import os
import easyocr
from ultralytics import YOLO
import time

# Set the device to GPU if available, otherwise use CPU
def process_video(input_video_path, output_video_path, model1_path, model2_path):
    # Load the first YOLOv8 model
    model1 = YOLO(model1_path)

    # Load the second YOLOv8 model
    model2 = YOLO(model2_path)

    # Set the language for OCR (update this based on your license plate language)
    ocr_reader = easyocr.Reader(['en'])

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the local save path
    save_path = "C:\\Users\\saadu\\PycharmProjects\\pythonProject1\\save"

    # Create VideoWriter object to save the output video in MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    # Set to store unique IDs for vehicles and number plates
    unique_vehicle_ids = set()
    unique_plate_ids = set()

    # Counters for vehicle and plate IDs
    vehicle_id_counter = 0
    plate_id_counter = 0

    # Dictionaries to store the best confidence and associated information for each ID
    best_vehicle_confidence_dict = {}
    best_plate_confidence_dict = {}

    # Initialize the highest confidence dynamically
    highest_confidence = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame for the first model
            results1 = model1.track(frame, persist=True, conf=0.1)

            # Run YOLOv8 inference on the frame for the second model
            results2 = model2.track(frame, persist=True, conf=0.1)

            # Draw bounding boxes for the first model (vehicle detection)
            annotated_frame = results1[0].plot()

            # Draw bounding boxes for the model
            if results1[0].boxes.id is not None:
                boxes = results1[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results1[0].boxes.id.cpu().numpy().astype(int)
                confidences = results1[0].boxes.conf.cpu().numpy()

                for box, vehicle_id, confidence in zip(boxes, ids, confidences):
                    # Check if the vehicle ID is already in the set
                    if vehicle_id not in unique_vehicle_ids:
                        # Add the vehicle ID to the set
                        unique_vehicle_ids.add(vehicle_id)

                        # Update the vehicle ID counter
                        vehicle_id_counter += 1

                        # Draw bounding box for the model on the frame
                        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (85, 45, 255), 2,
                                      lineType=cv2.LINE_AA)
                        # Display ID and confidence
                        cv2.putText(annotated_frame, f'Coqnf: {confidence:.2f}', (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Save the unique IDs
                        best_vehicle_confidence_dict[vehicle_id] = (confidence, box)

            # Display the last vehicle ID at the top of the frame
            cv2.putText(annotated_frame, f'Vehcile Count: {vehicle_id_counter}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw bounding boxes for the second model (license plate extraction)
            if results2[0].boxes.id is not None:
                boxes = results2[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results2[0].boxes.id.cpu().numpy().astype(int)
                confidences = results2[0].boxes.conf.cpu().numpy()

                for box, plate_id, confidence in zip(boxes, ids, confidences):
                    # Draw bounding box for the second model on the same frame
                    cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (85, 45, 255), 2,
                                  lineType=cv2.LINE_AA)

                    # Crop the image using the bounding box coordinates
                    cropped_img = frame[box[1]:box[3], box[0]:box[2]]

                    # Use easyocr to extract text from the cropped image
                    results_ocr = ocr_reader.readtext(cropped_img)

                    # Display the extracted text
                    if results_ocr:
                        text = results_ocr[0][1]
                        confidence = results_ocr[0][2]  # Confidence from OCR
                        cv2.putText(
                            annotated_frame,
                            f" {text} , {confidence:.2f}",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            [85, 45, 255],
                            2,
                            lineType=cv2.LINE_AA
                        )

                        # Update the plate ID counter
                        plate_id_counter += 1

                        # Update the highest confidence dynamically
                        highest_confidence = max(highest_confidence, confidence)

                        # Save the cropped image only when confidence is above the highest confidence
                        if confidence >= highest_confidence:
                            filename = f"cropped_img_{plate_id}conf{confidence:.2f}.jpg"
                            filepath = os.path.join(save_path, filename)
                            cv2.imwrite(filepath, cropped_img)

                    # Use easyocr to extract text from the cropped image
                    results_ocr = ocr_reader.readtext(cropped_img)

                    # Display the extracted text
                    if results_ocr:
                        text = results_ocr[0][1]
                        print(f"Plate ID: {plate_id} Conf: {confidence:.2f}, Extracted Text: {text}")

                        # Save the unique IDs
                        unique_plate_ids.add(plate_id)

                        # Update the best confidence and associated number plate for each ID
                        best_plate_confidence_dict[plate_id] = (confidence, text)

            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_info = f"FPS: {frame_count / elapsed_time:.2f}"
            cv2.putText(annotated_frame, fps_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save the combined frame to the output video
            out.write(annotated_frame)

            # Display the combined frame
            cv2.imshow("Combined Output", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, writer, and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print the total count of unique IDs and their corresponding confidences
    print("\nTotal Unique Vehicle IDs:")
    for unique_id in unique_vehicle_ids:
        confidence, box = best_vehicle_confidence_dict[unique_id]
        print(f"Vehicle ID: {unique_id}, Best Confidence: {confidence:.2f}, Bounding Box: {box}")

    print("\nTotal Unique Plate IDs:")
    for unique_id in unique_plate_ids:
        confidence, text = best_plate_confidence_dict[unique_id]
        print(f"Plate ID: {unique_id}, Best Confidence: {confidence:.2f}, Best Text: {text}")

# Process the video with both models
input_video_path = "E:\\data\\traffic.mp4"
output_video_path = 'output_video_combined.avi'
model1_path = 'vehicle_detection_model.pt'
model2_path = 'Number_plate.pt'
process_video(input_video_path, output_video_path, model1_path, model2_path)
