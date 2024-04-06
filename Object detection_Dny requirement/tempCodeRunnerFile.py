gmented_image = frame.copy()

        if not detect_results:
            return segmented_image  # Return the original frame if no objects are detected

        # Get the segmentation mask for the entire frame
        segmentation_mask = self.segment_model(frame)

        for result in detect_results[0]:
            class_id = result.boxes.cls[0].item()  # Convert to scalar value

            # Assuming class_id 0 represents the object of interest, like 'person'
            if class_id != 0:
                continue

            box = result.boxes.xyxy[0].cpu().numpy().astype(int)  # Assuming only one box per detection

            # Extract the segmentation mask for the detected object
            object_mask = segmentation_mask[0][box[1]:box[3], box[0]:box[2]]

            # Apply the segmentation mask to the object region
            segmented_roi = cv2.bitwise_and(frame[box[1]:box[3], box[0]:box[2]], 
                                            frame[box[1]:box[3], box[0]:box[2]], 
                                            mask=object_mask.astype(np.uint8))

            # Overlay the segmented ROI on the original frame
            segmented_image[box[1]:box[3], box[0]:box[2]] = segmented_roi

        return segmented_image
