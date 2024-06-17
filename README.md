# Privacy Management and its Implementation in Real-time videos

This project, Privacy Management was built with the goal to keep confidentional information private in real-time videos.We have implemented privacy using color-based inpainting, object-detection-based inpainting, segmentation-based inpating So we have divided project into total 7 tasks which helped us to built project step by step. 
Let's see how you can implement it in your machine

## Installation

Before running the script, ensure you have all the required libraries installed. You can do this by installing the dependencies listed in the `requirements.txt` file.

### Steps:

1. Clone the repository:

    ```sh
    git clone https://github.com/Sejalparate/privacy-management-dp.git
    ```

2. Navigate to the project directory:

    ```sh
    cd privacy-management-dp/
    ```

3. Install the required libraries:

    ```sh
    pip install -r requirements.txt
    ```

This will install all the dependencies specified in the `requirements.txt` file.



## Task 1 - Masked Video

This project demonstrates color detection using OpenCV. The script captures video from the webcam, converts it to HSV color space, and detects objects of a specific color (green) within the video feed. Detected objects are highlighted with a bounding box. So, in masked video we give color range according to your choice (in this code green colors range you can change according to your convience) then if particular that color range is present in camera feed then that particular potion gets masked.
You can run file by using command:
```sh
    python Masked_video.py
```
![Example Output](https://github.com/Sejalparate/privacy-management-dp/blob/main/Task%201%20-%20Masked%20Video/Masked_Video.jpg)
