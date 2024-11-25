import pygame
import numpy as np
import mediapipe as mp
import pygame.camera
import json
import torch
from PIL import Image
from datetime import datetime
from torchvision import transforms


# Initialize PyGame and Camera
pygame.init()
pygame.camera.init()

# Select Camera
camera = pygame.camera.Camera(pygame.camera.list_cameras()[0], (640, 480))
camera.start()

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=1)

# Create Display Window
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Face Detection")

#import faces data from faces.json

def load_json():
    with open("faces.json","r") as ri:
        retrived_data=json.load(ri)
    return retrived_data

global data
data=load_json()

def use_nn(image_input):
    #setup NN to compare faces
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    def load_image(image_path):
        image = Image.open(image_path)
        return transform(image).unsqueeze(0)

    input_tensor = transform(image_input).unsqueeze(0)

    max_similarity = -float('inf')
    found_img_path=""

    for entry in data:
        comparison_image = load_image(entry)
        similarity = torch.nn.functional.cosine_similarity(input_tensor.view(-1), comparison_image.view(-1), dim=0)
        
        if similarity.item() > max_similarity:
            max_similarity = similarity.item()
            found_img_path = entry

    if max_similarity >0.89:
        return(f'{max_similarity*100}%',found_img_path)
    else :
        return ("match not found","unauthorized")


#Some terms that i used:
"""
1. ROI= Region of intrest, it is the region that we want to compare
2. CNN= Convolutional Neural Network, it is a type of neural network that is designed to recognize patterns in images
"""






running = True
face_detected=False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f and face_detected:  # Save image when 'F' is pressed
                # Save each detected face region
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x_min = int(bbox.xmin * 640)  # Scale x-coordinate
                    y_min = int(bbox.ymin * 480)  # Scale y-coordinate
                    width = int(bbox.width * 640)  # Scale width
                    height = int(bbox.height * 480)  # Scale height

                    # Crop face region
                    face_roi = frame_rgb[y_min:y_min + height, x_min:x_min + width]

                    # Save the cropped face as a PIL image
                    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
                    face_image = Image.fromarray(face_roi)
                    face_image.save(f"images/{time}_face.jpg")

                    print(f"Face saved as {time}_face.jpg!")


            elif event.key == pygame.K_s and face_detected:  # Scan image when 'S' is pressed
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x_min = int(bbox.xmin * 640)  # Scale x-coordinate
                    y_min = int(bbox.ymin * 480)  # Scale y-coordinate
                    width = int(bbox.width * 640)  # Scale width
                    height = int(bbox.height * 480)  # Scale height

                    # Crop face region from the frame
                    face_roi = frame_rgb[y_min:y_min + height, x_min:x_min + width]

                    # Handle edge cases where bounding box might exceed image boundaries
                    face_roi = np.clip(face_roi, 0, 255)
                    
                    # Convert face ROI to PIL Image
                    image = Image.fromarray(face_roi)

                    # Perform neural network comparison
                    nn_output = use_nn(image)

                    if nn_output[0] == "match not found":
                        print("Person not found: Unauthorized access detected")
                    else:
                        print("Similarity detected:", nn_output[0])
                        print("Is Authorized:", data[nn_output[1]]["authorized"])

    
    # Capture Frame from Camera
    frame = camera.get_image()
    frame_surface = pygame.surfarray.array3d(frame)
    frame_rgb = np.rot90(frame_surface)  # Rotate PyGame frame to match RGB
    frame_rgb = np.flip(frame_rgb, axis=1)  # Flip horizontally to match webcam view
    frame_rgb = np.flip(frame_rgb, axis=0)  # Flip vertically to match webcam view

    # Detect Faces with Mediapipe
    results = face_detection.process(frame_rgb)

    # Convert frame back to PyGame surface
    frame_surface = pygame.surfarray.make_surface(np.flip(np.rot90(frame_rgb, 3), axis=1))

    # Draw Faces on Frame
    if results.detections:
        face_detected = True
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * 640)  # Scale x-coordinate
            y_min = int(bbox.ymin * 480)  # Scale y-coordinate
            width = int(bbox.width * 640)  # Scale width
            height = int(bbox.height * 480)  # Scale height

            # Crop face region from frame_rgb
            face_roi = frame_rgb[y_min:y_min + height, x_min:x_min + width]

            # Convert face ROI to PIL Image for NN
            face_image = Image.fromarray(face_roi)
            
            # Draw rectangle on the frame
            pygame.draw.rect(frame_surface, (255, 0, 0), (x_min, y_min, width, height), 2)

    else:
        face_detected = False

    # Display Frame on Screen
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

# Cleanup
camera.stop()
pygame.quit()