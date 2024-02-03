from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("../Videos/vid1.mp4")  # For Images
model = YOLO("best.pt")

classNames = ['article', 'chat-panel', 'comments', 'contentinfo', 'control-bar', 'form', 'items-grid', 'items-list',
              'login', 'media', 'menu-bar', 'menu-panel', 'option-panel', 'posts', 'search', 'side-panel', 'tab-bar']
myColor = (0, 0, 255)

while True:
    success, img = cap.read()

    # Check if the video has ended
    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

            myColor = (255, 1, 0)


            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                (max(0, x1), max(35, y1)), scale=2, thickness=2, colorB=myColor,
                                colorT=(255, 255, 255), colorR=myColor, offset=5)
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    # Resize the image to a smaller size
    resized_img = cv2.resize(img, (1280, 720))  # Adjust the size as needed

    # Display the resized image
    cv2.imshow("Image", resized_img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
