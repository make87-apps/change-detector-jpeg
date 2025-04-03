import logging

import cv2
import numpy as np
import make87
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG


class ImageChangeDetector:
    def __init__(self):
        self.previous_image_data = None

    def process_image_change(self, current_image_data):
        if self.previous_image_data is None:
            self.previous_image_data = current_image_data
            return False

        # Compare the current and the previous image
        current_image = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(current_image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY
        )
        previous_image = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(self.previous_image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY
        )

        # Detect changes
        difference = cv2.absdiff(current_image, previous_image) > 35
        # Count non-zero pixels (areas where changes were detected)
        changed_pixels = np.sum(difference > 0)

        # Total number of pixels in the image
        total_pixels = difference.size

        # Calculate the fraction of changed pixels
        change_fraction = changed_pixels / total_pixels

        if change_fraction > 0.3:
            self.previous_image_data = current_image_data
            return True
        return False


def main():
    make87.initialize()

    input_topic = make87.get_subscriber(name="JPEG_INPUT", message_type=ImageJPEG)
    output_topic = make87.get_publisher(name="JPEG_OUTPUT", message_type=ImageJPEG)
    detector = ImageChangeDetector()

    def callback(message: ImageJPEG):
        change_detected = detector.process_image_change(message.data)
        if change_detected:
            output_topic.publish(message)
            logging.info("Detected change and forwarded image.")

    input_topic.subscribe(callback)
    make87.loop()



if __name__ == "__main__":
    main()
