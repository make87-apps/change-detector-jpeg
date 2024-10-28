import logging

import cv2
import numpy as np
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from make87 import initialize, get_publisher_topic, get_subscriber_topic, resolve_topic_name


class ImageChangeDetector:
    def __init__(self):
        self.previous_image_data = None
        topic_name = resolve_topic_name(name="JPEG_OUTPUT")
        self.output_topic = get_publisher_topic(name=topic_name, message_type=ImageJPEG)

    def process_image_change(self, current_image_data):
        if self.previous_image_data is None:
            self.previous_image_data = current_image_data
            return False, None

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
            return True, current_image_data
        return False, None


def main():
    initialize()

    topic_name = resolve_topic_name("JPEG_INPUT")
    input_topic = get_subscriber_topic(name=topic_name, message_type=ImageJPEG)
    detector = ImageChangeDetector()

    def callback(message: ImageJPEG):
        change_detected, new_image_data = detector.process_image_change(message.data)
        if change_detected:
            new_message = ImageJPEG(data=new_image_data)
            detector.output_topic.publish(new_message)
            logging.info("Detected change and forwarded image.")

    input_topic.subscribe(callback)


if __name__ == "__main__":
    main()
