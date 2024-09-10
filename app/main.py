from datetime import datetime

from make87_messages.text.PlainText_pb2 import PlainText
from make87 import get_topic, topic_names, SubscriberTopic


def main():
    topic = get_topic(name=topic_names.INCOMING_MESSAGE)

    def callback(message: PlainText):
        received_dt = datetime.now()
        publish_dt = message.timestamp.ToDatetime()
        print(
            f"Received message '{message.body}'. Sent at {publish_dt}. Received at {received_dt}. Took {(received_dt - publish_dt).total_seconds()} seconds."
        )

    topic.subscribe(callback)


if __name__ == "__main__":
    main()
