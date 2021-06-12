from tflite_runtime.interpreter import Interpreter, load_delegate
import tensorflow as tf
import argparse
import cv2
import re
import numpy as np
import dlib

from trackable_object import TrackableObject
from centroidtracker import CentroidTracker


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape=[416, 416]):
    # from https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/9f16748aa3f45ff240608da4bd9b1216a29127f5/core/yolov4.py#L292
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(
        scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(
        scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(tf.constant(input_shape), dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


def detect_objects(interpreter, image, threshold, model_type):
    """Returns a list of detection results, each a dictionary of object info."""

    if model_type == 'tensorflow':
        set_input_tensor(interpreter, image)
        interpreter.invoke()

        # Get all output details
        boxes = get_output_tensor(interpreter, 0)
        classes = get_output_tensor(interpreter, 1)
        scores = get_output_tensor(interpreter, 2)
        count = int(get_output_tensor(interpreter, 3))
    elif model_type.startswith('yolo'):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], np.asarray(
            [image / 255.]).astype(np.float32))
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index'])
                for i in range(len(output_details))]
        if model_type == 'yolo':
            boxes, pred_conf = filter_boxes(
                pred[0], pred[1], score_threshold=0.25)
        elif model_type == 'yolov3-tiny':
            boxes, pred_conf = filter_boxes(
                pred[1], pred[0], score_threshold=0.25)
        boxes, scores, classes, count = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            score_threshold=threshold
        )
        boxes, scores, classes, count = boxes.numpy()[0], scores.numpy()[
            0], classes.numpy()[0], count.numpy()[0]

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': scores[i]
            }
            results.append(result)
    return results


def make_interpreter(model_file, use_edgetpu):
    model_file, *device = model_file.split('@')
    if use_edgetpu:
        return Interpreter(
            model_path=model_file,
            experimental_delegates=[
                load_delegate('libedgetpu.so.1',
                              {'device': device[0]} if device else {})
            ]
        )
    else:
        return Interpreter(model_path=model_file)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', type=str,
                        required=True, help='File path of .tflite file.')
    parser.add_argument('-l', '--labelmap', type=str,
                        required=True, help='File path of labels file.')
    parser.add_argument('-v', '--video_path', type=str, default='',
                        help='Path to video. If None camera will be used')
    parser.add_argument('-t', '--threshold', type=float,
                        default=0.5, help='Detection threshold')
    parser.add_argument('-roi', '--roi_position', type=float,
                        default=0.6, help='ROI Position (0-1)')
    parser.add_argument('-la', '--labels', nargs='+', type=str,
                        help='Label names to detect (default="all-labels")')
    parser.add_argument('-a', '--axis', default=True, action="store_false",
                        help='Axis for cumulative counting (default=x axis)')
    parser.add_argument('-e', '--use_edgetpu',
                        action='store_true', default=False, help='Use EdgeTPU')
    parser.add_argument('-s', '--skip_frames', type=int, default=20,
                        help='Number of frames to skip between using object detection model')
    parser.add_argument('-sh', '--show', default=True,
                        action="store_false", help='Show output')
    parser.add_argument('-sp', '--save_path', type=str, default='',
                        help='Path to save the output. If None output won\'t be saved')
    parser.add_argument('--type', choices=['tensorflow', 'yolo', 'yolov3-tiny'],
                        default='tensorflow', help='Whether the original model was a Tensorflow or YOLO model')
    args = parser.parse_args()

    labelmap = load_labels(args.labelmap)
    interpreter = make_interpreter(args.model, args.use_edgetpu)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[
        0]['shape']

    if args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")

    if args.save_path:
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fps, (width, height))

    counter = [0, 0, 0, 0]  # left, right, up, down
    total_frames = 0

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break

        height, width, _ = image_np.shape
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        status = "Waiting"
        rects = []

        if total_frames % args.skip_frames == 0:
            status = "Detecting"
            trackers = []

            image_pred = cv2.resize(image_np, (input_width, input_height))

            # Perform inference
            results = detect_objects(
                interpreter, image_pred, args.threshold, args.type)

            for obj in results:
                y_min, x_min, y_max, x_max = obj['bounding_box']
                if obj['score'] > args.threshold and (args.labels == None or labelmap[obj['class_id']] in args.labels):
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(
                        int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height))
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
        else:
            status = "Tracking"
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                x_min, y_min, x_max, y_max = int(pos.left()), int(
                    pos.top()), int(pos.right()), int(pos.bottom())

                if x_min < width and x_max < width and y_min < height and y_max < height and x_min > 0 and x_max > 0 and y_min > 0 and y_max > 0:
                    # add the bounding box coordinates to the rectangles list
                    rects.append((x_min, y_min, x_max, y_max))

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                if args.axis and not to.counted:
                    x = [c[0] for c in to.centroids]
                    direction = centroid[0] - np.mean(x)

                    if centroid[0] > args.roi_position*width and direction > 0 and np.mean(x) < args.roi_position*width:
                        counter[1] += 1
                        to.counted = True
                    elif centroid[0] < args.roi_position*width and direction < 0 and np.mean(x) > args.roi_position*width:
                        counter[0] += 1
                        to.counted = True

                elif not args.axis and not to.counted:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)

                    if centroid[1] > args.roi_position*height and direction > 0 and np.mean(y) < args.roi_position*height:
                        counter[3] += 1
                        to.counted = True
                    elif centroid[1] < args.roi_position*height and direction < 0 and np.mean(y) > args.roi_position*height:
                        counter[2] += 1
                        to.counted = True

                to.centroids.append(centroid)

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(
                image_np, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # Draw ROI line
        if args.axis:
            cv2.line(image_np, (int(args.roi_position*width), 0),
                     (int(args.roi_position*width), height), (0xFF, 0, 0), 5)
        else:
            cv2.line(image_np, (0, int(args.roi_position*height)),
                     (width, int(args.roi_position*height)), (0xFF, 0, 0), 5)

        # display count and status
        font = cv2.FONT_HERSHEY_SIMPLEX
        if args.axis:
            cv2.putText(image_np, f'Left: {counter[0]}; Right: {counter[1]}', (
                10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        else:
            cv2.putText(image_np, f'Up: {counter[2]}; Down: {counter[3]}', (
                10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(image_np, 'Status: ' + status, (10, 70), font,
                    0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

        if args.show:
            cv2.imshow('cumulative_object_counting', image_np)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        if args.save_path:
            out.write(image_np)

        total_frames += 1

    cap.release()
    if args.save_path:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
