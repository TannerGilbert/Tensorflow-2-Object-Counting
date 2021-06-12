import cv2
import numpy as np
import argparse
import tensorflow as tf
import dlib

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

from trackable_object import TrackableObject
from centroidtracker import CentroidTracker

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    tf.keras.backend.clear_session()
    model = tf.saved_model.load(model_path)
    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, category_index, cap, labels, roi_position=0.6, threshold=0.5, x_axis=True, skip_frames=20, save_path='', show=True):
    counter = [0, 0, 0, 0]  # left, right, up, down
    total_frames = 0

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # Check if results should be saved
    if save_path:
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fps, (width, height))

    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break

        height, width, _ = image_np.shape
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        status = "Waiting"
        rects = []

        if total_frames % skip_frames == 0:
            status = "Detecting"
            trackers = []

            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)

            for i, (y_min, x_min, y_max, x_max) in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][i] > threshold and (labels == None or category_index[output_dict['detection_classes'][i]]['name'] in labels):
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

                # add the bounding box coordinates to the rectangles list
                rects.append((x_min, y_min, x_max, y_max))

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                if x_axis and not to.counted:
                    x = [c[0] for c in to.centroids]
                    direction = centroid[0] - np.mean(x)

                    if centroid[0] > roi_position*width and direction > 0 and np.mean(x) < args.roi_position*width:
                        counter[1] += 1
                        to.counted = True
                    elif centroid[0] < roi_position*width and direction < 0 and np.mean(x) > args.roi_position*width:
                        counter[0] += 1
                        to.counted = True

                elif not x_axis and not to.counted:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)

                    if centroid[1] > roi_position*height and direction > 0 and np.mean(y) < args.roi_position*height:
                        counter[3] += 1
                        to.counted = True
                    elif centroid[1] < roi_position*height and direction < 0 and np.mean(y) > args.roi_position*height:
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
        if x_axis:
            cv2.line(image_np, (int(roi_position*width), 0),
                     (int(roi_position*width), height), (0xFF, 0, 0), 5)
        else:
            cv2.line(image_np, (0, int(roi_position*height)),
                     (width, int(roi_position*height)), (0xFF, 0, 0), 5)

        # display count and status
        font = cv2.FONT_HERSHEY_SIMPLEX
        if x_axis:
            cv2.putText(image_np, f'Left: {counter[0]}; Right: {counter[1]}', (
                10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        else:
            cv2.putText(image_np, f'Up: {counter[2]}; Down: {counter[3]}', (
                10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(image_np, 'Status: ' + status, (10, 70), font,
                    0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

        if show:
            cv2.imshow('cumulative_object_counting', image_np)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        if save_path:
            out.write(image_np)

        total_frames += 1

    cap.release()
    if save_path:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str,
                        required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str,
                        required=True, help='Path to Labelmap')
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
    parser.add_argument('-s', '--skip_frames', type=int, default=20,
                        help='Number of frames to skip between using object detection model')
    parser.add_argument('-sh', '--show', default=True,
                        action="store_false", help='Show output')
    parser.add_argument('-sp', '--save_path', type=str, default='',
                        help='Path to save the output. If None output won\'t be saved')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(
        args.labelmap, use_display_name=True)

    if args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")

    run_inference(detection_model, category_index, cap, labels=args.labels, threshold=args.threshold,
                  roi_position=args.roi_position, x_axis=args.axis, skip_frames=args.skip_frames, save_path=args.save_path, show=args.show)
