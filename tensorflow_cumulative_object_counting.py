import numpy as np
import argparse
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
import pathlib

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

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
    input_tensor = input_tensor[tf.newaxis,...]
    
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
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict


def cumulative_counting(image, output_dict, category_index, labels, roi_position, deviation, threshold, x_axis):
    if x_axis:
        return cumulative_counting_x_axis(image, output_dict, category_index, labels, roi_position, deviation, threshold)
    else:
        return cumulative_counting_y_axis(image, output_dict, category_index, labels, roi_position, deviation, threshold)


def cumulative_counting_x_axis(image, output_dict, category_index, labels, roi_position, deviation, threshold):
    directions = []
    for i, (y_min, x_min, y_max, x_max) in enumerate(output_dict['detection_boxes']):
        if output_dict['detection_scores'][i] > threshold and (labels == None or category_index[output_dict['detection_classes'][i]]['name'] in labels):
            if abs(((x_min+x_max)/2)-roi_position) < deviation:
                directions.append(((x_min+x_max)/2)-roi_position > 0)

    return directions


def cumulative_counting_y_axis(image, output_dict, category_index, labels, roi_position, deviation, threshold):
    directions = []
    for i, (y_min, x_min, y_max, x_max) in enumerate(output_dict['detection_boxes']):
        if output_dict['detection_scores'][i] > threshold and (labels == None or category_index[output_dict['detection_classes'][i]]['name'] in labels):
            if abs(((y_min+y_max)/2)-roi_position) < deviation:
                directions.append(((y_min+y_max)/2)-roi_position > 0)

    return directions


def run_inference(model, category_index, cap, labels, roi_position=0.6, deviation=0.005, threshold=0.5, x_axis=True):
    counter = 0
    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break

        height, width, _ = image_np.shape

        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        
        # Count objects
        directions = cumulative_counting(image_np, output_dict, category_index, labels, roi_position, deviation, threshold, x_axis)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        # Draw ROI line
        if len(directions) > 0:
            counter += sum(directions)
            if x_axis:
                cv2.line(image_np, (int(roi_position*width), 0), (int(roi_position*width), height), (0, 0xFF, 0), 5)
            else:
                cv2.line(image_np, (0, int(roi_position*height)), (width, int(roi_position*height)), (0, 0xFF, 0), 5)
        else:
            if x_axis:
                cv2.line(image_np, (int(roi_position*width), 0), (int(roi_position*width), height), (0xFF, 0, 0), 5)
            else:
                cv2.line(image_np, (0, int(roi_position*height)), (width, int(roi_position*height)), (0xFF, 0, 0), 5)

        # display count
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_np, 'Count: ' + str(counter), (10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.imshow('cumulative_object_counting', image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-v', '--video_path', type=str, default='', help='Path to video. If None camera will be used')
    parser.add_argument('-t', '--threshold', type=int, default=0.5, help='Detection threshold')
    parser.add_argument('-roi', '--roi_position', type=int, default=0.6, help='ROI Position (0-1)')
    parser.add_argument('-d', '--deviation', type=int, default=0.005, help='Deviation (0-1)')
    parser.add_argument('-la', '--labels', nargs='+', type=str, help='Label names to detect (default="all-labels")')
    parser.add_argument('-a', '--axis', default=True, action="store_false", help='Axis for cumulative counting (default=x axis)')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    if args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")

    run_inference(detection_model, category_index, cap, labels=args.labels, threshold=args.threshold, roi_position=args.roi_position, deviation=args.deviation, x_axis=args.axis) 