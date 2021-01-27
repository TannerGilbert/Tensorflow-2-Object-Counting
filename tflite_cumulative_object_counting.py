from tflite_runtime.interpreter import Interpreter, load_delegate
import argparse
import time
import cv2
import re
import numpy as np


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


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
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


def cumulative_counting(image, predictions, labelmap, labels, roi_position, deviation, threshold, x_axis):
    if x_axis:
        return cumulative_counting_x_axis(image, predictions, labelmap, labels, roi_position, deviation, threshold)
    else:
        return cumulative_counting_y_axis(image, predictions, labelmap, labels, roi_position, deviation, threshold)


def cumulative_counting_x_axis(image, predictions, labelmap, labels, roi_position, deviation, threshold):
    directions = []
    for obj in predictions:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        if obj['score'] > threshold and (labels == None or labelmap[obj['class_id']] in labels):
            if abs(((xmin+xmax)/2)-roi_position) < deviation:
                directions.append(((xmin+xmax)/2)-roi_position > 0)

    return directions


def cumulative_counting_y_axis(image, predictions, labelmap, labels, roi_position, deviation, threshold):
    directions = []
    for obj in predictions:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        if obj['score'] > threshold and (labels == None or labelmap[obj['class_id']] in labels):
            if abs(((ymin+ymax)/2)-roi_position) < deviation:
                directions.append(((ymin+ymax)/2)-roi_position > 0)

    return directions


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', type=str, required=True, help='File path of .tflite file.')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='File path of labels file.')
    parser.add_argument('-v', '--video_path', type=str, default='', help='Path to video. If None camera will be used')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('-roi', '--roi_position', type=float, default=0.6, help='ROI Position (0-1)')
    parser.add_argument('-d', '--deviation', type=float, default=0.005, help='Deviation (0-1)')
    parser.add_argument('-la', '--labels', nargs='+', type=str, help='Label names to detect (default="all-labels")')
    parser.add_argument('-a', '--axis', default=True, action="store_false", help='Axis for cumulative counting (default=x axis)')
    parser.add_argument('-e', '--use_edgetpu', action='store_true', default=False, help='Use EdgeTPU')
    args = parser.parse_args()

    labelmap = load_labels(args.labelmap)
    roi_position = args.roi_position
    interpreter = make_interpreter(args.model, args.use_edgetpu)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    if args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)
    counter = 0

    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break

        height, width, _ = image_np.shape

        image_pred = cv2.resize(image_np, (input_width ,input_height))

        # Perform inference
        results = detect_objects(interpreter, image_pred, args.threshold)

        # Count objects
        directions = cumulative_counting(image_np, results, labelmap, args.labels, args.roi_position, args.deviation, args.threshold, args.axis)
                
        for idx, obj in enumerate(results):
            # Prepare bounding box
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            image_np = cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (36,255,12), 2)

            # Annotate image with label and confidence score
            display_str = labelmap[obj['class_id']] + ": " + str(round(obj['score']*100, 2)) + "%"
            cv2.putText(image_np, display_str, (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        # Draw ROI line
        if len(directions) != 0:
            counter += len(directions)
            if args.axis:
                cv2.line(image_np, (int(roi_position*width), 0), (int(roi_position*width), height), (0, 0xFF, 0), 5)
            else:
                cv2.line(image_np, (0, int(roi_position*height)), (width, int(roi_position*height)), (0, 0xFF, 0), 5)
        else:
            if args.axis:
                cv2.line(image_np, (int(roi_position*width), 0), (int(roi_position*width), height), (0xFF, 0, 0), 5)
            else:
                cv2.line(image_np, (0, int(roi_position*height)), (width, int(roi_position*height)), (0xFF, 0, 0), 5)

        # display count
        cv2.putText(image_np, 'Count: ' + str(counter), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0xFF, 0xFF), 2)

        cv2.imshow('TFLITE Object Counting', image_np)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()