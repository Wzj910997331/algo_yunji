import onnxruntime
import numpy as np
import cv2
import torchvision

def preprocess_image(image_path, img_size=640):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (img_size, img_size))
    image_transposed = image_resized.transpose(2, 0, 1)  # HWC to CHW
    image_normalized = image_transposed[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return image, image_normalized

def post_process(output_80, output_40, output_20, conf_threshold=0.25, iou_threshold=0.45):
    outputs = [output_80, output_40, output_20]

    print(f'Shape of output_80: {output_80.shape}')
    
    boxes = []
    scores = []
    class_ids = []

    for output in outputs:
        output = output[0]
        grid_size = output.shape[1]
        output = output.transpose(1, 2, 0).reshape(-1, 21)
        
        conf = output[:, 4]
        mask = conf >= conf_threshold
        output = output[mask]
        
        if len(output) == 0:
            continue
        
        box = output[:, :4]
        score = output[:, 4] * output[:, 5:].max(axis=1)
        class_id = output[:, 5:].argmax(axis=1)
        
        boxes.append(box)
        scores.append(score)
        class_ids.append(class_id)
    
    if len(boxes) == 0:
        return [], [], []
    
    boxes = np.concatenate(boxes, axis=0)
    scores = np.concatenate(scores, axis=0)
    class_ids = np.concatenate(class_ids, axis=0)

    indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)
    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]
    
    return boxes, scores, class_ids

def draw_boxes(image, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class: {class_id} Conf: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main(image_path, onnx_model_path, output_image_path):
    original_image, preprocessed_image = preprocess_image(image_path)

    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed_image})
    output_80, output_40, output_20 = outputs
    print(f'Shape of output_80: {output_80.shape}')
    boxes, scores, class_ids = post_process(output_80, output_40, output_20)

    result_image = draw_boxes(original_image, boxes, scores, class_ids)
    cv2.imwrite(output_image_path, result_image)

if __name__ == "__main__":
    image_path = '1.jpeg'  # 替换为你自己的图像路径
    onnx_model_path = 'fire.onnx'  # 替换为你的ONNX模型路径
    output_image_path = 'result_image.jpg'  # 替换为你希望保存结果的路径
    main(image_path, onnx_model_path, output_image_path)
