import onnxruntime
import numpy as np
import cv2
import torch

class Detection:
    def __init__(self, bbox, cls, score):
        self.bbox = bbox
        self.cls = cls
        self.score = score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_proposals_yolov5(stride, feat, prob_threshold, letterbox_cols, letterbox_rows, anchors, prob_threshold_unsigmoid, cls_num=2):
    anchor_num = 3
    feat_w = letterbox_cols // stride
    feat_h = letterbox_rows // stride
    anchor_group = {8: 1, 16: 2, 32: 3}[stride]
    objects = []

    index = 0
    for a in range(anchor_num):
        for h in range(feat_h):
            for w in range(feat_w):
                # 获取置信度并进行 sigmoid 变换
                score = sigmoid(feat[index + 4])
                # 得到类别得分
                class_scores = feat[index + 5:index + 5 + cls_num]
                class_index = np.argmax(class_scores)
                class_score = class_scores[class_index]

                # 计算 final_score
                final_score = sigmoid(feat[index + 4]) * sigmoid(class_score)

                # 如果最终得分高于阈值则保留框
                if final_score > prob_threshold:
                    count += 1
                    dx = sigmoid(feat[index + 0])
                    dy = sigmoid(feat[index + 1])
                    dw = sigmoid(feat[index + 2])
                    dh = sigmoid(feat[index + 3])

                    # 计算框的坐标
                    pred_cx = (dx * 2.0 - 0.5 + w) * stride
                    pred_cy = (dy * 2.0 - 0.5 + h) * stride
                    anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0]
                    anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1]
                    pred_w = (dw ** 2) * 4.0 * anchor_w
                    pred_h = (dh ** 2) * 4.0 * anchor_h
                    x0 = pred_cx - pred_w * 0.5
                    y0 = pred_cy - pred_h * 0.5
                    x1 = pred_cx + pred_w * 0.5
                    y1 = pred_cy + pred_h * 0.5

                    # 保存检测结果
                    objects.append(Detection((x0, y0, x1, y1), class_index, final_score))
                index += 7
    return objects

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        inter_area = 0

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def nms(detections, iou_threshold):
    if not detections:
        return []

    detections = sorted(detections, key=lambda det: det.score, reverse=True)
    selected_detections = []

    while detections:
        current = detections.pop(0)
        selected_detections.append(current)
        detections = [
            det for det in detections
            if compute_iou(current.bbox, det.bbox) < iou_threshold
        ]

    return selected_detections

def preprocess_image(image_path, img_size=640):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (img_size, img_size))
    # image_transposed = image_resized.transpose(2, 0, 1)  # HWC to CHW
    image_transposed = image_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    image_normalized = image_transposed[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return image, image_normalized

def post_process(outputs, anchors, img_size=640, prob_threshold=0.4, iou_threshold=0.5):
    prob_threshold_unsigmoid = -np.log((1.0 / prob_threshold) - 1.0)
    all_detections = []

    strides = [8, 16, 32]  # 对应输出的步幅
    for i, output in enumerate(outputs):
        stride = strides[i]

        detections = generate_proposals_yolov5(stride, output.flatten(), prob_threshold, 
            img_size, img_size, anchors, prob_threshold_unsigmoid, cls_num=2)
        all_detections.extend(detections)

    final_detections = nms(all_detections, iou_threshold)
    return final_detections

def draw_boxes(image, detections):
    for det in detections:
        x0, y0, x1, y1 = map(int, det.bbox)
        print(f"x0: {x0}, y0: {y0} x1: {x1}, y1: {y1}")
       
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"Class: {det.cls} Conf: {det.score:.2f}"
        cv2.putText(image, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
      
def main(image_path, onnx_model_path, output_image_path):
    original_image, preprocessed_image = preprocess_image(image_path)

    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed_image})

    output_80, output_40, output_20 = outputs
    anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    # print(output_80.shape, output_40.shape, output_20.shape)
    # print(output_80)
    detections = post_process([output_80, output_40, output_20], anchors)

    result_image = draw_boxes(original_image, detections)
    cv2.imwrite(output_image_path, result_image)

if __name__ == "__main__":
    image_path = '1.jpeg'  # 替换为你自己的图像路径
    onnx_model_path = 'best.onnx'  # 替换为你的ONNX模型路径
    output_image_path = 'result_image.jpg'  # 替换为你希望保存结果的路径
    main(image_path, onnx_model_path, output_image_path)
