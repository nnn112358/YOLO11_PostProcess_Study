import cv2
import time
import yaml
import onnxruntime
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Detection:
    class_index: int
    confidence: float
    box: np.ndarray
    class_name: str

class YOLOv9:
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(
        self,
        model_path: str,
        original_size: Tuple[int, int] = (640, 640),
        score_threshold: float = 0.1,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.4,
        device: str = "CPU"
    ) -> None:
        """
        YOLOv9検出器を初期化します。
        
        引数:
            model_path: ONNXモデルファイルへのパス
            class_mapping_path: クラスマッピングYAMLファイルへのパス
            original_size: 元の画像サイズ (width, height)
            score_threshold: 物体検出スコアの閾値
            conf_threshold: 信頼度スコアの閾値
            iou_threshold: NMSにおけるIoUの閾値
            device: 推論を実行するデバイス ("CPU" または "CUDA")
        """
        self.model_path = Path(model_path)
        self.device = device.upper()
        self.score_threshold = score_threshold
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_width, self.image_height = original_size
        
        # モデルセッションの初期化
        self._create_session()
        
        # 高速処理のための画像サイズ配列を事前計算
        self.input_shape_array = np.array([self.input_width, self.input_height, 
                                         self.input_width, self.input_height])
        self.output_shape_array = np.array([self.image_width, self.image_height,
                                          self.image_width, self.image_height])
        
        # カラーパレットを一度だけ生成
        self.color_palette = np.random.uniform(0, 255, size=(len(self.COCO_CLASSES), 3))

    def _create_session(self) -> None:
        """ONNXランタイム推論セッションを作成します。"""
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        if self.device != "CPU":
            providers.insert(0, "CUDAExecutionProvider")
            
        self.session = onnxruntime.InferenceSession(
            str(self.model_path), 
            providers=providers,
            sess_options=opt_session
        )
        
        # モデルのプロパティをキャッシュ
        self.model_inputs = self.session.get_inputs()
        self.input_names = [input.name for input in self.model_inputs]
        self.input_shape = self.model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2:]
        
        self.output_names = [output.name for output in self.session.get_outputs()]

    @staticmethod
    def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
        """ボックスをxywh形式からxyxy形式に変換します。"""
        xyxy = np.copy(boxes)
        xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
        xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
        xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
        xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
        return xyxy

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """推論のための画像前処理を行います。"""
        # より効率的な色変換を使用
        if len(img.shape) == 3 and img.shape[2] == 3:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = img
            
        # 高速なリサイズにINTER_LINEARを使用
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height),
                           interpolation=cv2.INTER_LINEAR)
        
        # 正規化とトランスポーズを1ステップで実行
        input_tensor = resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
        return input_tensor

    def postprocess(self, outputs: np.ndarray) -> List[Detection]:
        """ネットワーク出力の後処理を行います。"""
        predictions = np.squeeze(outputs).T
        
        # 信頼度閾値でフィルタリング
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        
        if len(scores) == 0:
            return []
            
        # クラスIDの取得
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # ボックスのスケール変換
        boxes = predictions[:, :4]
        boxes = np.divide(boxes, self.input_shape_array, dtype=np.float32)
        boxes *= self.output_shape_array
        boxes = boxes.astype(np.int32)
        
        # NMSの適用
        indices = cv2.dnn.NMSBoxes(
            boxes, scores,
            score_threshold=self.score_threshold,
            nms_threshold=self.iou_threshold
        )
        
        # 検出オブジェクトの作成
        return [
            Detection(
                class_index=class_ids[i],
                confidence=scores[i],
                box=self._xywh2xyxy(boxes[i:i+1])[0],
                class_name=self.COCO_CLASSES[class_ids[i]]
            )
            for i in indices
        ]

    def detect(self, img: np.ndarray) -> List[Detection]:
        """入力画像に対して検出を実行します。"""
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]



        return self.postprocess(outputs)




    def draw_detections(self, img: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """検出結果を画像に描画します。"""
        img_copy = img.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.box.astype(int)
            color = self.color_palette[det.class_index].astype(int)
            
            # ボックスの描画
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color.tolist(), 2)
            
            # ラベルの準備
            label = f"{det.class_name}: {det.confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # ラベルの背景とテキストの描画
            label_y = max(y1 - 10, label_height)
            cv2.rectangle(
                img_copy,
                (x1, label_y - label_height),
                (x1 + label_width, label_y + 5),
                color.tolist(),
                -1
            )
            cv2.putText(
                img_copy, label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1,
                cv2.LINE_AA
            )
            
            # 検出結果をコンソールに表示
            print(f"検出: {det.class_name}, 信頼度: {det.confidence:.2f}, 位置: ({x1}, {y1}, {x2}, {y2})")
            
        return img_copy


if __name__ == "__main__":
    weight_path = "yolo11n.onnx"
    image_path = "./test_m5.jpg"
    
    # 検出器の読み込みと初期化
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {image_path}")
        
    h, w = image.shape[:2]
    detector = YOLOv9(
        model_path=weight_path,
        original_size=(w, h)
    )
    
    # 検出の実行
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = time.time() - start_time
    print(f"推論時間: {inference_time:.3f} 秒")
    



    # 結果の描画と表示
    result_image = detector.draw_detections(image, detections)
    
   # 画像を1/2サイズに縮小
    display_height = result_image.shape[0] // 2
    display_width = result_image.shape[1] // 2
    display_image = cv2.resize(result_image, (display_width, display_height))


    cv2.imshow("yolov9", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()