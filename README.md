onnx2json -if  yolo11n_post_without_weights.onnx -oj  yolo11n_post_without_weights.json -i 4
onnx2json -if  yolov8n_post_without_weights.onnx -oj  yolov8n_post_without_weights.json -i 4
onnx2json -if  yolov9t_post_without_weights.onnx -oj  yolov9t_post_without_weights.json -i 4
onnx2json -if  yolov10n_post_without_weights.onnx -oj  yolov10n_post_without_weights.json -i 4


 python onnx_clear_wight.py yolo11n_post.onnx
 python onnx_clear_wight.py yolov8n_post.onnx
 python onnx_clear_wight.py yolov9t_post.onnx
 python onnx_clear_wight.py yolov10n_post.onnx
