import onnxruntime
import numpy as np
import os

def run_onnx_inference(model_path, input_array):
    """
    ONNXモデルを実行して3つの出力を取得する関数
    
    Args:
        model_path (str): ONNXモデルファイルのパス
        input_array (numpy.ndarray): 入力画像データ
        
    Returns:
        tuple: 3つの出力テンソル
    """
    # セッションの初期化
    session = onnxruntime.InferenceSession(model_path)
    
    # 入力名と出力名の設定
    input_names = ["images"]
    output_names = [
        "/model.23/Concat_output_0",
        "/model.23/Concat_1_output_0",
        "/model.23/Concat_2_output_0"
    ]
    
    # 推論の実行
    outputs = session.run(
        output_names,
        {input_names[0]: input_array}
    )
    
    return outputs

# 使用例
if __name__ == "__main__":
    # モデルパスの設定
    model_path = "yolo11n_cut_backbone.onnx"
    
    # サンプル入力データの作成 (例: バッチサイズ1、3チャンネル、640x640の画像)
    sample_input = np.random.random((1, 3, 640, 640)).astype(np.float32)
    
    # モデルの実行
    outputs = run_onnx_inference(model_path, sample_input)
    
    # 各出力の形状を表示
    for i, output in enumerate(outputs):
        print(f"Output {i} shape:", output.shape)


    save_dir = "model_outputs"
# 追加した保存機能部分
    # 出力の保存
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, output in enumerate(outputs):
            save_path = os.path.join(save_dir, f"output_{i}.npy")
            np.save(save_path, output)
            print(f"Saved output {i} to {save_path}")


