fn main() {
    println!("cargo:rerun-if-changed=assets/models/mobilefacenet.onnx");
    println!("cargo:rerun-if-changed=assets/models/hand_landmarker_two_hands.onnx");
    println!("cargo:rerun-if-changed=assets/models/paddle_ocr.onnx");
    println!("cargo:rerun-if-changed=assets/models/latex_ocr.onnx");
}
