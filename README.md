# Air MNIST – Touchless Gesture-Based Digit Drawing & Recognition

A real-time, touchless drawing system that allows users to draw digits in mid-air using hand gestures and recognizes them using a CNN trained on MNIST-style representations.

No screen. No pen. Just air.

---

## Highlights

- **Input:** Live webcam feed with hand landmark trajectories  
- **Interaction:**  
  - Index finger → draw in mid-air  
  - Left-thumb gesture → clear canvas  
- **Model:** Convolutional Neural Network (CNN)  
- **Processing:** Noisy temporal hand trajectories → stabilized 2D spatial drawings  
- **Output:** Real-time digit prediction with visual feedback  
- **Performance:** Low-latency, real-time interaction  

---

## Tech Stack

- Python  
- OpenCV  
- MediaPipe (hand landmark tracking)  
- TensorFlow / Keras  
- NumPy  

---

## How It Works

- **Hand Tracking:** MediaPipe detects and tracks hand landmarks from a live webcam feed  
- **Gesture Input:** Index finger motion is used for mid-air drawing  
- **Stabilization:** Noisy temporal trajectories are smoothed and normalized  
- **Canvas Mapping:** Hand coordinates are projected onto a fixed 2D drawing space  
- **Inference:** Processed drawing is passed to a CNN for digit prediction  
- **Control:** Left-thumb gesture clears the canvas instantly  

---

## Why This Project

- Explore end-to-end ML systems beyond just model training  
- Handle real-world challenges like noisy human motion and unstable inputs  
- Design intuitive, touchless human–computer interaction  
- Balance accuracy, latency, and usability in a real-time system  

---

## Documentation

- **Preprocessing & smoothing:** `docs/preprocessing.md`  
- **Model architecture & training:** `docs/model.md`  
- **System design & limitations:** `docs/design_notes.md`  

---

## License

- MIT License  
- © Air MNIST Project



