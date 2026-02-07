Air MNIST – Touchless Gesture-Based Digit Drawing & Recognition

Real-time hand-gesture driven drawing system with deep-learning based digit recognition.

Highlights

Inputs: Live webcam feed, hand landmark trajectories
Interaction: Mid-air drawing using index finger; left-thumb gesture to clear canvas
Model: CNN trained on MNIST-style spatial representations
Processing: Temporal hand trajectories → stabilized 2D drawing space
Output: Real-time digit prediction with visual feedback
Stack: Python, OpenCV, MediaPipe, TensorFlow/Keras, NumPy
Performance: Low-latency real-time interaction; robust to noisy hand motion

Project Structure

Refer to README.md for setup instructions and system overview.
See docs/ for model details, preprocessing pipeline, and design notes.

Local Run (summary)

Environment: Python 3.x
Main entry: main.py
Hand tracking: MediaPipe + OpenCV
Model: Trained CNN (loaded at runtime)
Input: Webcam

Why this project

Air MNIST was built to explore end-to-end machine learning system design beyond just model training. While the CNN handled classification, the core challenge was capturing noisy human motion, tracking hand landmarks reliably, and converting unstable temporal trajectories into clean spatial representations suitable for learning.

The project introduces a gesture-driven interaction layer that supports both continuous and discontinuous drawing without physical contact, along with intuitive control gestures for seamless user experience. It highlights the importance of data pipelines, latency, robustness, and interaction design in real-time ML applications.

Documentation

Preprocessing & trajectory stabilization: docs/preprocessing.md
Model architecture & training notes: docs/model.md
Design decisions & limitations: docs/design_notes.md

License

MIT License
© Air MNIST Project


