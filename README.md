# Real-Time Emotion Recognition System

A lightweight PyTorch implementation of Mini-Xception for real-time emotion detection. Perfect for AI assistants, space applications, and multimodal systems.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage

#### Real-Time Emotion Stream (MVP)
```bash
# Simple emotion output
python emotion_stream.py --format simple

# JSON output for backend integration
python emotion_stream.py --format json

# With video window (for debugging)
python emotion_stream.py --format json --show-video
```

#### Visual Testing Demo
```bash
# Original demo with video window
python camera_demo.py

# Use Haar cascade detector instead of DNN
python camera_demo.py --haar
```

## 📊 Output Examples

**Simple Format:**
```
Emotion: Happy (Confidence: 0.849)
Emotion: Neutral (Confidence: 0.756)
```

**JSON Format:**
```json
{
  "emotion": "Happy",
  "confidence": 0.849,
  "probabilities": {
    "Angry": 0.05,
    "Disgust": 0.02,
    "Fear": 0.01,
    "Happy": 0.85,
    "Sad": 0.03,
    "Surprise": 0.02,
    "Neutral": 0.02
  },
  "face_detected": true,
  "timestamp": 1704112200.123
}
```

## 🎯 Features

- **Real-time processing**: ~17 FPS emotion detection
- **Lightweight**: Optimized Mini-Xception model (~268KB weights)
- **Space-ready**: CPU-compatible, minimal dependencies
- **Two interfaces**: MVP stream + visual demo
- **Multiple detectors**: DNN and Haar cascade face detection

## 📁 Project Structure

```
├── emotion_stream.py          # 🎯 Main MVP - JSON/simple output
├── camera_demo.py            # 🎥 Visual testing - video window
├── model/                    # Neural network implementation
├── face_detector/           # Face detection models
├── face_alignment/          # Face alignment and preprocessing
└── checkpoint/model_weights/ # Pre-trained model weights
```

## 🔧 Command Options

```bash
# Use Haar cascade detector (alternative to DNN)
python emotion_stream.py --format json --haar

# Use custom model weights
python emotion_stream.py --format json --pretrained path/to/weights.pth.tar

# Show help
python emotion_stream.py --help
```

## 🌟 Use Cases

- **AI Assistants**: Emotion-aware conversational systems
- **Space Applications**: Astronaut mental health monitoring
- **Multimodal Systems**: Integration with audio and text processing
- **Real-time Analytics**: Live emotion tracking and analysis

## 📚 References

- [Mini-Xception Paper](https://arxiv.org/pdf/1710.07557.pdf)
- [FER2013 Dataset](https://www.kaggle.com/deadskull7/fer2013)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.