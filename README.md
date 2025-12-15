# AI Mood-Based Music Generator from Facial Expressions

An AI-powered system that detects a user's mood from their facial expressions in real-time and generates/plays a music playlist that matches their current emotional state.

## 🎯 Project Overview

This project combines computer vision, deep learning, and audio playback to create an interactive system that:
- Detects real-time facial expressions using a laptop camera
- Classifies emotions into four categories: **Happy**, **Sad**, **Energetic**, **Calm**
- Plays music corresponding to the detected emotion automatically
- Tracks and visualizes mood changes over time
- Implements a modular, robust, and user-friendly system with proper exception handling

## 👥 Authors

- **Emaan Swati** (2024156)
- **Rabia Ashraf** (2024527)

**Course:** AI201L - Programming for AI Lab  
**Instructor:** Aamir Khan Maroofi

## 🛠️ Tools & Libraries

- **Streamlit** - Web interface
- **OpenCV** - Face detection and image processing
- **PyTorch** - Deep learning model for emotion recognition
- **NumPy** - Numerical operations
- **Pandas** - Data logging and analysis
- **Matplotlib** - Data visualization
- **Pygame** - Audio playback
- **Pillow** - Image processing

## 📁 Project Structure

```
AI-201/
│
├── app.py                 # Main Streamlit application
├── emotion_model.py       # PyTorch CNN model for emotion recognition
├── face_detector.py       # Face detection and emotion recognition system
├── music_player.py        # Music playback module
├── data_logger.py         # Data logging and visualization
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
└── music/                # Music directory (create this)
    ├── Happy/            # Happy emotion music files
    ├── Sad/              # Sad emotion music files
    ├── Energetic/        # Energetic emotion music files
    └── Calm/             # Calm emotion music files
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Webcam/camera
- Internet connection (for initial setup)

### Step 1: Clone or Download the Project

```bash
cd AI-201
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Music Directory Structure

```bash
# Windows
mkdir music
mkdir music\Happy
mkdir music\Sad
mkdir music\Energetic
mkdir music\Calm

# Linux/Mac
mkdir -p music/{Happy,Sad,Energetic,Calm}
```

### Step 5: Add Music Files

Add your music files (`.mp3`, `.wav`, or `.ogg` format) to the respective emotion folders:
- `music/Happy/` - Upbeat, cheerful songs
- `music/Sad/` - Melancholic, emotional songs
- `music/Energetic/` - High-energy, fast-paced songs
- `music/Calm/` - Relaxing, peaceful songs

**Note:** The system will work without music files, but music playback will be disabled.

## 🎮 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Start Camera**: Click the "📷 Start Camera" button in the sidebar
2. **Position Yourself**: Face the camera and ensure good lighting
3. **Enable Music**: Check "Enable Music Playback" to automatically play music based on detected emotions
4. **View Analytics**: Check "Show Charts" to see emotion distribution and trends
5. **Export Data**: Click "💾 Export Data" to save emotion logs as CSV

### Features

- **Real-time Emotion Detection**: See your current emotion displayed with confidence score
- **Automatic Music Playback**: Music changes automatically when your mood changes
- **Data Visualization**: 
  - Pie chart showing emotion distribution
  - Timeline showing emotion changes over time
  - Confidence trend analysis
- **Data Export**: Export all logged emotions to CSV for further analysis

## 🧠 Model Architecture

The emotion recognition model uses a Convolutional Neural Network (CNN) with:
- **Input**: 48x48 grayscale face images
- **Architecture**: 
  - 3 convolutional blocks with batch normalization
  - Max pooling and dropout for regularization
  - Fully connected layers for classification
- **Output**: 4 emotion classes (Happy, Sad, Energetic, Calm)

**Note**: The current implementation uses an untrained model for demonstration. For production use, you should:
1. Collect a dataset of labeled facial expressions
2. Train the model on this dataset
3. Save the trained weights
4. Load the trained model in `EmotionRecognizer`

## 🔧 Customization

### Adding a Pre-trained Model

To use a pre-trained model:

1. Save your trained model weights
2. Update `app.py` initialization:
```python
emotion_system = EmotionDetectionSystem(model_path="path/to/your/model.pth")
```

### Modifying Emotion Classes

To change emotion categories:

1. Update `emotion_labels` in `emotion_model.py`
2. Update `emotions` list in `data_logger.py`
3. Update `playlists` dictionary in `music_player.py`
4. Add corresponding music folders

### Adjusting Detection Sensitivity

Modify face detection parameters in `face_detector.py`:
```python
faces = self.face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,      # Adjust for detection sensitivity
    minNeighbors=5,       # Higher = fewer false positives
    minSize=(30, 30)      # Minimum face size
)
```

## ⚠️ Exception Handling

The system includes comprehensive error handling for:
- Camera access failures
- Face detection errors
- Model loading issues
- Music file missing/playback errors
- Data logging exceptions

All errors are caught and displayed to the user without crashing the application.

## 📊 Data Logging

The system logs:
- Timestamp of each detection
- Detected emotion
- Confidence score

Data can be:
- Viewed in real-time charts
- Exported to CSV
- Cleared for fresh start

## 🎵 Music Playback

- Supports `.mp3`, `.wav`, and `.ogg` formats
- Automatically loops tracks
- Switches tracks when emotion changes
- Can be paused/stopped manually

## 🐛 Troubleshooting

### Camera Not Working
- Check camera permissions
- Ensure no other application is using the camera
- Try a different camera index in `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`

### Music Not Playing
- Verify music files are in correct folders
- Check file formats (`.mp3`, `.wav`, `.ogg`)
- Ensure pygame is properly installed
- Check console for error messages

### Model Not Loading
- The system works with untrained models (random predictions)
- For better accuracy, train and save a model
- Check model path is correct

### Face Not Detected
- Ensure good lighting
- Face the camera directly
- Remove obstructions (glasses, masks)
- Adjust camera distance

## 📚 References

- [OpenCV Documentation](https://opencv.org)
- [PyTorch Documentation](https://pytorch.org)
- [NumPy Documentation](https://numpy.org)
- [Pandas Documentation](https://pandas.pydata.org)
- [Matplotlib Documentation](https://matplotlib.org)
- [Streamlit Documentation](https://streamlit.io)

## 📝 License

This project is created for educational purposes as part of AI201L Programming for AI Lab.

## 🙏 Acknowledgments

- Course Instructor: Aamir Khan Maroofi
- OpenCV community for face detection algorithms
- PyTorch team for deep learning framework

---

**Note**: This is a demonstration project. For production use, ensure proper model training, data privacy considerations, and performance optimization.
