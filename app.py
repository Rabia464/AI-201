"""
Main Streamlit Application
AI Mood-Based Music Generator from Facial Expressions
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime

from face_detector import EmotionDetectionSystem
from music_player import MusicPlayer
from data_logger import EmotionDataLogger


# Page configuration
st.set_page_config(
    page_title="AI Mood-Based Music Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-display {
        font-size: 3rem;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .happy { background-color: #FFF9C4; }
    .sad { background-color: #BBDEFB; }
    .energetic { background-color: #FFE0B2; }
    .calm { background-color: #C8E6C9; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_systems():
    """Initialize all systems (cached to avoid re-initialization)."""
    try:
        emotion_system = EmotionDetectionSystem(model_path=None)
        music_player = MusicPlayer(music_dir="music", mood_mode="change")  # Default to change mode
        data_logger = EmotionDataLogger()
        return emotion_system, music_player, data_logger
    except Exception as e:
        st.error(f"Error initializing systems: {e}")
        return None, None, None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎵 AI Mood-Based Music Generator</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Real-time Facial Expression Detection & Music Playback")
    
    # Initialize systems
    emotion_system, music_player, data_logger = initialize_systems()
    
    if emotion_system is None:
        st.error("Failed to initialize systems. Please check your setup.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Camera controls
        st.subheader("Camera")
        start_camera = st.button("📷 Start Camera", type="primary")
        stop_camera = st.button("⏹️ Stop Camera")
        
        # Music controls
        st.subheader("Music")
        enable_music = st.checkbox("Enable Music Playback", value=True)
        
        # Mood mode selection
        mood_mode = st.radio(
            "Music Mode:",
            ["Match Mood", "Change Mood"],
            index=1,  # Default to "Change Mood"
            help="Match Mood: Plays music matching your emotion\nChange Mood: Plays positive music to improve your mood (e.g., happy music when sad)"
        )
        
        # Update music player mode
        if music_player:
            mode_value = "change" if mood_mode == "Change Mood" else "match"
            music_player.set_mood_mode(mode_value)
        
        stop_music_btn = st.button("⏹️ Stop Music")
        
        # Visualization controls
        st.subheader("Visualization")
        show_charts = st.checkbox("Show Charts", value=True)
        chart_window = st.slider("Chart Time Window (minutes)", 1, 30, 5)
        show_debug = st.checkbox("Show Detection Debug", value=False,
                                 help="Display raw feature values driving the rule-based detector")
        
        # Data controls
        st.subheader("Data")
        clear_data_btn = st.button("🗑️ Clear Data")
        export_data_btn = st.button("💾 Export Data")
        
        # Statistics
        st.subheader("📊 Statistics")
        if data_logger.df is not None and len(data_logger.df) > 0:
            stats = data_logger.get_statistics()
            st.metric("Total Detections", stats['total_detections'])
            st.metric("Avg Confidence", f"{stats['average_confidence']:.2f}")
            st.metric("Most Common", stats['most_common_emotion'])
        else:
            st.info("No data collected yet")
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'frame_placeholder' not in st.session_state:
        st.session_state.frame_placeholder = None
    if 'last_emotion' not in st.session_state:
        st.session_state.last_emotion = None
    if 'detected_emotion' not in st.session_state:
        st.session_state.detected_emotion = None
    
    # Handle button clicks
    if start_camera:
        st.session_state.camera_active = True
    
    if stop_camera:
        st.session_state.camera_active = False
        if music_player:
            music_player.stop_music()
    
    if stop_music_btn:
        if music_player:
            music_player.stop_music()
    
    if clear_data_btn:
        data_logger.clear_data()
        st.success("Data cleared!")
        st.rerun()
    
    if export_data_btn:
        filename = f"emotion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        data_logger.export_data(filename)
        st.success(f"Data exported to {filename}!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        
        if st.session_state.camera_active:
            # Create placeholder for video frame
            frame_placeholder = st.empty()
            
            # Initialize camera
            try:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not open camera. Please check your camera connection.")
                    st.session_state.camera_active = False
                else:
                    st.info("🎥 Camera is active. Position your face in front of the camera.")
                    
                    # Process frames
                    while st.session_state.camera_active:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("Failed to read frame from camera")
                            break
                        
                        # Process frame for emotion detection
                        processed_frame, emotion, confidence, face_box = \
                            emotion_system.process_frame(frame)
                        
                        # Log emotion
                        if emotion and emotion != "No Face Detected" and emotion != "Unknown":
                            data_logger.log_emotion(emotion, confidence)
                            st.session_state.detected_emotion = emotion
                            
                            # Play music if enabled and emotion changed
                            if enable_music and emotion != st.session_state.last_emotion:
                                music_player.play_emotion_music(emotion)
                                st.session_state.last_emotion = emotion
                        
                        # Convert BGR to RGB for display
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(0.1)
                    
                    # Release camera
                    cap.release()
                    frame_placeholder.empty()
                    st.info("Camera stopped.")
            
            except Exception as e:
                st.error(f"Error with camera: {e}")
                st.session_state.camera_active = False
        
        else:
            st.info("👆 Click 'Start Camera' to begin emotion detection")
    
    with col2:
        st.subheader("😊 Current Emotion")
        
        # Display current emotion
        if emotion_system:
            current_emotion, current_confidence = emotion_system.get_current_emotion()
            
            if current_emotion and current_emotion != "Unknown":
                emotion_class = current_emotion.lower()
                st.markdown(
                    f'<div class="emotion-display {emotion_class}">'
                    f'<h2>{current_emotion}</h2>'
                    f'<p>Confidence: {current_confidence:.2%}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Waiting for face detection...")

            if show_debug:
                with st.expander("Detection debug"):
                    feat = emotion_system.get_last_features() if hasattr(emotion_system, "get_last_features") else {}
                    if feat:
                        st.json(feat)
                    else:
                        st.caption("No features yet. Start camera to see values.")
            
            # Music status
            st.subheader("🎵 Music Status")
            if music_player:
                mode_display = "Change Mood" if music_player.mood_mode == "change" else "Match Mood"
                st.caption(f"Mode: {mode_display}")
                
                if music_player.is_playing():
                    st.success(f"▶️ Playing: {music_player.get_current_track()}")
                    if music_player.mood_mode == "change":
                        detected = st.session_state.detected_emotion or emotion_system.get_current_emotion()[0]
                        playing = music_player.current_emotion
                        if detected and detected != playing and detected != "Unknown" and detected != "No Face Detected":
                            st.info(f"🎭 Detected: {detected} → Playing: {playing} (mood boost! ✨)")
                        else:
                            st.info(f"Emotion: {playing}")
                    else:
                        st.info(f"Emotion: {music_player.current_emotion}")
                else:
                    st.info("⏸️ No music playing")
            
            # Playlist info
            if music_player:
                st.subheader("📁 Playlist Info")
                playlist_info = music_player.get_playlist_info()
                for emotion, count in playlist_info.items():
                    st.metric(emotion, count)
    
    # Charts section
    if show_charts and data_logger.df is not None and len(data_logger.df) > 0:
        st.markdown("---")
        st.subheader("📊 Emotion Analytics")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Emotion Distribution")
            fig_dist = data_logger.plot_emotion_distribution()
            st.pyplot(fig_dist)
        
        with chart_col2:
            st.subheader("Confidence Trend")
            fig_conf = data_logger.plot_confidence_trend(window_minutes=chart_window)
            st.pyplot(fig_conf)
        
        st.subheader("Emotion Timeline")
        fig_timeline = data_logger.plot_emotion_timeline(window_minutes=chart_window)
        st.pyplot(fig_timeline)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>AI Mood-Based Music Generator | AI201L Programming for AI Lab</p>
        <p>Developed by: Emaan Swati (2024156) & Rabia Ashraf (2024527)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

