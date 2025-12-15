"""
Data Logging and Visualization Module
Handles emotion data logging and real-time visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit


class EmotionDataLogger:
    """
    Logs emotion detection data and provides visualization capabilities.
    """
    
    def __init__(self):
        """Initialize the data logger."""
        self.data = []
        self.df = None
        self.emotions = ['Happy', 'Sad', 'Energetic', 'Calm']
    
    def log_emotion(self, emotion, confidence, timestamp=None):
        """
        Log an emotion detection event.
        
        Args:
            emotion: Detected emotion
            confidence: Confidence score (0-1)
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.data.append({
            'timestamp': timestamp,
            'emotion': emotion,
            'confidence': confidence
        })
        
        # Update DataFrame
        self.df = pd.DataFrame(self.data)
    
    def get_emotion_counts(self):
        """
        Get count of each emotion detected.
        
        Returns:
            dict: Emotion counts
        """
        if self.df is None or len(self.df) == 0:
            return {emotion: 0 for emotion in self.emotions}
        
        counts = self.df['emotion'].value_counts().to_dict()
        # Ensure all emotions are present
        for emotion in self.emotions:
            if emotion not in counts:
                counts[emotion] = 0
        
        return counts
    
    def get_emotion_timeline(self, window_minutes=5):
        """
        Get emotion data for the last N minutes.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            DataFrame: Filtered emotion data
        """
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - pd.Timedelta(minutes=window_minutes)
        recent_data = self.df[self.df['timestamp'] >= cutoff_time]
        return recent_data
    
    def plot_emotion_distribution(self):
        """
        Create a pie chart showing emotion distribution.
        
        Returns:
            matplotlib figure
        """
        counts = self.get_emotion_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Filter out zero counts
        filtered_counts = {k: v for k, v in counts.items() if v > 0}
        
        if len(filtered_counts) == 0:
            ax.text(0.5, 0.5, 'No data yet', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Emotion Distribution')
            return fig
        
        colors = {
            'Happy': '#FFD700',      # Gold
            'Sad': '#4169E1',        # Royal Blue
            'Energetic': '#FF4500',  # Orange Red
            'Calm': '#32CD32'        # Lime Green
        }
        
        emotion_list = list(filtered_counts.keys())
        count_list = [filtered_counts[e] for e in emotion_list]
        color_list = [colors.get(e, '#808080') for e in emotion_list]
        
        ax.pie(count_list, labels=emotion_list, autopct='%1.1f%%',
               colors=color_list, startangle=90)
        ax.set_title('Emotion Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_emotion_timeline(self, window_minutes=5):
        """
        Create a line chart showing emotion changes over time.
        
        Args:
            window_minutes: Time window to display
            
        Returns:
            matplotlib figure
        """
        recent_data = self.get_emotion_timeline(window_minutes)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(recent_data) == 0:
            ax.text(0.5, 0.5, 'No data in the last {} minutes'.format(window_minutes),
                   ha='center', va='center', fontsize=14)
            ax.set_title('Emotion Timeline')
            return fig
        
        # Create time series for each emotion
        for emotion in self.emotions:
            emotion_data = recent_data[recent_data['emotion'] == emotion]
            if len(emotion_data) > 0:
                ax.scatter(emotion_data['timestamp'], 
                          [emotion] * len(emotion_data),
                          label=emotion, s=50, alpha=0.6)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Emotion', fontsize=12)
        ax.set_title(f'Emotion Timeline (Last {window_minutes} minutes)', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_confidence_trend(self, window_minutes=5):
        """
        Create a line chart showing confidence scores over time.
        
        Args:
            window_minutes: Time window to display
            
        Returns:
            matplotlib figure
        """
        recent_data = self.get_emotion_timeline(window_minutes)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(recent_data) == 0:
            ax.text(0.5, 0.5, 'No data in the last {} minutes'.format(window_minutes),
                   ha='center', va='center', fontsize=14)
            ax.set_title('Confidence Trend')
            return fig
        
        # Plot confidence for each emotion
        for emotion in self.emotions:
            emotion_data = recent_data[recent_data['emotion'] == emotion]
            if len(emotion_data) > 0:
                ax.plot(emotion_data['timestamp'], 
                       emotion_data['confidence'],
                       marker='o', label=emotion, linewidth=2, markersize=4)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title(f'Confidence Trend (Last {window_minutes} minutes)',
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def get_statistics(self):
        """
        Get summary statistics about detected emotions.
        
        Returns:
            dict: Statistics dictionary
        """
        if self.df is None or len(self.df) == 0:
            return {
                'total_detections': 0,
                'average_confidence': 0.0,
                'most_common_emotion': 'N/A',
                'emotion_counts': {}
            }
        
        stats = {
            'total_detections': len(self.df),
            'average_confidence': self.df['confidence'].mean(),
            'most_common_emotion': self.df['emotion'].mode()[0] if len(self.df['emotion'].mode()) > 0 else 'N/A',
            'emotion_counts': self.get_emotion_counts()
        }
        
        return stats
    
    def clear_data(self):
        """Clear all logged data."""
        self.data = []
        self.df = None
    
    def export_data(self, filename='emotion_data.csv'):
        """
        Export logged data to CSV file.
        
        Args:
            filename: Output filename
        """
        if self.df is not None and len(self.df) > 0:
            self.df.to_csv(filename, index=False)
            print(f"Data exported to {filename}")
        else:
            print("No data to export")

