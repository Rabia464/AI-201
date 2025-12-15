"""
Music Player Module
Handles mood-based music playback using pygame
"""

import pygame
import os
import random
from pathlib import Path


class MusicPlayer:
    """
    Manages music playback based on detected emotions.
    Maps emotions to playlists and handles track switching.
    """
    
    def __init__(self, music_dir="music", mood_mode="match"):
        """
        Initialize the music player.
        
        Args:
            music_dir: Directory containing emotion-based music folders
            mood_mode: "match" to match mood, "change" to change mood positively
        """
        self.music_dir = Path(music_dir)
        self.current_emotion = None
        self.current_track = None
        self.mood_mode = mood_mode  # "match" or "change"
        self.playlists = {
            'Happy': [],
            'Sad': [],
            'Energetic': [],
            'Calm': []
        }
        
        # Mood-changing mapping: if sad, play happy/energetic to cheer up
        self.mood_change_map = {
            'Sad': 'Happy',        # If sad, play happy music to cheer up
            'Happy': 'Happy',      # If happy, keep playing happy
            'Energetic': 'Energetic',  # If energetic, keep it up
            'Calm': 'Happy'        # If calm, play happy to energize
        }
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize pygame mixer: {e}")
            print("Music playback will be disabled.")
            self.initialized = False
        
        # Load playlists
        self._load_playlists()
    
    def _load_playlists(self):
        """Load music files from emotion-based folders."""
        if not self.initialized:
            return
        
        try:
            for emotion in self.playlists.keys():
                emotion_dir = self.music_dir / emotion
                if emotion_dir.exists() and emotion_dir.is_dir():
                    # Supported audio formats
                    audio_extensions = ['.mp3', '.wav', '.ogg']
                    for ext in audio_extensions:
                        self.playlists[emotion].extend(
                            list(emotion_dir.glob(f'*{ext}'))
                        )
                else:
                    # Create directory if it doesn't exist
                    emotion_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Created music directory: {emotion_dir}")
                    print(f"Please add {emotion} music files to: {emotion_dir}")
        
        except Exception as e:
            print(f"Error loading playlists: {e}")
    
    def _get_random_track(self, emotion):
        """
        Get a random track for the given emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Path object to music file or None
        """
        if emotion not in self.playlists:
            return None
        
        playlist = self.playlists[emotion]
        if len(playlist) == 0:
            return None
        
        return random.choice(playlist)
    
    def set_mood_mode(self, mode):
        """
        Set the mood mode: 'match' or 'change'
        
        Args:
            mode: 'match' to match detected mood, 'change' to change mood positively
        """
        if mode in ['match', 'change']:
            self.mood_mode = mode
            # Stop current music to apply new mode
            if self.initialized:
                try:
                    pygame.mixer.music.stop()
                    self.current_emotion = None
                except:
                    pass
    
    def play_emotion_music(self, emotion):
        """
        Play music corresponding to the detected emotion.
        Switches tracks if emotion changes.
        In 'change' mode, plays positive/energetic music to improve mood.
        
        Args:
            emotion: Detected emotion (Happy, Sad, Energetic, Calm)
            
        Returns:
            bool: True if music is playing, False otherwise
        """
        if not self.initialized:
            return False
        
        # Determine which emotion's music to play based on mode
        if self.mood_mode == "change":
            target_emotion = self.mood_change_map.get(emotion, emotion)
        else:
            target_emotion = emotion  # Match mode: play music matching detected emotion
        
        # If same target emotion, continue playing current track
        if target_emotion == self.current_emotion and pygame.mixer.music.get_busy():
            return True
        
        # Stop current music if emotion changed
        if self.current_emotion != target_emotion:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        
        # Get track for target emotion
        if target_emotion not in self.playlists:
            return False
        
        track = self._get_random_track(target_emotion)
        
        if track is None:
            print(f"No music files found for emotion: {emotion}")
            print(f"Please add music files to: {self.music_dir / emotion}")
            return False
        
        try:
            # Load and play new track
            pygame.mixer.music.load(str(track))
            pygame.mixer.music.play(-1)  # -1 means loop indefinitely
            self.current_emotion = target_emotion
            self.current_track = track
            
            if self.mood_mode == "change" and emotion != target_emotion:
                print(f"Detected: {emotion} → Playing: {target_emotion} music to change mood")
                print(f"Now playing: {track.name}")
            else:
                print(f"Now playing: {track.name} for emotion: {emotion}")
            return True
        
        except Exception as e:
            print(f"Error playing music: {e}")
            return False
    
    def stop_music(self):
        """Stop currently playing music."""
        if self.initialized:
            try:
                pygame.mixer.music.stop()
                self.current_emotion = None
                self.current_track = None
            except Exception as e:
                print(f"Error stopping music: {e}")
    
    def pause_music(self):
        """Pause currently playing music."""
        if self.initialized:
            try:
                pygame.mixer.music.pause()
            except Exception as e:
                print(f"Error pausing music: {e}")
    
    def unpause_music(self):
        """Resume paused music."""
        if self.initialized:
            try:
                pygame.mixer.music.unpause()
            except Exception as e:
                print(f"Error unpausing music: {e}")
    
    def is_playing(self):
        """Check if music is currently playing."""
        if not self.initialized:
            return False
        try:
            return pygame.mixer.music.get_busy()
        except:
            return False
    
    def get_current_track(self):
        """Get the name of currently playing track."""
        if self.current_track:
            return self.current_track.name
        return "No track playing"
    
    def get_playlist_info(self):
        """Get information about available playlists."""
        info = {}
        for emotion, tracks in self.playlists.items():
            info[emotion] = len(tracks)
        return info

