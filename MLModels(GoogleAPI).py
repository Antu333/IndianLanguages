import os
import io
import pyaudio
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
import pygame

# Set the path to your service account JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"Your GCP Credential Path"

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Initialize GCP clients
tts_client = texttospeech.TextToSpeechClient()
translate_client = translate.Client()

pygame.mixer.init(frequency=RATE, size=-16, channels=1)

# Supported Indian languages and their codes
SUPPORTED_LANGUAGES = {
    "as": "asm_Beng",  # Assamese ✅
    "bn": "ben_Beng",  # Bengali ✅
    "bho": "bho_Deva",  # Bhojpuri ❌ (Not supported in Google STT/TTS)
    "doi": "doi_Deva",  # Dogri ❌ (Not supported in Google STT/TTS)
    "gu": "guj_Gujr",  # Gujarati ✅
    "hi": "hin_Deva",  # Hindi ✅
    "kn": "kan_Knda",  # Kannada ✅
    "kok": "kok_Deva",  # Konkani ❌ (Not supported in Google STT/TTS)
    "mai": "mai_Deva",  # Maithili ❌ (Not supported in Google STT/TTS)
    "ml": "mal_Mlym",  # Malayalam ✅
    "mr": "mar_Deva",  # Marathi ✅
    "mni-Mtei": "mni_Beng",  # Meitei (Manipuri) ❌ (Not supported in Google STT/TTS)
    "ne": "npi_Deva",  # Nepali ✅
    "or": "ory_Orya",  # Odia ✅
    "pa": "pan_Guru",  # Punjabi ✅
    "sa": "san_Deva",  # Sanskrit ✅
    "sat": "sat_Olck",  # Santali ✅
    "sd": "snd_Arab",  # Sindhi ✅
    "ta": "tam_Taml",  # Tamil ✅
    "te": "tel_Telu",  # Telugu ✅
    "ur": "urd_Arab",  # Urdu ✅
    "dz": "dzo_Tibt",  # Dzongkha (Bhutanese) ✅
    "id": "ind_Latn",  # Indonesian ✅
    "en": "eng_Latn"   # English ✅
}


# Voice Map for WebNet (WaveNet) voices
VOICE_MAP = {
    "as": {
        "FEMALE": "as-IN-Wavenet-B"  # Assamese
    },
    "bn": {
        "MALE": "bn-IN-Wavenet-A",
        "FEMALE": "bn-IN-Wavenet-B"  # Bengali
    },
    "gu": {
        "MALE": "gu-IN-Wavenet-A",
        "FEMALE": "gu-IN-Wavenet-B"  # Gujarati
    },
    "hi": {
        "MALE": "hi-IN-Wavenet-A",
        "FEMALE": "hi-IN-Wavenet-B"  # Hindi
    },
    "kn": {
        "MALE": "kn-IN-Wavenet-A",
        "FEMALE": "kn-IN-Wavenet-B"  # Kannada
    },
    "ml": {
        "MALE": "ml-IN-Wavenet-A",
        "FEMALE": "ml-IN-Wavenet-B"  # Malayalam
    },
    "mr": {
        "MALE": "mr-IN-Wavenet-A",
        "FEMALE": "mr-IN-Wavenet-B"  # Marathi
    },
    "ne": {
        "FEMALE": "ne-NP-Wavenet-B"  # Nepali
    },
    "or": {
        "FEMALE": "or-IN-Wavenet-B"  # Odia
    },
    "pa": {
        "MALE": "pa-IN-Wavenet-A",
        "FEMALE": "pa-IN-Wavenet-B"  # Punjabi
    },
    "ta": {
        "MALE": "ta-IN-Wavenet-A",
        "FEMALE": "ta-IN-Wavenet-B"  # Tamil
    },
    "te": {
        "MALE": "te-IN-Wavenet-A",
        "FEMALE": "te-IN-Wavenet-B"  # Telugu
    },
    "ur": {
        "MALE": "ur-IN-Wavenet-A",
        "FEMALE": "ur-IN-Wavenet-B"  # Urdu
    },
    "id": {
        "MALE": "id-ID-Wavenet-A",
        "FEMALE": "id-ID-Wavenet-B"  # Indonesian
    },
    "en": {
        "MALE": "en-IN-Wavenet-A",
        "FEMALE": "en-IN-Wavenet-B"  # English (India)
    }
}


def get_language_code(language_name):
    """Returns the language code for the given language name."""
    for code, name in SUPPORTED_LANGUAGES.items():
        if name.lower() == language_name.lower():
            return code
    return None

def get_target_language(current_language):
    """Returns the target language for translation."""
    return target_language if current_language == source_language else source_language

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""
    
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = io.BytesIO()
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=lambda in_data, frame_count, time_info, status_flags: self.record_audio(in_data),
        )
        self._is_recording = True

    def record_audio(self, in_data):
        """Save recorded audio data to buffer."""
        self._buff.write(in_data)
        return None, pyaudio.paContinue  # Must return two values

    def close(self):
        """Clean up resources (closing the stream and PyAudio instance)."""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._buff.close()
        self._audio_interface.terminate()

    def generator(self):
        """Stream audio chunks from the buffer."""
        while self._is_recording:
            data = self._buff.getvalue()
            if len(data) > 0:
                yield data  # Send audio chunk to the transcriber
                self._buff.seek(0)
                self._buff.truncate()

    def stop_recording(self):
        """Stop recording."""
        self._is_recording = False

def transcribe_streaming(source_language, gender):
    """Transcribe and translate audio stream."""
    streaming_config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=source_language,
            enable_automatic_punctuation=True,
        ),
        interim_results=False,
    )

    stream = MicrophoneStream(RATE, CHUNK)
    
    try:
        client = speech.SpeechClient()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in stream.generator()
        )
        responses = client.streaming_recognize(streaming_config, requests)

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            print(f"{source_language} Transcript: {transcript}")

            if result.is_final:
                stream.stop_recording()

                target_language = get_target_language(source_language)
                translated_text = translate_text(transcript, target_language)
                print(f"Translated Text ({target_language}): {translated_text}")
                synthesize_speech(translated_text, target_language, gender)  # Pass gender here
                break
    except UnicodeDecodeError as e:
        print(f"Error during transcription: {e}")
    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        stream.close()  # Ensure cleanup



def translate_text(text, target_language):
    """Translate text using GCP Translation API."""
    translation = translate_client.translate(text, target_language=target_language)
    return translation["translatedText"]


def synthesize_speech(text, language_code, gender):
    
    # Get the WebNet voice name based on language code and gender
    if language_code not in VOICE_MAP:
        print(f"Voice map not found for {language_code}. Defaulting to male WebNet voice.")
        voice_name = VOICE_MAP["en-IN"]["MALE"]  # Default to en-IN Male
    else:
        voice_map = VOICE_MAP[language_code]
        gender = gender.strip().upper()  # Trim spaces and ensure uppercase
        print(f"Processed gender: {gender}")

    if gender == "MALE":
        voice_name = voice_map.get("MALE", voice_map["FEMALE"])  # Default to FEMALE if MALE key missing
    elif gender == "FEMALE":
        voice_name = voice_map.get("FEMALE", voice_map["MALE"])  # Default to MALE if FEMALE key missing
    else:
        print(f"Gender {gender} not recognized. Defaulting to male WebNet voice.")
        voice_name = voice_map["MALE"]

    # Create the VoiceSelectionParams using the WebNet voice
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,  # Use the selected WebNet voice
    )

    # Create the text input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Define audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1,
        pitch=1
    )

    # Generate speech
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    # Get the audio content and play it
    audio_content = response.audio_content
    with open('test_output.wav', 'wb') as f:
        f.write(audio_content)
    play_audio(audio_content)  # Ensure you have a function to handle audio playback


def play_audio(audio_content):
    """Play audio directly using pygame mixer."""
    print("Playing audio...")  # Debug log
    try:
        # Initialize pygame mixer and play audio
        pygame.mixer.music.load(io.BytesIO(audio_content))
        pygame.mixer.music.play()

        # Wait until the audio is done playing before continuing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Check every 10ms

    except Exception as e:
        print(f"Error during audio playback: {e}")


# Your main program
if __name__ == "__main__":
    try:
        # Get source and target languages from user
        source_language_name = input("Enter the source language (e.g., Hindi, Bengali, English): ")
        target_language_name = input("Enter the target language (e.g., Hindi, Bengali, English): ").upper()
        gender = input("Enter Preferred gender for voice: ")  # Ask for gender input
        source_language = get_language_code(source_language_name)
        target_language = get_language_code(target_language_name)

        if not source_language or not target_language:
            print("Unsupported language. Please choose from the supported languages.")
            exit(1)

        while True:
            input(f"User 1 ({source_language_name}) - Press Enter to start recording...")
            print("User 1 is speaking...")
            transcribe_streaming(source_language, gender)  # Pass gender to the function
            
            input(f"User 2 ({target_language_name}) - Press Enter to reply...")
            print("User 2 is speaking...")
            transcribe_streaming(target_language, gender)  # Pass gender to the function
            
            print("Conversation cycle complete. Restarting...")
    except KeyboardInterrupt:
        print("Stopping conversation...")
