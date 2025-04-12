from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io
import os
import pyaudio
import asyncio
import base64
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate

# Set up FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\\Users\\Antu Sanbui\\Desktop\\project\\GoogleApi\\SA_S2T.json"

# Initialize GCP clients
tts_client = texttospeech.TextToSpeechClient()
translate_client = translate.Client()
speech_client = speech.SpeechClient()

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

SUPPORTED_LANGUAGES = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "ur": "Urdu", "gu": "Gujarati", "kn": "Kannada", "or": "Odia", "pa": "Punjabi", "ml": "Malayalam", "en": "English"}

VOICE_MAP = {lang: {"MALE": f"{lang}-IN-Wavenet-B", "FEMALE": f"{lang}-IN-Wavenet-A"} for lang in SUPPORTED_LANGUAGES}

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class SynthesisRequest(BaseModel):
    text: str
    language_code: str
    gender: str

@app.post("/translate")
def translate_text_api(request: TranslationRequest):
    translation = translate_client.translate(request.text, target_language=request.target_language)
    return {"translatedText": translation["translatedText"]}

@app.post("/synthesize")
def synthesize_speech_api(request: SynthesisRequest):
    voice_name = VOICE_MAP.get(request.language_code, VOICE_MAP["en"]).get(request.gender.upper(), VOICE_MAP["en"]["MALE"])
    voice = texttospeech.VoiceSelectionParams(language_code=request.language_code, name=voice_name)
    synthesis_input = texttospeech.SynthesisInput(text=request.text)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")
    return {"audioContent": audio_base64}

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    queue = asyncio.Queue()

    async def receive_audio():
        """Receives audio from WebSocket and adds it to the queue."""
        try:
            while True:
                data = await websocket.receive_bytes()
                if data:
                    await queue.put(speech.StreamingRecognizeRequest(audio_content=data))
        except WebSocketDisconnect:
            print("WebSocket disconnected, stopping audio reception.")
        finally:
            # Do not send None, instead send an empty request to indicate end of stream
            await queue.put(speech.StreamingRecognizeRequest())

    async def audio_generator():
        """Generator function that yields audio from the queue."""
        while True:
            request = await queue.get()
            if request.audio_content == b"":  # End of stream
                break
            yield request

    audio_task = asyncio.create_task(receive_audio())

    try:
        responses = speech_client.streaming_recognize(streaming_config, audio_generator())

        async for response in responses:
            for result in response.results:
                if result.alternatives:
                    transcript = result.alternatives[0].transcript
                    await websocket.send_json({"transcript": transcript})

    except WebSocketDisconnect:
        print("WebSocket closed by client.")
    except Exception as e:
        print(f"Error in streaming: {e}")
    finally:
        audio_task.cancel()  # Ensure audio task stops
        await websocket.close()

@app.get("/supported-languages")
def get_supported_languages():
    return SUPPORTED_LANGUAGES

if __name__ == '__main__':
    uvicorn.run(app, port=5000)
