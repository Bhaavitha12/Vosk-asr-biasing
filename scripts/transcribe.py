import os    # For directory and file handling
import json    # For reading and writing JSON files
import wave    # For reading WAV audio files
from vosk import Model, KaldiRecognizer   # Vosk ASR model and decoder


MODEL_PATH = "model/vosk-model-small-en-us-0.15"
AUDIO_DIR = "audio/test_audio"
OUTPUT_FILE = "output/predictions.json"


# Load the Vosk model
model = Model(MODEL_PATH)

# Dictionary to store filename: transcription mapping
results = {}

# Loop through all files in the audio directory
for file in os.listdir(AUDIO_DIR):

    if not file.endswith(".wav"):
        continue
    
    # Open the WAV file in read-binary mode
    wf = wave.open(os.path.join(AUDIO_DIR, file), "rb")

    # Creating obj with (the loaded model,the sample rate of the audio file)
    rec = KaldiRecognizer(model, wf.getframerate())

    # Enable word-level output (adds timestamps and word details internally)
    rec.SetWords(True)

    # Read and process audio in chunks (streaming style decoding)
    while True:

        # Read 4000 frames of audio at a time
        data = wf.readframes(4000)

        # If no data left, break the loop
        if len(data) == 0:
            break

        # Feed audio chunk to the recognizer
        # Internally updates decoding hypothesis
        rec.AcceptWaveform(data)

    # Get the final recognition result (JSON string)
    # Convert JSON string → Python dictionary
    # Extract only the "text" field
    text = json.loads(rec.FinalResult()).get("text", "")

    # Store transcription using filename as key
    results[file] = text


# Write all predictions to output JSON file
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)


print("Transcription DONE")
