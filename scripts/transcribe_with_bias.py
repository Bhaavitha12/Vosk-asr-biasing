# Import required libraries
import os          # used to work with folders/files
import json        # used to read/write JSON data
import wave        # used to read .wav audio files
from vosk import Model, KaldiRecognizer   # Vosk speech recognition classes

# Path to the Vosk model we downloaded
MODEL_PATH = "model/vosk-model-small-en-us-0.15"

# Folder where the test audio files are stored
AUDIO_DIR = "audio/test_audio"

# Text file containing bias words (one word per line)
BIAS_FILE = "bias_words.txt"

# File where we will save the transcription results
OUTPUT_FILE = "output/predictions_with_bias.json"


# -------------------- LOAD BIAS WORDS --------------------

# Open the bias words file and read each line
with open(BIAS_FILE) as f:
    # Remove spaces/newlines and convert words to lowercase
    # Also ignore empty lines
    bias_words = [line.strip().lower() for line in f if line.strip()]


# -------------------- LOAD VOSK MODEL --------------------

# Load the speech recognition model
model = Model(MODEL_PATH)

# Dictionary to store results for all audio files
results = {}


# -------------------- PROCESS EACH AUDIO FILE --------------------

# Loop through all files inside the audio folder
for file in os.listdir(AUDIO_DIR):

    # Skip files that are not .wav format
    if not file.endswith(".wav"):
        continue

    # Open the audio file
    wf = wave.open(os.path.join(AUDIO_DIR, file), "rb")

    # Create recognizer object
    # It takes the model and the sample rate of the audio
    rec = KaldiRecognizer(model, wf.getframerate())

    # Enable word-level timestamps in output
    rec.SetWords(True)


    # -------------------- APPLY BIAS WORDS --------------------

    # Convert bias words list into JSON format
    # This tells the decoder to prefer these words during recognition
    rec.SetGrammar(json.dumps(bias_words))


    # -------------------- READ AUDIO IN CHUNKS --------------------

    # We read the audio file in small chunks instead of all at once
    while True:
        data = wf.readframes(4000)

        # If no more audio is left, stop the loop
        if len(data) == 0:
            break

        # Feed audio chunk into the recognizer
        rec.AcceptWaveform(data)


    # -------------------- GET FINAL RESULT --------------------

    # Get the final recognition output
    # It returns JSON so we parse it
    text = json.loads(rec.FinalResult()).get("text", "")

    # Store the transcription using filename as key
    results[file] = text


# -------------------- SAVE RESULTS --------------------

# Write all transcriptions to a JSON file
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)


# Print completion message
print("Bias-based transcription DONE")