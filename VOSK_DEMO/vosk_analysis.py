import wave
import json
from vosk import Model, KaldiRecognizer

# -------- CONFIG --------
AUDIO_FILE = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/audio/finance/earnings22_0000.wav"
MODEL_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/model/vosk-model-small-en-us-0.15"   # path to vosk model

# -------- LOAD MODEL --------
print("Step 1: Loading Vosk model...")
model = Model(MODEL_PATH)
print("Model loaded\n")

# -------- LOAD AUDIO --------
print(" Step 2: Opening audio file...")
wf = wave.open(AUDIO_FILE, "rb")

print(f"Channels: {wf.getnchannels()}")
print(f"Sample Rate: {wf.getframerate()}")
print(f"Total Frames: {wf.getnframes()}\n")

if wf.getnchannels() != 1:
    raise ValueError("Audio must be mono")

# -------- INIT RECOGNIZER --------
print("Step 3: Initializing recognizer...")
rec = KaldiRecognizer(model, wf.getframerate())
print("Recognizer ready\n")

# -------- PROCESS AUDIO --------
print("Step 4: Processing audio frames...\n")

final_text = ""

while True:
    data = wf.readframes(4000)

    if len(data) == 0:
        print( "No more audio frames left")
        break

    print(f"Read {len(data)} bytes")

    if rec.AcceptWaveform(data):
        print(" Accepted waveform chunk")

        result = json.loads(rec.Result())
        print(f"Intermediate Result: {result}")

        final_text += result.get("text", "") + " "
    else:
        partial = json.loads(rec.PartialResult())
        print(f"Partial Result: {partial.get('partial', '')}")

print("\n🔹 Step 5: Finalizing recognition...")
final_result = json.loads(rec.FinalResult())
print(f"Final Chunk Result: {final_result}")

final_text += final_result.get("text", "")

# -------- OUTPUT --------
print("\n Step 6: Final Transcript")
print(final_text.strip())

print("\n Done")
