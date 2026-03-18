import os
import json
import wave
from vosk import Model, KaldiRecognizer

MODEL_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/model/vosk-model-small-en-us-0.15"
AUDIO_DIR = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/audio/finance"
OUTPUT_DIR = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/INTERMEDIATE_ANALYSIS"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = Model(MODEL_PATH)

def process_audio(file_path, file_name):
    wf = wave.open(file_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    partials = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            results.append(res)
        else:
            partial = json.loads(rec.PartialResult())
            partials.append(partial)

    final_res = json.loads(rec.FinalResult())

    output = {
        "file": file_name,
        "results": results,
        "partials": partials,
        "final": final_res
    }

    with open(os.path.join(OUTPUT_DIR, file_name + ".json"), "w") as f:
        json.dump(output, f, indent=2)


for file in os.listdir(AUDIO_DIR):
    if file.endswith(".wav"):
        print(f"Processing {file}")
        process_audio(os.path.join(AUDIO_DIR, file), file)

print("Done saving intermediate outputs.")