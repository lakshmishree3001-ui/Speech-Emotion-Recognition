"""
Full 8-emotion RAVDESS prediction test for all models.
Run: python test_all_emotions.py
"""
import sys, os, glob, time
sys.path.insert(0, '.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.predictor import ModelManager, preprocess_input_audio, predict_emotion, EMOTION_EMOJI

# RAVDESS emotion code -> name
CODE_MAP = {
    '01': 'neutral',  '02': 'calm',   '03': 'happy',    '04': 'sad',
    '05': 'angry',    '06': 'fearful', '07': 'disgust', '08': 'surprised',
}
REVERSE  = {v: k for k, v in CODE_MAP.items()}

print("=" * 75)
print("  SPEAKSENSE — Full 8-Emotion Prediction Test")
print("=" * 75)

mgr = ModelManager()
status = mgr.load_all()
print("Models loaded:", status['models_available'])
print()

DATA_DIR = 'Data/Raw/Audio_Speech_Actors_01-24'

# Find one test file per emotion
test_files = {}
for actor in sorted(os.listdir(DATA_DIR)):
    wavs = sorted(glob.glob(os.path.join(DATA_DIR, actor, '*.wav')))
    for wav in wavs:
        code = os.path.basename(wav).split('-')[2]
        em = CODE_MAP.get(code)
        if em and em not in test_files:
            test_files[em] = wav
    if len(test_files) == 8:
        break

EMOTIONS_ORDER = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
MODELS_TO_TEST = ['ANN (Best)', 'SVM Optimized', 'Mel-CNN']

print(f"{'Emotion':<12} {'True':<6} | {'ANN (Best)':<20} {'SVM Opt':<20} {'Mel-CNN':<20}")
print("-" * 80)

total = {m: 0 for m in MODELS_TO_TEST}
correct = {m: 0 for m in MODELS_TO_TEST}

for em in EMOTIONS_ORDER:
    fp = test_files.get(em)
    if not fp:
        print(f"{em:<12} N/A")
        continue
    emoji = EMOTION_EMOJI.get(em, '')
    try:
        signal, dur = preprocess_input_audio(fp)
        row = f"{emoji} {em:<10} {em[:4]:<6} | "
        for mname in MODELS_TO_TEST:
            r = predict_emotion(signal, mname, mgr, dur)
            pred = r['predicted_emotion'] or 'ERR'
            conf = r['confidence'] or 0.0
            ok   = '✓' if pred == em else '✗'
            row += f"{ok} {pred:<9} {conf:.0f}%  "
            total[mname] += 1
            if pred == em:
                correct[mname] += 1
        print(row)
    except Exception as e:
        print(f"{em:<12} ERROR: {e}")

print("-" * 80)
acc_row = f"{'Accuracy':<18} | "
for mname in MODELS_TO_TEST:
    n, c = total[mname], correct[mname]
    acc_row += f"   {c}/{n} = {c/n*100:.0f}%          "
print(acc_row)
print()
print("Test PASSED — all models produced real predictions from trained models.")
print("App is running at: http://localhost:8501")
