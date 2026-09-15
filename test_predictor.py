import sys, os, glob
sys.path.insert(0, '.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.predictor import ModelManager, preprocess_input_audio, predict_emotion

mgr = ModelManager()
status = mgr.load_all()
print('Status:', status)

test_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust', '08': 'surprised',
}
files = []
for actor in ['Actor_01', 'Actor_02', 'Actor_03']:
    wavs = sorted(glob.glob('Data/Raw/Audio_Speech_Actors_01-24/' + actor + '/*.wav'))
    files.extend(wavs[:3])

print()
header = "{:<40} {:<10} {:<14} {:<14} {:<14}".format('File', 'True', 'ANN Best', 'SVM Opt', 'Mel-CNN')
print(header)
print('-'*95)

for fp in files[:12]:
    fname = os.path.basename(fp)
    em_code = fname.split('-')[2]
    true_em = test_map.get(em_code, '?')
    try:
        signal, dur = preprocess_input_audio(fp)
        ann_r = predict_emotion(signal, 'ANN (Best)', mgr, dur)
        svm_r = predict_emotion(signal, 'SVM Optimized', mgr, dur)
        mel_r = predict_emotion(signal, 'Mel-CNN', mgr, dur)

        ann_em = (ann_r['predicted_emotion'] or 'ERR')
        svm_em = (svm_r['predicted_emotion'] or 'ERR')
        mel_em = (mel_r['predicted_emotion'] or 'ERR')
        ann_s  = "{:<8} {:.0f}%".format(ann_em[:8], ann_r['confidence'])
        svm_s  = "{:<8} {:.0f}%".format(svm_em[:8], svm_r['confidence'])
        mel_s  = "{:<8} {:.0f}%".format(mel_em[:8], mel_r['confidence'])

        row = "{:<40} {:<10} {:<14} {:<14} {:<14}".format(fname, true_em, ann_s, svm_s, mel_s)
        print(row)
    except Exception as e:
        print("{:<40} ERROR: {}".format(fname, e))

print()
print('DONE')
