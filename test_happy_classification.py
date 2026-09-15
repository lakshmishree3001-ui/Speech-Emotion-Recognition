import sys, glob, os
from src.predictor import preprocess_input_audio, predict_emotion, ModelManager

mgr = ModelManager()
mgr.load_all()

happy_files = glob.glob('Data/Raw/Audio_Speech_Actors_01-24/**/03-01-03-*.wav', recursive=True)
print(f"Total Happy files in dataset: {len(happy_files)}")

print(f"{'Filename':<32} | {'ANN (Best)':<16} | {'SVM Opt':<16} | {'Mel-CNN':<16}")
print("-" * 88)

ann_correct = 0
svm_correct = 0
mel_correct = 0
total = 16

for fp in happy_files[:total]:
    sig, dur = preprocess_input_audio(fp)
    ann_r = predict_emotion(sig, 'ANN (Best)', mgr, dur)
    svm_r = predict_emotion(sig, 'SVM Optimized', mgr, dur)
    mel_r = predict_emotion(sig, 'Mel-CNN', mgr, dur)
    
    ann_p = ann_r['predicted_emotion']
    svm_p = svm_r['predicted_emotion']
    mel_p = mel_r['predicted_emotion']
    
    if ann_p == 'happy': ann_correct += 1
    if svm_p == 'happy': svm_correct += 1
    if mel_p == 'happy': mel_correct += 1
    
    fn = os.path.basename(fp)
    print(f"{fn:<32} | {ann_p:<8} {ann_r['confidence']:.0f}%   | {svm_p:<8} {svm_r['confidence']:.0f}%   | {mel_p:<8} {mel_r['confidence']:.0f}%")

print("-" * 88)
print(f"Accuracy on Happy files: ANN: {ann_correct}/{total} ({ann_correct/total*100:.0f}%), SVM: {svm_correct}/{total} ({svm_correct/total*100:.0f}%), Mel-CNN: {mel_correct}/{total} ({mel_correct/total*100:.0f}%)")
