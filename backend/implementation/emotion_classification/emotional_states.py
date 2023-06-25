from transformers import pipeline
import json
import pickle

emotional_keywords = []
with open('emotional_keywords.txt', 'r') as f:
    for line in f:
        emotional_keywords.append(line.rstrip('\n'))

emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
print("feeling" in emotional_keywords)

for e in emotions:
    if e not in emotional_keywords:
        emotional_keywords.append(e)

emotion_scores = {e : 0 for e in emotions}
classified_keywords = {e : [] for e in emotions}
emotional_keywords_dict = {}

emotion_classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)

for kw in emotional_keywords:
    emotion_prediction = emotion_classifier(kw)[0]
    for d in emotion_prediction:
        emotion = d['label']
        score = d['score']
        emotion_scores[emotion] = score
    max_emotion = max(emotion_scores, key=emotion_scores.get)
    classified_keywords[max_emotion].append(kw)
    emotional_keywords_dict[kw] = max_emotion

print(emotional_keywords_dict)
with open('emotional_keywords.pickle', 'wb') as f:
    pickle.dump(emotional_keywords_dict, f) # kw to emoton

with open('classified_emotional_keywords.json', 'w') as f:
    f.write(json.dumps(classified_keywords)) # emotion to list of kw