from implementation.keyword_predictor.global_planning import GlobalPlanning
import torch
from itertools import chain
import nltk
nltk.download('punkt')
import json
from backend.chatbot_helper_functions import *

if __name__ == "__main__":
  use_numberbatch = False
  global_planning = GlobalPlanning(use_numberbatch)

  generator, tokenizer = load_generator_blenderbot()
  predictor = load_predictor(global_planning)
  emotion_classifier = load_emotion_classfier()
  kw_extractor = load_kw_extractor()
  emotional_keywords_dict = load_emotional_keywords_dict()
  sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  with open('keyword_predictor/classified_emotional_keywords.json') as f:
    emotional_kw_dict = json.load(f)

  emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
  emotion_scores = {e : 0 for e in emotions}

  # Chat with the bot
  bos = tokenizer.bos_token
  kw_token = tokenizer.additional_special_tokens[0]

  initial_prompt = "Hi! What did you do today?"
  prevConcepts = ["today"]
  input_ids_history = []
  input_ids_history.append(tokenizer.encode(bos + initial_prompt))

  conversation_history = ["Blenderbot: " + initial_prompt]
  targets_list = []
  emotion_scores_list = []
  user_concepts_list = []
  prediction_words_list = []
  emotional_kw = ""
  kw_emotion = ""
  max_emotion = ""

  is_success = False
  user_input = ""

  i = 1

  print(f"\nBot 2: {initial_prompt}")

  while i <= 20:
    user_input = input(">> You: ")
    if user_input == ":q":
      break
    conversation_history.append("User: " + user_input)

    # Calculate emotion scores
    emotion_prediction = emotion_classifier(user_input)[0]
    curr_emotion_scores = get_emotion_scores_dict(emotion_prediction)
    emotion_scores_list.append(curr_emotion_scores.copy())
    cumulative_emotion_scores = {k: sum(d[k] for d in emotion_scores_list[-3:]) for k in curr_emotion_scores} 
    max_emotion = max(cumulative_emotion_scores, key=cumulative_emotion_scores.get)
    target = emotional_kw_dict[max_emotion] + ["emotion"]
    targets_list.append(max_emotion)

    is_target_reached_pair = is_target_reached(emotional_keywords_dict.keys(), user_input)
    if is_target_reached_pair[0]:
      is_success = True
      emotional_kw = is_target_reached_pair[1]
      kw_emotion = emotional_keywords_dict[emotional_kw]
      break

    head = extract_concepts_yake(user_input, global_planning, kw_extractor, prevConcepts)
    user_concepts_list.append(head)
    prevConcepts = head
    global_graph, target_graph, context_id= build_graph(user_input, head, target, global_planning)
    inputs = build_model_input(global_graph, target_graph, context_id, predictor)

    logits = predictor.predict(**inputs)['logits'][0]
    prediction_idx = get_topk_predicted_idx(logits, k=10)
    prediction_ids = inputs['nodes_ids'][0][prediction_idx]
    prediction_words = [global_planning.glove.itos[id] for id in prediction_ids]
    prediction_words = [w for w in prediction_words if w not in blacklist][:3]
    prediction_words_list.append(prediction_words)

    # generate response
    user_input_ids = tokenizer.encode(bos + user_input)
    input_ids_history.append(user_input_ids)

    keywords = kw_token + kw_token.join(prediction_words)
    encoded_kw = tokenizer.encode(keywords)

    input_id_max_length = 128 - len(encoded_kw)

    input_ids = encoded_kw + list(chain.from_iterable(input_ids_history[-5:]))[-input_id_max_length:]
    assert len(input_ids) <= 128

    input_ids = torch.LongTensor(input_ids).unsqueeze(0)

    output_ids = generator.generate(input_ids=input_ids, max_length=128)
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    sentences = sent_tokenizer.tokenize(response)
    response = ' '.join(sent.capitalize() for sent in sentences)

    print("Bot 2: " + response)
    conversation_history.append("Blenderbot: " + response)
    input_ids_history.append(tokenizer.encode(bos + response))
    bot_concepts = extract_concepts_yake(response, global_planning, kw_extractor, prevConcepts)
    prevConcepts = bot_concepts
    i += 1

  i = min(i, 20)
  print(f"Number of turns: {i}")
  print("Please complete survey at: https://imperial.eu.qualtrics.com/jfe/form/SV_aY2Uj1K9gTvm7u6")
  write_to_file(conversation_history, targets_list, emotion_scores_list, user_concepts_list, prediction_words_list, emotional_kw, kw_emotion, max_emotion, i, is_success)