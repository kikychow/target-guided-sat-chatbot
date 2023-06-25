import torch
from itertools import chain

import json
from backend.chatbot_helper_functions import *

if __name__ == "__main__":
  simulator, simulator_tokenizer = load_blenderbot()
  initial_prompts = load_initial_prompts()
  generator, tokenizer = load_blenderbot()

  emotion_classifier = load_emotion_classfier()
  emotional_keywords_dict = load_emotional_keywords_dict()

  with open('keyword_predictor/classified_emotional_keywords.json') as f:
    emotional_kw_dict = json.load(f)

  emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

  for id in range(1, 501):
    print(f"\n>>>>> Prompt {id}")
    emotion_scores = {e : 0 for e in emotions}

    # Chat with the bot
    bos = tokenizer.bos_token

    prevConcepts = ["today"]
    input_ids_history = []
    simulator_input_ids_history = []

    conversation_history = []
    emotional_kw = ""
    kw_emotion = ""
    max_emotion = ""
    emotion_scores_list = []

    is_success = False
    user_input = ""

    i = 1

    while i <= 20:
      if i == 1:
        user_input = initial_prompts[id-1]
      else:
        simulator_input_ids = list(chain.from_iterable(simulator_input_ids_history[-5:]))[-128:]
        simulator_input_ids = torch.LongTensor(simulator_input_ids).unsqueeze(0)
        simulator_output_ids = simulator.generate(simulator_input_ids, max_length=1000)
        decoded_output = simulator_tokenizer.batch_decode(simulator_output_ids, skip_special_tokens=True)
        user_input = decoded_output[0]
      print(f">> {i} Simulator: {user_input}")
      conversation_history.append("Simulator: " + user_input)

      # Calculate emotion scores
      emotion_prediction = emotion_classifier(user_input)[0]
      curr_emotion_scores = get_emotion_scores_dict(emotion_prediction)
      emotion_scores_list.append(curr_emotion_scores.copy())
      cumulative_emotion_scores = {k: sum(d[k] for d in emotion_scores_list[-3:]) for k in curr_emotion_scores} 
      max_emotion = max(cumulative_emotion_scores, key=cumulative_emotion_scores.get)

      is_target_reached_pair = is_target_reached(emotional_keywords_dict.keys(), user_input)
      if is_target_reached_pair[0]:
        is_success = True
        emotional_kw = is_target_reached_pair[1]
        kw_emotion = emotional_keywords_dict[emotional_kw]
        break

      # generate response
      user_input_ids = tokenizer.encode(bos + user_input)
      input_ids_history.append(user_input_ids)
      simulator_input_ids_history.append(simulator_tokenizer.encode(simulator_tokenizer.bos_token + user_input))

      simulator_input_ids = list(chain.from_iterable(simulator_input_ids_history[-5:]))[-128:]
      simulator_input_ids = torch.LongTensor(simulator_input_ids).unsqueeze(0)
      simulator_output_ids = simulator.generate(simulator_input_ids, max_length=1000)
      decoded_output = simulator_tokenizer.batch_decode(simulator_output_ids, skip_special_tokens=True)
      response = decoded_output[0]

      print("Baseline: " + response)
      conversation_history.append("Baseline: " + response)
      input_ids_history.append(tokenizer.encode(bos + response))
      simulator_input_ids_history.append(simulator_tokenizer.encode(simulator_tokenizer.bos_token + response))

      i += 1

    i = min(i, 20)
    print(f"Number of turns: {i}")
    append_to_file_baseline(id, conversation_history, emotional_kw, kw_emotion, max_emotion, i, is_success)
