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

  simulator, simulator_tokenizer = load_blenderbot()
  initial_prompts = load_initial_prompts()
  generator, tokenizer = load_generator_gpt2()
  predictor = load_predictor(global_planning)
  emotion_classifier = load_emotion_classfier()
  kw_extractor = load_kw_extractor()
  emotional_keywords_dict = load_emotional_keywords_dict()
  sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

  with open('keyword_predictor/classified_emotional_keywords.json') as f:
    emotional_kw_dict = json.load(f)

  emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

  for id in range(101, 121):
    emotion_scores = {e : 0 for e in emotions}

    # Chat with the bot
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    kw_token = tokenizer.additional_special_tokens[0]
    s1_id = tokenizer.additional_special_tokens_ids[1]
    s2_id = tokenizer.additional_special_tokens_ids[2]

    initial_prompt = "Hi! What did you do today?"
    prevConcepts = ["today"]
    input_ids_history = []
    simulator_input_ids_history = []

    conversation_history = []
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
      target = emotional_kw_dict[max_emotion] + ["emotion"]
      targets_list.append(max_emotion)

      is_target_reached_pair = is_target_reached(emotional_keywords_dict.keys(), user_input)
      if is_target_reached_pair[0]:
        is_success = True
        emotional_kw = is_target_reached_pair[1]
        kw_emotion = emotional_keywords_dict[emotional_kw]
        break

      # keyword predictions
      # head = extract_concepts(user_input, global_planning)
      head = extract_concepts_yake(user_input, global_planning, kw_extractor, prevConcepts)
      user_concepts_list.append(head)
      prevConcepts = head
      global_graph, target_graph, context_id= build_graph(user_input, head, target, global_planning)
      inputs = build_model_input(global_graph, target_graph, context_id, predictor)

      logits = predictor.predict(**inputs)['logits'][0]
      prediction_idx = get_topk_predicted_idx(logits, k=10)
      prediction_ids = inputs['nodes_ids'][0][prediction_idx]
      prediction_words = [global_planning.glove.itos[id] for id in prediction_ids ]
      prediction_words = [w for w in prediction_words if w not in blacklist][:3]
      # print('prediction words: ', prediction_words)
      prediction_words_list.append(prediction_words)

      # generate response
      user_input_ids = [s1_id] + tokenizer.encode(user_input)
      input_ids_history.append(user_input_ids)
      simulator_input_ids_history.append(simulator_tokenizer.encode(simulator_tokenizer.bos_token + user_input))


      keywords = kw_token + kw_token.join(prediction_words)
      encoded_kw = tokenizer.encode(keywords)

      input_ids = encoded_kw + [bos_id] + list(chain.from_iterable(input_ids_history[-5:])) + [s2_id]

      token_type_ids = [[s1_id] * len(hist) if h % 2 == 0 else [s2_id] * len(hist) for h, hist in enumerate(input_ids_history[-5:])]
      token_type_ids = [s2_id] * len(encoded_kw) + [s1_id] + list(chain.from_iterable(token_type_ids)) + [s2_id]
      input_len = len(input_ids)

      assert len(input_ids) == len(token_type_ids)


      input_ids = torch.LongTensor(input_ids[-1000:]).unsqueeze(0)
      token_type_ids = torch.LongTensor(token_type_ids[-1000:]).unsqueeze(0)

      output_ids = nucleus_sampling(input_ids, token_type_ids, input_len, generator, tokenizer)
      response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
      sentences = sent_tokenizer.tokenize(response)
      response = ' '.join(sent.capitalize() for sent in sentences)


      print("GPT2: " + response)
      conversation_history.append("GPT2: " + response)
      input_ids_history.append([s2_id] + tokenizer.encode(response))
      simulator_input_ids_history.append(simulator_tokenizer.encode(simulator_tokenizer.bos_token + response))

      bot_concepts = extract_concepts_yake(response, global_planning, kw_extractor, prevConcepts)
      prevConcepts = bot_concepts
      i += 1

    i = min(i, 20)
    print(f"GPT2: Are you feeling the emotion {kw_emotion} ({emotional_kw})? target: {max_emotion}")
    print(f"Number of turns: {i}")
    append_to_file(id, conversation_history, targets_list, emotion_scores_list, user_concepts_list, prediction_words_list, emotional_kw, kw_emotion, max_emotion, i, is_success)