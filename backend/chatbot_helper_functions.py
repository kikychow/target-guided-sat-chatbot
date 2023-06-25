from transformers import GPT2Tokenizer, GPT2LMHeadModel, BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline
from keyword_predictor.global_planning import GlobalPlanning
from keyword_predictor.keyword_predictor import KeywordPredictor
from transformers.utils import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv

import re
import yake

import nltk
nltk.download('punkt')
from nltk import word_tokenize
import datetime
import json
import pickle

import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
blacklist = set(["from", "as", "hey", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or", "the", "a", "of", "for", "is", "was", "the", "-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be","mine","us","em",
                 "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz",
                 "people", "person", "lot", "nothing", "place", "being", "thing", "good"])

def load_blenderbot():
    mname = "facebook/blenderbot-400M-distill"
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    return model, tokenizer

def load_initial_prompts():
    initial_prompts = []
    with open('simulation_initial_prompts.pickle', 'rb') as f:
        initial_prompts = pickle.load(f)
    return initial_prompts

def load_generator_blenderbot():
    mname = "models/model_blenderbot"
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    return model, tokenizer

def load_generator_gpt2():
    mname = "models/model_gpt2_5epochs_more_data"
    model = GPT2LMHeadModel.from_pretrained(mname)
    tokenizer = GPT2Tokenizer.from_pretrained(mname)
    return model, tokenizer


def load_emotion_classfier():
   emotion_classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
   return emotion_classifier

def load_predictor(global_planning):
  model_state_dict_path = "models/keyword_predictor_state_dict_model_8_10_1e-6.pt"
  predictor_model = KeywordPredictor(global_planning)
  predictor_model.load_state_dict(torch.load(model_state_dict_path))
  predictor_model.eval() # must be set before inference
  return predictor_model

def suppress_transformer_warnings():
   logging.set_verbosity_error()

def extract_concepts(user_input, global_planning):
  """
  Returns a list of concepts extracted from the user input

  :param user_input: user utterance
  :type user_input: string 

  :param global_planning: global planning
  :type global_planning: GlobalPlanning

  :return: list of concepts
  :rtype: [string]
  """
  # user_input = "I am listening to music."
  # user_input = input()
  # concepts = list(global_planning.conceptnet_graph.nodes)
  sent = user_input.lower()
  doc = nlp(sent)
  head = set()
  for t in doc:
    if global_planning.word_embedding_exists(t.lemma_) and  global_planning.word_exists_in_conceptnet(t.lemma_) and t.lemma_ not in blacklist:
      if t.pos_ == "NOUN" or t.pos_ == "VERB":
        head.add(t.lemma_)
  if len(head) == 0:
    for t in doc:
      if global_planning.word_embedding_exists(t.lemma_) and  global_planning.word_exists_in_conceptnet(t.lemma_) and t.lemma_ not in blacklist:
        head.add(t.lemma_)
  print("Start concepts: ", head)
  return head

def load_kw_extractor():
  language = "en"
  max_ngram_size = 1
  deduplication_thresold = 0.9
  deduplication_algo = 'seqm'
  windowSize = 1
  numOfKeywords = 3
  kw_extractor = yake.KeywordExtractor(lan=language, 
                                      n=max_ngram_size, 
                                      dedupLim=deduplication_thresold, 
                                      dedupFunc=deduplication_algo, 
                                      windowsSize=windowSize, 
                                      top=numOfKeywords)
  return kw_extractor

def load_emotional_keywords_dict():
  with open('keyword_predictor/emotional_keywords.pickle', 'rb') as f:
    emotional_keyowrds_dict = pickle.load(f)
  return emotional_keyowrds_dict

def is_valid_concept(concept, global_planning):
  return global_planning.word_embedding_exists(concept) and  global_planning.word_exists_in_conceptnet(concept) and concept not in blacklist

def extract_concepts_yake(user_input, global_planning, kw_extractor, prevConcepts):
    concepts = []
    sent = user_input.lower()
    keywords = kw_extractor.extract_keywords(sent)
    if len(keywords) != 0:
       concepts = [kw[0] for kw in keywords if is_valid_concept(kw[0], global_planning)]
    
    if len(concepts) == 0:
      concepts = prevConcepts
    return concepts


def build_graph(user_input, head, target, global_planning):
  """
  Returns (global, target, context)

  :param user_input: user utterance
  :type user_input: string 

  :param conceptnet_graph: conceptnet knowledge graph
  :type conceptnet_graph: nx graph

  :return: list of concepts
  :rtype: [string]
  """
  
  global_graph = nx.Graph()
  target_graph = []
  context_id = []

  ### Global graph
  all_paths = []
  all_shortest_paths = []

  # Build global_graph and find all paths
  for s in head:
    for t in target:
      all_paths.extend(global_planning.find_path(s, t, global_graph))
  # print("Global graph created: ", global_graph)
  
  # find all shortest paths


  # print("All paths: ", all_paths)

  # Add word id to graph
  global_graph_nodes = list(global_graph.nodes)
  for n in global_graph_nodes:
      global_graph.nodes[n]['x'] = global_planning.glove.stoi[n]

  ### Target graph
  for t in target:
    if t in global_graph_nodes:
      target_graph.append(global_planning.glove.stoi[t])
      for n in global_graph.neighbors(t):
        target_graph.append(global_planning.glove.stoi[n])

  ### Context
  sent = user_input.lower()
  for word in word_tokenize(sent):
    if word in global_planning.glove.stoi:
      context_id.append(global_planning.glove.stoi[word])

  return global_graph, target_graph, context_id

def build_model_input(global_graph, target_graph, context_id, predictor):
  GCNNet = predictor.gcn_graph_encoder # graph encoder
  embedding_layer = predictor.embedding_layer
  context_encoder = predictor.context_encoder

  global_graph_data = from_networkx(global_graph)

  global_graphs_embeddings = [GCNNet(embedding_layer(global_graph_data.x), global_graph_data.edge_index)]

  target_ids_embeddings = [embedding_layer(torch.tensor(target_graph)).mean(0).repeat(global_graph_data.num_nodes, 1)]

  context_ids_embeddings = [context_encoder(embedding_layer(torch.tensor(context_id)))[1].reshape(-1).repeat(global_graph_data.num_nodes, 1)]

  # label = [torch.ones(global_graph_data.num_nodes, dtype=int) * -100]

  # inputs = torch.cat((global_graphs_embeddings, target_graphs_embeddings, context_embeddings), 1)
  # inputs = inputs.unsqueeze(0)
  # print(inputs.shape)
  inputs = {'nodes_ids':[global_graph_data.x], 'global_graphs':global_graphs_embeddings, 'target_ids':target_ids_embeddings, 'context_embeddings':context_ids_embeddings}
  return inputs


def predict_keywords(model, inputs, global_graph_nodes):
  logits = model(**inputs)['logits']
  # print(">>", logits[0].softmax(1).argmax(1))
  # preds_idx = logits[0].softmax(1).argmax(1).tolist()
  # preds = global_graph_nodes[preds_idx]

  _, pred_topk_idx = logits[0].softmax(1)[:, 1].topk(5)
  preds= global_graph_nodes[pred_topk_idx]
  # print(">>> predictions: ", preds)
  
  return preds

def get_predictions(logits):
  # Return prediction matrix
  all_preds = logits.softmax(1).argmax(1)
  return all_preds

def get_all_predicted_idx(logits):
  # Return all node index that are classified as 1
  all_preds_idx = (logits.softmax(1).argmax(1)==1).nonzero().flatten()
  return all_preds_idx

def get_topk_predicted_idx(logits, k):
  # Return topk node index that are classified as 1
  _, topk_preds_idx = logits.softmax(1)[:, 1].topk(k)
  return topk_preds_idx

def get_words_from_idx(graph_nodes, idx): 
  return graph_nodes[idx]

def is_target_reached(targets, utterance):
  sent = " " + utterance.lower()
  for target in targets:
    if re.search(r'\b' + target + r'\b', sent):
        return True, target
  return False, ""

def add_emotion_scores(emotion_scores, emotion_predictions):
  for d in emotion_predictions:
    emotion = d['label']
    score = d['score']
    emotion_scores[emotion] += score
  return emotion_scores

def get_emotion_scores_dict(emotion_predictions):
  emotion_scores = {}
  for d in emotion_predictions:
    emotion = d['label']
    score = d['score']
    emotion_scores[emotion] = score
  return emotion_scores

def nucleus_sampling(input_ids, token_type_ids, input_len, model, tokenizer):
  output_ids = []
  top_p = 0.9
  for pos in range(input_len, 1024):
    output = model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, pos-1]  # (1, V)
    output = F.softmax(output, dim=-1)  # (1, V)
    
    sorted_probs, sorted_idxs = torch.sort(output, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
    idx_remove = cumsum_probs > top_p
    idx_remove[:, 1:] = idx_remove[:, :-1].clone()
    idx_remove[:, 0] = False
    sorted_probs[idx_remove] = 0.0
    sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, V)
    
    probs = torch.zeros(output.shape).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
    idx = torch.multinomial(probs, 1)  # (1, 1)
    
    idx_item = idx.squeeze(-1).squeeze(-1).item()
    output_ids.append(idx_item)
    
    if idx_item == tokenizer.eos_token_id:
        break
        
    input_ids = torch.cat((input_ids, idx), dim=-1)
    next_type_id = torch.LongTensor([[tokenizer.additional_special_tokens_ids[2]]])
    token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
    assert input_ids.shape == token_type_ids.shape
      
  return output_ids

def write_to_file(conversation_history, targets, emotion_scores, user_concepts, predicted_words, emotional_kw, kw_emotion, max_emotion, turns, is_success):
  filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  data = {}
  data['timestamp'] = filename
  data['dialogue'] = conversation_history
  data['targets'] = targets
  data['emotion_scores'] = emotion_scores
  data['user_concepts'] = user_concepts
  data['predicted_words'] = predicted_words
  data['emotional_kw'] = emotional_kw
  data['kw_emotion'] = kw_emotion
  data['max_emotion'] = max_emotion
  data['turns'] = turns
  data['success'] = is_success
  with open(f'human_trial/blenderbot/{filename}.json', 'w') as f:
    json.dump(data, f, indent=4)
  with open(f'human_trial/human_trial_blenderbot_result.json', 'a') as f:
    f.write(json.dumps(data))
    f.write('\n')

def append_to_file(id, conversation_history, targets, emotion_scores, user_concepts, predicted_words, emotional_kw, kw_emotion, max_emotion, turns, is_success):
  data = {}
  data['id'] = id
  data['dialogue'] = conversation_history
  data['targets'] = targets
  data['emotion_scores'] = emotion_scores
  data['user_concepts'] = user_concepts
  data['predicted_words'] = predicted_words
  data['emotional_kw'] = emotional_kw
  data['kw_emotion'] = kw_emotion
  data['max_emotion'] = max_emotion
  data['turns'] = turns
  data['success'] = is_success
  with open(f'simulation/simulation_gpt2_result.json', 'a') as f:
    f.write(json.dumps(data))
    f.write('\n')

def append_to_file_baseline(id, conversation_history, emotional_kw, kw_emotion, max_emotion, turns, is_success):
  data = {}
  data['id'] = id
  data['dialogue'] = conversation_history
  data['emotional_kw'] = emotional_kw
  data['kw_emotion'] = kw_emotion
  data['max_emotion'] = max_emotion
  data['turns'] = turns
  data['success'] = is_success
  with open(f'simulation/simulation_baseline_result.json', 'a') as f:
    f.write(json.dumps(data))
    f.write('\n')