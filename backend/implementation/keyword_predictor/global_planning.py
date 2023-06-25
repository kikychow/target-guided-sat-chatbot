import networkx as nx
import datetime
from tqdm import tqdm
import pickle
import numpy as np
import gensim
from torch_geometric.utils.convert import from_networkx
import torchtext
import torch.nn.functional as F
import os

blacklist = set(["from", "as", "hey", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or", "the", "a", "of", "for", "is", "was", "the", "-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be","mine","us","em",
                 "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz",
                 ])

def package_path(*paths, package_directory=os.path.dirname(os.path.abspath(__file__))):
    return os.path.join(package_directory, *paths)

def conceptnet_csv_to_graph():
    graph = nx.Graph()
    with open('conceptnet.en.csv', "r", encoding="utf8") as f:
        for line in tqdm(f):
            ls = line.rstrip('\n').split('\t')
            relation = ls[0]
            subject = ls[1]
            object = ls[2]
            weight = float(ls[3])

            if subject == object: # delete loops
                continue
            if relation == "hascontext":
                continue
            graph.add_edge(subject, object, relation=relation, weight=weight)
    # Dump graph
    with open('conceptnet_graph.gpickle', 'wb') as f:
        pickle.dump(graph, f)

class GlobalPlanning():
    def __init__(self, use_numberbatch):
        self.conceptnet_graph = self.load_graph()
        print("Graph: ", self.conceptnet_graph)
        self.word_vectors = None
        if use_numberbatch:
            self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format('numberbatch-en-19.08.txt', binary=False)
        self.glove = torchtext.vocab.GloVe(name='6B', dim=300)

    def load_graph(self):
        time_start = datetime.datetime.now()
        print(f"load_graph started at: {time_start}")

        # Load graph
        with open(package_path("conceptnet_graph.gpickle"), 'rb') as f:  # notice the r instead of w
            conceptnet_graph = pickle.load(f)

        time_end = datetime.datetime.now()
        print(f"load_graph ended at: {time_end}, total: {time_end - time_start}")

        return conceptnet_graph
    
    def find_path(self, head, target, global_graph):
        if not self.word_exists_in_conceptnet(head) or not self.word_embedding_exists(head) or not self.word_exists_in_conceptnet(target) or not self.word_embedding_exists(target):
            return []
        # head to target
        curr_path = [head]
        res_path = []
        edges = self.tree_search(head, head, target, curr_path, global_graph,[])
        res_path.extend(edges)

        # target to head
        curr_path = [target]
        res_path = []
        edges = self.tree_search(target, target, head, curr_path, global_graph,[])
        res_path.extend(edges)

        return res_path

    # Recursive function
    def tree_search(self, word, head, target, curr_path, global_graph, edges):
        if word and (len(curr_path) >= 3 or word == target):
            edge = " ".join(curr_path)
            edges.append(edge)
            return edges

        new_choices = self.getNeighbours(word, head, target, curr_path, global_graph)

        if len(new_choices) == 0:
            edge = " ".join(curr_path)
            edges.append(edge)
        
        for new_word_pair in new_choices:
            if new_word_pair[0] not in curr_path:
                curr_path.append(new_word_pair[0])
                self.tree_search(new_word_pair[0], head, target, curr_path, global_graph, edges)
                curr_path.pop()
        return edges

    def getNeighbours(self, word, head, target, curr_path, global_graph):
        neighbours = []
        if not self.word_exists_in_conceptnet(word) or not self.word_embedding_exists(word):
            return []
        for n in list(self.conceptnet_graph.neighbors(word)):
            # if n not in curr_path and self.word_embedding_exists(n) and n not in blacklist:
            if n not in curr_path and self.word_embedding_exists(n):
                neighbours.append(n)
        # Select top K concepts most similar to the head entity
        #head_similarity_list = [(n, self.word_vectors.similarity(head, n)) for n in neighbours]
        head_similarity_list = [(n, F.cosine_similarity(self.glove[head].unsqueeze(0), self.glove[n].unsqueeze(0))) for n in neighbours]
        sorted_head_similarity_list = sorted(head_similarity_list, key=lambda x : x[1], reverse=True)[:10]
        # print("head similarity: ", sorted_head_similarity_list)

        # Select top K concepts most similar to the target entity
        #target_similarity_list = [(n, self.word_vectors.similarity(target, n)) for n in neighbours]
        target_similarity_list = [(n, F.cosine_similarity(self.glove[target].unsqueeze(0), self.glove[n].unsqueeze(0))) for n in neighbours]
        sorted_target_similarity_list = sorted(target_similarity_list, key=lambda x : x[1], reverse=True)[:10]
        # print("target similarity: ", sorted_target_similarity_list)

        new_words = sorted_head_similarity_list + sorted_target_similarity_list 
        for w in new_words:
            global_graph.add_edge(word, w[0])
        return new_words
    
    def word_embedding_exists(self, word):
        if word in self.glove.stoi and (self.word_vectors is None or word in self.word_vectors.key_to_index):
            return True
        return False
    
    def word_exists_in_conceptnet(self, word):
        if word in self.conceptnet_graph:
            return True
        return False

# def load_embeddings(conceptnet_graph):
#     conceptnet_embeddings = {}
#     with open('numberbatch-en-19.08.txt', 'r') as f:
#         for line in tqdm(f):
#             ls = line.rstrip('\n').split(' ')
#             word = ls[0]
#             if word in conceptnet_graph:
#                 conceptnet_embeddings[word] = np.array(ls[1:], dtype=np.float64)
#     return conceptnet_embeddings


if __name__ == "__main__":
    # csv_to_graph()

    # conceptnet_graph = load_graph()
    # word_vectors = gensim.models.KeyedVectors.load_word2vec_format('numberbatch-en-19.08.txt', binary=False)
    # word_vectors = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.300d.txt', binary=False)

    global_graph = nx.Graph()
    head = "art"
    target = "outside"
    use_numberbatch = False
    global_planning = GlobalPlanning(use_numberbatch)
    res = global_planning.find_path(head, target, global_graph)
    # print(">>>>> all paths: ", res)
    print(">>>>> shortest paths", [p for p in nx.all_shortest_paths(global_graph, 'art', 'outside')])

    # for n in list(global_graph.nodes):
    #     global_graph.nodes[n]['x'] = global_planning.glove.stoi[n]

    
    # data = from_networkx(global_graph)
    # print(data.keys)
    # print(data.edge_index)
    # print(data.x)
    # print(data)
    # text = ["i", "love", "you"]
    # print(global_planning.glove.stoi[text])
    
