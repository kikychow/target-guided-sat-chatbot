import json
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import nltk
import tensorflow_hub as hub

class EvaluationMetrics():
  def __init__(self, evaluation, model, embed=None):
    self.model = model
    path = f'{evaluation}/{evaluation}_{model}_result.json'
    if model == "gpt2":
      self.subtitle = f"{evaluation.capitalize()} - {model.upper()}"
      self.subtitle = "Simulation - Model A (GPT-2)"
    else:
      self.subtitle = f"{evaluation.capitalize()} - {model.capitalize()}"
      # self.subtitle = "Simulation - Model B (BlenderBot)"
    if evaluation == "simulation":
      self.data_json = self.load_json_data(path)
      self.total_n = len(self.data_json)
      self.num_of_turns = self.get_num_of_turns()
      self.num_of_turns_freq = self.get_num_of_turns_freq()
      self.num_of_turns_avg = self.get_num_of_turns_avg()
      self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
      self.user_bot_relatedness_list = []
      self.embed = embed

    if evaluation == "trial":
      if model == "blenderbot":
        self.subtitle = "Human Trial - Model B (BlenderBot)"
        trial_relevant = [0, 0, 1, 9, 5]
        trial_natural = [0, 0, 2, 10, 3]
        trial_grammar = [0, 1, 1, 8, 6]
        trial_empathetic = [0, 1, 1, 5, 8]
        trial_feeling = [0, 3, 3, 7, 2]
        self.all_trial = [("Relevant Responses", trial_relevant),
                          ("Engaging and Natural", trial_natural),
                          ("Correct Grammar", trial_grammar),
                          ("Empathetic Response", trial_empathetic),
                          ("Interested in Feeling", trial_feeling)]
      elif model == "gpt2":
        self.subtitle = "Human Trial - Model A (GPT-2)"
        trial_relevant = [0, 9, 3, 3, 0]
        trial_natural = [1, 6, 5, 3, 0]
        trial_grammar = [0, 3, 3, 8, 1]
        trial_empathetic = [0, 4, 3, 6, 2]
        trial_feeling = [0, 6, 3, 5, 1]
        self.all_trial = [("Relevant Responses", trial_relevant),
                          ("Engaging and Natural", trial_natural),
                          ("Correct Grammar", trial_grammar),
                          ("Empathetic Responses", trial_empathetic),
                          ("Interested in Feeling", trial_feeling)]
      elif model == "baseline":
        self.subtitle = "Human Trial - Baseline"
        trial_relevant = [0, 0, 0, 9, 6]
        trial_natural = [0, 0, 4, 7, 4]
        trial_grammar = [0, 0, 0, 3, 12]
        trial_empathetic = [1, 4, 5, 3, 2]
        trial_feeling = [3, 6, 5, 1, 0]
        self.all_trial = [("Relevant Responses", trial_relevant),
                          ("Engaging and Natural", trial_natural),
                          ("Correct Grammar", trial_grammar),
                          ("Empathetic Responses", trial_empathetic),
                          ("Interested in Feeling", trial_feeling)]
  def get_mean(self, trial_list):
    mean = 0
    for i, n in enumerate(trial_list):
      score = i + 1
      mean += score * n
    mean = mean / 15
    return mean

  def load_json_data(self, path):
    with open(path) as f:
        data_json = [json.loads(row) for row in f]
    return data_json

  def get_task_success_rate(self):
    success_count = 0
    for data in self.data_json:
        if data['success']:
            success_count += 1
    task_success_rate = success_count / self.total_n
    return success_count, task_success_rate

  def get_num_of_turns(self):
    num_of_turns_list = []
    for data in self.data_json:
      if data['success']:
        num_of_turns_list.append(data['turns'])
      else:
        num_of_turns_list.append(21)

    return num_of_turns_list

  def get_num_of_turns_freq(self):
    res = [0] * 21
    for n in self.num_of_turns:
        res[n-1] += 1
    return res

  def get_num_of_turns_avg(self):
    return np.average(self.num_of_turns)

  def plot_num_of_turns(self):
    # Create the figure and axes objects, specify the size and the dots per inches
    fig, ax = plt.subplots(figsize=(3.5,8), dpi = 96)

    # Plot bars
    y = range(1, 22)
    y_labels = list(range(1, 21))
    y_labels.append(">20")
    x = self.num_of_turns_freq
    x_labels = range(0, 101, 10)
    bar = ax.barh(y, x, zorder=2, color='#2196f3')
    # bar1 = ax.bar(x, y, width=0.8, zorder=2, color='#2196f3')

    # Create the grid
    ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
    ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)

    # Reformat x-axis label and tick labels
    ax.set_xlabel('Number of Conversations', fontsize=12, labelpad=10) # No need for an axis label
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=10, labelrotation=0)
    ax.set_xticks(x_labels) # Map integers numbers from the series to labels list

    # Reformat y-axis
    ax.set_ylabel('Number of Turns', fontsize=12, labelpad=0)
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=10)
    ax.set_yticks(y, y_labels)
    # ax.set_yticks(y, [])

    # Add label on top of each bar
    ax.bar_label(bar, labels=[f'{e:,.0f}' for e in x], padding=3, color='black', fontsize=10, zorder=4)

    average = self.num_of_turns_avg
    plt.axhline(y=average, color = '#ff5a5f', linewidth=3, zorder=3)
    # Determine the y-limits of the plot
    ymin, ymax = ax.get_ylim()

    # Calculate a suitable y position for the text label
    y_pos = average/ymax + 0.03
    # Annotate the average line
    ax.text(0.88, y_pos, f'Average = {average:.2f}', ha='right', va='center', transform=ax.transAxes, size=12, zorder=3)

    # Remove the spines
    ax.spines[['top','right','bottom']].set_visible(False)

    # Add in title and subtitle
    ax.text(x=0.12, y=.93, s="Number of Turns", transform=fig.transFigure, ha='left', fontsize=14, weight='bold', alpha=.8)
    ax.text(x=0.12, y=.90, s=self.subtitle, transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)

    # Set a white background
    fig.patch.set_facecolor('white')

    plt.show()


  def plot_trial(self, title, trial):
    # Create the figure and axes objects, specify the size and the dots per inches
    fig, ax = plt.subplots(figsize=(3.5,5), dpi = 96)

    # Plot bars
    y = range(1,6)
    y_labels = ["Strongly\ndisagree", "Somewhat\ndisagree", "Neither agree\nnor disagree", "Somewhat\nagree", "Strongly\nagree"]
    x = trial
    x_labels = range(0, 16, 5)
    bar = ax.barh(y, x, zorder=2, color='#2196f3')
    # bar1 = ax.bar(x, y, width=0.8, zorder=2, color='#2196f3')

    # Create the grid
    ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
    ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)

    # Reformat x-axis label and tick labels
    ax.set_xlabel('Number of Responses', fontsize=12, labelpad=10) # No need for an axis label
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=10, labelrotation=0)
    ax.set_xticks(x_labels) # Map integers numbers from the series to labels list

    # Reformat y-axis
    # ax.set_ylabel('Frequency', fontsize=12, labelpad=10)
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12)
    ax.set_yticks(y, y_labels)
    # ax.set_yticks(y, [])

    # Add label on top of each bar
    ax.bar_label(bar, labels=[f'{e:,.0f}' for e in x], padding=3, color='black', fontsize=10, zorder=4)

    average = self.get_mean(trial)
    plt.axhline(y=average, color = '#ff5a5f', linewidth=3, zorder=3)
    # Determine the y-limits of the plot
    ymin, ymax = ax.get_ylim()
    # Calculate a suitable y position for the text label
    y_pos = average/ymax + 0.03
    # Annotate the average line
    ax.text(0.88, y_pos, f'Average = {average:.2f}', ha='right', va='center', transform=ax.transAxes, size=12, zorder=3)

    # Remove the spines
    ax.spines[['top','right','bottom']].set_visible(False)

    # Add in title and subtitle
    ax.text(x=0.12, y=.95, s=title, transform=fig.transFigure, ha='left', fontsize=14, weight='bold', alpha=.8)
    ax.text(x=0.12, y=.90, s=self.subtitle, transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)

    # Adjust the margins around the plot area
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None)

    # Set a white background
    fig.patch.set_facecolor('white')

    plt.show()

  def plot_all_trial(self):
    for title, trial in self.all_trial:
      self.plot_trial(title, trial)

  def utterance_similarity(self, u1, u2):
    u1_embed = self.embed([u1])
    u2_embed = self.embed([u2])
    return np.inner(u1_embed, u2_embed).item()

  def user_bot_relatedness(self):
    dialogues = self.get_dialogues()
    for user_utterances, bot_utterances in dialogues:
      for user_utterance, bot_utterance in zip(user_utterances, bot_utterances):
        bot_sentences = self.sent_tokenizer.tokenize(bot_utterance)
        self.user_bot_relatedness_list.append(self.utterance_similarity(user_utterance, bot_sentences[0]))
    return self.user_bot_relatedness_list

  def get_dialogues(self):
    dialogues = []
    for data in self.data_json:
      dialogue = data['dialogue']
      user_tag_len = len("Simulator: ")
      bot_tag_len = len(self.model) + 2
      user_utterances = [u[user_tag_len:].strip() for i, u in enumerate(dialogue) if i % 2 == 0]
      bot_utterances = [u[bot_tag_len:].strip() for i, u in enumerate(dialogue) if i % 2 != 0]
      if len(user_utterances) != len(bot_utterances):
        user_utterances = user_utterances[:-1]
      dialogues.append((user_utterances, bot_utterances))
    return dialogues

if __name__ == "__main__":
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    simulation_gpt2 = EvaluationMetrics("simulation", "gpt2", embed)
    simulation_blenderbot = EvaluationMetrics("simulation", "blenderbot", embed)
    simulation_baseline = EvaluationMetrics("simulation", "baseline", embed)

    trial_gpt2 = EvaluationMetrics("trial", "gpt2")
    trial_blenderbot = EvaluationMetrics("trial", "blenderbot")
    trial_baseline = EvaluationMetrics("trial", "baseline")

    ### Task success rate
    simulation_blenderbot_task_success_rate = simulation_blenderbot.get_task_success_rate()
    print(simulation_blenderbot_task_success_rate)
    simulation_gpt2_task_success_rate = simulation_gpt2.get_task_success_rate()
    print(simulation_gpt2_task_success_rate)
    simulation_baseline_task_success_rate = simulation_baseline.get_task_success_rate()
    print(simulation_baseline_task_success_rate)

    simulation_blenderbot.plot_num_of_turns()
    simulation_gpt2.plot_num_of_turns()
    simulation_baseline.plot_num_of_turns()

    trial_gpt2.plot_all_trial()
    trial_blenderbot.plot_all_trial()
    trial_baseline.plot_all_trial()

    print(simulation_blenderbot.get_dialogues())
    relatedness = simulation_blenderbot.user_bot_relatedness()
    print(len(relatedness))
    print(relatedness)
    print(np.average(relatedness))
    relatedness = simulation_gpt2.user_bot_relatedness()
    print(len(relatedness))
    print(relatedness)
    print(np.average(relatedness))
    relatedness = simulation_baseline.user_bot_relatedness()
    print(len(relatedness))
    print(relatedness)
    print(np.average(relatedness))

    

    
