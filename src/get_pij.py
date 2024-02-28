import difflib
from Bio.Align import PairwiseAligner
from config.config_paths import PIJ
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pattern = r'\|--\|'


def load_data(file_name):
    ocr_outputs = []
    correct_labels = []
    with open(os.path.join(PIJ, file_name), 'r') as file:
        lines = [line.strip() for line in file]
    for line in lines:
        if line.startswith("output"):
            ocr_outputs.append(line.replace("output: ", ""))
        if line.startswith("label"):
            correct_labels.append(line.replace("label:  ", ""))
    return ocr_outputs, correct_labels


def word_matching(ocr_outputs, correct_labels):
    matches = []
    for ocr, correct in zip(ocr_outputs, correct_labels):
        # Splitting the sentences into words
        ocr_words = ocr.split(" ")
        correct_words = correct.split(" ")
        
        # Finding matches and alignments
        s = difflib.SequenceMatcher(None, correct_words, ocr_words)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag in ['replace', 'delete', 'insert']:
                if i1 != i2:
                    correct = correct_words[i1:i2]
                    correct = "*" + "*".join(correct).strip() + "*"
                if j1 != j2:
                    ocr = ocr_words[j1:j2]
                    ocr = "*" + "*".join(ocr).strip()+ "*"
                if ocr!="" and correct!="":
                    matches.append((correct, ocr))
    return matches


def record_character_mismatches(matches):
    aligner = PairwiseAligner()
    mismatches = dict()
    for pair in matches:
        a1 = pair[0]
        a2 = pair[1]
        align_score = aligner.score(a1, a2)
        score = align_score/(len(a1)+len(a2))*2
        if score>0.8:
            print("SCORE: ", score)
            alignment = next(aligner.align(a1, a2))

            # Convert alignment to string and split into lines for analysis
            alignment_str = str(alignment).split('\n')

            # Assuming alignment_str[0] is the target, alignment_str[1] is the match line, and alignment_str[2] is the query
            target = alignment_str[0]
            match_line = alignment_str[1]
            query = alignment_str[2]

            target = target[6:].strip()[2:]
            target = target[:target.rfind(" ")]
            match_line = match_line.strip()[2:]
            match_line = match_line[:match_line.rfind(" ")]
            query = query[6:].strip()[2:]
            query = query[:query.rfind(" ")]
            print(target)
            print(match_line)
            print(query)
            mismatch_char = re.finditer(pattern, match_line)

            # Print all matches found
            for match in mismatch_char:
                start = match.start()+1
                end = match.end()-1
                target_wrong = target[start:end]
                query_wrong = query[start:end]
                if target_wrong.count("-")==1 and query_wrong.count("-")==1:
                    key = (query_wrong.replace("-", ""), target_wrong.replace("-", ""))
                    if key in mismatches:
                        mismatches[key] += 1
                    else:
                        mismatches[key] = 1
                # print(target_wrong, "/////", query_wrong)

    return mismatches

def heatmap(mismatches, threshold):
    mismatches = {k:v for k,v in mismatches.items() if v>threshold}
    # Create a list of all unique characters involved in mismatches
    characters = list(set([char for pair in mismatches.keys() for char in pair]))
    characters.sort() # Sort characters for consistent ordering

    # Create an empty frequency matrix
    freq_matrix = np.zeros((len(characters), len(characters)))

    # Fill the frequency matrix with the counts from `mismatches`
    char_index = {char: i for i, char in enumerate(characters)} # Map characters to indices
    for (incorrect, correct), count in mismatches.items():
        i, j = char_index[incorrect], char_index[correct]
        freq_matrix[i, j] = count

    # Normalize the matrix if desired (here we're just using raw counts)
    # For normalization, you could divide each cell by the total number of mismatches, for example

    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(freq_matrix, annot=False, cmap='YlOrBr', xticklabels=characters, yticklabels=characters)
    # sns.heatmap(freq_matrix, annot=False, cmap='flag', xticklabels=characters, yticklabels=characters)
    plt.title('Heatmap of Character Misrecognitions')
    plt.xlabel('Correct Characters')
    plt.ylabel('Misrecognized Characters')
    plt.show()
    return


def get_pij(threshold=14, plot=False):
    ocr_outputs_all = []
    correct_labels_all = []
    files = [f for f in os.listdir(PIJ) if f.endswith("txt")]
    # for file_name in ["gw_cv1_train.txt", "gw_cv1_test.txt", "gw_cv1_val.txt", "iam_train.txt"]:
    for file_name in files:
        ocr_outputs, correct_labels = load_data(file_name)
        ocr_outputs_all += ocr_outputs
        correct_labels_all += correct_labels
    matches = word_matching(ocr_outputs_all, correct_labels_all)
    # print(matches)
    mismatches = record_character_mismatches(matches)
    # Sort the dictionary by its values, resulting in a list of tuples
    sorted_misread_list = sorted(mismatches.items(), key=lambda item: item[1])

    # Convert the sorted list of tuples back to a dictionary if needed
    sorted_misread_dict = dict(sorted_misread_list)
    print(sorted_misread_dict) 
    if plot:
        heatmap(mismatches, threshold)

if __name__=="__main__":
    get_pij(plot=True)
