import nltk
import json
from collections import defaultdict
import inflect
import pandas as pd
import argparse
from tqdm import tqdm


def update_count(word):
    if word not in count_dict:
        count_dict[word] = 1
    else:
        count_dict[word] += 1

def find_key_for_word(word):
    for key, synonyms in sim_dict.items():
        if word.lower() in synonyms:
            return key
    return None

def tokens_nons(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    nn_tagged = []
    for word, pos in tagged:
        if pos in ['NN', 'NNP']:
            nn_tagged.append(word)
        elif pos == 'NNS':
            singular_word = p.singular_noun(word)
            if singular_word:
                nn_tagged.append(singular_word.lower())
    return tagged, nn_tagged

def find_compound_nouns(tokens):
    compound_nouns1 = []
    compound_nouns2 = []
    compound_nouns3 = []
    compound_nouns4 = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1:
            if tokens[i][1] in ['JJ', 'VB', 'RB'] and tokens[i+1][1] in ['NN', 'NNP','NNS']:  # 형용사 + 명사
                compound_nouns1.append([f"{tokens[i][0]} {tokens[i+1][0]}", [i, i+1]])

            if tokens[i][1] in ['NN', 'NNP','NNS'] and tokens[i+1][1] in ['NN', 'NNP','NNS']:  # 명사 + 명사
                compound_nouns3.append([f"{tokens[i][0]} {tokens[i+1][0]}", [i, i+1]])   
        if i < len(tokens) - 2:
            if tokens[i][1] in ['NN', 'NNP','NNS'] and tokens[i+1][1] in ['NN', 'NNP','NNS'] and tokens[i+2][1] in ['NN', 'NNP','NNS']:  # 명사 + 명사
                compound_nouns2.append([f"{tokens[i][0]} {tokens[i+1][0]}", [i, i+1, i+2]])
                
        if tokens[i][1] in ['NN', 'NNP','NNS', 'JJ']:
            compound_nouns4.append([f"{tokens[i][0]}", [i]])
            
        i += 1
            
    return compound_nouns1, compound_nouns2, compound_nouns3, compound_nouns4

def update_counts(args, key_caption,captions,captions2):
    for idx, (caption, caption2) in enumerate(tqdm(zip(captions, captions2), total=len(captions))):
        tagged, _ = tokens_nons(caption.lower())
        tagged2, _ = tokens_nons(caption2.lower())
        compound_noun = find_compound_nouns(tagged)
        compound_noun2 = find_compound_nouns(tagged2)
        
        all_compound_noun = compound_noun + compound_noun2
        index_list = []
        ex_nouns = []
        for compound in all_compound_noun:
            for comp in compound:

                singular_word = p.singular_noun(comp[0])
                if singular_word == False:
                    key = find_key_for_word(comp[0])
                else:
                    key = find_key_for_word(singular_word)
                if key:
                    skip_this_compound = False
                    for com in comp[1]: 
                        if com in index_list:
                            skip_this_compound = True
                            break
                    if skip_this_compound:
                        continue
                    index_list.extend(comp[1])
                    ex_nouns.append(key)
                    word_counts[key] += 1
        
        key_list = list(set(ex_nouns))
        for key in key_list:
            word_counts[key] += 1
            if key == None:
                continue

            update_count(key)

        for i, word1 in enumerate(key_list):
            for j, word2 in enumerate(key_list):
                if i != j:
                    co_occurrence_counts[word1][word2] += 1

        index = [list(sim_dict.keys()).index(key) for key in key_list]

        if args.dataset in ['coco2014']:
            key_id = key_caption[idx][0].split('COCO_train2014_')[-1].split('.')[0]
        else:
            key_id = key_caption[idx][0].split('.')[0]

        if key_id not in persudo_label_name_dict:
            persudo_label_name_dict[key_id] = []
        persudo_label_name_dict[key_id].append(key_list)

        if key_id not in persudo_label_dict:
            persudo_label_dict[key_id] = []
        persudo_label_dict[key_id].append(index)

    return word_counts, co_occurrence_counts, persudo_label_dict, persudo_label_name_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='coco2014', choices=['voc2007', 'voc2012', 'coco2014', 'nuswide'])
    args = parser.parse_args()

    with open(f'caption/{args.dataset}_synonyms.json', 'r', encoding='utf-8') as f:
        sim_dict = json.load(f)

    p = inflect.engine()
    synonyms_dict = {}
    word_counts = defaultdict(int)
    count_dict = {}
    co_occurrence_counts = defaultdict(lambda: defaultdict(int))
    persudo_label_dict = {}
    persudo_label_name_dict = {}

    import nltk
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

    file_name1 = f'caption/{args.dataset}/{args.dataset}_captions.json'
    file_name2 = f'caption/{args.dataset}/{args.dataset}_re_captions.json'

    with open(f'{file_name1}', 'r') as f:
        captions_data = json.load(f)

    with open(f'{file_name2}', 'r') as f:
        captions_data2 = json.load(f)

    key_caption = [[k,captions_data[k][j]] for j in range(6) for k in captions_data.keys()]
    captions = [captions_data[k][j] for j in range(6) for k in captions_data.keys()]
    captions2 = [captions_data2[k][j] for j in range(6) for k in captions_data2.keys()]

    word_counts, co_occurrence_counts, persudo_label_dict, persudo_label_name_dict = update_counts(args, key_caption, captions, captions2)

    with open(f'caption/{args.dataset}/train_co_occurrence_counts.json', 'w') as file:
        json.dump(co_occurrence_counts, file)
    with open(f'caption/{args.dataset}/train_word_counts.json', 'w') as file:
        json.dump(word_counts, file)
    with open(f'caption/{args.dataset}/train_count_dict.json', 'w') as file:
        json.dump(count_dict, file)
    with open(f'caption/{args.dataset}/train_label_dictionary.json', 'w') as file:
        json.dump(persudo_label_dict, file)
