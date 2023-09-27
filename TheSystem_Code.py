import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


file_path = '/Users/peibo1/Desktop/BMI 550/Assignment/Assignment1GoldStandardSet.xlsx' # Replace this with your input file path
annotated_data_df = pd.read_excel(file_path)


symptom_dict = {}
infile = open('/Users/peibo1/Desktop/BMI 550/Assignment/COVID-Twitter-Symptom-Lexicon.txt') # Replace this with your COVID Sympton Lexicon
for line in infile:
    items = line.strip().split('\t')
    standard_symptom = items[0].lower()
    cui = items[1]
    symptom_expression = items[2].lower()
    symptom_dict[standard_symptom] = cui
    symptom_dict[symptom_expression] = cui


negation_triggers = ['no', 'not', 'without', 'absence of', 'cannot', "couldn't",
                     'could not', "didn't", 'did not', 'denied', 'denies', 'free of',
                     'negative for', 'never had', 'resolved', 'exclude', 'with no',
                     'rule out', 'free', 'aside from', 'except', 'apart from']


def preprocess(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def in_scope(neg_end, text, symptom_expression):
    negated = False
    text_following_negation = text[neg_end:]
    tokenized_text_following_negation = text_following_negation.split()
    three_terms_following_negation = ' '.join(tokenized_text_following_negation[:min(len(tokenized_text_following_negation), 3)])
    match_object = re.search(r'\b' + re.escape(symptom_expression) + r'\b', three_terms_following_negation)
    if match_object:
        period_check = re.search('\.', three_terms_following_negation)
        next_negation = 1000
        for neg in negation_triggers:
            if re.search(r'\b' + re.escape(neg) + r'\b', text_following_negation):
                index = text_following_negation.find(neg)
                if index < next_negation:
                    next_negation = index
        if period_check and period_check.start() > match_object.start() and next_negation > match_object.start():
            negated = True
        elif not period_check:
            negated = True
    return negated

output_df = pd.DataFrame(columns=['ID', 'TEXT', 'Symptom CUIs', 'Negation Flag'])
rows_list = []

for index, row in annotated_data_df.iterrows():
    post_id = row['ID']
    text = row['TEXT']
    sentences = sent_tokenize(text)
    cui_set = set()
    for sentence in sentences:
        for symptom, cui in symptom_dict.items():
            for match in re.finditer(r'\b' + re.escape(lemmatizer.lemmatize(symptom)) + r'\b', preprocess(sentence)):
                is_negated = False
                for neg in negation_triggers:
                    for neg_match in re.finditer(r'\b' + re.escape(lemmatizer.lemmatize(neg)) + r'\b', preprocess(sentence)):
                        is_negated = in_scope(neg_match.end(), preprocess(sentence), lemmatizer.lemmatize(symptom))
                        if is_negated:
                            break
                    if is_negated:
                        break
                if not match:
                    words = preprocess(sentence).split()
                    for word in words:
                        if fuzz.ratio(word, lemmatizer.lemmatize(symptom)) >= 90:
                            is_negated = any(re.search(r'\b' + re.escape(lemmatizer.lemmatize(neg)) + r'\b', preprocess(sentence)) for neg in negation_triggers)
                            break
                cui_set.add((cui, int(is_negated)))

    cui_list, neg_flag_list = zip(*list(cui_set)) if cui_set else ([], [])
    rows_list.append({
        'ID': post_id,
        'TEXT': text,
        'Symptom CUIs': '$$$'.join(cui_list) + '$$$',
        'Negation Flag': '$$$'.join(map(str, neg_flag_list)) + '$$$'
    })

output_df = pd.concat([output_df, pd.DataFrame(rows_list)], ignore_index=True)
output_df['Symptom CUIs'] = '$$$' + output_df['Symptom CUIs']
output_df['Negation Flag'] = '$$$' + output_df['Negation Flag']

output_excel_file_path = '/Users/peibo1/Desktop/BMI 550/Assignment/Results_Assignment1GoldStandardSet.xlsx' #Replace this with your output file path
output_df.to_excel(output_excel_file_path, index=False)
output_df