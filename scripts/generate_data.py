import os, requests, random, sys, re
from operator import itemgetter
sys.path.append('./')
from utils.gpt_3_utils import q_and_a_gpt3
from tqdm import tqdm

conceptnet_uri = ("https://conceptnet5.media.mit.edu/data/5.4/assoc/list/en"
                  "/{concept_one},{concept_two}?filter=/c/en&limit=1000")

def remove_beginning_articles(string):
    # Split the string into a list of words
    words = string.split()

    # Create a list of articles
    articles = ['a', 'an', 'the', 'some', 'many', 'few', 'several']

    # If the first word is an article, remove it
    if words[0].lower() in articles:
        words = words[1:]

    # Join the list of words back into a single string and return it
    return ' '.join(words)

def extract_random_concepts_from_conceptnet(num_concepts):
    base_url = 'https://api.conceptnet.io'
    # seed_topics = ['apple', 'banana', 'iPhone', 'milk', 'Santa', 'dog', 'cat', 'fish', 'bird', 'car', 'truck', 'plane', 'train', 'boat', 'ship', 'bicycle', 'motorcycle', 'bus', 'airplane', 'rocket', 'computer', 'laptop', 'phone', 'television', 'radio', 'camera', 'book', 'pen', 'pencil', 'eraser', 'notebook', 'paper', 'chair', 'table', 'bed', 'sofa', 'couch', 'lamp', 'light', 'fan', 'heater', 'air conditioner', 'refrigerator', 'oven', 'stove', 'microwave', 'toaster', 'sink', 'toilet', 'shower', 'bathtub', 'bathroom', 'kitchen', 'bedroom', 'living room', 'dining room', 'closet', 'garage', 'yard', 'garden', 'park', 'forest', 'mountain', 'ocean', 'lake', 'river', 'beach', 'desert', 'snow', 'rain', 'sun', 'moon', 'star', 'cloud', 'wind', 'fire', 'water', 'earth', 'air', 'tree', 'flower', 'grass', 'leaf']
    # seed_topics = ['apple', 'banana', 'iPhone', 'milk', 'Santa', 'dog', 'cat', 'fish', 'bird', 'car']
    seed_topics = ['apple', 'banana']
    num_concepts_per_seed = num_concepts // len(seed_topics)
    all_topics = []
    all_topics += seed_topics
    for seed_topic in tqdm(seed_topics, desc='seed topics'):
        limit = 2 * num_concepts_per_seed # to account for duplicates and ones discarded
        request_uri = f'{base_url}/query?start=/c/en/{seed_topic}&rel=/r/RelatedTo&limit={limit}'
        print(f'Calling ConceptNet API: {request_uri}')
        response = requests.get(request_uri)
        if response.status_code == 200:
            concepts = response.json()['edges']
            related_concepts_to_seed = [seed_topic]
            while (len(related_concepts_to_seed) < num_concepts_per_seed) and (len(concepts) > 0):
                concept = random.choice(concepts)
                concept_label = concept['end']['label']
                if concept_label not in all_topics:
                    related_concepts_to_seed.append(concept_label)
                    all_topics.append(concept_label)
                concepts.remove(concept)
        else:
            print('Failed to retrieve concepts from ConceptNet')
    return all_topics

def generate_concept_definitions(concepts):
    # Create an empty list to store the concept definitions
    concept_definitions = {}
    # Iterate over the concepts to generate definitions using GPT-3 API
    for concept in tqdm(concepts, desc='Generating concept definitions'):
        # Create a prompt for GPT-3
        prompt="Q: What is Apple? Describe briefly.\nA: An apple is a spherical fruit, which is usually sweet and can be in different colors of red, green, and yellow."
        question = "Q: What is " + concept + "? Describe briefly."
        # Generate answer using GPT-3
        text_answer, response = q_and_a_gpt3(question, prompt)
        # Replace concept in text_answer with 'X' and case insensitive
        pattern = re.compile(rf'{concept}', re.IGNORECASE)
        updated_text = str(re.sub(pattern, "X", text_answer)).strip()
        # Add the concept definition to the list of concept definitions
        concept_definitions[concept] = updated_text
    return concept_definitions

if __name__ == "__main__":
    num_concepts = 100
    # Types of scenarios/concepts to generate
    scenarios = ['concept']
    # Create folder for the generated data
    target_folder = 'data/negated_cs'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # Iterating over scenarios
    for scenario in scenarios:
        if scenario is 'concept':
            # If output file exists ask user if they want to overwrite it
            if os.path.exists(os.path.join(target_folder, 'concept.tsv')):
                overwrite = input('Output file already exists. Overwrite? (y/n): ')
                if overwrite is not 'y':
                    continue
            # Create a TSV file for the generated data
            tsv_file = os.path.join(target_folder, 'concept.tsv')
            # Create headers of the TSV file
            with open(tsv_file, 'w') as f:
                f.write('head\trelation\ttail\n')
            # Extract concepts from conceptnet
            # concepts = extract_concepts_from_conceptnet(num_concepts)
            concepts = extract_random_concepts_from_conceptnet(num_concepts)
            # Generate concept definitions for the concepts
            concept_definitions = generate_concept_definitions(concepts)
            # Write the concept definitions to the TSV file
            with open(tsv_file, 'a') as f:
                for concept, definition in concept_definitions.items():
                    if definition is not '':
                        f.write(definition + '\t' + 'Is' + '\t' + concept + '\n')
            # Print where the generated data is stored
            print('Generated data stored at: ' + tsv_file)