import requests, random
from tqdm import tqdm

# Set the base URL for the ConceptNet API
base_url = 'https://api.conceptnet.io'
num_concepts_per_seed = 10

# Seed topics
# seed_topics = ['apple', 'banana', 'iPhone', 'milk', 'Santa', 'dog', 'cat', 'fish', 'bird', 'car', 'truck', 'plane', 'train', 'boat', 'ship', 'bicycle', 'motorcycle', 'bus', 'airplane', 'rocket', 'computer', 'laptop', 'phone', 'television', 'radio', 'camera', 'book', 'pen', 'pencil', 'eraser', 'notebook', 'paper', 'chair', 'table', 'bed', 'sofa', 'couch', 'lamp', 'light', 'fan', 'heater', 'air conditioner', 'refrigerator', 'oven', 'stove', 'microwave', 'toaster', 'sink', 'toilet', 'shower', 'bathtub', 'bathroom', 'kitchen', 'bedroom', 'living room', 'dining room', 'closet', 'garage', 'yard', 'garden', 'park', 'forest', 'mountain', 'ocean', 'lake', 'river', 'beach', 'desert', 'snow', 'rain', 'sun', 'moon', 'star', 'cloud', 'wind', 'fire', 'water', 'earth', 'air', 'tree', 'flower', 'grass', 'leaf']
# seed_topics = ['apple', 'banana', 'iPhone', 'milk', 'Santa', 'dog', 'cat', 'fish', 'bird', 'car']
seed_topics = ['apple', 'banana']
all_topics = []
all_topics += seed_topics
for seed_topic in tqdm(seed_topics, desc='seed topics'):
    limit = 2 * num_concepts_per_seed # to account for duplicates and ones discarded
    request_uri = f'{base_url}/query?start=/c/en/{seed_topic}&rel=/r/RelatedTo&limit={limit}'
    response = requests.get(request_uri)
    if response.status_code == 200:
        concepts = response.json()['edges']
        related_concepts_to_seed = []
        while len(related_concepts_to_seed) < num_concepts_per_seed:
            concept = random.choice(concepts)
            concept_label = concept['end']['label']
            if concept_label not in all_topics:
                related_concepts_to_seed.append(concept_label)
                all_topics.append(concept_label)
    else:
        print('Failed to retrieve concepts from ConceptNet')
import IPython; IPython. embed(); exit(1)
