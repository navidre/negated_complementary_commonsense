import os, requests, random
from operator import itemgetter

conceptnet_uri = ("https://conceptnet5.media.mit.edu/data/5.4/assoc/list/en"
                  "/{concept_one},{concept_two}?filter=/c/en&limit=1000")

def extract_concepts_from_conceptnet():
    # TODO: Update to the new API calls
    # This method is adjusted from here: https://gist.github.com/williardx/828f0f2c5cd8ce06a3f203da926821b2
    all_current_topics = [["apocalypse", "desert_island"], ["apple", "banana"], ["santa_claus", "christmas"], ["table", "chair"], ["laptop", "iphone"]]
    all_topics = []
    num_topics = 10

    for current_topics in all_current_topics:
        all_topics += current_topics

        for i in range(num_topics):
            # Choose a random similarity score to decide how to similar the next
            # term should be to the two terms we passed to /assoc
            similarity_level = random.random()

            req = conceptnet_uri.format(
                    concept_one=current_topics[0],
                    concept_two=current_topics[1]
                )
            response = requests.get(req)
            similar_terms = sorted(response.json()["similar"], key=itemgetter(1))
            num_terms = len(similar_terms)

            for j in range(num_terms):
                # Choose the least dissimilar word defined by similarity_level or
                # the last word in the list, whichever is first
                if similar_terms[j][1] > random.random() or j == (num_terms - 1):
                    next_topic = similar_terms[j][0].replace("/c/en/", "")
                    break

            current_topics.pop(0)
            current_topics.append(next_topic)
            all_topics.append(next_topic)
            import IPython; IPython. embed(); exit(1)

def generate_negated_concepts(target_folder):
    pass

if __name__ == "__main__":
    
    # Types of scenarios/concepts to generate
    scenarios = ['concept']
    # Create folder for the generated data
    target_folder = 'data/negated_cs'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # Iterating over scenarios
    for scenario in scenarios:
        if scenario is 'concept':
            # Extract concepts from conceptnet
            concepts = extract_concepts_from_conceptnet()