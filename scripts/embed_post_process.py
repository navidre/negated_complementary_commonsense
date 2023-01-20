import os, ast
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from dotenv import load_dotenv
import openai
load_dotenv(f'{Path().resolve()}/.env')
openai.api_key = os.environ['OPENAI_API_KEY']

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

HOME = str(Path.home())

OUR_METHOD_RESULT_PATH = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_negated_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated.tsv'
target_file_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_negated_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_with_ada_embeddings.csv'
figure_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_negated_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_with_ada_embeddings.pdf'

def get_embedding(text, model="text-embedding-ada-002"):
    # Check if text is a string
    if not isinstance(text, str):
        return None
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# If target file exists, load it and return
if os.path.exists(target_file_path):
    print(f"Loading embeddings from {target_file_path}...")
    df = pd.read_csv(target_file_path)
    try:
        # Ignore nan values
        df = df[~df.ada_embedding.isna()]
        df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)
    except:
        import IPython; IPython. embed(); exit(1)
else:
    # Calculating embeddings
    print(f"Calculating embeddings for {OUR_METHOD_RESULT_PATH}...")
    df = pd.read_csv(OUR_METHOD_RESULT_PATH, sep='\t')
    df["combined"] = (
        "Question: " + df.prompt.str.strip() + "; Answer: " + df.generated_tail.str.strip()
    )
    # Get combined column value from row zero of the dataframe

    # Apply get_embedding to each question
    df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    # Save dataframe to file
    df.to_csv(target_file_path, index=False)

# Now using the embeddings, we can do some cool stuff like clustering, etc.

# Create a t-SNE model and transform the data
# matrix = df.ada_embedding.apply(ast.literal_eval).to_list()
matrix = df.ada_embedding.apply(np.ndarray.tolist).to_list()
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

# five colors from green to red
colors = ['#00FF00', '#00FF7F', '#FF0000', '#FF0000', '#FF0000']
x = [x for x,y in vis_dims]
y = [y for x,y in vis_dims]
color_indices = df.majority_vote.values - 1

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
plt.title("Question+Answer embeddings visualized using t-SNE")
plt.savefig(figure_path)