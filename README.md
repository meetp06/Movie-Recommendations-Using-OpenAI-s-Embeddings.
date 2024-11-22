# Movie Recommendations Using OpenAI's Embeddings

## Overview

This project aims to create a personalized movie recommendation system using OpenAI's embedding models. By embedding movie plots into vector representations, we can measure the semantic similarity between movies and provide tailored recommendations based on user queries.

The recommendation system has various applications, such as:
- Enhancing user satisfaction and retention on streaming platforms.
- Improving user engagement for e-commerce websites selling movies.
- Simplifying the movie selection process for users.

---

## Features
- Load and preprocess a movie dataset with detailed attributes like title, release year, cast, and plot.
- Generate embeddings for movie plots using OpenAI's **text-embedding-3-small** model.
- Compute similarity scores between user queries and movie embeddings.
- Provide the top 5 personalized movie recommendations.
- Export recommendations and query results for future use.

---

## Requirements

- Python 3.8 or above
- Libraries:
  - `datasets==3.0.0`
  - `openai==1.16.2`
  - `pandas==1.5.3`
  - `numpy`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/meetp06/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your OpenAI API Key:
   Replace `"your-api-key-here"` in the script with your actual OpenAI API key.

---

## Dataset

The project uses the dataset [AIatMongoDB/embedded_movies](https://huggingface.co/datasets/AIatMongoDB/embedded_movies), which contains:
- Movie titles
- Release year
- Cast
- Plot summaries
- Pre-generated embeddings (replaced with new ones in this project)

The dataset is loaded using Hugging Face's `datasets` library.

---

## Usage

### 1. Load the Dataset
Run the script to load the dataset and preprocess it:
```python
from datasets import load_dataset
dataset = load_dataset("AIatMongoDB/embedded_movies")
```

### 2. Preprocess Data
Remove missing values and unnecessary columns:
```python
dataset_df = dataset_df.dropna(subset=['plot'])
dataset_df = dataset_df.drop(columns=['plot_embedding'])
```

### 3. Generate New Embeddings
Use OpenAI's API to generate embeddings for movie plots:
```python
import openai
openai.api_key = "your-api-key-here"

# Generate embeddings for plots
dataset_df['plot_embedding_optimized'] = dataset_df['plot'].apply(get_embedding)
```

### 4. Query for Recommendations
Provide a query (e.g., `"What are the best action movies?"`) and get personalized recommendations:
```python
response, source_information = handle_user_query(query, dataset_df)
```

### 5. Save Results
Export recommendations and source information to a text file:
```python
with open("response.txt", "w") as file:
    file.write(response)
    file.write("\n\nSource Information:\n")
    file.write(source_information)
```

---

## Example Query
```python
query = "What are the best action movies?"
response, source_information = handle_user_query(query, dataset_df)
print(response)
```

**Sample Output**:
```
Recommended Movies:
1. Title: Mad Max: Fury Road, Plot: In a post-apocalyptic wasteland...
2. Title: Die Hard, Plot: A cop tries to save hostages during a Christmas party...
```

---

## Estimated Costs
The estimated cost for using OpenAI's API in this project ranges between **$4 and $8** depending on the number of queries and the embedding model used.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- OpenAI for providing powerful embedding models.
- Hugging Face for hosting the `AIatMongoDB/embedded_movies` dataset.
```
