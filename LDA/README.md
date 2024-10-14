# Project Overview
This repository performs LDA (Latent Dirichlet Allocation) topic modeling for preprocessed movie plot data. It includes two main components: text preprocessing and topic modeling using LDA. The final output includes topic distribution per document(movie plot) and word distribution per topic.

# preprocessing_LDA.py
`'preprocessing_LDA.py'` is designed to preprocess the crawled movie plots before applying the LDA topic modeling. Make sure that this script must be run after the movie plots have been are crawled.

## Example Command
```python
python preprocessing_LDA.py
```
Since this script has no options, you do not have to specify any command-line arguments when running it.

# LDA.py
Once the movie plot data is preprocessed, you can run `'LDA.py'` to perform LDA topic modeling. 

## Example Command
```python
python LDA.py --num_topics <number_of_topics> --top_n <top_n_words_per_topic>
```
- `num_topics`: Number of distinct topics that the LDA model will identify from the given text data.
-  `top_n`: The number of top words per topic to be displayed (default: 30)

# Stopwords
To remove custom stopwords during the preprocessing stage, prepare a `'stopwords.csv'` file containing a list of customary stopwords. The script will automatically remove these custom stopwords from the data.

Make sure that a `'stopwords.csv'` file should follow a predetermined format.
- Use 'stopwords' as the column name in the first row.
- From the second row onward, add stopwords one per line.

# Note
- Ensure that the `'preprocesed_data.csv'` file is generated before running the LDA script.
- You can adjust the number of topics and the number of words to be displayed to customize the LDA results.