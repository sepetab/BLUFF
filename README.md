# BLUFF: Sarcasm Detection on News Headlines
### COMP9444 22T2

This is the submission folder for DreamTeamV3. The primary notebook is `BLUFF.ipynb` which contains descriptions of the problem, dataset, approach used, results, and discussion. Multiple models were constructed for this project, and the code used to train each model can be found in separate notebooks in `models/`.

Sarcasm detector in news headlines. Training dataset and further information found (https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection).

Download glove embedding (https://drive.google.com/file/d/1WIG_GVbzcA5AEz-4186JvDVtev_dwTOC/view?usp=sharing). Unzip and place in Data folder.

The `model_probabilities/` directory contains the predicted probabilities of each model on the same test set (`data/test.csv`), saved using the `pickle` library. These files are loaded in `BLUFF.ipynb` in the Results section for use in analysing model performance and constructing an ensemble model. 

Reference for DistlBERT model: Mccormickml.com. (2019). BERT Fine-Tuning Tutorial with PyTorch · Chris McCormick. [online] Available at: https://mccormickml.com/2019/07/22/BERT-fine-tuning/.

Reference for ELMo model: Analytics Vidhya. (2019). What is ELMo | ELMo For text Classification in Python. [online] Available at: https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/.

Reference for Fasttext: Fasttext.cc. (2019). Text classification · fastText. [online] Available at: https://fasttext.cc/docs/en/supervised-tutorial.html.

Reference for Word2Vec and GloVe: kaggle.com. (n.d.). Sarcasm Detection with GloVe/Word2Vec(83%Accuracy). [online] Available at: https://www.kaggle.com/code/madz2000/sarcasm-detection-with-glove-word2vec-83-accuracy [Accessed 2 Aug. 2022].

‌
