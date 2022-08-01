# BLUFF: Sarcasm Detection on News Headlines
### COMP9444 22T2

This is the submission folder for DreamTeamV3. The primary notebook is `BLUFF.ipynb` which contains descriptions of the problem, dataset, approach used, results, and discussion. Multiple models were constructed for this project, and the code used to train each model can be found in separate notebooks in `models/`.

The `model_probabilities/` directory contains the predicted probabilities of each model on the same test set (`data/test.csv`), saved using the `pickle` library. These files are loaded in `BLUFF.ipynb` in the Results section for use in analysing model performance and constructing an ensemble model. 