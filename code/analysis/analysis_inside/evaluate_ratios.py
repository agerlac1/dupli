# *** Evaluation ***
""" Script receives the dataframe with the processed and analyzed testdata. 
Uses the given similarity-scores for each pair to decide if the pair is part of the class duplicate or non-duplicate.
Similarity-scores are rated with a threshold. 

# SET Measurements
    --> The threshold is defined by a class object and varies with each method.
    Threshold, depending on source_url, delivers best result and is set via trial and error for each method.

# CLASSIFY data
    --> The pairs that get classified as duplicates are stored in an output df called df_pairs. 

# EVALUATE data
    --> The function __classification() is used for a classification report in logging. Only used for masterthesis.
    --> The function __plotting() is used for visualization of the results. Only used for masterthesis.


Inspired by: https://www.kaggle.com/currie32/predicting-similarity-tfidfvectorizer-doc2vec """

# ## Imports
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import logging

# ## Class Threshold
class Threshold:
    """ Class to manage the setter and getter for the threshold-objects depending on the methods and source_urls."""
    # init method
    def __init__(self, name, df):
        self.name = name
        self.df = df
        self.threshold = None
        self.threshold_source = None

    # getter method for thresholds depending on source_url
    def get_threshold_source(self): 
        return self.threshold_source

    # getter method for threshold
    def get_threshold(self): 
        return self.threshold  

    # setter method for thresholds depending on source_url 
    def set_threshold_source(self):
        df = self.df
        step_key = self.name
        if self.name == "countvec":
            threshold = (df[step_key].mean() + df[step_key].std()+ df[step_key].std()+ df[step_key].std())  
        elif self.name =="doc2vec":
            threshold = (df[step_key].mean()+ df[step_key].std()+ df[step_key].std()/2)
        elif self.name == "levenshtein":
            threshold = (df[step_key].mean() + df[step_key].std()+ df[step_key].std())
        elif self.name == "tfidf":
            threshold = (df[step_key].mean() + df[step_key].std()+ df[step_key].std()+ df[step_key].std())
        elif self.name == "shingling":
            threshold = (df[step_key].mean() + df[step_key].std()+ df[step_key].std()+ df[step_key].std())
        self.threshold_source = threshold

    # setter method for threshold
    def set_threshold(self):
        df = self.df
        step_key = self.name
        if self.name == "countvec":
            threshold = (df[step_key].mean())     
        elif self.name == "doc2vec":
            threshold = (df[step_key].mean()+ df[step_key].std())
        elif self.name == "levenshtein":
            threshold = (df[step_key].mean()+ df[step_key].std()) 
        elif self.name == "tfidf":
            threshold = (df[step_key].mean()+ df[step_key].std()/2)
        elif self.name == "shingling":
            threshold = (df[step_key].mean())
        self.threshold = threshold


# ## Functions
def starter(df: pd.DataFrame, step_key: str) -> pd.DataFrame:

    """Function iterates over dataframe and evaluates the final data. Returns a df with the chosen duplicate pairs.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe contains one table from the input Dictionary to be evaluated.
    step_key: str
        String with the step_key to define which parts of the program need to be used.
   
    Raises
    ------
    KeyError
        Check if a specific column exists in the DataFrame

    Returns
    -------
    df_pairs: pd.Dataframe
        Dataframe with the Output-duplicates (pairwise). """

    # Set Output DataFrame to store found duplicate-pairs
    df_pairs = pd.DataFrame()

    # Initiate Threshold-object to set different values depending on metadata "source_url"
    thres_obj = Threshold(step_key, df)
    thres_obj.set_threshold_source()
    thres_obj.set_threshold()

    # Set scores-list
    scores = list()

    # A. SET Measurements
    # --------------------
    """ Iterate over df, check if a pair has the same source_website and set threshold (higher for same urls and lower for differnt urls) """
    for ind in df.index:
        if ind%2 == False:
            if df['source_website'][ind] == df['source_website'][ind+1]:
                # get higher threshold
                threshold = thres_obj.get_threshold_source()
            else:
                # get lower threshold
                threshold = thres_obj.get_threshold()
        else: 
            pass

        """# If you want to use generalized threshold for all methods and not a source related one, use the following line.
        threshold = (df[step_key].mean()+ df[step_key].std())"""

        # B. CLASSIFY the data
        # --------------------
        """ Check if a job-ad has a score over or under the threshold and store the positives in df_pairs."""
        if np.count_nonzero(float(df[step_key][ind]) >= threshold):
            scores.append(1)
            """ The following line is commented out because it is only needed for masterthesis.
            df.at[ind, 'Shingling_duplicate'] = 1"""
            df_pairs = df_pairs.append(df.iloc[ind], ignore_index=True)
        else:
            scores.append(0)
            # following line commented out only for masterthesis needed
            """ df.at[ind, 'Shingling_duplicate'] = 0
            df_pairs = df_pairs.append(df.iloc[ind], ignore_index=True) """

    # C. EVALUATE data
    # -----------------
    """ Get Classification report in logging. Only needed for annotated testdata. In real implementation it is not used or commented out. """
    __classification(step_key, thres_obj, df, scores)

    """Get Plotting of results in logging. Only needed for annotated testdata. In real implementaiton it is not used or commented out. """
    __plotting(step_key, df)
    return df_pairs

# Classification Report
def __classification(step_key, thres_obj, df, scores):
    """ Logs a classification report in the Logging-file (Logger.log). 
        Contains:
            * threshold-settings
            * mean
            * median
            * standard derivation
            * accuracy
            * confusion matrix
            * classification report """

    # Check if input-dataframe contains annotated data, if function is not called.
    if 'testset_id' in df:
        try:
            logging.info('\n\n***** EVALUATION: CLASSIFICATION REPORT *****')
            # Only count every second row because one pair = 2 rows and 2 scores (but only one needs to be counted)
            accuracy = accuracy_score(df.duplicate[df.index % 2 != 0], scores[0::2]) * 100
            logging.info(f'''\n\n{step_key}:
            threshold set to: {thres_obj.get_threshold_source()} & {thres_obj.get_threshold()}
            MEAN {df[step_key].mean()}
            MEDIAN {df[step_key].median()}
            STANDARD DEVIATION {df[step_key].std()}
            \nAccuracy score is {round(accuracy)}%.
            \nConfusion Matrix:\n{confusion_matrix(df.duplicate[df.index % 2 != 0], scores[0::2])}
            \nClassification Report:\n{classification_report(df.duplicate[df.index % 2 != 0], scores[0::2])}''')
            logging.info('\n\n***** /EVALUATION: CLASSIFICATION REPORT *****')
        except KeyError:
            logging.warning('Classfication Report was not generated, because no annotation was given.')
            pass
    else:
        logging.warning('Classification report was not generated because of missing annotation. Check again if needed.')

# Plotting
def __plotting(step_key, df):
    """ Function plots the distribution of Similarity-Scores for a chosen method. """
    
    # Check if input-dataframe contains annotated data, if function is not called.
    if 'testset_id' in df:
        try:
            logging.getLogger('matplotlib.font_manager').disabled = True
            sns.set_theme(style="ticks", color_codes=True) 
            x = df[step_key][df.index % 2 != 0] # Excludes every 2rd row starting from 0
            sns.violinplot(x=df[step_key], y=df.duplicate, orient='h', linewidth=3, data=x)
            ax = plt.gca()
            ax.set_yticks([0,1]) 
            ax.set_yticklabels(['Nicht-Dubletten', 'Dubletten'])
            plt.axvline((df[step_key].mean()+ df[step_key].std()) , color='r')
            plt.xlabel(step_key)
            plt.xlim(0,1)
            #plt.show()
            plt.savefig(f'temp/temps_analysis_in/{step_key}_plots.png')
            plt.close()
            logging.info(f'Plotting used to visualize the results. Saved in file temp/temp_analysis_in/{step_key}_plots.png')
        except KeyError:
            logging.warning('while plotting, some column or value is missing. Check again if plotting is needed.')
    else:
        logging.warning('Plotting visualization was not generated, because no annotation was given.')
        pass