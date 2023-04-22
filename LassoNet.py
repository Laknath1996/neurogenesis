
import pandas as pd
from scipy.stats import ranksums
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from lassonet import LassoNetClassifierCV
import seaborn as sns
sns.set_theme()
import matplotlib
matplotlib.rc('font', **{'size':16})
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc

# Load the dataset
df_original = pd.read_csv("GW3_data.csv")
df_meta_original = pd.read_csv("GW3_meta.csv")

# orginal dataframe
df_original = df_original.drop(columns=["Unnamed: 0"])
df_meta_original = df_meta_original.drop(columns=["Unnamed: 0"])

df = df_original
df_meta = df_meta_original

# arrange data and labels (GZ = 0, CP = 1)
X = df.to_numpy()
layers = df_meta["Layer"].to_numpy()
Y = np.zeros(len(layers))
Y[layers == "CP"] = 1 # labels

# split the data into train (60%) and test (40%)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=1996)
tr_idx, te_idx = next(sss.split(df, Y))
df_train = df.iloc[tr_idx]
df_test = df.iloc[te_idx]
Y_train, Y_test = Y[tr_idx], Y[te_idx]
X_train, X_test = df_train.to_numpy(), df_test.to_numpy()

# record meta data
gene_names = df.keys().to_numpy()
num_genes = len(gene_names)

bio_gene_names = ['SOX2', 'PAX6', 'NEUROG1', 'NEUROG2', 'ASCL1', 'NOTCH1']
D_bio = [np.where(gene_names == name)[0][0] for name in bio_gene_names]

# Perform Wilcoxon Rank Sum Test
p_val_list = []
for i in range(num_genes):
    data = X_train[:, i]
    labels = Y_train
    _, p_val = ranksums(data[labels==0], data[labels==1])
    p_val_list.append(p_val)

# Estimating D using Bonferroni correction
D = []
p_vals = []
alpha = 0.05
M = num_genes       # number of hypothesis tests performed (= number of genes)
bc = alpha / M      # Bonferroni correction
for k, p in enumerate(p_val_list):
    if p < bc:
        D.append(k)
        p_vals.append(p)
sorted_D = np.array(D)[np.argsort(p_vals)]
selected_gene_names = gene_names[sorted_D].tolist()

pool = [sorted_D[:10], sorted_D[:100], sorted_D[:1000], sorted_D[:10], D_bio, np.union1d(sorted_D[:10], D_bio)]
for k, selected_genes in enumerate(pool):
    X_train_selected = X_train[:, selected_genes]
    model = LassoNetClassifierCV()
    model.fit(X_train_selected, Y_train)
    # Predict the labels
    X_test_selected = X_test[:, selected_genes]
    Y_pred = model.predict(X_test_selected)
    metrics=[]

    # Print the best model score and lambda value
    print("Best model scored", model.score(X_test, Y_test))
    print("Lambda =", model.best_lambda_)

    accuracy = accuracy_score(Y_test, Y_pred)
    auc_roc = roc_auc_score(Y_test, X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)


metrics.append([accuracy, auc_roc, sensitivity, specificity])
print("| metric | Accuracy \t | AUC ROC \t | Sensitivity \t | Specificity \t |")
print("| mean \t | {:.4f} \t | {:.4f} \t | {:.4f} \t | {:.4f} \t |".format(*np.mean(metrics, axis=0)))

metrics = np.array(metrics)
