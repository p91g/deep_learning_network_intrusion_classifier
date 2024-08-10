import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.optimize import linprog
import statsmodels 

cols="""duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""

columns=[]
for c in cols.split(','):
    if(c.strip()):
       columns.append(c.strip())

columns.append('target')
print(columns)
print(len(columns))

attacks_types = {
    'normal': 'normal',
'back': 'dos',
'buffer_overflow': 'u2r',
'ftp_write': 'r2l',
'guess_passwd': 'r2l',
'imap': 'r2l',
'ipsweep': 'probe',
'land': 'dos',
'loadmodule': 'u2r',
'multihop': 'r2l',
'neptune': 'dos',
'nmap': 'probe',
'perl': 'u2r',
'phf': 'r2l',
'pod': 'dos',
'portsweep': 'probe',
'rootkit': 'u2r',
'satan': 'probe',
'smurf': 'dos',
'spy': 'r2l',
'teardrop': 'dos',
'warezclient': 'r2l',
'warezmaster': 'r2l',
}

path = "C:/Users/patri/Documents/kdd/kdd_files/kddcup.data_10_percent.gz"
df = pd.read_csv(path,names=columns)

list(df.columns)

#Adding Attack Type column

df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])

df.head()

df.to_csv("C:/Users/patri/Documents/kdd/kdd_df.csv")


### READ IN CSV ###

df = pd.read_csv("C:/Users/patri/Documents/kdd/kdd_df.csv")
df = df.drop('Unnamed: 0', axis=1)
df.drop_duplicates(keep='first', inplace = True)

print('Null values in dataset are', len(df[df.isnull().any(1)]))

df.head()

# convert attack type to categorical
df["Attack Type"] = df["Attack Type"].astype("category")

# check types of attack types
output = df['Attack Type'].values
labels = set(output)
print('The different type of attack type labels are:', labels)
print('='*125)
print('No. of different attack type labels are:', len(labels))

# check types of service
output = df['service'].values
labels = set(output)
print('The different type of attack type labels are:', labels)
print('='*125)
print('No. of different attack type labels are:', len(labels))

# number and % of attack types
counts = df['Attack Type'].value_counts()
percentages = counts / counts.sum() * 100
print(pd.concat([counts, percentages], axis=1, keys=['Count', 'Percentage']))


## CORRELATED FEATURES ##

corr_matrix=df.drop(['num_outbound_cmds','is_host_login'],axis=1).corr()
corr_matrix2=df2.corr()

sns.heatmap(corr_matrix)

plt.show()

f, ax = plt.subplots(figsize =(20,20)) 
sns.heatmap(corr_matrix,ax = ax, cmap ="YlGnBu", linewidths = 0.1, annot=True, annot_kws={"fontsize":6}, fmt='.2f')
plt.show()

# filter matrix to only show correlations above 0.5 (not including self-correlations)
filtered_corr_matrix = corr_matrix2.mask(np.tril(np.ones(corr_matrix2.shape)).astype(np.bool_)).where(np.abs(corr_matrix2) > 0.8)

print("\nFiltered correlation matrix:")
print(filtered_corr_matrix)

# create dictionary of variable pairs with correlations above 0.5
corr_dict = {(col1, col2): corr for col1, row in filtered_corr_matrix.iteritems() for col2, corr in row.iteritems() if not pd.isnull(corr)}

# print dictionary for correlated variables
for key, value in corr_dict.items():
    print(f"{key}: {value}\n")

## DROP CORRELATED FEATURES ##
df2 = df


'''('num_root', 'num_compromised'): 0.9942146069064515'''
df2.drop('num_root', axis=1, inplace=True)

'''('srv_serror_rate', 'serror_rate'): 0.996362754828705
('srv_rerror_rate', 'rerror_rate'): 0.9913169525713014'''
df2.drop('srv_rerror_rate', axis=1, inplace=True)

'''('dst_host_serror_rate', 'serror_rate'): 0.996842731371667'''
df2.drop('dst_host_serror_rate', axis=1, inplace=True)

'''('dst_host_rerror_rate', 'rerror_rate'): 0.9755139232818265'''
df2.drop('dst_host_rerror_rate', axis=1, inplace=True)

'''('dst_host_srv_serror_rate', 'serror_rate'): 0.9951521034735604'''
df2.drop('dst_host_srv_serror_rate', axis=1, inplace=True)

'''('flag_S0', 'srv_serror_rate'): 0.9979906380074327'''
df2.drop('flag_S0', axis=1, inplace=True)

'''('flag_REJ', 'rerror_rate'): 0.9613974311373755'''
df2.drop('flag_REJ', axis=1, inplace=True)

list(df.columns)

# PLOT FEATURES CORRELATED WITH ATTACKS

sns.barplot(x=df['dst_host_serror_rate'], y=df['Attack Type'])

sns.barplot(x=df['serror_rate'], y=df['Attack Type'])

sns.barplot(x=df['srv_serror_rate'], y=df['Attack Type'])

sns.barplot(x=df['dst_host_same_srv_rate'], y=df['Attack Type'])

sns.barplot(x=df['dst_host_srv_serror_rate'], y=df['Attack Type'])

sns.barplot(x=df['count'], y=df['Attack Type'])

sns.barplot(x=df['same_srv_rate'], y=df['Attack Type'])

sns.barplot(x=df['is_guest_login'], y=df['Attack Type'])

sns.barplot(x=df['protocol_type'], y=df['Attack Type'])

sns.countplot(data=df, x='protocol_type', y='Attack Type', col='Attack Type', kind='bar', col_wrap=3)

plt.show()

# VISUALISATION FUNCTIONS # 

## TSNE PLOT ## 
from sklearn.manifold import TSNE 

text_cols = df.select_dtypes(include='object').columns.tolist()
text_cols
df_text_drops = df.drop(columns=text_cols, axis=1)


#attack type feature mapping
amap = {'dos':0,'normal':1,'probe':2,'r2l':3,'u2r':4}
df['Attack Type'] = df['Attack Type'].map(amap)
df_text_drops = df.drop(columns=text_cols, axis=1)


print(Y.dtypes)

def tsne_func(data, label, no_components, perplexity_value, n_iter_value):
    print('TSNE with perplexity={} and no. of iterations={}'.format(perplexity_value, n_iter_value))
    tsne = TSNE(n_components=no_components, perplexity=perplexity_value, n_iter=n_iter_value)
    tsne_df1 = tsne.fit_transform(data)
    print(tsne_df1.shape)
    tsne_df1 = np.vstack((tsne_df1.T, Y)).T
    tsne_data1 = pd.DataFrame(data=tsne_df1, columns=['feature1', 'feature2', 'Output'])
    sns.FacetGrid(tsne_data1, hue='Output', size=6).map(plt.scatter, 'feature1', 'feature2').add_legend()
    plt.show()

tsne_func(data=df_text_drops, label=Y, no_components=2, perplexity_value=100, n_iter_value=500)

# BAR CHARTS
def bar_graph(feature):
    df[feature].value_counts().plot(kind="bar")

def bar_graph_pc(feature):
    counts = df[feature].value_counts()
    percentages = counts / counts.sum() * 100
    percentages.plot(kind="bar")  

# HISTOGRAMS # 
def plot_histograms(df, cols):
    for col in cols:
        plt.hist(df[col])
        plt.title(col)
        plt.show()

def pairplot(data, label, columns=[]):
    sns.pairplot(data, hue=label, height=4, diag_kind='hist',   vars=columns, plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'})
    plt.show()

# VISUALISATION OF PREDICTORS # 
df2.iloc[:, 1:30].hist(bins=50, figsize=(5,5))
plt.show()

# PAIR PLOTS # 

df_pairs1 = df[['srv_count', 'serror_rate', 'Attack Type', 'duration']]    

sns.pairplot(df_pairs1, hue='Attack Type') 
plt.show()   

# VISUALISATION OF TARGETS  # 

pairplot(df, 'Attack Type', columns = ['duration', 'src_bytes', 'dst_bytes', 'land'])
  
bar_graph('Attack Type')

bar_graph_pc('Attack Type')

plt.xlabel('Attack Type')
plt.ylabel('Occurances per type')
plt.title('Distribution of attack types in train data')
plt.show()


#attack type feature mapping
amap = {'dos':0,'normal':1,'probe':2,'r2l':3,'u2r':4}
df['Attack Type'] = df['Attack Type'].map(amap)


### LINEAR SEPARABILITY ####

Y = df[['Attack Type']]
Y = Y.values
X = df_text_drops.drop('Attack Type', 'land', 'logged_in', 'root_shell', 'is_guest_login', 'num_outbound_cmds', 'is_host_login', axis=1)
X = X.values


# Formulate linear programming problem
c = np.zeros(X.shape[1] + 1)
A = -Y.reshape(-1, 1) * np.hstack((X, np.ones((X.shape[0], 1))))
b = -np.ones(X.shape[0])
res = linprog(c, A_ub=A, b_ub=b, method = 'revised simplex')


# Check if dataset is linearly separable
if res['success']:
    print('Dataset is linearly separable.')
else:
    print('Dataset is not linearly separable.')

df
X = df.drop(['Attack Type', 'target'], axis=1)
Y = Y.drop('index', axis=1)

## SPLIT TO TRAINING / TEST ## 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# DROP TEXT COLUMNS # 
print(df.dtypes)

text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
text_cols

X_train = X_train.drop(columns=text_cols, axis=1)
X_test = X_test.drop(columns=text_cols, axis=1)

# identify text cols
text_cols = df.select_dtypes(include='object').columns.tolist()
text_cols
''''['protocol_type', 'service', 'flag', 'target']'''

# manually create list 
text_cols = ['protocol_type', 'service', 'flag']

# create text col df
x_train_text_col_df = X_train[text_cols]
x_test_text_col_df = X_test[text_cols]

# dummy vars for all text cols
x_train_text_col_df_dummies = pd.get_dummies(x_train_text_col_df)
x_test_text_col_df_dummies = pd.get_dummies(x_test_text_col_df)

# drop target column
text_col_df = text_col_df.drop(['target', 'Attack Type'], axis=1)
text_col_df 

# BOOLEAN COLUMNS # 
X_train_bools = X_train[['land', 'logged_in', 'root_shell', 'is_guest_login']]
X_test_bools = X_test[['land', 'logged_in', 'root_shell', 'is_guest_login']]

# Boolean dummy variables
x_train_bool_col_df_dummies = pd.get_dummies(X_train_bools)
x_test_bool_col_df_dummies = pd.get_dummies(X_test_bools)

# DROP  BOOLEAN COLS
X_train = X_train.drop(columns=['land', 'logged_in', 'root_shell', 'is_guest_login', 'num_outbound_cmds', 'is_host_login'], axis=1)
X_test = X_test.drop(columns=['land', 'logged_in', 'root_shell', 'is_guest_login', 'num_outbound_cmds', 'is_host_login'], axis=1)

# don't include these
'''num_outbound_cmds
is_host_login'''

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

Y_train = Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)

# Y 
# create a new column 'Attack_Normal' based on the values in 'Attack Type'
Y1_train = Y_train.copy()
Y1_test = Y_test.copy()

Y1_test

# Attack or normal columns
Y1_train.loc[:, 'target'] = Y1_train['Attack Type'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
Y1_test.loc[:, 'target'] = Y1_test['Attack Type'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

Y1_train_drop_attack_type = Y1_train.drop(['Unnamed: 0','Attack Type'], axis = 1).rename(columns={'Attack_Normal':'target'}).reset_index(drop=True)
Y1_test_drop_attack_type = Y1_test.drop(['Unnamed: 0','Attack Type'], axis = 1).rename(columns={'Attack_Normal':'target'}).reset_index(drop=True)

# Attack or normal  dummy variables
Y1_train_dummy_attack_normal = pd.get_dummies(Y1_train_drop_attack_type)
Y1_test_dummy_attack_normal = pd.get_dummies(Y1_test_drop_attack_type)

Y1_train_dummy_attack_normal
Y1_test_dummy_attack_normal

Y1_train_dummy_attack_normal.to_csv("C:/Users/patri/Documents/kdd/Y1_train_dummy_attack_normal.csv")
Y1_test_dummy_attack_normal.to_csv("C:/Users/patri/Documents/kdd/Y1_test_dummy_attack_normal.csv")

# Attack type dummy variables
Y1_train_dummy = pd.get_dummies(Y1_train)
Y1_test_dummy = pd.get_dummies(Y1_test)

X_train.to_csv("C:/Users/patri/Documents/kdd/X_train.csv")
X_test.to_csv("C:/Users/patri/Documents/kdd/X_test.csv")

X_train_std.to_csv("C:/Users/patri/Documents/kdd/X_train_std.csv")
X_test_std.to_csv("C:/Users/patri/Documents/kdd/X_test_std.csv")

Y1_train.to_csv("C:/Users/patri/Documents/kdd/Y1_train.csv")
Y1_test.to_csv("C:/Users/patri/Documents/kdd/Y1_test.csv")

Y1_train_dummy.to_csv("C:/Users/patri/Documents/kdd/Y1_train_dummy.csv")
Y1_test_dummy.to_csv("C:/Users/patri/Documents/kdd/Y1_test_dummy.csv")

## READ IN Y
Y1_train = pd.read_csv("C:/Users/patri/Documents/kdd/Y1_train.csv")
Y1_test  = pd.read_csv("C:/Users/patri/Documents/kdd/Y1_test.csv")

Y1_train
Y1_test

## READ IN X_TRAIN / X_TEST FULL DF
X_train_joined_back = pd.read_csv("C:/Users/patri/Documents/kdd/X_train_joined_back.csv")
X_test_joined_back = pd.read_csv("C:/Users/patri/Documents/kdd/X_test_joined_back.csv")

X_train_joined_back
X_test_joined_back

# filter boolean cols
X_train_bools = X_train_joined_back[['land', 'logged_in', 'root_shell', 'is_guest_login']]
X_test_bools = X_test_joined_back[['land', 'logged_in', 'root_shell', 'is_guest_login']]

X_train_bools
X_test_bools

# filter text cols
X_train_text_cols = X_train_joined_back.iloc[:,37:]
X_test_text_cols = X_test_joined_back.iloc[:,37:]

X_train_text_cols
X_test_text_cols

## READ IN X_TRAIN with non-bool and text variables

X_train = pd.read_csv("C:/Users/patri/Documents/kdd/X_train.csv")

X_test = pd.read_csv("C:/Users/patri/Documents/kdd/X_test.csv")

# join bools to X_cbrt_stds
X_train_text_bools = pd.concat([X_train_cbrt_std, X_train_bools], axis=1)
X_test_text_bools = pd.concat([X_test_cbrt_std, X_test_bools], axis=1)

# join texts to X_cbrt_stds
X_train_text_bools = pd.concat([X_train_text_bools, X_train_text_cols], axis=1)
X_test_text_bools = pd.concat([X_test_text_bools, X_test_text_cols], axis=1)

X_train_text_bools
X_test_text_bools

X_train_text_bools.to_csv("C:/Users/patri/Documents/kdd/X_train_cbrt_std_text_bools.csv")
X_test_text_bools.to_csv("C:/Users/patri/Documents/kdd/X_test_cbrt_std_text_bools.csv")

X_train_text_bools = pd.read_csv("C:/Users/patri/Documents/kdd/X_train_cbrt_std_text_bools.csv")
X_test_text_bools = pd.read_csv("C:/Users/patri/Documents/kdd/X_test_cbrt_std_text_bools.csv")


# drop cols in training set from test set
X_test_text_bools = X_test_text_bools.drop(['service_tftp_u', 'service_pm_dump', 'service_red_i'], axis=1)
X_test_text_bools
'service_tftp_u', 'service_pm_dump', 'service_red_i'

# CUBE TRANSFORM # 
''' many variables shows extreme skew and kurtosis'''
X_train_cbrt = X_train.apply(lambda x: np.cbrt(x) if np.issubdtype(x.dtype, np.number) else x)
X_test_cbrt = X_test.apply(lambda x: np.cbrt(x) if np.issubdtype(x.dtype, np.number) else x)


# Save cbrt df
X_train_cbrt.to_csv("C:/Users/patri/Documents/kdd/X_train_cbrt.csv")
X_test_cbrt.to_csv("C:/Users/patri/Documents/kdd/X_test_cbrt.csv")


## DESCRIPTIVE STATISTICS ##
from statsmodels.stats.descriptivestats import describe

# convert df to array
X_train_vals = df_cbrt.values

# describe function 
X_train_desc = describe(pca_train_4_join)
X_train_desc

X_train_desc_df = pd.DataFrame(X_train_desc)

X_train_desc_df.columns = df_cbrt.columns
X_train_desc_df

# SKEW #
skew = X_train_desc_df.loc['skew',:]
skew

# KURTOSIS # 
kurtosis = X_train_desc_df.loc['kurtosis',:]
kurtosis

# Save descriptive statistics
X_train_desc_df.to_csv("C:/Users/patri/Documents/kdd/X_train_desc_df.csv")

## SCALING / NORMALISATION ##
X_train_cbrt_std = (X_train_cbrt - X_train_cbrt.mean()) / X_train_cbrt.std()
X_test_cbrt_std = (X_test_cbrt - X_test_cbrt.mean()) / X_test_cbrt.std()

X_train_cbrt_std.to_csv("C:/Users/patri/Documents/kdd/X_train_cbrt_std.csv")
X_test_cbrt_std.to_csv("C:/Users/patri/Documents/kdd/X_test_cbrt_std.csv")

# JOIN TEXT AND BOOL COLUMNS BACK TO STANDARDISED DFs


''''''
# COLUMNS NOT TO STANDARDISE # 
'''land, logged_in, root_shell, is_guest_login'''
# COLUMNS TO REMOVE
'''num_outbound_cmds, is_host_login'''

# reset indexes
X_train_bools = X_train_bools.reset_index(drop=True)
X_test_bools = X_test_bools.reset_index(drop=True)

x_train_text_col_df_dummies = x_train_text_col_df_dummies.reset_index(drop=True)
x_test_text_col_df_dummies = x_test_text_col_df_dummies.reset_index(drop=True)

# JOIN DFs #
X_train__cbrt_joined_back = pd.concat([X_train_cbrt_std, X_train_bools], axis=1)
X_train__cbrt_joined_back = pd.concat([X_train__cbrt_joined_back, x_train_text_col_df_dummies], axis=1)

X_test_cbrt_joined_back = pd.concat([X_test_cbrt_std, X_test_bools], axis=1)
X_test_cbrt_joined_back = pd.concat([X_test_cbrt_joined_back, x_test_text_col_df_dummies], axis=1)


X_train_joined_back.to_csv("C:/Users/patri/Documents/kdd/X_train_joined_back.csv")
X_test_joined_back.to_csv("C:/Users/patri/Documents/kdd/X_test_joined_back.csv")

# PCA # 
from sklearn.decomposition import PCA


# check ideal number of principal components
pca = PCA().fit(X_train_text_bools)
# plot the cumulative explained variance ratio as a function of the number of components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

pca_5 = PCA(n_components=5)
pca_10 = PCA(n_components=10)

# fit 
pca_5.fit(X_train_text_bools)
pca_5.fit(X_test_text_bools)

pca_10.fit(X_train_text_bools)
pca_10.fit(X_test_text_bools)

# transform train
pca_train_10 = pca_10.transform(X_train_text_bools)
pca_train_10 = pd.DataFrame(pca_train_10)

# transform test
pca_test_10 = pca_10.transform(X_test_text_bools)
pca_test_10 = pd.DataFrame(pca_test_10)

# rename cols
pca_train_10 = pca_train_10.rename(columns={0:'pca1', 1:'pca2', 2:'pca3', 3:'pca4', 4:'pca5', 5:'pca6', 6:'pca7', 7:'pca8', 8:'pca9', 9:'pca10'})
pca_test_10 = pca_test_10.rename(columns={0:'pca1', 1:'pca2', 2:'pca3', 3:'pca4', 4:'pca5', 5:'pca6', 6:'pca7', 7:'pca8', 8:'pca9', 9:'pca10'})


pca_train.to_csv("C:/Users/patri/Documents/kdd/pca_train.csv")
pca_test.to_csv("C:/Users/patri/Documents/kdd/pca_test.csv")

pca_train_10.to_csv("C:/Users/patri/Documents/kdd/pca_train_10.csv")
pca_test_10.to_csv("C:/Users/patri/Documents/kdd/pca_test_10.csv")

pca_train_10 = pd.read_csv("C:/Users/patri/Documents/kdd/pca_train_10.csv")
pca_test_10 = pd.read_csv("C:/Users/patri/Documents/kdd/pca_test_10.csv")

# K-Means # 
from sklearn.cluster import MiniBatchKMeans
# assume X is your data matrix
wcss = []  # within-cluster sum of squares

for i in range(1, 30):
    kmeans = MiniBatchKMeans(n_clusters=i, init='k-means++', batch_size=128, max_iter=100, random_state=0)
    kmeans.fit(X_train_text_bools)
    wcss.append(kmeans.inertia_)
    
# plot the elbow graph
plt.plot(range(1, 30), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.axvline(x=6, color='r', linestyle='--')
plt.show()

kmeans_6 = MiniBatchKMeans(n_clusters=6, random_state=0, batch_size=128, max_iter=100)

# fit training
kmeans_6.fit(X_train_text_bools)

train_cluster = kmeans_6.predict(X_train_text_bools)
train_cluster = pd.DataFrame(train_cluster)

train_cluster = train_cluster.rename(columns={0:'cluster'})
train_cluster["cluster"] = train_cluster["cluster"].astype("category")

train_cluster_dummies = pd.get_dummies(train_cluster)
train_cluster_dummies

# fit test
kmeans_6.fit(X_test_text_bools)
test_cluster = kmeans_6.predict(X_test_text_bools)
test_cluster = pd.DataFrame(test_cluster)

test_cluster = test_cluster.rename(columns={0:'cluster'})
test_cluster["cluster"] = test_cluster["cluster"].astype("category")

test_cluster_dummies = pd.get_dummies(test_cluster)
test_cluster_dummies

# SAVE CLUSTERS
train_cluster_dummies.to_csv("C:/Users/patri/Documents/kdd/train_cluster_6_dummies.csv")
test_cluster_dummies.to_csv("C:/Users/patri/Documents/kdd/test_cluster_6_dummies.csv")

# READ IN CLUSTER CSVs
train_cluster_6_dummies = pd.read_csv("C:/Users/patri/Documents/kdd/train_cluster_6_dummies.csv")
test_cluster_6_dummies = pd.read_csv("C:/Users/patri/Documents/kdd/test_cluster_6_dummies.csv")

## CONCAT X_train / X_test with clusters and PCA

X_train_cbrt_std_text_bools_pca_clusters = pd.concat([X_train_text_bools, train_cluster_6_dummies], axis = 1)
X_train_cbrt_std_text_bools_pca_clusters = pd.concat([X_train_cbrt_std_text_bools_pca_clusters, pca_train_10], axis = 1)

X_test_cbrt_std_text_bools_pca_clusters = pd.concat([X_test_text_bools, test_cluster_6_dummies], axis = 1)
X_test_cbrt_std_text_bools_pca_clusters = pd.concat([X_test_cbrt_std_text_bools_pca_clusters, pca_test_10], axis = 1)

# SAVE X_train / X_test with clusters and PCA
X_train_cbrt_std_text_bools_pca_clusters.to_csv("C:/Users/patri/Documents/kdd/X_train_cbrt_std_text_bools_pca_clusters.csv")
X_test_cbrt_std_text_bools_pca_clusters.to_csv("C:/Users/patri/Documents/kdd/X_test_cbrt_std_text_bools_pca_clusters.csv")


list(X_train_cbrt_std_text_bools_pca_clusters.columns)
X_test_cbrt_std_text_bools_pca_clusters

# additional features # 

# source bytes - number of bytes from source to destination
X_train_cbrt_std_text_bools_pca_clusters['src_dst_minus_bytes'] = X_train_cbrt_std_text_bools_pca_clusters['src_bytes'] - X_train_cbrt_std_text_bools_pca_clusters['dst_bytes']
X_test_cbrt_std_text_bools_pca_clusters['src_dst_minus_bytes'] = X_test_cbrt_std_text_bools_pca_clusters['src_bytes'] - X_test_cbrt_std_text_bools_pca_clusters['dst_bytes']


# same / different server - i.e. % of connections to same minus different service
X_train_cbrt_std_text_bools_pca_clusters['same_minus_diff_srv_rate'] = X_train_cbrt_std_text_bools_pca_clusters['same_srv_rate'] - X_train_cbrt_std_text_bools_pca_clusters['diff_srv_rate']
X_test_cbrt_std_text_bools_pca_clusters['same_minus_diff_srv_rate'] = X_test_cbrt_std_text_bools_pca_clusters['same_srv_rate'] - X_test_cbrt_std_text_bools_pca_clusters['diff_srv_rate']


# % connections to the same service / number of connections to same service in past 2 seconds
X_train_cbrt_std_text_bools_pca_clusters['pc_cons_same_div_cons_same_2s'] = X_train_cbrt_std_text_bools_pca_clusters['same_srv_rate'] / X_train_cbrt_std_text_bools_pca_clusters['srv_count']
X_test_cbrt_std_text_bools_pca_clusters['pc_cons_same_div_cons_same_2s'] = X_test_cbrt_std_text_bools_pca_clusters['same_srv_rate'] / X_test_cbrt_std_text_bools_pca_clusters['srv_count']

# number of compromised conditions divided by number of root accesses
X_train_cbrt_std_text_bools_pca_clusters['num_compr_div_num_root'] = X_train_cbrt_std_text_bools_pca_clusters['num_compromised'] / X_train_cbrt_std_text_bools_pca_clusters['num_root']
X_test_cbrt_std_text_bools_pca_clusters['num_compr_div_num_root'] = X_test_cbrt_std_text_bools_pca_clusters['num_compromised'] / X_test_cbrt_std_text_bools_pca_clusters['num_root']


X_train_cbrt_std_text_bools_pca_clusters
X_test_cbrt_std_text_bools_pca_clusters

# JOIN DFs #
X_train = X_train.reset_index()

X_train_extra_cols = pd.concat([X_train, train_cluster_dummies], axis=1)
X_train_extra_cols = pd.concat([X_train_extra_cols, pca_train1], axis=1)
list(X_train_extra_cols.columns)
X_train_extra_cols.to_csv("C:/Users/patri/Documents/kdd/X_train_extra_cols.csv")


#### CHECKS #########

# check how many types of service
unique_service = df['service'].unique()
print(unique_service)


# CHECK ZERO COLUMNS Loop over columns
for col in X_train.columns:
    # Check if the sum of the column equals 0
    if X_train[col].sum() == 0:
        print(col)
    else:
        print('none')    
'''num_outbound_cmds
is_host_login'''

# CHECK WHICH COLS HAVE 1 or 0 - BOOLEAN
for col_name in X.columns:
    # check if the values in the column are either 0 or 1
    if set(X[col_name].unique()) == {0, 1} or set(X[col_name].unique()) == {0.00, 1.0}:
        print(col_name)


# CHECK MISSING COLS BETWEEN DFs
# Assuming df1 and df2 are your dataframes
# Get the column names for each dataframe

X_train_text_bools
X_test_text_bools

df1_columns = set(X_train_text_bools.columns)
df2_columns = set(X_test_text_bools.columns)

# Find columns that are in df1 but not in df2
columns_in_df1_only = df1_columns - df2_columns

# Find columns that are in df2 but not in df1
columns_in_df2_only = df2_columns - df1_columns

# Print the results
print(f"Columns in df1 only: {columns_in_df1_only}")
print(f"Columns in df2 only: {columns_in_df2_only}")
'''Columns in df2 only: {'service_tftp_u', 'service_pm_dump', 'service_red_i'}'''

# PCA 2 ##

X_train = pd.read_csv('C:/Users/patri/Documents/kdd/X_train_cbrt_std_text_bools_pca_clusters.csv')
X_test = pd.read_csv('C:/Users/patri/Documents/kdd/X_test_cbrt_std_text_bools_pca_clusters.csv')

list(X_train.columns)

pca_train = X_train[['num_root', 'srv_rerror_rate', 'dst_host_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 
                    'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH']]

X_train2 = X_train.drop(['num_root', 'srv_rerror_rate', 'dst_host_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 
                    'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH', 
                    'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10'], axis=1)

pca_test = X_test[['num_root', 'srv_rerror_rate', 'dst_host_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 
                    'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH']]

X_test2 = X_test.drop(['num_root', 'srv_rerror_rate', 'dst_host_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 
                    'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH', 
                    'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10'], axis=1)

from sklearn.decomposition import PCA
pca = PCA().fit(pca_train)
# plot the cumulative explained variance ratio as a function of the number of components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

pca_4 = PCA(n_components=4)

# fit 
pca_4.fit(pca_train)
pca_4.fit(pca_test)

# transform train
pca_train_4 = pca_4.transform(pca_train)
pca_train_4 = pd.DataFrame(pca_train_4)

pca_test_4 = pca_4.transform(pca_test)
pca_test_4 = pd.DataFrame(pca_test_4)


# rename cols
pca_train_4 = pca_train_4.rename(columns={0:'pca1', 1:'pca2', 2:'pca3', 3:'pca4'})
pca_test_4 = pca_test_4.rename(columns={0:'pca1', 1:'pca2', 2:'pca3', 3:'pca4'})

pca_train_4
pca_test_4

# join
pca_train_4_join = pd.concat([X_train2, pca_train_4], axis=1)
pca_test_4_join = pd.concat([X_test2, pca_test_4], axis=1)

pca_train_4_join.to_csv("C:/Users/patri/Documents/kdd/pca_train_4.csv")
pca_test_4_join.to_csv("C:/Users/patri/Documents/kdd/pca_test_4.csv")

pca_train_4_join
pca_test_4_join


## SPLIT TO TRAINING / TEST ## 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)