# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 18:30:53 2020

@author: prathee
"""
import nltk
import pandas as pd ## for feautre engineering
import numpy as np  ## for feautre engineering
import matplotlib as mpl ## for plot
import matplotlib.pyplot as plt ## for plot
#import seaborn as sns  ## for plot
import datetime, nltk, warnings 
import matplotlib.cm as cm
from sklearn.decomposition import NMF
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing,cross_validation, metrics, feature_selection
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import SparsePCA 
from IPython.display import display, HTML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import math
import plotly.graph_objs as go
import seaborn as sns
color = sns.color_palette()
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
#mpl.rcParams["patch.force_edgecolor"] = True
mpl.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)


df_initial = pd.read_csv('data.csv', encoding="ISO-8859-1",
                          dtype={'InvoiceID': str})
                        
display(df_initial.shape)

# Top 10 observations of our data:
display(df_initial.head(10))
display(df_initial.describe())


#____________________________________________________________
# converting invoiceDate to datetime
df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate']) 

#____________________________________________________________
# some info on columns types and find number of columns with null values
tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))

# displaying the observations with missing values and thier percentage count
display(tab_info)

df_initial.dropna(axis = 0, subset = ['CustomerID'], inplace = True)
print('Dataframe dimensions:', df_initial.shape)

#____________________________________________________________
# checking to see if we have removed all observations with missing values
tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
# checking the dataset status after dropping the missing values
display(tab_info)


print('Duplicate Entries: {}'.format(df_initial.duplicated().sum()))
df_initial[(df_initial.InvoiceNo == 536412) & (df_initial.StockCode == 21448) & (df_initial.Quantity == 2) ]

# dropping values with duplicate entries
df_initial.drop_duplicates(inplace = True) 
df_initial.shape


temp = df_initial[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False)
countries = temp['Country'].value_counts()
print('The Online Retail Company covers : {} countries'.format(len(countries)))

data = dict(type='choropleth',
locations = countries.index,
locationmode = 'country names', z = countries,
text = countries.index, colorbar = {'title':' Number of Orders'},
colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],    
reversescale = False)
#_______________________
layout = dict(title='Number of orders per country',
geo = dict(showframe = True, projection={'type':'Mercator'}))
#______________
#choromap = go.Figure(data = [data], layout = layout)
#iplot(choromap, validate=False)



df_uk = df_initial[df_initial.Country == 'United Kingdom']
df_uk.shape


print(pd.DataFrame([{'products': len(df_uk['StockCode'].value_counts()),    
               'transactions': len(df_uk['InvoiceNo'].value_counts()),
               'customers': len(df_uk['CustomerID'].value_counts()),  
              }], columns = ['products', 'transactions', 'customers'], index = ['quantity']))

temp = df_uk.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
nb_products_per_basket = temp.rename(columns = {'InvoiceDate':'Number of products'})
nb_products_per_basket[:4].sort_values('CustomerID')


# counting all the invoiceNo that has a C present in it
nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x: int('C' in str(x)))
display(nb_products_per_basket[:5])
#______________________________________________________________________________________________
n1 = nb_products_per_basket['order_canceled'].sum()
n2 = nb_products_per_basket.shape[0]
print('Number of canceled orders: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))

display(df_uk.sort_values('CustomerID')[:2])

df_check = df_uk[df_uk['Quantity'] < 0][['CustomerID','Quantity',
                                                   'StockCode','Description','UnitPrice']]
for index, col in  df_check.iterrows():
    if df_uk[(df_uk['CustomerID'] == col[0]) & (df_uk['Quantity'] == -col[1]) 
                & (df_uk['Description'] == col[3])].shape[0] == 0: 
        print(df_check.loc[index])
        display(15*'-'+'>'+' Discounted Products also contribute to Negative Quantity')
        break
df_check = df_uk[(df_uk['Quantity'] < 0) & (df_uk['Description'] != 'Discount')][
                                 ['CustomerID','Quantity','StockCode',
                                  'Description','UnitPrice']]

for index, col in  df_check.iterrows():
    if df_uk[(df_uk['CustomerID'] == col[0]) & (df_uk['Quantity'] == -col[1]) 
                & (df_uk['Description'] == col[3])].shape[0] == 0: 
        print(index, df_check.loc[index])
        print(20*'-'+'>'+' HYPOTHESIS NOT FULFILLED')
        break
    
df_cleaned = df_uk.copy()

df_cleaned['QuantityCanceled'] = 0

entry_to_remove = [] 
entry_before_dec2010 = []

for index, col in  df_uk.iterrows():
    if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
    df_test = df_uk[(df_uk['CustomerID'] == col['CustomerID']) &
                         (df_uk['StockCode']  == col['StockCode']) & 
                         (df_uk['InvoiceDate'] < col['InvoiceDate']) & 
                         (df_uk['Quantity']   > 0)].copy()
    #_________________________________
    # Cancelation WITHOUT counterpart
    if (df_test.shape[0] == 0): 
        entry_before_dec2010.append(index)
    #________________________________
    # Cancelation WITH a counterpart
    elif (df_test.shape[0] == 1): 
        index_order = df_test.index[0]
        df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
        entry_to_remove.append(index)        
    #______________________________________________________________
    # Various counterparts exist in orders: we delete the last one
    elif (df_test.shape[0] > 1): 
        df_test.sort_index(axis=0 ,ascending=False, inplace = True)        
        for ind, val in df_test.iterrows():
            if val['Quantity'] < -col['Quantity']: continue
            df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index) 
            break
# df_cleaned_copy = df_cleaned
df_cleaned[(df_cleaned.CustomerID == 17315) & (df_cleaned.Description == '36 PENCILS TUBE RED RETROSPOT')]        

df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)
df_cleaned.drop(entry_before_dec2010, axis = 0, inplace = True)

list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]
#for code in list_special_codes:
    #print("{:<15} -> {:<30}".format(code, df_cleaned[df_cleaned['StockCode'] == code]['Description'].unique()[0]))

is_noun = lambda pos: pos[:2] == 'NN'

def keywords_inventory(dataframe, colonne = 'Description'):
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys   = []
    count_keywords  = dict()
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        
        for t in nouns:
            t = t.lower() ; racine = stemmer.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1                
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    print("number of keywords in variable '{}': {}".format(colonne,len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords
df_products = pd.DataFrame(df_uk['Description'].unique()).rename(columns = {0:'Description'})
#nltk.download('all')
keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_products)

list_products = []
for k,v in count_keywords.items():
    list_products.append([keywords_select[k],v])
list_products.sort(key = lambda x:x[1], reverse = True)
lists = sorted(list_products, key = lambda x:x[1], reverse = True)
#_______________________________
plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(7, 25))
y_axis = [i[1] for i in lists[:50]]
x_axis = [k for k,i in enumerate(lists[:50])]
x_label = [i[0] for i in lists[:50]]
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 13)
plt.yticks(x_axis, x_label)
plt.xlabel("No. of occurences", fontsize = 18, labelpad = 10)
ax.barh(x_axis, y_axis, align = 'center')
ax = plt.gca()
ax.invert_yaxis()
#_______________________________________________________________________________________
plt.title("Frequency of Words occurence in Description",bbox={'facecolor':'k', 'pad':5}, color='w',fontsize = 18)
plt.show()


df_cleaned['Revenue'] = (df_cleaned.Quantity - df_cleaned.QuantityCanceled)* df_cleaned.UnitPrice
df_cleaned.head(10)

#___________________________________________ 
# Renamed the Revenue column as Basket_Price
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['Revenue'].sum()
basket_price = temp.rename(columns = {'Revenue':'Basket_Price'})


#_____________________
# Order Date
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
basket_price.head()


#______________________________________
# selecting entities generating positive revenue  :
basket_price = basket_price[basket_price['Basket_Price'] > 0]

display(basket_price.shape)
display(basket_price.sort_values('CustomerID')[1:5])

#____________________
# Purchase count
price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
count_price = []
for i, price in enumerate(price_range):
    if i == 0: continue
    val = basket_price[(basket_price['Basket_Price'] < price) &
                       (basket_price['Basket_Price'] > price_range[i-1])]['Basket_Price'].count()
    count_price.append(val)

print('purchase count of observation in each price_range: {}'.format(count_price))
#____________________________________________
# Representation of the number of purchases / amount       
plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(11, 6))
colors = ['yellowgreen', 'gold', 'wheat', 'c', 'violet', 'royalblue','firebrick']
labels = [ '{}-{}'.format(price_range[i-1], s) for i,s in enumerate(price_range) if i != 0]
sizes  = count_price
explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
ax.pie(sizes, explode = explode, labels=labels, colors = colors,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow = False, startangle=0)
ax.axis('equal')
f.text(0.5, 1.01, "Basket Price Range", ha='center', fontsize = 18);

list_products = []
for k,v in count_keywords.items():
    word = keywords_select[k]
    if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
    if len(word) < 3 or v < 13: continue
    if ('+' in word) or ('/' in word): continue
    list_products.append([word, v])
#______________________________________________________    
list_products.sort(key = lambda x:x[1], reverse = True)

listz_products = df_cleaned['Description'].unique()
X = pd.DataFrame()
for key, occurence in list_products:
    X.loc[:, key] = list(map(lambda x:int(key.upper() in x), listz_products))
X.head()


threshold = [0, 1, 2, 3, 5, 10]
label_col = []
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '>{}'.format(threshold[i])
    else:
        col = '{}<-<{}'.format(threshold[i],threshold[i+1])
    label_col.append(col)
    X.loc[:, col] = 0

for i, prod in enumerate(listz_products):
    prix = df_cleaned[ df_cleaned['Description'] == prod]['UnitPrice'].mean()
    j = 0
    while prix > threshold[j]:
        j+=1
        if j == len(threshold): break
    X.loc[i, label_col[j-1]] = 1
X_copy = X.copy(deep = True)
X.head(20)
X.shape

matrix = X.as_matrix()
for n_clusters in range(3,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    
n_clusters = 5
silhouette_avg = -1
while silhouette_avg < 0.145:
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    
    #km = kmodes.KModes(n_clusters = n_clusters, init='Huang', n_init=2, verbose=0)
    #clusters = km.fit_predict(matrix)
    #silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    
def graph_component_silhouette(n_clusters, lim_x, mat_size, sample_silhouette_values, clusters):
    #plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
    #____________________________
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax1.set_xlim([lim_x[0], lim_x[1]])
    ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
        #___________________________________________________________________________________
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.spectral(float(i) / n_clusters)        
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.8)
        #____________________________________________________________________
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color = 'red', fontweight = 'bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))
        #______________________________________
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  
#____________________________________
# define individual silouhette scores
sample_silhouette_values = silhouette_samples(matrix, clusters)
#__________________
# and do the graph
graph_component_silhouette(n_clusters, [-0.07, 0.33], len(X), sample_silhouette_values, clusters)


liste_produits = df_cleaned['Description'].unique()
#print(liste_produits[0:2])
Xx = pd.DataFrame()
for key, occurence in list_products:
    Xx.loc[:, key] = list(map(lambda x:int(key.upper() in x), liste_produits))
#print(X[0:1])



liste = pd.DataFrame(liste_produits)
liste_words = [word for (word, occurence) in list_products]

occurence = [dict() for _ in range(n_clusters)]

for i in range(n_clusters):
    liste_cluster = liste.loc[clusters == i]
    for word in liste_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))
        
        
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)

def make_wordcloud(liste, increment):
    ax1 = fig.add_subplot(4,2,increment)
    words = dict()
    trunc_occurences = liste[0:150]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    
    wordcloud = WordCloud(width=1000,height=400, background_color='lightgrey', 
                          max_words=1628,relative_scaling=1,
                          color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    plt.title('cluster n{}'.format(increment-1))

fig = plt.figure(1, figsize=(14,14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
for i in range(n_clusters):
    list_cluster_occurences = occurence[i]

    tone = color[i] # define the color of the words
    liste = []
    for key, value in list_cluster_occurences.items():
        liste.append([key, value])
    liste.sort(key = lambda x:x[1], reverse = True)
    make_wordcloud(liste, i+1)
    
    
    
corresp = dict()
for key, val in zip (listz_products, clusters):
    corresp[key] = val 

df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)
df_cleaned[['InvoiceNo', 'Description', 
            'categ_product']][:10]


for i in range(5):
    col = 'categ_{}'.format(i)        
    df_temp = df_cleaned[df_cleaned['categ_product'] == i]
    price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
    price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
    df_cleaned.loc[:, col] = price_temp
    df_cleaned[col].fillna(0, inplace = True)
#__________________________________________________________________________________________________
df_cleaned[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3','categ_4']][:5]
#total.to_excel (r'export_dataframe.xlsx', index = None, header=True)
__________________________________________
# sum of purchases / user & order
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['Revenue'].sum()
basket_price = temp.rename(columns = {'Revenue':'Basket Price'})
#____________________________________________________________
# percentage of the price of the order / product category
for i in range(5):
    col = 'categ_{}'.format(i) 
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
    basket_price.loc[:, col] = temp 
#_____________________
# date of the order
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
#______________________________________
# selection of significant entries:
basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID', ascending = True)[:5]
#total.to_excel (r'export_dataframe.xlsx', index = None, header=True)




transactions_per_user=basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(['count','sum'])
for i in range(5):
    col = 'categ_{}'.format(i)
    transactions_per_user.loc[:,col] = basket_price.groupby(by=['CustomerID'])[col].sum() /\
                                            transactions_per_user['sum']*100

transactions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending = True)[:5]

# calculating the Recency(last purchase date) & the First_date of purchase
last_date = basket_price['InvoiceDate'].max().date()

first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
last_purchase      = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

test  = first_registration.applymap(lambda x:(last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x:(last_date - x.date()).days)

transactions_per_user.loc[:, 'Recency'] = test2.reset_index(drop = False)['InvoiceDate']
transactions_per_user.loc[:, 'First Purchase'] = test.reset_index(drop = False)['InvoiceDate']

transactions_per_user[:5]

# renaming the columns
selected_customers = transactions_per_user.rename(index=str, columns={"count": "Frequency", "sum": "Monetary"})     
selected_customers = selected_customers[['CustomerID','Recency','Frequency', 'Monetary','First Purchase','categ_0', 'categ_1','categ_2','categ_3','categ_4']]

rfm_data = selected_customers[['CustomerID','Recency','Frequency', 'Monetary','First Purchase']]
product_data = selected_customers[['CustomerID', 'Monetary','categ_0', 'categ_1','categ_2','categ_3','categ_4']]
product_data.head()
#total.to_excel (r'export_dataframe.xlsx', index = None, header=True)


rfm_clustering = selected_customers[['Recency','Frequency', 'Monetary']]

# selecting features for clustering based on previous purchased products
product_clustering = selected_customers[['categ_0','categ_1', 'categ_2','categ_3','categ_4']]

display(rfm_clustering.describe())

recency = go.Box(
    y= rfm_clustering.Recency,
    name = 'Recency'
    
)
frequency = go.Box(
    y=rfm_clustering.Frequency,
    name = 'Frequency'
    
)
monetary = go.Box(
    y=rfm_clustering.Monetary,
    name = 'Monetary'
)
data =[recency, frequency, monetary]

iplot(data)


# Squre Root Transformation
df_sqrt = rfm_clustering
sqrt_df = df_sqrt.apply(np.sqrt)

matrix = sqrt_df.as_matrix()
scaler = StandardScaler()
scaler.fit(matrix)
print('variables mean values: \n' + 90*'-' + '\n' , scaler.mean_)
scaled_matrix = scaler.transform(matrix)


from sklearn.cluster import KMeans
wcss = []
for i in range(2,15):
 kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
 kmeans.fit(scaled_matrix)
 wcss.append(kmeans.inertia_)

plt.plot(range(2,15), wcss)
plt.title('The elbow method')
plt.xlabel('The number of clusters')
plt.ylabel('WCSS')
plt.show()

for n_clusters in range(3,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    
n_clusters = 5
kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=100)
kmeans.fit(scaled_matrix)
clusters_clients = kmeans.predict(scaled_matrix)
silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
print('score de silhouette: {:<.3f}'.format(silhouette_avg))

pd.DataFrame(pd.Series(clusters_clients).value_counts(), columns = ['nb. of customers']).T

sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)
#____________________________________
# define individual silouhette scores
sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)
#__________________
# and do the graph
graph_component_silhouette(n_clusters, [-0.15, 0.55], len(scaled_matrix), sample_silhouette_values, clusters_clients)

rfm_data.loc[:, 'cluster'] = clusters_clients
rfm_data.head()

cls0 =  rfm_data[rfm_data.cluster == 0]
cls1 =  rfm_data[rfm_data.cluster == 1]
cls2 =  rfm_data[rfm_data.cluster == 2]
cls3 =  rfm_data[rfm_data.cluster == 3]
cls4 =  rfm_data[rfm_data.cluster == 4]

trace0 = go.Box(
    y= cls0.Recency,
    name = 'cluster0_Recency'
)

trace1 = go.Box(
    y=cls1.Recency,
    name = 'cluster1_Recency'
)
trace2 = go.Box(
    y=cls2.Recency,
    name = 'cluster2_Recency'
)
trace3 = go.Box(
    y=cls3.Recency,
    name = 'cluster3_Recency'
)
trace4 = go.Box(
    y=cls4.Recency,
    name = 'cluster4_Recency'
)

data =[trace0,trace1,trace2,trace3,trace4]

iplot(data)


cls0 =  rfm_data[rfm_data.cluster == 0]
cls1 =  rfm_data[rfm_data.cluster == 1]
cls2 =  rfm_data[rfm_data.cluster == 2]
cls3 =  rfm_data[rfm_data.cluster == 3]
cls4 =  rfm_data[rfm_data.cluster == 4]

trace0 = go.Box(
    y= cls0.Frequency,
    name = 'cluster0_Frequency'
)

trace1 = go.Box(
    y=cls1.Frequency,
    name = 'cluster1_Frequency'
)
trace2 = go.Box(
    y=cls2.Frequency,
    name = 'cluster2_Frequency'
)
trace3 = go.Box(
    y=cls3.Frequency,
    name = 'cluster3_Frequency'
)
trace4 = go.Box(
    y=cls4.Frequency,
    name = 'cluster4_Frequency'
)

data =[trace0,trace1,trace2,trace3,trace4]

iplot(data)

cls0 =  rfm_data[rfm_data.cluster == 0]
cls1 =  rfm_data[rfm_data.cluster == 1]
cls2 =  rfm_data[rfm_data.cluster == 2]
cls3 =  rfm_data[rfm_data.cluster == 3]
cls4 =  rfm_data[rfm_data.cluster == 4]

trace0 = go.Box(
    y= cls0.Monetary,
    name = 'cluster0_Monetary'
)

trace1 = go.Box(
    y=cls1.Monetary,
    name = 'cluster1_Monetary'
)
trace2 = go.Box(
    y=cls2.Monetary,
    name = 'cluster2_Monetary'
)
trace3 = go.Box(
    y=cls3.Monetary,
    name = 'cluster3_Monetary'
)
trace4 = go.Box(
    y=cls4.Monetary,
    name = 'cluster4_Monetary'
)

data =[trace0,trace1,trace2,trace3,trace4]

iplot(data)

merged_df = pd.DataFrame()
for i in range(n_clusters):
    test = pd.DataFrame(rfm_data[rfm_data['cluster'] == i].mean())
    test = test.T.set_index('cluster', drop = True)
    test['size'] = rfm_data[rfm_data['cluster'] == i].shape[0]
    merged_df = pd.concat([merged_df, test])
#_____________________________________________________

print('total number of customers:', merged_df['size'].sum())

merged_df = merged_df.sort_values('Monetary')
merged_df
#total.to_excel (r'export_dataframe.xlsx', index = None, header=True)

def _scale_data(data, ranges):
    (x1, x2) = ranges[0]
    d = data[0]
    return [(d - y1) / (y2 - y1) * (x2 - x1) + x1 for d, (y1, y2) in zip(data, ranges)]

class RadarChart():
    def __init__(self, fig, location, sizes, variables, ranges, n_ordinate_levels = 6):

        angles = np.arange(0, 360, 360./len(variables))

        ix, iy = location[:] ; size_x, size_y = sizes[:]
        
        axes = [fig.add_axes([ix, iy, size_x, size_y], polar = True, 
        label = "axes{}".format(i)) for i in range(len(variables))]

        _, text = axes[0].set_thetagrids(angles, labels = variables)
        
        for txt, angle in zip(text, angles):
            if angle > -1 and angle < 181:
                txt.set_rotation(angle - 90)
            else:
                txt.set_rotation(angle - 270)
        
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid("off")
        
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],num = n_ordinate_levels)
            grid_label = [""]+["{:.0f}".format(x) for x in grid[1:-1]]
            ax.set_rgrids(grid, labels = grid_label, angle = angles[i])
            ax.set_ylim(*ranges[i])
        
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
                
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, *args, **kw):
        self.ax.legend(*args, **kw)
        
    def title(self, title, *args, **kw):
        self.ax.text(0.9, 1, title, transform = self.ax.transAxes, *args, **kw)
        
fig = plt.figure(figsize=(20,10))

attributes = ['Recency','Frequency', 'Monetary']
ranges = [[0.1, 373], [0.1, 71], [0.01, 10000]]
index  = [0, 1, 2, 3, 4]

n_groups = n_clusters ; i_cols = 3
i_rows = n_groups//i_cols
size_x, size_y = (1/i_cols), (1/i_rows)

for ind in range(n_clusters):
    ix = ind%3 ; iy = i_rows - ind//3
    pos_x = ix*(size_x + 0.05) ; pos_y = iy*(size_y + 0.05)            
    location = [pos_x, pos_y]  ; sizes = [size_x, size_y] 
    #______________________________________________________
    data = np.array(merged_df.loc[index[ind], attributes])    
    radar = RadarChart(fig, location, sizes, attributes, ranges)
    radar.plot(data, color = 'b', linewidth=2.0)
    radar.fill(data, alpha = 0.2, color = 'b')
    radar.title(title = 'cluster {}'.format(index[ind]), color = 'r')
    ind += 1
matrix = product_clustering.as_matrix()
scaler = StandardScaler()
scaler.fit(matrix)
print('variables mean values: \n' + 90*'-' + '\n' , scaler.mean_)
scaled_matrix = scaler.transform(matrix)
    
from sklearn.cluster import KMeans
wcss = []
for i in range(2,15):
 kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
 kmeans.fit(scaled_matrix)
 wcss.append(kmeans.inertia_)

plt.plot(range(2,15), wcss)
plt.title('The elbow method')
plt.xlabel('The number of clusters')
plt.ylabel('WCSS')
plt.show()


n_clusters = 6
kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=100)
kmeans.fit(scaled_matrix)
clusters_clients = kmeans.predict(scaled_matrix)
silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
print('score de silhouette: {:<.3f}'.format(silhouette_avg))

pd.DataFrame(pd.Series(clusters_clients).value_counts(), columns = ['nb. of customers']).T

product_data.loc[:, 'cluster'] = clusters_clients
product_data.head(10)
#total.to_excel (r'export_dataframe.xlsx', index = None, header=True)

merged_df = pd.DataFrame()
for i in range(n_clusters):
    test = pd.DataFrame(product_data[product_data['cluster'] == i].mean())
    test = test.T.set_index('cluster', drop = True)
    test['size'] = product_data[product_data['cluster'] == i].shape[0]
    merged_df = pd.concat([merged_df, test])
#_____________________________________________________
merged_df.drop('CustomerID', axis = 1, inplace = True)
print('total number of customers:', merged_df['size'].sum())

merged_df = merged_df.sort_values('Monetary')
total=merged_df
total.to_excel (r'export_dataframe.xlsx', index = None, header=True)



def _scale_data(data, ranges):
    (x1, x2) = ranges[0]
    d = data[0]
    return [(d - y1) / (y2 - y1) * (x2 - x1) + x1 for d, (y1, y2) in zip(data, ranges)]

class RadarChart():
    def __init__(self, fig, location, sizes, variables, ranges, n_ordinate_levels = 6):

        angles = np.arange(0, 360, 360./len(variables))

        ix, iy = location[:] ; size_x, size_y = sizes[:]
        
        axes = [fig.add_axes([ix, iy, size_x, size_y], polar = True, 
        label = "axes{}".format(i)) for i in range(len(variables))]

        _, text = axes[0].set_thetagrids(angles, labels = variables)
        
        for txt, angle in zip(text, angles):
            if angle > -1 and angle < 181:
                txt.set_rotation(angle - 90)
            else:
                txt.set_rotation(angle - 270)
        
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid("off")
        
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],num = n_ordinate_levels)
            grid_label = [""]+["{:.0f}".format(x) for x in grid[1:-1]]
            ax.set_rgrids(grid, labels = grid_label, angle = angles[i])
            ax.set_ylim(*ranges[i])
        
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
                
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, *args, **kw):
        self.ax.legend(*args, **kw)
        
    def title(self, title, *args, **kw):
        self.ax.text(0.9, 1, title, transform = self.ax.transAxes, *args, **kw)

fig = plt.figure(figsize=(20,10))

attributes = ['categ_0','categ_1', 'categ_2','categ_3','categ_4']
ranges = [[0.01, 75], [0.01, 75], [0.01, 75], [0.01, 75], [0.01, 75]]
index  = [0, 1, 2, 3, 4, 5]

n_groups = n_clusters ; i_cols = 3
i_rows = n_groups//i_cols
size_x, size_y = (1/i_cols), (1/i_rows)

for ind in range(n_clusters):
    ix = ind%3 ; iy = i_rows - ind//3
    pos_x = ix*(size_x + 0.05) ; pos_y = iy*(size_y + 0.05)            
    location = [pos_x, pos_y]  ; sizes = [size_x, size_y] 
    #______________________________________________________
    data = np.array(merged_df.loc[index[ind], attributes])    
    radar = RadarChart(fig, location, sizes, attributes, ranges)
    radar.plot(data, color = 'b', linewidth=2.0)
    radar.fill(data, alpha = 0.2, color = 'b')
    radar.title(title = 'cluster {}'.format(index[ind]), color = 'r')
    ind += 1

