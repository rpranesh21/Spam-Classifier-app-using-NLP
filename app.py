from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('spam.csv', encoding = 'latin-1')


    # In[6]:


    df.head()


    # In[9]:


    df.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)


    # In[10]:


    df['label'] = df['v1'].map({'ham':0, 'spam':1})


    # In[11]:


    df.head()


    # In[12]:


    df.columns = ['class', 'message', 'label']


    # In[13]:


    df.head()


    # In[14]:


    X = df['message']
    y = df['label']


    # In[18]:

    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)


    # In[22]:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    # In[24]:

    from sklearn.naive_bayes import MultinomialNB
    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    NB.score(X_test, y_test)
    #y_pred = NB.predict(X_test)
    #print(classification_report(y_test, y_pred))

    #joblib.dump(NB, 'NB_spam_filer.pkl')

    #NB_spam_model = open('NB_spam_filter.pkl', 'rb')
    #NB = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data =[message]
        vect = cv.transform(data).toarray()
        my_prediction = NB.predict(vect)
    return render_template('result.html',prediction= my_prediction)

if __name__=='__main__':
    app.run(debug=True)
