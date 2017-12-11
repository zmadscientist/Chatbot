import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from io import StringIO
import requests
import json
import pandas as pd
import numpy as np

# @hidden_cell
# This function accesses a file in your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def get_object_storage_file_with_credentials_e70a1bd410d84e0dadb938f93dae4cf9(container, filename):
    """This functions returns a StringIO object containing
    the file content from Bluemix Object Storage."""

    url1 = ''.join(['https://identity.open.softlayer.com', '/v3/auth/tokens'])
    data = {'auth': {'identity': {'methods': ['password'],
            'password': {'user': {'name': 'member_3744b85b9eef3133282178da294e5e487ee9c803','domain': {'id': '7527e49e459a4de4a41373a7e8fb444f'},
            'password': 'Z=d26kG~HZsP=t5('}}}}}
    headers1 = {'Content-Type': 'application/json'}
    resp1 = requests.post(url=url1, data=json.dumps(data), headers=headers1)
    resp1_body = resp1.json()
    for e1 in resp1_body['token']['catalog']:
        if(e1['type']=='object-store'):
            for e2 in e1['endpoints']:
                        if(e2['interface']=='public'and e2['region']=='dallas'):
                            url2 = ''.join([e2['url'],'/', container, '/', filename])
    s_subject_token = resp1.headers['x-subject-token']
    headers2 = {'X-Auth-Token': s_subject_token, 'accept': 'application/json'}
    resp2 = requests.get(url=url2, headers=headers2)
    return StringIO(resp2.text)

def Translate_Question(Input):
    df_a_csv = pd.read_csv(get_object_storage_file_with_credentials_e70a1bd410d84e0dadb938f93dae4cf9('ChatbotPOC', 'Comcast Answers.csv'))
    f = open('DecisionTreeCorpus.pkl','rb')
    corpus = pickle.load(f)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    f = open('DecisionTreeModel.pkl','rb')
    clf2 = pickle.load(f)
    test = vectorizer.transform([Input])
    prediction = clf2.predict(test)
    return df_a_csv[0:].Answer[df_a_csv.Label == prediction[0]].iloc[0]
    #return prediction
