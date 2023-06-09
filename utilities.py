import re
import csv
import joblib
import pandas as pd

breed = pd.read_csv("./data/breed.csv",sep=';')
name = pd.read_csv("./data/name.csv")

def extract_letters(s):
    regex = r'[hA-ZÄÖÅ]+'
    letters = re.findall(regex, s)
    return ','.join(letters)

def process_text(text):
    if isinstance(text, str):
        processed = re.sub('[éè\s]', '', text.lower()).strip().rstrip(',')
        return processed
    else:
        return text

def formatDf(data):
    df = data.copy()
    if 'settleornot' not in df.columns: 
        df['quantity'] = df['quantity'].fillna(df['quantityFloat'])
        df['code'] = df['code'].apply(lambda x: extract_letters(str(x)))
        df['kind'] = df['kind'].str.replace('okänt','Okänt').replace('Okänd','Okänt').replace('marsvin','Marsvin').replace('råtta','Råtta')
        df['breed'] = df['breed'].fillna('None').apply(process_text)  
        df['name'] = df['name'].apply(process_text)  
        df = pd.merge(df, breed, on='breed', how='left')
        df = pd.merge(df, name, on='name', how='left')
        df['settleornot'] = df['settlementAmount'].apply(lambda x: 1.0 if x>0 else 0.0)
        df = df.drop(['breed','name','quantityFloat'], axis=1)
    return df
    
def read_frame_as_list(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row[0]) 
    return data

def loadEncoder(veriDf, frame):
    data = formatDf(veriDf)
    columns = {'proc_name', 'code', 'kind', 'proc_breed'}
    frameDf = pd.DataFrame(0, index=range(len(data)), columns=frame)

    df = pd.concat([data, frameDf], axis=1)
    for row in df.index: 
        for col in columns:
            value = df.loc[row, col]
            if value in frame:
                df.loc[row, value] = 1
    df = df.drop(columns, axis=1)
    return df

def load_mean_predict(df,weight_rfc, weight_knn, weight_dtc):
    frame = read_frame_as_list('./data/frame_df.csv')
    encodedData = loadEncoder(df, frame)
    X = encodedData.drop(['settlementAmount','insuranceCaseId','settleornot'], axis=1)
    X.fillna(X.mean(), inplace=True) 
    y = encodedData['settleornot']
       
    pro_knn = joblib.load('./model/model_knn.joblib').predict_proba(X)[:, 1]
    pro_dtc = joblib.load('./model/model_dtc.joblib').predict_proba(X)[:, 1]
    pro_rfc = joblib.load('./model/model_rfc.joblib').predict_proba(X)[:, 1]
    prob = (pro_rfc * weight_rfc + pro_knn * weight_knn + pro_dtc * weight_dtc)

    settled = pd.DataFrame({'Actual': y, 'Probability': prob })
    row_pred = pd.concat([encodedData['insuranceCaseId'], settled], axis=1)
    row_pred['Correct'] = row_pred.groupby(['insuranceCaseId','Actual'])['Probability'].transform(lambda x: int(all(x >= 0.85)))
    row_pred = row_pred.groupby(['insuranceCaseId', 'Actual', 'Correct'])['Probability'].mean().to_frame().reset_index()

    X_cases = encodedData.groupby('insuranceCaseId').size().reset_index(name='BillingRows')
    ensamble = pd.merge(row_pred, X_cases, on='insuranceCaseId', how='left')
    ensamble['Compensable'] = ensamble['Correct'].apply(lambda x: True if x==1 else False)
    ensamble = ensamble[['insuranceCaseId','BillingRows','Actual','Probability','Correct','Compensable']]

    return ensamble