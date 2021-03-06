import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier




st.write("""
# HealthCare App: Predict chance of Heart Attack

""")



st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV input file](https://github.com/update-ankur/Health-Care/blob/main/Dataset/heart.csv)
""")

#collect data from User

def user_input_feature():
    age=st.sidebar.slider("Age",10,100,step=1)
    sex=st.sidebar.selectbox("Sex",('Male','Female'))
    cp=st.sidebar.slider("Chest pain type",0,3,1)
    trestbps=st.sidebar.slider("resting blood pressure",0,200,1)
    chol=st.sidebar.slider("serum cholestoral in mg/dl",0,564,1)
    fbs=st.sidebar.slider("fasting blood sugar",0,1,1)
    restecg=st.sidebar.slider("resting electrocardiographic results",0,2,1)
    thalach=st.sidebar.slider("maximum heart rate achieved",0,564,1)
    exang=st.sidebar.slider("exercise induced angina",0,2,1)
    oldpeak=st.sidebar.slider("oldpeak = ST depression induced by exercise relative to rest",0.0,10.0,.1)
    slope=st.sidebar.slider("the slope of the peak exercise ST segment",0,2,1)
    ca=st.sidebar.slider("number of major vessels (0-3) colored by flourosopy",0,3,1)
    thal=st.sidebar.slider("number of major vessels (0-3) colored by flourosopy",0,2,1)

    data = {'age':age,
            'sex':sex,
            'cp':cp,
            'trestbps':trestbps,
            'chol':chol,
            'fbs':fbs,
            'restecg':restecg,
            'thalach':thalach,
            'exang':exang,
            'oldpeak':oldpeak,
            'slope':slope,
            'ca':ca,
            'thal':thal
            }
    features=pd.DataFrame(data,index=[0])
    return features

upload_file=st.sidebar.file_uploader("upload your csv file.")
if upload_file is not None:
    input_df=pd.read_csv(upload_file)
else:
    input_df=user_input_feature()

# data=pd.DataFrame(columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])

st.subheader('User Input Feature')
if upload_file is not None:
    st.write(input_df)
else:
    st.write(input_df)

if input_df['sex'].tolist()[0]=='Male':
    input_df['sex']=1
else:
    input_df['sex']=0

old_data=pd.read_csv('heart.csv')
old_data=old_data.drop(['target'],axis=1)

dataset=pd.concat([input_df,old_data],axis=0)

data=pd.get_dummies(dataset,columns=['cp','fbs','restecg','exang','slope','ca','thal'])
data['trestbps']=data['trestbps']/200
data['chol']=data['chol']/564
data['thalach']=data['thalach']/564

data=data.drop(['age','cp_0','fbs_0','restecg_0','exang_0','slope_0','ca_0','thal_0'],axis=1)
data=data[:1]

mymodel=pickle.load(open('model.pkl','rb'))
prediction=mymodel.predict(data)


st.subheader('Prediction: ')
if prediction>0.6:
    st.write("More Chance of Heart Attack")
else:
    st.write("Less Chance of Heart Attack")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}

            footer {
            visibility: hidden;
            }
            footer:after {
            content:'Made with ❤️ by Ankur Singh';
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
