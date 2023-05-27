#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Machine learning predictors of risk of death within 7 days in patients with non-traumatic subarachnoid hemorrhage in the intensive care unit: A multicenter retrospective study')
st.title('Machine learning predictors of risk of death within 7 days in patients with non-traumatic subarachnoid hemorrhage in the intensive care unit: A multicenter retrospective study')

#%%set variables selection
st.sidebar.markdown('## Variables')
# features = ["age","mingcs","gcsmotor","gcsverbal",'gcseyes','heartrate','sysbp','resprate','tempc',
#             'spo2','aniongap','bicarbonate','creatinine','glucose','potassium','sodium','wbc','rbc',
#             'ph','nely']
age = st.sidebar.slider("Age (years)",20,100,value=40, step=1)
mingcs = st.sidebar.slider("GCS",3,15,value = 10,step = 1)
gcsmotor = st.sidebar.slider("Gcsmotor",1,6,value = 4,step = 1)
gcsverbal = st.sidebar.slider("Gcsverbal",0,5,value = 4,step = 1)
gcseyes = st.sidebar.slider("Gcseyes",1,4,value = 3,step = 1)
heartrate = st.sidebar.slider("Heartrate",50,180,value = 75,step = 1)
sysbp = st.sidebar.slider("Sysbp",100,250,value = 120,step = 1)
resprate = st.sidebar.slider("Resprate",15,70,value = 30,step = 1)
tempc = st.sidebar.slider("Tempc",35.0,42.0,value = 37.5,step = 0.1)
spo2 = st.sidebar.slider("Spo2",10,100,value = 98,step = 1)
aniongap = st.sidebar.slider("Aniongap",9,35,value = 16,step = 1)
bicarbonate = st.sidebar.slider("Bicarbonate",10,35,value = 20,step = 1)
creatinine = st.sidebar.slider("Creatinine",20.0,200.0,value = 100.0,step = 0.1)
glucose = st.sidebar.slider("Glucose",4.0,35.0,value = 7.0,step = 0.1)
potassium = st.sidebar.slider("Potassium",1.0,6.0,value = 4.0,step = 0.1)
sodium = st.sidebar.slider("Sodium",120,170,value = 140,step = 1)
wbc = st.sidebar.slider("Wbc",1.0,48.0,value = 10.0,step = 0.1)
rbc = st.sidebar.slider("Rbc",1.00,7.00,value = 4.00,step = 0.01)
ph = st.sidebar.slider("ph",6.00,8.00,value = 7.00,step = 0.01)
nely = st.sidebar.slider("NLR",0.00,98.00,value = 10.00,step = 0.01)


#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
# map = {'Squamous cell carcinoma':0,'Undifferentiated carcinoma':1}
# Pathology_type =map[Pathology_type]
# 数据读取，特征标注
#%%load model
ab_model = joblib.load('subarachnoid_hemorrhagemlp_model.pkl')

#%%load data
hp_train = pd.read_csv('train.csv')
features =["age","mingcs","gcsmotor","gcsverbal",'gcseyes','heartrate','sysbp','resprate','tempc',
            'spo2','aniongap','bicarbonate','creatinine','glucose','potassium','sodium','wbc','rbc',
            'ph','nely']
target = 'group'
y = np.array(hp_train[target])
sp = 0.5

is_t = (ab_model.predict_proba(np.array([[age,mingcs,gcsmotor,gcsverbal,gcseyes,heartrate,sysbp,resprate,tempc,
            spo2,aniongap,bicarbonate,creatinine,glucose,potassium,sodium,wbc,rbc,
            ph,nely]]))[0][1])> sp
prob = (ab_model.predict_proba(np.array([[age,mingcs,gcsmotor,gcsverbal,gcseyes,heartrate,sysbp,resprate,tempc,
            spo2,aniongap,bicarbonate,creatinine,glucose,potassium,sodium,wbc,rbc,
            ph,nely]]))[0][1])*1000//1/10
    

if is_t:
    result = 'High Risk Group'
else:
    result = 'Low Risk Group'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Group':
        st.balloons()
    st.markdown('## Probability of High Risk group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[age,mingcs,gcsmotor,gcsverbal,gcseyes,heartrate,sysbp,resprate,tempc,
                spo2,aniongap,bicarbonate,creatinine,glucose,potassium,sodium,wbc,rbc,
                ph,nely]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0  
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = ab_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of MLP model')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of MLP model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of MLP model')
    ab_prob = ab_model.predict(X)
    cm = confusion_matrix(y, ab_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of MLP model")
    disp1 = plt.show()
    st.pyplot(disp1)


