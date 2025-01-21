import streamlit as st
import pickle
import numpy as np
import math

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

screen_size = st.selectbox('Scrensize in inches', [10.0, 18.0, 13.0])

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

#CPU_speed = st.number_input('Speed of CPU')

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['OS'].unique())


if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,ssd,hdd,gpu,os])

    query = query.reshape(1,12)
    #predicted_price = np.exp(pipe.predict(query)[0])
    #st.title("The predicted price of this configuration is " + str(int(predicted_price)))

    raw_prediction = pipe.predict(query)[0]
    st.write(f"Raw prediction: {raw_prediction}")

    # Handle cases where the prediction might result in an infinite value
    try:
        predicted_price = np.exp(raw_prediction)
        if raw_prediction > 50000:  # Check if the result is infinity
            st.title("The predicted price is extremely high and cannot be computed accurately.")
        if raw_prediction <= 50000:
            predicted_price = np.exp(raw_prediction)
            st.title("The predicted price of this configuration is " + str(int(predicted_price)))
    except OverflowError:
        st.title("")
