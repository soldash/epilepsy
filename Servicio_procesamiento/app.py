from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS 
import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
import os

import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch
from scipy.signal import find_peaks
import pyhrv.frequency_domain as fd
import pyhrv.time_domain as td
import numpy as np

from sagemaker.tensorflow import TensorFlowPredictor
from sagemaker.sklearn import SKLearnPredictor
import joblib

endpoint_name = 'tensorflow-inference-2023-09-02-23-00-24-140'
predictor = TensorFlowPredictor(endpoint_name=endpoint_name)

decision_tree_endpoint_name = 'sagemaker-scikit-learn-2023-09-14-04-04-44-788'
decision_tree_predictor = SKLearnPredictor(endpoint_name=decision_tree_endpoint_name)


correct_order = ['RRI', 'SDNN', 'HR', 'HR_std', 'NN50', 'pNN50', 'HF', 'LF', 'Total_power', 'LF_HF']
fbands = {'ulf': (0.0, 0.1), 'vlf': (0.1, 0.2), 'lf': (0.2, 0.3), 'hf': (0.3, 0.4)}

# Inicializa el cliente de SNS
sns = boto3.client('sns')
# ARN de tu tema en SNS
topic_arn = 'arn:aws:sns:us-east-2:027331693661:paciente0'

# Funciones de filtrado
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_filter(data, cutoff, Q, fs):
    nyq = 0.5 * fs
    freq = cutoff / nyq
    b, a = iirnotch(freq, Q)
    y = lfilter(b, a, data)
    return y

#Funciones de procesamiento
def detect_peaks(data, prominence=(300,8000), width=1, distance=50):
    """Detect peaks in the given data."""
    peaks, _ = find_peaks(data, prominence=prominence, width=width, distance=distance)
    return peaks

def compute_RRI(dataframe, peaks):
    """Compute the R-R interval (RRI)."""
    df = dataframe.copy()
    df['RRI'] = 0
    for i in range(len(peaks)):
        if i != 0:
            RRI = df["time"][peaks[i]] - df["time"][peaks[i-1]]
            RRI = RRI.total_seconds() * 1000
            df.loc[peaks[i-1]:peaks[i], ("RRI")] = RRI
    return df, RRI

def compute_SDNN(subset):
    """Compute the SDNN."""
    subset.drop_duplicates(subset="RRI", keep='last', inplace=True)
    return subset.std()["RRI"]

def compute_HR_for_last_30_seconds(sampleDf, peaks):
    """
    Compute the HR for the last 30 seconds of the given DataFrame.

    Parameters:
    - sampleDf (pd.DataFrame): The input DataFrame with 'time' and 'filtered_ECG' columns.
    - peaks (np.array): An array with the indices of the detected peaks.

    Returns:
    - sampleDf (pd.DataFrame): The updated DataFrame with the 'HR' column added.
    """
    
    # Identifica el tiempo de la última muestra
    last_time = sampleDf['time'].iloc[-1]
    sampleDf['HR'] = 0

    # Identifica el tiempo 30 segundos antes de la última muestra
    start_time = last_time - pd.Timedelta(seconds=30)
    subset_indices = sampleDf[sampleDf['time'] >= start_time].index
    start_index = subset_indices[0] if len(subset_indices) > 0 else None

    for index in range(len(sampleDf) - 1, start_index - 1, -1):
        time = sampleDf['time'].iloc[index]
        start_data = time - pd.Timedelta(seconds=60)

        subset_indices = sampleDf[sampleDf['time'] >= start_data].index
        start_index_hr = subset_indices[0] if len(subset_indices) > 0 else None

        if start_index_hr is not None:
            sampleDf.loc[index, "HR"] = len(peaks[(peaks > start_index_hr) & (peaks < index)])

    return sampleDf

def get_neronal_network_prediction(X):
    X_df = pd.DataFrame([X], columns=correct_order) 
    X_scaled = scaler.transform(X_df)
    raw_result = predictor.predict(X_scaled[0].tolist())
    prediction_value = raw_result['predictions'][0][0]
    rounded_result = round(prediction_value)
    return rounded_result

def get_decision_tree_prediction(X):
    result = decision_tree_predictor.predict([X])
    return result[0]

def load_scaler_from_s3(bucket_name, file_name):
    s3 = boto3.client('s3')
    with open(file_name, 'wb') as f:
        s3.download_fileobj(bucket_name, file_name, f)
    scaler = joblib.load(file_name)
    return scaler


app = Flask(__name__)
CORS(app)

bucket_name = 'neuronalnetwork'
file_name = 'scaler_neuronal_net.pkl'
scaler = load_scaler_from_s3(bucket_name, file_name)

client = boto3.client('timestream-query')

@app.route('/ping')
def hello_world():
    return '¡pong!'

@app.route('/getPredictVariables', methods=['GET'])
def get_predict_variables():
    client = boto3.client('timestream-query')

    try:
        # Obtener el último tiempo registrado
        query_latest_time = 'SELECT time FROM "ECG_Records"."ECG" ORDER BY time DESC LIMIT 1'
        response_latest_time = client.query(QueryString=query_latest_time)

        latest_time = response_latest_time['Rows'][0]['Data'][0]['ScalarValue']
        print(latest_time)

        # Usar el último tiempo para hacer la consulta de 2 minutos hacia atrás
        query_string = f'SELECT ECG, time FROM "ECG_Records"."ECG" WHERE time BETWEEN timestamp \'{latest_time}\' - interval \'130\' second AND timestamp \'{latest_time}\' ORDER BY time ASC'
        response = client.query(QueryString=query_string)

        columns = [col['Name'] for col in response['ColumnInfo']]
        rows = []

        for row in response['Rows']:
            data_row = [value[list(value.keys())[0]] for value in row['Data']]
            rows.append(data_row)
        df = pd.DataFrame(rows, columns=columns)

        df['ECG'] = df['ECG'].astype(float)
        df['time'] = pd.to_datetime(df['time'])

        # Suponiendo que la frecuencia de muestreo es 500 Hz 
        fs = 500.0

        # Ajuste de parámetros
        lowpass_cutoff = 80.0
        highpass_cutoff = 0.1
        notch_cutoff = 60.0 
        order = 5 

        df['ECG_centered'] = df['ECG'] - df['ECG'].mean()

        # Filtra la señal ECG
        filtered_ecg_values = df['ECG_centered'].values
        filtered_ecg_values = lowpass_filter(filtered_ecg_values, lowpass_cutoff, fs, order)
        filtered_ecg_values = highpass_filter(filtered_ecg_values, highpass_cutoff, fs, order)
        filtered_ecg_values = notch_filter(filtered_ecg_values, notch_cutoff, 60, fs)



        # Añadir la señal filtrada como otra columna
        df['filtered_ECG'] = filtered_ecg_values

        df['filtered_ECG'] = -df['filtered_ECG']

        # Sample usage:
        sampleDf = df
        peaks = detect_peaks(sampleDf["filtered_ECG"])

        predict_variables = {
            'RRI': 0,
            'SDNN': 0,
            'HR': 0,
            'HR_std': 0,
            'NN50': 0,
            'pNN50': 0,
            'HF': 0,
            'LF': 0,
            'Total_power': 0,
            'LF_HF': 0
        }

        sampleDf, last_RRI = compute_RRI(sampleDf, peaks)
        predict_variables["RRI"] = last_RRI
        sampleDf = sampleDf[sampleDf["RRI"] > 0]

        last_time = sampleDf['time'].iloc[-1]
        subset = sampleDf[sampleDf['time'] > (last_time - pd.Timedelta(seconds=30))].copy()
        predict_variables["SDNN"] = compute_SDNN(subset)

        #NN50 y pNN50
        resultNN50 = td.nn50(subset["RRI"])
        predict_variables["NN50"] = resultNN50["nn50"]
        predict_variables["pNN50"] = resultNN50["pnn50"]

        #HR
        sampleDf = compute_HR_for_last_30_seconds(sampleDf, peaks)
        predict_variables["HR"] = sampleDf[sampleDf['HR'] > 0]['HR'].iloc[-1]
        predict_variables["HR_std"] = sampleDf[sampleDf['HR'] > 0]['HR'].std()

        #Frecuencia
        result = fd.welch_psd(sampleDf["RRI"], mode='dev', fbands=fbands)
        predict_variables["HF"] = result[0]["fft_abs"][3]
        predict_variables["LF"] = result[0]["fft_abs"][2]
        predict_variables["Total_power"] = result[0]["fft_total"]
        predict_variables["LF_HF"] = result[0]["fft_ratio"]

        """
        
        
        predict_variables = {
            'RRI': 678,
            'SDNN': 41.233267,
            'HR': 93,
            'HR_std': 0.574147,
            'NN50': 4,
            'pNN50': 15.384615,
            'HF': 135.113971,
            'LF': 17.492237,
            'Total_power': 252.425314,
            'LF_HF': 0.129463
        }
        """

        # Orden correcto de las variables
        X_new = [predict_variables[p] for p in correct_order]

        # Obtiene la predicción usando las variables en el orden correcto
        prediction = get_neronal_network_prediction(X_new)

        decision_tree_prediction = get_decision_tree_prediction(X_new)
        decision_tree_prediction = 1
        prediction = 1
        if prediction == 10 :
            # Publica el mensaje
            response = sns.publish(
                TopicArn=topic_arn,
                Message='¡Alerta! Condición de emergencia detectada para el paciente 0.',
                Subject='Alerta de Emergencia'
            )
        print(response)
        # Agrega la predicción al resultado
        predict_variables["prediction"] = prediction
        predict_variables["decision_tree_prediction"] = decision_tree_prediction


        print(predict_variables)
        for key, value in predict_variables.items():
            if isinstance(value, (np.float64, np.float32)):
                predict_variables[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                predict_variables[key] = int(value)

        return jsonify(predict_variables)

    except (BotoCoreError, ClientError) as e:
        return jsonify({"error": str(e)}), 500


@app.route('/export_csv', methods=['GET'])
def export_csv():
    try:
        # Obtener el último tiempo registrado
        query_latest_time = 'SELECT time FROM "ECG_Records"."ECG" ORDER BY time DESC LIMIT 1'
        response_latest_time = client.query(QueryString=query_latest_time)
        
        latest_time = response_latest_time['Rows'][0]['Data'][0]['ScalarValue']

        # Usar el último tiempo para hacer la consulta de 2 minutos hacia atrás
        query_string = f'SELECT ECG, time FROM "ECG_Records"."ECG" WHERE time BETWEEN timestamp \'{latest_time}\' - interval \'2\' minute AND timestamp \'{latest_time}\' ORDER BY time DESC'
        response = client.query(QueryString=query_string)

        columns = [col['Name'] for col in response['ColumnInfo']]
        rows = []

        for row in response['Rows']:
            data_row = [value[list(value.keys())[0]] for value in row['Data']]
            rows.append(data_row)

        df = pd.DataFrame(rows, columns=columns)
        
        # Guardar el DataFrame en un archivo .csv
        csv_filename = "timestream_data.csv"
        df.to_csv(csv_filename, index=False)
        
        return send_from_directory(os.getcwd(), csv_filename, as_attachment=True)

    except (BotoCoreError, ClientError) as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)

