#!/usr/bin/python3
import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertConfig
from tensorflow.keras.utils import plot_model
from flask import Flask, request, render_template, Response, jsonify, make_response

app = Flask(__name__)

model = tf.keras.models.load_model('/mnt/tensorflow')

# Check its architecture
# model.summary()

max_seq_length = 64
CLASSES = [1, 2, 3, 4, 5]
config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(CLASSES),
    id2label={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
    label2id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def predict(text):
    encode_plus_tokens = tokenizer.encode_plus(
                            text,
                            pad_to_max_length=True,
                            max_length=max_seq_length,
                            truncation=True,
                            return_tensors='tf')
    
    input_ids = encode_plus_tokens['input_ids']
    input_mask = encode_plus_tokens['attention_mask']
    
    outputs = model.predict(x=(input_ids,input_mask))
    
    prediction = [{"label":config.id2label[item.argmax()], \
                   "socre":item.max().item()} for item in outputs]

    return prediction[0]


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/text", methods=['GET', 'POST'])
def predict_server():
    try:
        if request.method == 'GET':
            return predict(request.args.get('value'))
        elif request.method == 'POST':
            inputText = request.form['value']
            predictValue = predict(inputText) 
            return render_template("index.html", predictValue=predictValue, inputText=inputText)
        else:
            return abort(400)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)

