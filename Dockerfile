FROM nvcr.io/nvidia/tensorflow:23.07-tf2-py3
RUN git clone https://github.com/developer-onizuka/MachineLearningOnAWS
RUN cd MachineLearningOnAWS
RUN pip3 install transformers seaborn matplotlib==3.7.3 flask
COPY AmazonCustomerReviewAPI.py /tmp
COPY templates/ /tmp/templates
ENTRYPOINT /tmp/AmazonCustomerReviewAPI.py
