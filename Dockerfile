FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter

RUN pip install -U pip pyyaml pandas tqdm -i https://pypi.douban.com/simple
RUN pip install keras==2.2.4 keras-bert==0.68.1 cn2an==0.3.6 thulac==0.2.0 zhon==1.1.5 -i https://pypi.douban.com/simple

ADD ./model/chinese_wwm_L-12_H-768_A-12 /competition/model/chinese_wwm_L-12_H-768_A-12
ADD ./model/task1.h5 /competition/model/
ADD ./model/task2.h5 /competition/model/
ADD ./code /competition/code
ADD ./run.sh /competition/
WORKDIR /competition

CMD ["sh", "run.sh"]
