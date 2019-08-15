FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter

RUN pip install -U pip pyyaml pandas tqdm -i https://pypi.douban.com/simple
RUN pip install keras==2.2.4 keras-bert==0.68.1 cn2an==0.3.6 thulac==0.2.0 zhon==1.1.5 -i https://pypi.douban.com/simple

ADD . /competition
WORKDIR /competition

CMD ["sh", "run.sh"]
