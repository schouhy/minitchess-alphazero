FROM python:3.8

RUN apt-get update && apt-get install -y git

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt 

ENV ERLYX_VERSION=0f6409c3ff6aea417d89c7e8922db99be7edc606
RUN pip install git+https://git@github.com/schouhy/erlyx.git@$ERLYX_VERSION

ENV PYTHON_CHESS_VERSION=1f7c95a8a58a260b22cb56b37be44ffd3b5b5363
RUN pip install git+https://git@github.com/schouhy/python-chess.git@$PYTHON_CHESS_VERSION

# ENV MINITCHESS_ALPHAZERO_VERSION=5f83fc81137713fb26c9032581b054c683269dc3
# RUN pip install git+https://git@github.com/schouhy/minitchess-alphazero.git@$MINITCHESS_ALPHAZERO_VERSION
# RUN cd /tmp && \
# 	git clone https://github.com/schouhy/erlyx.git && \
# 	cd erlyx && \
# 	git checkout 0f6409c3ff6aea417d89c7e8922db99be7edc606 && \
# 	cp -r erlyx /app/erlyx
# RUN cd /tmp && \
# 	git clone https://github.com/schouhy/python-chess.git && \
# 	cd python-chess && \
# 	git checkout a4cf4ddda739f50568b0fd78662a3c3185b426c5 && \
# 	cp -r chess /app/chess
# RUN cd /tmp && \
# 	git clone https://github.com/schouhy/minitchess-alphazero.git exp && \
# 	cp -r exp /app/exp 
# 
# RUN wget https://raw.githubusercontent.com/schouhy/minitchess-alphazero/main/exp/moves_dict.json
RUN pip install jsonpickle
RUN pip install paho-mqtt
RUN pip install gunicorn
COPY exp /app/exp
RUN mv exp/moves_dict.json .
COPY app /app/app
RUN touch log
ARG MINITCHESS_ALPHAZERO_VERSION=0
ENV MINITCHESS_ALPHAZERO_VERSION $MINITCHESS_ALPHAZERO_VERSION

EXPOSE 5000
CMD ["python", "-m", "app.puppet"]
