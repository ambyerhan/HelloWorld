#!/bin/bash
python main.py \
		--epoch 4 \
		--method 1 \
		--minibatch 3 \
		--src ./data/src.txt \
		--swvocab ./data/src.dict \
		--mode 1 \
		--model ./model/1/2017.06.22-1
