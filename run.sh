#
THEANO_FLAGS="optimizer=None" python main.py \
		--method 1 \
		--shuffle 1 \
		--src ./data/src.txt \
		--swvocab ./data/src.dict
