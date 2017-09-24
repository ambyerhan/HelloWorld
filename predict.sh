#
THEANO_FLAGS="optimizer=None" python main.py \
				--test ./test/test.txt \
				--mode 2 \
				--model ./model/1/2017.06.22-1 \
				--swvocab ./data/src.dict
