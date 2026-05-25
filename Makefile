model=gemma4:31b

help:
	@echo "make SPA2MSLG"
	@echo "make MSLG2SPA"

train: SPA2MSLG_train MSLG2SPA_train

SPA2MSLG_train:
	python bin/predict.py \
		--model $(model) \
		--prompt data/SPA2MSLG_prompt.txt \
		--input data/MSLG_SPA_train.txt \
		--reverse \
	| tee results/SPA2MSLG.txt 

SPA2MSLG_eval:
	cat results/SPA2MSLG.txt \
	| grep -v -P "^T: MSLG" \
	| grep -1 -P "^T: " \
	| python bin/text2csv.py \
	| tee /dev/stderr \
	| python bin/eval.py

MSLG2SPA_train:
	python bin/predict.py \
		--model $(model) \
		--prompt data/MSLG2SPA_prompt.txt \
		--input data/MSLG_SPA_train.txt \
	| tee results/MSLG2SPA.txt

MSLG2SPA_eval:
	cat results/MSLG2SPA.txt \
	| grep -v -P "^T: SPA" \
	| grep -1 -P "^T: " \
	| python bin/text2csv.py \
	| tee /dev/stderr \
	| python bin/eval.py

test: SPA2MSLG_test MSLG2SPA_test

SPA2MSLG_test:
	python bin/predict.py \
		--model $(model) \
		--prompt data/SPA2MSLG_prompt.txt \
		--input data/SPA2MSLG_test.txt \
		--output results/UAM_ChineseRoom_SPA2MSLG.txt \
	| tee results/UAM_ChineseRoom_SPA2MSLG.out.txt

MSLG2SPA_test:
	python bin/predict.py \
		--model $(model) \
		--prompt data/MSLG2SPA_prompt.txt \
		--input data/MSLG2SPA_test.txt \
		--output results/UAM_ChineseRoom_MSLG2SPA.txt \
	| tee results/UAM_ChineseRoom_MSLG2SPA.out.txt





