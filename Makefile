model=gemma4:31b

help:
	@echo "make SPA2MSLG"
	@echo "make MSLG2SPA"

train: SPA2MSLG_train MSLG2SPA_train

SPA2MSLG_train:
	python bin/predict.py \
		--model $(model) \
		--prompt data/SPA2MSLG_prompt.2.txt \
		--input data/MSLG_SPA_train.txt \
		--reverse \
	| tee results/SPA2MSLG.7.txt 

MSLG2SPA_train:
	python bin/predict.py \
		--model $(model) \
		--prompt data/MSLG2SPA_prompt.2.txt \
		--input data/MSLG_SPA_train.txt \
	| tee results/MSLG2SPA.7.txt 

test: SPA2MSLG_test MSLG2SPA_test

SPA2MSLG_test:
	python bin/predict.py \
		--model $(model) \
		--prompt data/SPA2MSLG_prompt.2.txt \
		--input data/SPA2MSLG_test.txt \
		--output results/UAM_ChineseRoom_SPA2MSLG.1.txt \
	| tee results/UAM_ChineseRoom_SPA2MSLG.1.out.txt

MSLG2SPA_test:
	python bin/predict.py \
		--model $(model) \
		--prompt data/MSLG2SPA_prompt.2.txt \
		--input data/MSLG2SPA_test.txt \
		--output results/UAM_ChineseRoom_MSLG2SPA.1.txt \
	| tee results/UAM_ChineseRoom_MSLG2SPA.1.out.txt




