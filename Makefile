model=gemma4:31b

help:
	@echo "make SPA2MSLG"
	@echo "make MSLG2SPA"

SPA2MSLG:
	python bin/test-suite.py \
		--model $(model) \
		--test_suite data/MSLG_SPA_train.csv \
		--prompt data/SPA2MSLG.2.txt \
	| tee results/SPA2MSLG.5.txt 

MSLG2SPA:
	python bin/test-suite.py \
		--model $(model) \
		--test_suite data/MSLG_SPA_train.csv \
		--reverse \
		--prompt data/MSLG2SPA.2.txt \
	| tee results/MSLG2SPA.5.txt 

