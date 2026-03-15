model=openai/gpt-oss-20b

all:

help:
	@echo "make ..."

test1:
	python bin/test-suite.py \
		--model $(model) \
		--test_suite data/test-suite1.csv \
	| tee results/test-suite1.out.txt

test2:
	python bin/test-suite.py \
		--model $(model) \
		--test_suite data/test-suite2.csv \
	| tee results/test-suite2.out.txt

esp-lsm:
	python bin/test-suite.py \
		--model $(model) \
		--test_suite data/esp-lsm_glosses_corpus.csv \
	| tee results/esp-lsm_glosses_corpus.out.txt

