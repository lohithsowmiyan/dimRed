
FILES = data/optimize/config/SS-K.csv data/optimize/config/SS-L.csv data/optimize/config/SS-M.csv \
        data/optimize/config/SS-N.csv data/optimize/config/SS-O.csv data/optimize/config/SS-P.csv \
        data/optimize/config/SS-Q.csv data/optimize/config/SS-R.csv data/optimize/config/SS-T.csv \
        data/optimize/config/SS-U.csv data/optimize/config/SS-V.csv data/optimize/config/SS-W.csv

# Replace WARMS with filtered paths based on FILES
DIMS = $(patsubst data/optimize/config/%.csv,var/out/dims/%.csv,$(FILES))

# General rule for processing files
var/out/dims/%.csv: data/optimize/config/%.csv
	@echo $<
	python3 ./dim.py --epochs 500 --dataset $< | tee $@

demo:
	mkdir -p var/out/dims
	$(MAKE) -j $(DIMS)

