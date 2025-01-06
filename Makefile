
FILES = data/optimize/config/SS-K.csv data/optimize/config/SS-L.csv data/optimize/config/SS-M.csv \
        data/optimize/config/SS-N.csv data/optimize/config/SS-O.csv data/optimize/config/SS-P.csv \
        data/optimize/config/SS-Q.csv data/optimize/config/SS-R.csv data/optimize/config/SS-T.csv \
        data/optimize/config/SS-U.csv data/optimize/config/SS-V.csv data/optimize/config/SS-W.csv


FILES2 = data/optimize/config/SQL_AllMeasurements.csv data/optimize/config/X264_AllMeasurements.csv \
         data/optimize/process/xomo_flight.csv data/optimize/process/xomo_ground.csv \
         data/optimize/process/xomo_osp.csv data/optimize/process/xomo_osp2.csv

# Replace WARMS with filtered paths based on FILES
DIMS = $(patsubst data/optimize/config/%.csv,var/out/dims/%.csv,$(filter data/optimize/config/%.csv,$(FILES2))) \
       $(patsubst data/optimize/process/%.csv,var/out/dims/%.csv,$(filter data/optimize/process/%.csv,$(FILES2)))

# General rule for processing config files
var/out/dims/%.csv: data/optimize/config/%.csv
	@echo "Processing config file: $<"
	python3 ./dim.py --epochs 500 --dataset $< | tee $@

# General rule for processing process files
var/out/dims/%.csv: data/optimize/process/%.csv
	@echo "Processing process file: $<"
	python3 ./dim.py --epochs 500 --dataset $< | tee $@

demo:
	mkdir -p var/out/dims
	$(MAKE) -j $(DIMS)

