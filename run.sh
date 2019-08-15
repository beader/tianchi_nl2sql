#!/bin/bash

cd code

python task1.py infer --test_data_file /tcdata/test.json --test_table_file /tcdata/test.tables.json --model_weights ../model/task1.12-0.852.h5 --batch_size 256 --output_file ../task1_output.json
python task2.py infer --test_data_file /tcdata/test.json --test_table_file /tcdata/test.tables.json --model_weights ../model/task2.h5 --batch_size 256 --synthesis_with_task1_output 1 --task1_output ../task1_output.json --submit_output ../result.json
