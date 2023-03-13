#!/bin/bash
python3 ./DirectVoxGO/run.py --json_path $1 --dump_images $2 --config DirectVoxGO/configs/nerf/hotdog.py --render_only --render_test --ft_path ./hw4_1_fine.tar