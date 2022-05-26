python ./eval/test_AdaSR.py --data_dir ./dataset/test4k --quality 41 --sr_model_name edsr
python ./eval/test_AdaSR.py --data_dir ./dataset/test4k --quality 60 --sr_model_name edsr
python ./eval/test_AdaSR.py --data_dir ./dataset/test4k --quality 80 --sr_model_name edsr

python ./eval/test_AdaSR.py --data_dir ./dataset/test4k --quality 41 --sr_model_name fsrcnn
python ./eval/test_AdaSR.py --data_dir ./dataset/test4k --quality 60 --sr_model_name fsrcnn
python ./eval/test_AdaSR.py --data_dir ./dataset/test4k --quality 80 --sr_model_name fsrcnn

python ./eval/test_AdaSR.py --data_dir ./dataset/test4k --quality 41 --sr_model_name carn
python ./eval/test_AdaSR.py --data_dir ./dataset/test4k --quality 60 --sr_model_name carn
python ./eval/test_AdaSR.py --data_dir ./dataset/test4k --quality 80 --sr_model_name carn
