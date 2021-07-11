# train atm
# python main_train.py --taskname  20news_clean  --n_topic 100 --num_epochs 50000 --language en --clean_data True --bkpt_continue True
# python main_train.py --taskname  20news_clean  --n_topic 100 --num_epochs 50000 --language en --clean_data True ----bkpt_continue True

# train c_atm
python catm_main.py --taskname  20news_clean  --n_topic 20 --num_epochs 50000 --language en --clean_data True
# python catm_main.py --taskname  20news_clean  --n_topic 20 --num_epochs 50000 --language en --clean_data True --bkpt_continue True