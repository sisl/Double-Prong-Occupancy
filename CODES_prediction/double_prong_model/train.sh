python train_t_1.py -o trained_t_1_model -ps prednet -pd prednet
python train_t_5.py -o trained_t_5_model -f trained_t_1_model -ps prednet -pd prednet # fine-tuning

# Please add the '-g' argument if wish to train on more than the default number of gpu of 1. e.g.
# python train_t_1.py -o trained_t_1_model -ps prednet -pd prednet -g 3 -> train on 3 gpus
