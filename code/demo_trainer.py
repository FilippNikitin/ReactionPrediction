import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from onmt.train_single import main as single_main
from onmt.new_opts import Opts


opt = Opts(
        data="data/USP",
        save_model="models/demo_model",
        batch_size=256,
        gpu_ranks=[0],
        valid_steps=1000,
        valid_batch_size=100,
        decay_steps=1000,
        start_decay_steps=10000,
        learning_rate_decay=0.09,
        optim="adam",
        learning_rate=0.001,
        save_checkpoint_steps=10000,
        src_word_vec_size=2,
        tgt_word_vec_size=2,
        encoder_type="brnn",
        enc_rnn_size=256,
        enc_layers=2,
        decoder_type="brnn",
        dec_rnn_size=256,
        dec_layers=2
        )

single_main(opt, -1)
