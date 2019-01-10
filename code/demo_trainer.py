
from onmt.train_single import main as single_main
from onmt.new_opts import Opts


opt = Opts(
        data="data/USP",
        save_model="models/demo_model",
        epochs=10,
        batch_size=256,
        gpu_ranks=[0, 1],
        valid_steps=100,
        valid_batch_size=1000,
        decay_steps=1000,
        start_decay_steps=10000,
        learning_rate_decay=0.9,
        optim="adam",
        learning_rate=0.001,
        save_checkpoint_steps=1000
        )

single_main(opt, -1)
