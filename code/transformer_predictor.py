import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from glob import glob
from tqdm import tqdm
from onmt.translate.translator import build_translator
from onmt.new_opts import TranslateOpts

ckpt_paths = glob("models/transformer_ckpts/*.pt")
num_steps = [ckpt_path.split("_")[-1].split(".")[0] for ckpt_path in ckpt_paths]

for ckpt, step in tqdm(zip(ckpt_paths, num_steps)):
    opt = TranslateOpts(
        gpu=0,
        models=[ckpt],
        src="data/USP_src-test.txt",
        tgt="data/USP_tgt-test.txt",
        replace_unk=False,
        verbose=False,
        output="data/transformer_prediction/" + str(step) + ".txt",
        batch_size=256,
        n_best=5,
    )
    translator = build_translator(opt, report_score=True)
    translator.translate(
        src_path=opt.src,
        tgt_path=opt.tgt,
        src_dir=opt.src_dir,
        batch_size=opt.batch_size,
        attn_debug=opt.attn_debug
    )
