import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from onmt.translate.translator import build_translator
from onmt.new_opts import TranslateOpts


opt = TranslateOpts(
    gpu=-1,
    models=["models/demo_model_step_20000.pt"],
    src="data/USP_src-test.txt",
    tgt="data/USP_tgt-test.txt",
    replace_unk=False,
    verbose=True,
    output="data/USP_demo_pred_20k.txt",
    batch_size=512,
    n_best=5,
)
translator = build_translator(opt, report_score=True)
translator.translate(src_path=opt.src,
                     tgt_path=opt.tgt,
                     src_dir=opt.src_dir,
                     batch_size=opt.batch_size,
                     attn_debug=opt.attn_debug
                     )
