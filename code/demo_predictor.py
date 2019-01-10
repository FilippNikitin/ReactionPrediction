from onmt.translate.translator import build_translator
from onmt.new_opts import TranslateOpts


opt = TranslateOpts(
        gpu=-1,
        models=["demo_model_step_29000.pt"],
        src="data/USP_src-valid.txt",
        tgt="data/USP_tgt-valid.txt",
        replace_unk=True,
        verbose=True,
        output="data/USP_demo_pred.txt",
        )
translator = build_translator(opt, report_score=True)
translator.translate(src_path=opt.src,
                     tgt_path=opt.tgt,
                     src_dir=opt.src_dir,
                     batch_size=opt.batch_size,
                     attn_debug=opt.attn_debug
                     )
