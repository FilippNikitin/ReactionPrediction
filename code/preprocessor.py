from onmt.new_opts import PreprocessorOpts
from onmt.preprocess import build_save_dataset, build_save_vocab
import onmt.inputters as inputters


def separate_data(data_path, name_to_save, mode):
    lines = open(data_path).read().split('\n')
    with open(name_to_save + "_src-" + mode + ".txt", "w") as f1, \
            open(name_to_save + "_tgt-" + mode + ".txt", "w") as f2:
        for line in lines[3: len(lines) - 1]:
            input_text, target_text, *c = line.split('\t')
            input_text,_ = input_text.split('>')
            input_text = input_text.split()
            target_text = target_text.split()
            f1.write(" ".join(input_text) + " \n")
            f2.write(" ".join(target_text) + "\n")


if __name__ == "__main__":
    data_path = "data/US_patents_1976-Sep2016_1product_reactions_train.csv"
    name_to_save = "data/USP"
    mode = "train"
    separate_data(data_path, name_to_save, mode)

    data_path = "data/US_patents_1976-Sep2016_1product_reactions_valid.csv"
    name_to_save = "data/USP"
    mode = "valid"
    separate_data(data_path, name_to_save, mode)

    opt = PreprocessorOpts(
                            train_src=name_to_save + "_src-train.txt",
                            train_tgt=name_to_save + "_tgt-train.txt",
                            valid_src=name_to_save + "_src-valid.txt",
                            valid_tgt=name_to_save + "_tgt-valid.txt",
                            save_data="data/USP"
                          )

    src_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_src, 'src')

    tgt_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_tgt, 'tgt')

    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

    train_dataset_files = build_save_dataset('train', fields, opt)

    build_save_dataset('valid', fields, opt)
    build_save_vocab(train_dataset_files, fields, opt)
