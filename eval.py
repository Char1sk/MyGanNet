import os

from options.eval_options import EvalOptions
from utils.fid_score import get_fid, get_folders_from_list, get_paths_from_list
from utils.fsim_score import get_file_list_from_file_list, get_file_list_from_folder, eval_avg_fsim


def main():
    opts = EvalOptions().parse()
    if opts.metric.lower() == 'fid':
        path_label_test = tuple(get_paths_from_list(opts.data_folder, opts.test_photo_list))
        path_pred_test = opts.pred_folder
        fid = get_fid(path_label_test, path_pred_test, path=opts.inception_model)
        print(f"FID: {fid:>6.2f}")
        
    elif opts.metric.lower() == 'fsim':
        l1 = get_file_list_from_file_list(opts.data_folder, opts.test_photo_list)
        l2 = get_file_list_from_folder(opts.pred_folder)
        fsim = eval_avg_fsim(l1, l2)
        print(f"FSIM: {fsim:>6.4f}")
        
    else:
        print(f"Metric {opts.metric} is not defined.")


if __name__ == '__main__':
    main()
