from utils.fid_score import get_fid, get_folders_from_list, get_paths_from_list


def main():
    path_all = get_folders_from_list('../Datasets/CUFS-CAGAN-New/', 'files/list_fid.txt')
    path_label_train = get_paths_from_list('../Datasets/CUFS-CAGAN-New/', 'files/list_train.txt')
    path_label_test  = tuple(get_paths_from_list('../Datasets/CUFS-CAGAN-New/', 'files/list_test.txt'))
    path_pred_train = './Images/Images/Train/700/'
    path_pred_test  = './Images/Images/Test/700/'
    model_path = '../Models/pt_inception.pth'
    fid = get_fid([path_pred_test, path_label_test], path=model_path)
    print(f'FID: {fid}')


if __name__ == '__main__':
    main()