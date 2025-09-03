from config import FLAGS
from train import train_main, inference
from LLM4DSE.dse import LMEAExplorer, EAExplorer, SAExplorer, ExhaustiveExplorer, ACOExplorer, LMSAExplorer, LMACOExplorer
from saver import saver
import os.path as osp
from utils import get_root_path
from os.path import join
import config
from programl_data import get_data_list
from dse_database.gen_dataset import MyOwnDataset


TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL

if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'COLORS-3')
    if not FLAGS.force_regen or FLAGS.subtask == 'dse':
        dataset = MyOwnDataset()
    else:
        pragma_dim = 0
        dataset, pragma_dim = get_data_list()

    if FLAGS.subtask == 'inference':
        inference(dataset)
    elif FLAGS.subtask == 'dse':
        for dataset in ['poly', '']:#
            path = join(get_root_path(), 'dse_database', dataset, 'config')
            path_graph = join(get_root_path(), 'dse_database', 'programl', dataset, 'processed')
            KERNELS = ['gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d']
            # KERNELS = ['bicg', 'doitgen', 'gemm-p', 'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d']
            for kernel in KERNELS:
                # if 'md' not in kernel:
                #     continue
                # if kernel in dse_unseen_kernel:
                saver.info('#################################################################')
                saver.info(f'Starting DSE for {kernel}')
                if FLAGS.explorer == 'LMEA':
                    LMEAExplorer(path, kernel, path_graph, run_dse = True)
                elif FLAGS.explorer == 'LMACO':
                    LMACOExplorer(path, kernel, path_graph, run_dse=True)
                elif FLAGS.explorer == 'LMSA':
                    LMSAExplorer(path, kernel, path_graph, run_dse=True)
                elif FLAGS.explorer == 'EA':
                    EAExplorer(path, kernel, path_graph, run_dse = True)
                elif FLAGS.explorer == 'SA':
                    SAExplorer(path, kernel, path_graph, run_dse = True)
                elif FLAGS.explorer == 'Exhastive':
                    ExhaustiveExplorer(path, kernel, path_graph, run_dse = True)
                else:
                    ACOExplorer(path, kernel, path_graph, run_dse = True)
                saver.info('#################################################################')
                saver.info(f'')
    else:
        train_main(dataset, 0)


    saver.close()