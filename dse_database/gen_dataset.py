from LLM4DSE.dse import GNNModel
from utils import get_root_path
import os.path as osp
from os.path import join, basename
import config
from config import FLAGS
from glob import glob, iglob
from src.utils import get_root_path, MLP, print_stats, get_save_path, \
    create_dir_if_not_exists, plot_dist, load
from collections import Counter, OrderedDict
import redis, pickle
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from src.saver import saver
import networkx as nx
from src.programl_data import _encode_X_dict, _encode_edge_dict, _encode_X_torch, _encode_edge_torch, \
    finte_diff_as_quality, create_edge_index
import math
import torch
import numpy as np
from shutil import rmtree
from torch_geometric.data import Data
from copy import deepcopy
from torch_geometric.data import Dataset

tag = 'new_speedup'

TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL

ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL

MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
                'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
                'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']

code_path = join(get_root_path(), 'two_tower_dataset/code')
graph_path = join(get_root_path(), 'two_tower_dataset/graph')
ENCODER_PATH = join(get_root_path(), 'passion/save_models_and_data')

db_path = []
for benchmark in FLAGS.benchmarks:
    db_path.append(f'../dse_database/{benchmark}/databases/**/*')

GEXF_FOLDER = join(get_root_path(), 'dse_database', 'programl', '**', 'processed', '**')

GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf')])




if __name__ == '__main__':
    database = redis.StrictRedis(host='localhost', port=6379)
    print(graph_path)
    bench = ['machsuite', 'poly']
    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')
    # create a redis database
    database = redis.StrictRedis(host='localhost', port=6379)

    ntypes = Counter()
    ptypes = Counter()
    numerics = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()

    if FLAGS.encoder_path != None:
        encoders = load(FLAGS.encoder_path)
        enc_ntype = encoders['enc_ntype']
        enc_ptype = encoders['enc_ptype']
        enc_itype = encoders['enc_itype']
        enc_ftype = encoders['enc_ftype']
        enc_btype = encoders['enc_btype']

        enc_ftype_edge = encoders['enc_ftype_edge']
        enc_ptype_edge = encoders['enc_ptype_edge']

    else:
        ## handle_unknown='ignore' is crucial for handling unknown variables of new kernels
        enc_ntype = OneHotEncoder(handle_unknown='ignore')
        enc_ptype = OneHotEncoder(handle_unknown='ignore')
        enc_itype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype = OneHotEncoder(handle_unknown='ignore')
        enc_btype = OneHotEncoder(handle_unknown='ignore')

        enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
        enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')
    data_list = []

    all_gs = OrderedDict()

    X_ntype_all = []
    X_ptype_all = []
    X_itype_all = []
    X_ftype_all = []
    X_btype_all = []

    edge_ftype_all = []
    edge_ptype_all = []
    tot_configs = 0
    num_files = 0
    init_feat_dict = {}
    gf = ['/home/wslcccc/passion/dse_database/programl/poly/processed/atax_processed_result.gexf', '/home/wslcccc/passion/dse_database/programl/poly/processed/heat-3d_processed_result.gexf']
    # for gexf_file in tqdm(gf):
    for gexf_file in tqdm(GEXF_FILES[0:]):
        if FLAGS.dataset == 'machsuite' or 'programl' in FLAGS.dataset:
            proceed = False
            for k in ALL_KERNEL:
                if k in gexf_file:
                    proceed = True
                    break
            if not proceed:
                continue
            # pass
        else:
            raise NotImplementedError()

        g = nx.read_gexf(gexf_file)
        g.variants = OrderedDict()
        # extract file name
        gname = basename(gexf_file).split('.')[0]
        saver.log_info(gname)
        n = basename(gexf_file).split('_')[0]
        all_gs[n] = g

        # get .db paths
        if FLAGS.dataset == 'programl':
            db_paths = []
            for db_p in db_path:
                paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f]
                db_paths.extend(paths)
            if db_paths is None:
                saver.warning(f'No database found for {n}. Skipping.')
                continue
        else:
            raise NotImplementedError()

        # clear choosing key-values
        database.flushdb()
        saver.log_info(f'db_paths for {n}:')
        for d in db_paths:
            saver.log_info(f'{d}')
        if len(db_paths) == 0:
            saver.log_info(f'{n} has no db_paths')

        assert len(db_paths) >= 1

        # load the database and get the keys
        # the key for each entry shows the value of each of the pragmas in the source file
        for idx, file in enumerate(db_paths):
            f_db = open(file, 'rb')
            data = pickle.load(f_db)
            database.hmset(0, data)
            max_idx = idx + 1
            f_db.close()

        # get keys
        keys = [k.decode('utf-8') for k in database.hkeys(0)]
        lv2_keys = [k for k in keys if 'lv2' in k]
        saver.log_info(f'num keys for {n}: {len(keys)} and lv2 keys: {len(lv2_keys)}')

        got_reference = False
        res_reference = 0
        max_perf = 0
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))

            if type(obj) is int or type(obj) is dict:
                continue
            # if key[0:3] == 'lv1' or obj.perf == 0:  # obj.ret_code.name == 'PASS':
            #     continue
            if obj.perf > max_perf:
                max_perf = obj.perf
                got_reference = True
                res_reference = obj
        if res_reference != 0:
            saver.log_info(f'reference point for {n} is {res_reference.perf}')
        else:
            saver.log_info(f'did not find reference point for {n} with {len(keys)} points')

        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            # if FLAGS.task == 'regression' and key[0:3] == 'lv1':  # or obj.perf == 0:#obj.ret_code.name == 'PASS':
            #     continue
            # if FLAGS.task == 'regression' and not FLAGS.invalid and obj.perf == 0:
            #     continue
            # print(obj.point)
            xy_dict = _encode_X_dict(
                g, ntypes=ntypes, ptypes=ptypes, itypes=itypes, ftypes=ftypes, btypes=btypes, numerics=numerics,
                obj=obj)
            edge_dict = _encode_edge_dict(
                g, ftypes=ftypes_edge, ptypes=ptypes_edge)

            if FLAGS.task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        if obj.perf > 0:
                            y = FLAGS.normalizer / obj.perf
                            y = math.log2(y)
                        else:
                            y = obj.perf
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                    elif tname == 'quality':
                        y = finte_diff_as_quality(obj, res_reference)
                    elif 'util' in tname or 'total' in tname:
                        y = obj.res_util[tname]
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            else:
                raise NotImplementedError()

            vname = key

            g.variants[vname] = (xy_dict, edge_dict)
            X_ntype_all += xy_dict['X_ntype']
            X_ptype_all += xy_dict['X_ptype']
            X_itype_all += xy_dict['X_itype']
            X_ftype_all += xy_dict['X_ftype']
            X_btype_all += xy_dict['X_btype']

            edge_ftype_all += edge_dict['X_ftype']
            edge_ptype_all += edge_dict['X_ptype']

        tot_configs += len(g.variants)
        num_files += 1
        saver.log_info(f'{n} g.variants {len(g.variants)} tot_configs {tot_configs}')
        saver.log_info(f'\tntypes {len(ntypes)}')
        saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
        saver.log_info(f'\tnumerics {len(numerics)} {numerics}')

        for gname, g in all_gs.items():
            edge_index = create_edge_index(g)
            saver.log_info('edge_index created', gname)
            data_list = []

            if gname in MACHSUITE_KERNEL:
                SAVE_DIR = join(graph_path, 'machsuite', gname)
                SAVE_DIR_1 = join(code_path, 'machsuite', gname)
                CODE_FILES = join(get_root_path(), 'dse_database', 'programl', 'machsuite', gname, gname + '.c')
            else:
                SAVE_DIR = join(graph_path, 'poly', gname)
                SAVE_DIR_1 = join(code_path, 'poly', gname)
                CODE_FILES = join(get_root_path(), 'dse_database', 'programl', 'poly', gname, gname + '.c')

            code_list = []
            with open(CODE_FILES, 'r') as file:
                fc = file.read()
            for nums, (vname, d) in enumerate(g.variants.items()):
                d_node, d_edge = d
                X = _encode_X_torch(d_node, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
                edge_attr = _encode_edge_torch(d_edge, enc_ftype_edge, enc_ptype_edge)

                if FLAGS.task == 'regression':
                    data_list.append(Data(
                        x=X,
                        edge_index=edge_index,
                        perf=d_node['perf'],
                        actual_perf=d_node['actual_perf'],
                        quality=d_node['quality'],
                        util_BRAM=d_node['util-BRAM'],
                        util_DSP=d_node['util-DSP'],
                        util_LUT=d_node['util-LUT'],
                        util_FF=d_node['util-FF'],
                        total_BRAM=d_node['total-BRAM'],
                        total_DSP=d_node['total-DSP'],
                        total_LUT=d_node['total-LUT'],
                        total_FF=d_node['total-FF'],
                        edge_attr=edge_attr,
                        kernel=gname
                    ))
                else:
                    raise NotImplementedError()
                fcc = deepcopy(fc)
                kd = [i for i in vname[4:].split('.')]
                kd_d = {}
                for j in kd:
                    jl = j.split('-')
                    if jl[1] == 'NA':
                        kd_d[jl[0]] = ''
                    else:
                        kd_d[jl[0]] = jl[1]
                for k, v in kd_d.items():
                    fcc = fcc.replace('auto' + '{' + k + '}', v)

                with open(join(SAVE_DIR_1, f'{nums + 1}.c'),'w') as f:
                    f.write(fcc)


            saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
            rmtree(SAVE_DIR)
            create_dir_if_not_exists(SAVE_DIR)
            for i in tqdm(range(len(data_list))):
                torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i)))

            nns = [d.x.shape[0] for d in data_list]
            print_stats(nns, 'number of nodes')
            ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
            print_stats(ads, 'avg degrees')
            saver.log_info('dataset[0].num_features', data_list[0].num_features)
            for target in TARGETS:
                if not hasattr(data_list[0], target.replace('-', '_')):
                    saver.warning(f'Data does not have attribute {target}')
                    continue
                ys = [getattr(d, target.replace('-', '_')).item() for d in data_list]
                # if target == 'quality':
                #     continue
                plot_dist(ys, f'{target}_ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
                saver.log_info(f'{target}_ys', Counter(ys))


class MyOwnDataset():
    def __init__(self):
        pass

    @property
    def raw_file_names(self):
        # return ['some_file_1', 'some_file_2', ...]
        return []

    @property
    def processed_file_names(self):
        gp = {}
        cp = {}
        for k in ALL_KERNEL:
            if k in MACHSUITE_KERNEL:
                gt = join(graph_path, 'machsuite', k)
                ct = join(code_path, 'machsuite', k)
            else:
                gt = join(graph_path, 'poly', k)
                ct = join(code_path, 'poly', k)
            gp[k] = glob(join(gt, '*.pt'))
            cp[k] = glob(join(ct, '*.c'))
        return gp, cp

    def download(self):
        pass

    # Download to `self.raw_dir`.

    def process(self):
        # i = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     data = Data(...)
        #
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #
        #     torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
        #     i += 1
        pass

    def len(self):
        pass

    def __len__(self):
        return self.len()

    def get(self, idx):
        pass

    @staticmethod
    def get_data(idx, k):
        if k in MACHSUITE_KERNEL:
            gt = join(graph_path, 'machsuite', k)
            ct = join(code_path, 'machsuite', k)
        else:
            gt = join(graph_path, 'poly', k)
            ct = join(code_path, 'poly', k)
        data = torch.load(osp.join(gt, 'data_{}.pt'.format(idx)))
        with open(join(ct, f'{idx}.c')) as f:
            fc = f.read()
        code_d = fc
        return data, code_d







