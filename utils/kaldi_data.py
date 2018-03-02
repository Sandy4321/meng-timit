import bisect
import os
import struct

import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset

def read_next_utt(scp_line, ark_fd=None):
    # From https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py
    if scp_line == '' or scp_line == None:
        return '', None
    utt_id, path_pos = scp_line.replace('\n','').split(' ')
    path, pos = path_pos.split(':')

    if ark_fd is not None:
        if ark_fd.name != path.split(os.sep)[-1]:
            # New ark file now -- close and get new descriptor
            ark_fd.close()
            ark_fd = open(path, 'rb')
    else:
        ark_fd = open(path, 'rb')
    ark_fd.seek(int(pos),0)
    header = struct.unpack('<xcccc', ark_fd.read(5))
    if header[0] != b'B':
        print("Input .ark file is not binary", flush=True)
        exit(1)

    rows = 0; cols= 0
    m, rows = struct.unpack('<bi', ark_fd.read(5))
    n, cols = struct.unpack('<bi', ark_fd.read(5))

    tmp_mat = np.frombuffer(ark_fd.read(rows * cols * 4), dtype=np.float32)
    utt_mat = np.reshape(tmp_mat, (rows, cols))

    return utt_id, utt_mat, ark_fd

def write_kaldi_ark(ark_fd, utt_id, arr):
    # From https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py
    mat = np.asarray(arr, dtype=np.float32, order='C')
    utt_id_bytes = bytes(utt_id, 'utf-8')
    rows, cols = mat.shape

    ark_fd.write(struct.pack('<%ds'%(len(utt_id)), utt_id_bytes))
    ark_fd.write(struct.pack('<cxcccc', bytes(' ', 'utf-8'),
                                        bytes('B', 'utf-8'),
                                        bytes('F', 'utf-8'),
                                        bytes('M', 'utf-8'),
                                        bytes(' ', 'utf-8')))
    ark_fd.write(struct.pack('<bi', 4, rows))
    ark_fd.write(struct.pack('<bi', 4, cols))
    ark_fd.write(mat)



# Dataset class to support loading just features from Kaldi files
# Only for sequential use! Does not support random access
class KaldiDataset(Dataset):
    def __init__(self, scp_path):
        super(KaldiDataset, self).__init__()

        # Load in Kaldi files
        self.scp_path = scp_path
        self.scp_file = open(self.scp_path, 'r')

        # Determine how many utterances and features are included
        self.num_utts = 0
        self.num_feats = 0
        self.ark_fd = None
        for scp_line in self.scp_file:
            self.num_utts += 1

            utt_id, path_pos = scp_line.replace('\n','').split(' ')
            path, pos = path_pos.split(':')

            if self.ark_fd is None or self.ark_fd.name != path.split(os.sep)[-1]:
                if self.ark_fd is not None:
                    self.ark_fd.close()
                self.ark_fd = open(path, 'rb')
            self.ark_fd.seek(int(pos),0)
            headerstr = self.ark_fd.read(5)
            header = struct.unpack('<xcccc', headerstr)
            if header[0] != b'B':
                print("Input .ark file is not binary", flush=True)
                exit(1)

            rows = 0; cols= 0
            rowstr = self.ark_fd.read(5)
            m, rows = struct.unpack('<bi', rowstr)
            colstr = self.ark_fd.read(5)
            n, cols = struct.unpack('<bi', colstr)
            self.num_feats += rows
        self.ark_fd.close()
        self.ark_fd = None

        self.num_feats = int(self.num_feats)

        # Track where we are with respect to feature index
        self.scp_file.seek(0)
        self.current_utt_id = None
        self.current_feat_mat = None
        self.current_feat_idx = 0

    def __del__(self):
        self.scp_file.close()
        if self.ark_fd is not None:
            self.ark_fd.close()
        
    def __len__(self):
        return self.num_feats

    def __getitem__(self, idx):
        if self.current_feat_mat is None:
            scp_line = self.scp_file.readline()
            self.current_utt_id, self.current_feat_mat, self.ark_fd = read_next_utt(scp_line,
                                                                                    ark_fd=self.ark_fd)
            self.current_feat_idx = 0
        feats_tensor = torch.FloatTensor(self.current_feat_mat[self.current_feat_idx, :])

        # Target is identical to feature tensor
        target_tensor = feats_tensor.clone()

        # Update where we are in the feature matrix
        self.current_feat_idx += 1
        if self.current_feat_idx == len(self.current_feat_mat):
            self.current_utt_id = None
            self.current_feat_mat = None
            self.current_feat_idx = 0

        if idx == len(self) - 1:
            # Reset back to the beginning
            self.scp_file.seek(0)
        
        return (feats_tensor, target_tensor)



# Utterance-by-utterance loading of Kaldi files
# Includes utterance ID data data for evaluation and decoding
class KaldiEvalDataset(Dataset):
    def __init__(self, scp_path):
        super(KaldiEvalDataset, self).__init__()

        # Load in Kaldi files
        self.scp_path = scp_path
        self.scp_file = open(self.scp_path, 'r')

        # Determine how many utterances are included
        self.utt_ids = []
        for scp_line in self.scp_file:
            utt_id, path_pos = scp_line.replace('\n','').split(' ')
            self.utt_ids.append(utt_id)
        self.num_utts = len(self.utt_ids)

        # Reset files
        self.scp_file.seek(0)

    # Utterance-level
    def __len__(self):
        return self.num_utts

    def utt_id(self, utt_idx):
        assert(utt_idx < len(self))
        return self.utt_ids[utt_idx]
    
    # Full utterance, not one frame
    def __getitem__(self, idx):
        # Get next utt from SCP file
        scp_line = self.scp_file.readline()
        utt_id, feat_mat, ark_fd = read_next_utt(scp_line)
        feats_tensor = torch.FloatTensor(feat_mat)
        
        # Target is identical to feature tensor
        target_tensor = feats_tensor.clone()

        if idx == len(self) - 1:
            # Reset back to the beginning
            self.scp_file.seek(0)
        
        # Return utterance ID info as well
        return (feats_tensor, target_tensor, utt_id)



# Concat version of KaldiEvalDataset
class KaldiEvalConcatDataset(ConcatDataset):
    def utt_id(self, utt_idx):
        assert(utt_idx < len(self))
        dataset_idx = bisect.bisect_right(self.cummulative_sizes, utt_idx)
        if dataset_idx == 0:
            sample_idx = utt_idx
        else:
            sample_idx = utt_idx - self.cummulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].utt_id(sample_idx)
