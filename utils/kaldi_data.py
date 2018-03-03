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
# Do not use Pytorch's built-in shuffle in DataLoader -- use the optional arguments here instead
class KaldiDataset(Dataset):
    def __init__(self, scp_path, left_context=0, right_context=0, shuffle_utts=False, shuffle_feats=False, include_lookup=False):
        super(KaldiDataset, self).__init__()

        self.left_context = left_context
        self.right_context = right_context

        # Load in Kaldi files
        self.scp_path = scp_path

        self.include_lookup = include_lookup
        if self.include_lookup:
            # Not great performance-wise; only include if using for analysis
            self.uttid_2_scpline = dict()
        
        # Determine how many features are included
        self.num_feats = 0
        self.ark_fd = None
        self.scp_lines = []
        for scp_line in self.scp_file:
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

            self.scp_lines.append(scp_line)
            if self.include_lookup:
                self.uttid_2_scpline[utt_id] = scp_line

        self.ark_fd.close()
        self.ark_fd = None

        self.num_feats = int(self.num_feats)

        self.ark_fd.close()
        self.ark_fd = None
        
        # Set up shuffling of utterances within SCP (if enabled)
        self.shuffle_utts = shuffle_utts
        if self.shuffle_utts:
            # Shuffle SCP list in place
            np.random.shuffle(self.scp_lines)

        # Set up shuffling of frames within utterance (if enabled)
        self.shuffle_feats = shuffle_feats

        # Track where we are with respect to feature index
        self.current_utt_id = None
        self.current_feat_mat = None
        self.current_utt_idx = 0
        self.current_feat_idx = 0

    def __del__(self):
        if self.ark_fd is not None:
            self.ark_fd.close()
        
    def __len__(self):
        return self.num_feats

    def __getitem__(self, idx):
        if self.current_feat_mat is None:
            scp_line = self.scp_lines[self.current_utt_idx]
            self.current_utt_id, feat_mat, self.ark_fd = read_next_utt(scp_line,
                                                                           ark_fd=self.ark_fd)

            # Duplicate frames at start and end of utterance (as in Kaldi)
            self.current_feat_mat = np.empty((feat_mat.shape[0] + self.left_context + self.right_context,
                                              feat_mat.shape[1]))
            self.current_feat_mat[self.left_context:self.left_context + feat_mat.shape[0], :] = feat_mat
            for i in range(self.left_context):
                self.current_feat_mat[i, :] = feat_mat[0, :]
            for i in range(self.right_context):
                self.current_feat_mat[self.left_context + feat_mat.shape[0] + i, :] = feat_mat[feat_mat.shape[0] - 1, :]

            # Shuffle features if enabled
            if self.shuffle_feats:
                np.random.shuffle(self.current_feat_mat)
            
            self.current_feat_idx = 0

        feats_tensor = torch.FloatTensor(self.current_feat_mat[self.current_feat_idx:self.current_feat_idx + self.left_context + self.right_context + 1, :])
        feats_tensor = feats_tensor.view((self.left_context + self.right_context + 1, -1))

        # Target is identical to feature tensor
        target_tensor = feats_tensor.clone()

        # Update where we are in the feature matrix
        self.current_feat_idx += 1
        if self.current_feat_idx == len(self.current_feat_mat) - self.left_context - self.right_context:
            self.current_utt_id = None
            self.current_feat_mat = None
            self.current_feat_idx = 0
            self.current_utt_idx += 1

        if idx == len(self) - 1:
            # We've seen all of the data (i.e. one epoch) -- shuffle SCP list in place
            np.random.shuffle(self.scp_lines)
            self.current_utt_idx = 0

        return (feats_tensor, target_tensor)

    # Get specific utterance 
    def feats_for_uttid(self, utt_id):
        if not self.include_lookup:
            raise RuntimeError("Lookup table not built for this dataset; initialize with include_lookup=True to do so")

        utt_id, feat_mat, ark_fd = read_next_utt(self.uttid_2_scpline[utt_id])
        return feat_mat

# Utterance-by-utterance loading of Kaldi files
# Includes utterance ID data data for evaluation and decoding
# Do not use Pytorch's built-in shuffle in DataLoader -- use the optional arguments here instead
class KaldiEvalDataset(Dataset):
    def __init__(self, scp_path, shuffle_utts=False, shuffle_feats=False):
        super(KaldiEvalDataset, self).__init__()

        # Load in Kaldi files
        self.scp_path = scp_path
        self.uttid_2_scpline = dict()
        with open(self.scp_path, 'r') as scp_file:
            # Determine how many utterances are included
            # Also sets up shuffling of utterances within SCP if desired
            self.scp_lines = list(map(lambda line: line.replace('\n', ''), scp_file.readlines()))
            self.utt_ids = []
            for scp_line in self.scp_lines:
                utt_id = scp_line.split(" ")[0]
                self.utt_ids.append(utt_id)
                self.uttid_2_scpline[utt_id] = scp_line
            
        self.shuffle_utts = shuffle_utts
        if self.shuffle_utts:
            # Shuffle SCP list in place
            np.random.shuffle(self.scp_lines)

        # Set up shuffling of feats within utterance
        self.shuffle_feats = shuffle_feats

    # Utterance-level
    def __len__(self):
        return len(self.utt_ids)

    def utt_id(self, utt_idx):
        assert(utt_idx < len(self))
        return self.utt_ids[utt_idx]
    
    # Get next utterance
    def __getitem__(self, idx):
        # Get next utt from SCP file
        scp_line = self.scp_lines[idx]
        utt_id, feat_mat, ark_fd = read_next_utt(scp_line)
        if self.shuffle_feats:
            # Shuffle features in-place
            np.random.shuffle(feat_mat)
        feats_tensor = torch.FloatTensor(feat_mat)
        
        # Target is identical to feature tensor
        target_tensor = feats_tensor.clone()

        if idx == len(self) - 1 and self.shuffle_utts:
            # We've seen all of the data (i.e. one epoch) -- shuffle SCP list in place
            np.random.shuffle(self.scp_lines)
        
        # Return utterance ID info as well
        return (feats_tensor, target_tensor, utt_id)

    # Get specific utterance 
    def feats_for_uttid(self, utt_id):
        utt_id, feat_mat, ark_fd = read_next_utt(self.uttid_2_scpline[utt_id])
        return feat_mat
