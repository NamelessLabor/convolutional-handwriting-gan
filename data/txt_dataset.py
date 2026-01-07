# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import random
from PIL import Image
import torch
from data.base_dataset import BaseDataset, get_transform
from data.text_dataset import RegularCollator


class TxtDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--collate', action='store_false', default=True,
                            help='use regular collate function in data loader')
        parser.add_argument('--txt_list', type=str, required=True,
                            help='path to txt list file')
        parser.add_argument('--txt_format', type=str, default='tsv',
                            choices=['tsv', 'pair_lines'],
                            help='format of txt list file')
        parser.add_argument('--txt_resize', type=str, default='charResize',
                            choices=['charResize', 'keepRatio', 'noResize'],
                            help='LMDB-style resize policy for txt images')
        parser.add_argument('--txt_init_gap', type=int, default=0,
                            help='initial gap for LMDB-style preprocessing')
        parser.add_argument('--txt_h_gap', type=int, default=0,
                            help='vertical gap for LMDB-style preprocessing')
        parser.add_argument('--txt_charminW', type=int, default=16,
                            help='minimum character width for LMDB-style preprocessing')
        parser.add_argument('--txt_charmaxW', type=int, default=17,
                            help='maximum character width for LMDB-style preprocessing')
        parser.add_argument('--no_txt_discard_wide', action='store_false',
                            dest='txt_discard_wide', default=True,
                            help='do not discard images that are too wide for LMDB-style preprocessing')
        parser.add_argument('--no_txt_discard_narr', action='store_false',
                            dest='txt_discard_narr', default=True,
                            help='do not discard images that are too narrow for LMDB-style preprocessing')
        return parser

    def __init__(self, opt, target_transform=None):
        BaseDataset.__init__(self, opt)
        self.samples = self._load_txt_list(opt.txt_list, opt.txt_format)
        self.transform = get_transform(opt, grayscale=(opt.input_nc == 1))
        self.target_transform = target_transform
        if opt.collate:
            self.collate_fn = TxtCollator(opt)
        else:
            self.collate_fn = RegularCollator(opt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        try:
            img = Image.open(img_path)
        except IOError:
            print('Corrupted image for %s' % img_path)
            return self[index + 1]

        img = self._apply_lmdb_preprocess(img, label, img_path)
        if img is None:
            return self[index + 1]

        img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'img': img, 'label': label, 'img_path': img_path}

    def _apply_lmdb_preprocess(self, img, label, img_path):
        if self.opt.txt_resize not in ['charResize', 'keepRatio']:
            return img

        width, height = img.size
        new_height = self.opt.imgH - (self.opt.txt_h_gap * 2)
        len_word = len(label)
        if height == 0 or len_word == 0:
            return img
        width = int(width * self.opt.imgH / height)
        new_width = width
        if self.opt.txt_resize == 'charResize':
            avg_char_width = width / len_word
            if (avg_char_width > (self.opt.txt_charmaxW - 1)) or (avg_char_width < self.opt.txt_charminW):
                if self.opt.txt_discard_wide and avg_char_width > 3 * (self.opt.txt_charmaxW - 1):
                    print('%s has a width larger than max image width' % img_path)
                    return None
                if self.opt.txt_discard_narr and avg_char_width < (self.opt.txt_charminW / 3):
                    print('%s has a width smaller than min image width' % img_path)
                    return None
                new_width = len_word * random.randrange(self.opt.txt_charminW, self.opt.txt_charmaxW)

        img = img.resize((new_width, new_height))
        init_w = int(random.normalvariate(self.opt.txt_init_gap, self.opt.txt_init_gap / 2)) if self.opt.txt_init_gap > 0 else 0
        new_img = Image.new("RGB", (new_width + self.opt.txt_init_gap, self.opt.imgH), color=(255, 255, 255))
        new_img.paste(img, (abs(init_w), self.opt.txt_h_gap))
        return new_img

    def _load_txt_list(self, list_path, list_format):
        list_path = os.path.abspath(list_path)
        data_root = os.path.abspath(self.root)
        if list_format == 'pair_lines':
            return self._parse_pair_lines(list_path, data_root)
        if list_format == 'tsv':
            return self._parse_tsv(list_path, data_root)
        raise ValueError(f'Unsupported txt_format: {list_format}')

    def _parse_pair_lines(self, list_path, data_root):
        samples = []
        with open(list_path, 'r', encoding='utf-8') as handle:
            lines = [line.rstrip('\n') for line in handle]

        idx = 0
        total = len(lines)
        while idx < total:
            img_line = lines[idx].strip()
            idx += 1
            if not img_line or img_line.startswith('#'):
                continue
            if idx >= total:
                raise ValueError('pair_lines format expects image path and label lines.')
            label_line = lines[idx].rstrip('\n')
            idx += 1
            img_path = self._resolve_path(img_line, data_root)
            samples.append((img_path, label_line))
        return samples

    def _parse_tsv(self, list_path, data_root):
        samples = []
        with open(list_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '\t' in line:
                    img_part, label_part = line.split('\t', 1)
                else:
                    parts = line.split(maxsplit=1)
                    img_part = parts[0]
                    label_part = parts[1] if len(parts) > 1 else ''
                img_path = self._resolve_path(img_part, data_root)
                samples.append((img_path, label_part))
        return samples

    def _resolve_path(self, path, data_root):
        if os.path.isabs(path):
            return path
        return os.path.join(data_root, path)


class TxtCollator(object):
    def __init__(self, opt):
        self.resolution = opt.resolution

    def __call__(self, batch):
        img_path = [item['img_path'] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        imgs = torch.ones(
            [len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)],
            dtype=torch.float32,
        )
        for idx, item in enumerate(batch):
            imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
        item = {'img': imgs, 'img_path': img_path}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        return item
