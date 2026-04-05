import glob
import os
import os.path as osp
import re

from .bases import BaseImageDataset

class BallShow(BaseImageDataset):
    """
    自定义数据集加载类
    """
    # 这里对应 data/ 下的文件夹名称
    dataset_dir = 'BallShow'
    dataset_aliases = ('BallShow', 'Occluded_Duke')
    img_pattern = re.compile(r'([-\d]+)_c(\d+)(?:s(\d+))?')

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(BallShow, self).__init__()
        self.dataset_dir = self._resolve_dataset_dir(root)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        self.camid2label = self._build_camid_mapping()

        # 处理文件夹
        # relabel=True 表示训练集会将 ID 重新映射为 0, 1, 2...
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print(f"=> BallShow loaded from {self.dataset_dir}")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        # 获取数据集统计信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)
        self.num_total_cams = len(self.camid2label)

    def _resolve_dataset_dir(self, root):
        if isinstance(root, (list, tuple)):
            if len(root) != 1:
                raise RuntimeError('BallShow expects a single dataset root, got {}'.format(root))
            root = root[0]

        root = osp.abspath(osp.expanduser(root))
        split_names = ('bounding_box_train', 'query', 'bounding_box_test')
        candidates = [root]
        candidates.extend(osp.join(root, name) for name in self.dataset_aliases)

        for candidate in candidates:
            if all(osp.isdir(osp.join(candidate, split_name)) for split_name in split_names):
                return candidate

        raise RuntimeError(
            "BallShow dataset was not found under '{}'. Checked: {}".format(
                root, ', '.join(candidates)
            )
        )

    def _check_before_run(self):
        """检查文件夹是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _build_camid_mapping(self):
        camids = set()
        for dir_path in (self.train_dir, self.query_dir, self.gallery_dir):
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            for img_path in img_paths:
                match = self.img_pattern.search(osp.basename(img_path))
                if match is None:
                    continue
                camids.add(int(match.group(2)))

        if not camids:
            raise RuntimeError('No valid BallShow images found under {}'.format(self.dataset_dir))

        return {camid: label for label, camid in enumerate(sorted(camids))}

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # 支持 .jpg
        # 如果有 .png 图片，可以追加: img_paths += glob.glob(osp.join(dir_path, '*.png'))

        pid_container = set()
        for img_path in sorted(img_paths):
            # 先遍历一遍获取所有 ID，用于 relabel
            match = self.img_pattern.search(osp.basename(img_path))
            if match is None:
                continue
            pid = int(match.group(1))
            if pid == -1:
                continue
            pid_container.add(pid)

        # 建立 ID 到 0~N 的映射
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            match = self.img_pattern.search(osp.basename(img_path))
            if match is None:
                raise RuntimeError('Unexpected BallShow filename format: {}'.format(img_path))

            pid = int(match.group(1))
            camid = int(match.group(2))

            if pid == -1:
                continue  # 过滤垃圾图片

            camid = self.camid2label[camid]

            if relabel:
                pid = pid2label[pid]

            # 如果是测试集，不进行 relabel，直接用原始 PID (需加上偏移量 pid_begin)
            else:
                pid = self.pid_begin + pid

            # dataset 格式: (图片路径, ID, 摄像头ID, TrackID)
            # TrackID 暂时不用，固定为 1
            dataset.append((img_path, pid, camid, 1))

        return dataset
