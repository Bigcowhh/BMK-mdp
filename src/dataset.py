import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import json
import os
import pandas as pd

class MaichartDataset(Dataset):
    def __init__(self, serialized_dir, labels_csv, cache_size=None):
        self.serialized_dir = serialized_dir
        
        # 读取CSV并清理数据
        self.labels_data = pd.read_csv(labels_csv)
        self.labels_data = self.labels_data.dropna(subset=['song_id', 'level_index', 'difficulty_constant','total_notes', 'json_filename'])
        
        # 重置索引以确保连续的整数索引
        self.labels_data = self.labels_data.reset_index(drop=True)

        # TouchArea映射
        self.touch_area_mapping = {" ": 0, "A": 1, "D": 2, "E": 3, "B": 4, "C": 5} # 从外到内

        # 初始化编码器
        self._setup_encoders()
        
        # 缓存机制
        self.cache_size = cache_size
        self.cache = {}  # 存储处理后的数据
        self.cache_access_order = []  # 记录访问顺序，用于LRU淘汰

    def _setup_encoders(self):
        """设置note类型和位置的编码器"""
        # Note类型编码器
        self.NOTE_TYPES = ['Tap', 'Hold', 'Slide', 'Touch', 'TouchHold']
        self.note_type_encoder = OneHotEncoder(
            sparse_output=False,
            dtype=np.float32,
            handle_unknown='ignore'
        )
        self.note_type_encoder.fit(np.array(self.NOTE_TYPES).reshape(-1, 1))
        
        # 位置编码器（假设位置范围是1-8）
        self.positions = list(range(1, 9))  # maimai有8个位置
        self.position_encoder = OneHotEncoder(
            sparse_output=False,
            dtype=np.float32,
            handle_unknown='ignore'
        )
        self.position_encoder.fit(np.array(self.positions).reshape(-1, 1))

    def _manage_cache(self, key):
        """管理缓存，实现LRU淘汰策略"""
        # 如果key已在缓存中，更新访问顺序
        if key in self.cache:
            self.cache_access_order.remove(key)
            self.cache_access_order.append(key)
            return
        
        # 如果缓存已满，删除最久未使用的项
        if self.cache_size is not None and len(self.cache) >= self.cache_size:
            oldest_key = self.cache_access_order.pop(0)
            del self.cache[oldest_key]
        
        # 添加新key到访问顺序
        self.cache_access_order.append(key)

    def _extract_note_features(self, note, time):
        """
        从单个note中提取特征向量
        
        Args:
            note: 包含note信息的字典
            time: note的时间戳
            
        Returns:
            np.ndarray: 21维的特征向量
        """
        # 编码note类型和位置
        note_type_encoded = self.note_type_encoder.transform([[note['noteType']]])[0]
        position_encoded = self.position_encoder.transform([[note['startPosition']]])[0]
        
        # 提取其他特征
        hold_time = note.get('holdTime', 0)
        is_break = int(note['isBreak'])
        is_ex = int(note['isEx'])
        is_slide_break = int(note['isSlideBreak'])
        slide_start_time = note['slideStartTime']
        slide_end_time = slide_start_time + note['slideTime']
        touch_area = self.touch_area_mapping[note['touchArea']]
        
        # 组合特征向量
        feature_vector = np.concatenate([
            [time],             # 1维
            note_type_encoded,  # 5维
            position_encoded,   # 8维
            [hold_time],        # 1维
            [is_break],         # 1维
            [is_ex],            # 1维
            [is_slide_break],   # 1维
            [slide_start_time], # 1维
            [slide_end_time],   # 1维
            [touch_area]        # 1维
        ])  # 总共 21维
        
        return feature_vector

    def _extract_sequence_features(self, json_data):
        """
        从JSON数据中提取整个谱面的note序列特征
        
        Args:
            json_data: 包含谱面数据的JSON对象
            
        Returns:
            list: note特征向量的列表
        """
        note_groups = json_data.get('notes', [])
        note_features_sequence = []
        
        for note_group in note_groups:
            time = note_group['Time']
            notes = note_group['Notes']
            
            for note in notes:
                feature_vector = self._extract_note_features(note, time)
                note_features_sequence.append(feature_vector)
        
        return note_features_sequence

    def _extract_sequence_features_vectorized(self, json_data):
        """
        向量化提取整个谱面的note序列特征
        
        Args:
            json_data: 包含谱面数据的JSON对象
            
        Returns:
            np.ndarray: (num_notes, 21) 的特征矩阵
        """
        note_groups = json_data.get('notes', [])
        if not note_groups:
            raise ValueError(f"未找到{json_data}的note group信息")
        
        # 收集所有notes数据
        all_times = []
        all_notes_data = []
        
        for note_group in note_groups:
            time = note_group['Time']
            notes = note_group['Notes']
            
            for note in notes:
                all_times.append(time)
                all_notes_data.append(note)
        
        if not all_notes_data:
            raise ValueError(f"未找到{json_data}的note信息")
        
        num_notes = len(all_notes_data)
        
        # 向量化提取所有note类型
        note_types = np.array([note['noteType'] for note in all_notes_data]).reshape(-1, 1)
        note_types_encoded = self.note_type_encoder.transform(note_types)  # (num_notes, 5)
        
        # 向量化提取所有位置
        positions = np.array([note['startPosition'] for note in all_notes_data]).reshape(-1, 1)
        positions_encoded = self.position_encoder.transform(positions)  # (num_notes, 8)
        
        # 向量化提取其他特征
        times_array = np.array(all_times, dtype=np.float32)  # (num_notes,)
        hold_times = np.array([note.get('holdTime', 0) for note in all_notes_data], dtype=np.float32)
        is_break = np.array([int(note['isBreak']) for note in all_notes_data], dtype=np.float32)
        is_ex = np.array([int(note['isEx']) for note in all_notes_data], dtype=np.float32)
        is_slide_break = np.array([int(note['isSlideBreak']) for note in all_notes_data], dtype=np.float32)
        slide_start_times = np.array([note['slideStartTime'] for note in all_notes_data], dtype=np.float32)
        slide_times = np.array([note['slideTime'] for note in all_notes_data], dtype=np.float32)
        slide_end_times = slide_start_times + slide_times
        touch_areas = np.array([self.touch_area_mapping[note['touchArea']] for note in all_notes_data], dtype=np.float32)
        
        # 组合所有特征 - 向量化拼接
        feature_matrix = np.column_stack([
            times_array,           # (num_notes, 1)
            note_types_encoded,    # (num_notes, 5)
            positions_encoded,     # (num_notes, 8)
            hold_times,            # (num_notes, 1)
            is_break,              # (num_notes, 1)
            is_ex,                 # (num_notes, 1)
            is_slide_break,        # (num_notes, 1)
            slide_start_times,     # (num_notes, 1)
            slide_end_times,       # (num_notes, 1)
            touch_areas            # (num_notes, 1)
        ])  # 总共 (num_notes, 21)
        
        return feature_matrix

    def __getitem__(self, index):
        # 从CSV中获取第index行的数据
        row = self.labels_data.iloc[index]
        json_filename = row['json_filename']
        difficulty_constant = float(row['difficulty_constant'])
        
        # 检查缓存中是否已有处理好的数据
        cache_key = json_filename
        if cache_key in self.cache:
            # 缓存命中，更新访问顺序并返回缓存数据
            self._manage_cache(cache_key)
            note_features_tensor = self.cache[cache_key]
        else:
            # 缓存未命中，读取并处理JSON文件
            json_file_path = os.path.join(self.serialized_dir, json_filename)
            
            # 检查文件是否存在
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(f"JSON文件不存在: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON解析失败: {json_file_path}") from e

            # 使用向量化方法提取谱面特征序列
            note_features_matrix = self._extract_sequence_features_vectorized(json_data)

            # 将谱面数据转换为张量
            note_features_tensor = torch.from_numpy(note_features_matrix)
            
            # 将处理好的数据存入缓存
            self._manage_cache(cache_key)
            self.cache[cache_key] = note_features_tensor

        difficulty_constant_tensor = torch.tensor(difficulty_constant, dtype=torch.float32)
        return note_features_tensor, difficulty_constant_tensor

    def __len__(self):
        return len(self.labels_data)
    
    def get_cache_stats(self):
        """
        获取缓存统计信息
        
        Returns:
            dict:
                - cache_size: 当前缓存中的条目数
                - max_cache_size: 最大缓存大小（如果未设置则为无限制）
                - cache_usage: 缓存使用百分比
                - cached_files: 当前缓存中的文件名列表
        """
        if self.cache_size is None:
            cache_usage_percent = 100.0 if len(self.cache) > 0 else 0.0
            max_cache_display = "无限制"
        else:
            cache_usage_percent = len(self.cache) / self.cache_size * 100
            max_cache_display = self.cache_size
        return {
            'cache_size': len(self.cache),
            'max_cache_size': max_cache_display,
            'cache_usage': cache_usage_percent,
            'cached_files': list(self.cache.keys())
        }
    

def collate_fn(batch):
    """
    自定义的collate_fn，用于处理变长序列。
    - 对note序列进行padding，使其在batch内长度一致。
    - 将标签堆叠成一个tensor。
    """
    # 1. 分离序列和标签
    # batch中的每个元素是 (note_features_tensor, difficulty_constant_tensor)
    sequences, labels = zip(*batch)

    # 2. 对序列进行padding
    # pad_sequence期望一个tensor列表
    # batch_first=True使输出的形状为 (batch_size, sequence_length, feature_dim)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # 3. 将标签堆叠成一个tensor
    # torch.stack(labels) 会创建一个 [batch_size] 的1D张量
    # .view(-1, 1) 将其转换为 [batch_size, 1] 以匹配模型输出
    labels_tensor = torch.stack(labels).view(-1, 1)

    return padded_sequences, labels_tensor

class CollateWithStatsWrapper:
    """
    包装器类，允许在需要时获取统计信息，同时保持与现有代码的兼容性
    """
    def __init__(self):
        self.last_stats = None
    
    def __call__(self, batch):
        if not batch:
            raise ValueError("batch为0")
        sequences, labels = zip(*batch)
        
        # 收集统计信息
        seq_lengths = [seq.size(0) for seq in sequences]
        stats = {
            'batch_size': len(sequences),
            'min_seq_length': min(seq_lengths),
            'max_seq_length': max(seq_lengths),
            'avg_seq_length': sum(seq_lengths) / len(seq_lengths),
            'total_notes': sum(seq_lengths),
            'padding_ratio': 0
        }
        
        # Padding
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        labels_tensor = torch.stack(labels).view(-1, 1)
        
        # 计算padding比例
        total_elements = padded_sequences.numel()
        padding_elements = (padded_sequences == 0).sum().item()
        stats['padding_ratio'] = padding_elements / total_elements if total_elements > 0 else 0
        
        # 存储统计信息
        self.last_stats = stats
        
        return padded_sequences, labels_tensor
    
    def get_last_stats(self):
        """获取最后一个batch的统计信息"""
        return self.last_stats

def analyze_sequence_lengths():
    """
    分析数据集中序列长度的分布，找出padding比例过高的原因
    """
    import matplotlib.pyplot as plt
    
    base_dir = os.path.dirname(os.path.abspath(''))
    serialized_dir = os.path.join(base_dir, "data", "serialized")
    csv_path = os.path.join(base_dir, "data", "song_info.csv")
    
    dataset = MaichartDataset(serialized_dir, csv_path)
    
    print("=== 序列长度分布分析 ===")
    print(f"数据集总大小: {len(dataset)}")
    
    # 收集序列长度统计
    sequence_lengths = []
    feature_dims = []
    sample_count = min(100, len(dataset))  # 分析前100个样本
    
    print(f"分析前 {sample_count} 个样本...")
    
    for i in range(sample_count):
        try:
            note_features, difficulty = dataset[i]
            seq_len = note_features.shape[0]
            feat_dim = note_features.shape[1] if len(note_features.shape) > 1 else 0
            
            sequence_lengths.append(seq_len)
            feature_dims.append(feat_dim)
            
            if i < 10:  # 显示前10个样本的详细信息
                print(f"  样本 {i}: 序列长度={seq_len}, 特征维度={feat_dim}, 难度={difficulty:.2f}")

            if i < 3: # 显示前3个样本的特征矩阵
                print(f"  样本 {i} 特征矩阵:\n{note_features.numpy()}")
                
        except Exception as e:
            print(f"  样本 {i} 处理出错: {e}")
            sequence_lengths.append(0)
            feature_dims.append(0)
    
    # 统计分析
    sequence_lengths = np.array(sequence_lengths)
    feature_dims = np.array(feature_dims)
    
    print(f"\n序列长度统计:")
    print(f"  最小长度: {np.min(sequence_lengths)}")
    print(f"  最大长度: {np.max(sequence_lengths)}")
    print(f"  平均长度: {np.mean(sequence_lengths):.1f}")
    print(f"  中位数长度: {np.median(sequence_lengths):.1f}")
    print(f"  标准差: {np.std(sequence_lengths):.1f}")
    
    print(f"\n特征维度统计:")
    print(f"  特征维度: {np.unique(feature_dims)}")


    
    # 计算不同批次大小的padding比例
    batch_sizes = [4, 8, 16, 32]
    print(f"\n不同批次大小的padding分析:")
    
    for batch_size in batch_sizes:
        total_padding_ratio = 0
        num_batches = 0
        
        for i in range(0, len(sequence_lengths), batch_size):
            batch_lengths = sequence_lengths[i:i+batch_size]
            if len(batch_lengths) == 0:
                continue
                
            max_len = np.max(batch_lengths)
            total_elements = len(batch_lengths) * max_len
            actual_elements = np.sum(batch_lengths)
            
            if total_elements > 0:
                padding_ratio = 1 - (actual_elements / total_elements)
                total_padding_ratio += padding_ratio
                num_batches += 1
        
        avg_padding = total_padding_ratio / num_batches if num_batches > 0 else 0
        print(f"  批次大小 {batch_size}: 平均padding比例 {avg_padding:.3f}")
    
    # 找出异常长的序列
    print(f"\n异常长序列分析:")
    percentile_95 = np.percentile(sequence_lengths, 95)
    percentile_99 = np.percentile(sequence_lengths, 99)
    
    print(f"  95%分位数: {percentile_95:.1f}")
    print(f"  99%分位数: {percentile_99:.1f}")
    
    long_sequences = sequence_lengths[sequence_lengths > percentile_95]
    print(f"  超过95%分位数的序列数量: {len(long_sequences)}")
    print(f"  这些序列长度: {sorted(long_sequences)}")
    
    return sequence_lengths, feature_dims
