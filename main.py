import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
from pathlib import Path
import torch
from typing import List
import pandas as pd
from entities import *
from recall import RecallRecommender
from rough_ranking import RoughRankingRecommender
from fine_ranking import FineRankingRecommender
from rearrangement import MmrDiversity
from history_features import HistoryConfig, HistorySequenceEncoder
from methods.registry import ACTIVE_METHODS, HISTORY_IMPROVEMENT_METHODS, build_method_registry

class RecommenderSystem:
    def __init__(self, interactions_path: str | Path = "data/splits/interactions_train.csv", methods=None,
                 history_config: HistoryConfig | None = None):
        # 需要用到的各种数据
        self.items=[]
        self.id_item_dict={}
        self.item_idx2id={}
        self.users=[]
        self.interactions=[] # [(user_id,item_id,rating)]

        self.df_items=pd.DataFrame()
        self.df_users=pd.DataFrame()
        self.df_interactions=pd.DataFrame() # 只包含 user_id 和 item_id 的 Dataframe
        self.df_train=pd.DataFrame()
        self.interactions_path = Path(interactions_path)
        self.method_names = dict(ACTIVE_METHODS)
        if methods:
            self.method_names.update(methods)
        self.methods = build_method_registry(self.method_names)
        # 各阶段的推荐器
        self.rough_ranking_recommender=RoughRankingRecommender()
        self.fine_ranking_recommender=FineRankingRecommender()
        self.rearrangement_recommender=MmrDiversity()
        self.history_config = history_config or HistoryConfig()
        # 自动初始化数据
        self.prepare_data()

        self.recall_recommender = RecallRecommender(self.item_idx2id)

    def prepare_data(self):
        print("data preparing...")
        print(
            f"PyTorch={torch.__version__}, CUDA compiled={torch.version.cuda}, "
            f"CUDA available={torch.cuda.is_available()}",
            flush=True,
        )
        if torch.cuda.is_available():
            print(
                f"CUDA device={torch.cuda.get_device_name(0)}",
                flush=True,
            )
        # clip用于提取图片和文本特征，拼接成为物品内容特征向量
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model = clip_model.to(device)
        print(
            f"CLIP parameters device={next(clip_model.parameters()).device}",
            flush=True,
        )
        # 准备物品数据
        self.df_items = pd.read_csv("data/items_new.csv", encoding="utf-8")
        self.df_items['item_keywords'] = self.df_items['item_keywords'].apply(lambda x: tuple(x.split(';')))
        print("开始构建物品索引...")
        for row in self.df_items.itertuples(index=True):  # index=True 获取原始的 DataFrame 索引
            item_idx = row.Index
            # 构建 discrete_features 和 continuous_features 字典
            discrete_features = {
                'city': row.city,
                'name': row.name,
                'author': row.author,
                'item_categories': row.item_categories,
                'item_keywords': row.item_keywords
            }
            continuous_features = {'price': row.price}
            # 创建 Item 对象
            item = Item(
                item_idx,
                row.item_id,
                row.name,
                row.author,
                row.item_categories,
                row.item_keywords,
                discrete_features,
                continuous_features,
                row.create_time,
                row.image,
                row.description
            )
            # 填充字典和列表
            self.id_item_dict[item.item_id] = item
            self.item_idx2id[item_idx] = item.item_id
            self.items.append(item)
        # 图片下载和 CPU 预处理并行，CLIP 按批次使用 GPU 推理
        calculate_content_features_batch(
            self.items,
            clip_model,
            preprocess,
            cache_dir="data/image_cache",
        )
        self.history_encoder = HistorySequenceEncoder(self.items, self.history_config)
        print("物品索引构建完成")
        # 准备用户数据
        self.df_users = pd.read_csv("data/users_new.csv", encoding="utf-8")
        self.df_users['user_categories'] = self.df_users['user_categories'].fillna('').apply(lambda x: tuple(x.split(';')))
        self.df_users['user_keywords'] = self.df_users['user_keywords'].fillna('').apply(lambda x: tuple(x.split(';')))
        print("开始构建用户画像...")
        # itertuples 返回命名元组，访问速度极快
        for row in self.df_users.itertuples(index=True):
            # 获取原始索引
            user_idx = row.Index
            # 构建离散和连续特征字典
            discrete_features = {
                'gender': row.gender,
                'user_categories': row.user_categories,
                'user_keywords': row.user_keywords
            }
            continuous_features = {'age': row.age}
            # 创建 UserProfile 对象
            user = UserProfile(
                user_idx,
                row.user_id,
                row.user_categories,
                row.user_keywords,
                discrete_features,
                continuous_features,
                50
            )
            self.users.append(user)
        print("用户画像构建完成")
        # 准备交互数据
        if not self.interactions_path.exists():
            raise FileNotFoundError(
                f"Training interactions not found: {self.interactions_path}. "
                "Run data_split.py before starting the recommendation system."
            )
        self.df_interactions = pd.read_csv(self.interactions_path, encoding="utf-8")
        self.interactions=list(zip(
            self.df_interactions['user_id'],
            self.df_interactions['item_id'],
            self.df_interactions['rating']
        ))
        # Build the online user history from training interactions only. The
        # held-out test split is intentionally never loaded here.
        interaction_history = self.df_interactions.copy()
        interaction_history['_parsed_datetime'] = pd.to_datetime(
            interaction_history['datetime'], errors='coerce'
        )
        interaction_history = interaction_history.sort_values(
            ['user_id', '_parsed_datetime']
        )
        users_by_id = {user.user_id: user for user in self.users}
        for row in interaction_history.itertuples(index=False):
            user = users_by_id.get(row.user_id)
            if user is not None:
                user.add_interaction(row.item_id)
        # 准备双塔模型，三塔模型和精排多目标模型训练数据(根据交互记录)
        df_merge = pd.merge(self.df_interactions, self.df_users, how='left', on='user_id')
        self.df_train = pd.merge(df_merge, self.df_items, how='left', on='item_id')
        print("data finished\n")

    def fit(self):
        """离线计算"""
        # 召回
        self.recall_recommender.fit(self.df_train,self.items,self.interactions,self.df_interactions
                                    ,list(self.df_users['user_id']),list(self.df_items['item_id']))
        # 粗排
        self.rough_ranking_recommender.train_three_towers_model(self.df_train)
        self.rough_ranking_recommender.calculate_item_features(self.df_items)
        # 精排
        self.fine_ranking_recommender.train_multi_task_model(self.df_train)
        # 重排
        self.rearrangement_recommender.build_cosine_similarity_matrix(self.items)

    def fit_with_weights(self,twin_towers_model_weights_path,light_gcn_weights_path,
                         three_towers_model_weights_path,multi_task_model_path):
        """
        离线计算（模型直接加载训练好的权重）
        """
        # 召回
        self.recall_recommender.fit_with_weights(self.df_train, self.items, self.interactions, self.df_interactions
                                                ,list(self.df_users['user_id']), list(self.df_items['item_id'])
                                                ,twin_towers_model_weights_path,light_gcn_weights_path)
        # 粗排
        self.rough_ranking_recommender.load_three_towers_model(self.df_train,three_towers_model_weights_path)
        self.rough_ranking_recommender.calculate_item_features(self.df_items)
        # 精排
        self.fine_ranking_recommender.load_multi_task_model(self.df_train,multi_task_model_path)
        # 重排
        self.rearrangement_recommender.build_cosine_similarity_matrix(self.items)

    def fit_with_weight_dir(self, weights_dir: str | Path):
        """Load an experiment's exported model weights."""
        weights_dir = Path(weights_dir)
        self.fit_with_weights(
            weights_dir / "improved_twin_towers_model.pth",
            weights_dir / "lightgcn.pth",
            weights_dir / "three_towers_model.pth",
            weights_dir / "multi_task_model.pth",
        )

    def recommend(self,user_id,hour,is_weekend,is_holiday)->List[int]:
        """
        在线推荐
        Args:
            user_id: 用户ID
            hour: 当前时间(小时)
            is_weekend: 当前是否是周末
            is_holiday: 当前是否是节假日
        """
        user_profile = next((user for user in self.users if user.user_id == user_id),None)
        if user_profile is None:
            raise KeyError(f"Unknown user_id: {user_id}")
        history_ids = user_profile.get_last_n_interactions(10_000)
        df_user_profile=self.df_users.loc[self.df_users['user_id']==user_id]
        # ============= 召回 ================
        recall_items_ids=self.methods['recall'](
            self.recall_recommender,
            user_profile,
            history_encoder=self.history_encoder,
        )
        # ============= 粗排 ================
        # 提取召回的id列表对应的行
        df_recall_items = self.df_items[self.df_items['item_id'].isin(recall_items_ids)].reset_index(drop=True)
        # 创建场景特征
        df_scene = pd.DataFrame([[hour,is_weekend,is_holiday]],columns=['hour','is_weekend','is_holiday'])
        rough_ranking_ids = self.methods['rough_ranking'](
            self.rough_ranking_recommender,
            df_user_profile.copy(),
            df_recall_items,
            df_scene,
            history_encoder=self.history_encoder,
            history_ids=history_ids,
        )
        # ============= 精排 ================
        df_rough_items = df_recall_items[df_recall_items['item_id'].isin(rough_ranking_ids)].reset_index(drop=True)
        df_user = pd.concat([df_user_profile] * len(df_rough_items)).reset_index(drop=True)
        df_multi_task = pd.concat([df_rough_items, df_user], axis=1)
        # 添加上当前的场景特征
        df_multi_task['hour'] = [hour] * len(df_multi_task)
        df_multi_task['is_weekend'] = [is_weekend] * len(df_multi_task)
        df_multi_task['is_holiday'] = [is_holiday] * len(df_multi_task)
        item_id_score = self.methods['fine_ranking'](
            self.fine_ranking_recommender,
            df_multi_task,
            history_encoder=self.history_encoder,
            history_ids=history_ids,
        )
        # ============= 重排 ================
        print("rearrangement...")
        rearrangement_items = []
        for item in self.items:
            if item.item_id in item_id_score:
                item.relevance_score = item_id_score[item.item_id]
                rearrangement_items.append(item)
        final_selected_items_ids=self.methods['rearrangement'](
            self.rearrangement_recommender, rearrangement_items
        )
        # print("最终推荐的物品ID列表：", final_selected_items_ids)
        print("recommend finished\n")
        return final_selected_items_ids

def main(weights_dir="model_weights", methods=None, history_config=None):
    """主流程"""
    recommender_system = RecommenderSystem(methods=methods, history_config=history_config)
    # 各模型权重路径
    twin_towers_model_weights_path = "model_weights/improved_twin_towers_model.pth"
    light_gcn_weights_path = "model_weights/lightgcn.pth"
    three_towers_model_weights_path = "model_weights/three_towers_model.pth"
    multi_task_model_path = "model_weights/multi_task_model.pth"
    # 采用预训练权重直接加载模型
    if Path(weights_dir).resolve() == Path("model_weights").resolve():
        recommender_system.fit_with_weights(
            twin_towers_model_weights_path,
            light_gcn_weights_path,
            three_towers_model_weights_path,
            multi_task_model_path,
        )
    else:
        recommender_system.fit_with_weight_dir(weights_dir)
    # 假设对第一个用户做推荐，此时是15时，是周末，不是节假日
    recommender_system.recommend("U00001",15,1,0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the split-based recommendation system")
    parser.add_argument('--weights-dir', default='model_weights')
    parser.add_argument('--method-profile', choices=['baseline', 'history'], default='baseline')
    parser.add_argument('--history-mode', choices=['din', 'sim_soft', 'sim_hard', 'hybrid'], default='hybrid')
    parser.add_argument('--history-L', type=int, default=20)
    parser.add_argument('--history-alpha', type=float, default=2.0)
    args = parser.parse_args()
    selected_methods = HISTORY_IMPROVEMENT_METHODS if args.method_profile == 'history' else ACTIVE_METHODS
    main(
        weights_dir=args.weights_dir,
        methods=selected_methods,
        history_config=HistoryConfig(
            mode=args.history_mode,
            max_length=args.history_L,
            hard_threshold_alpha=args.history_alpha,
        ),
    )
