# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :model_scope_download.py
# @Time        :2025/1/21 20:50
# @Author      :JasonMa
# @ProjectName :neural_collaborative_filtering
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('Qwen/Qwen2.5-3B-Instruct', cache_dir='E:/Models/')
