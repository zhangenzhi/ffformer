# TODO

## Dataset

- [ ] **samples_per_scene**: 当前 `__len__` 返回场景数（47），每个 epoch 每个场景只随机裁剪一次。
  如果场景数很少（如 1-2 个），训练样本严重不足。
  应在 `ForAINetV2Dataset.__init__` 加 `samples_per_scene` 参数，让 `__len__` 返回
  `len(scenes) * samples_per_scene`，`__getitem__` 用 `idx % len(scenes)` 取场景。
