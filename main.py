import mc_gurk_classification

# To compute the X_tensors from videos using PerceiverIO, videos_paths should be specified, and X_save_path can be specified to cache the obtain tensors
# To use a cached X_tensor, one should instead specify the X_cache_path,

model, X, Y, accuracy = mc_gurk_classification.training_pipeline(
    labels_path="dataset/labels/train_labels_a+v.txt",
    videos_paths=mc_gurk_classification.sorted_videos_paths('dataset/train_sets/train_bafava'),
    #X_cache_path="cache/tensors/X_bafava_a+av.pt",
    epochs=100000,
    learning_rate=0.00002,
    model_save_path="cache/models/lr_bafava_a+av.pt",
    # X_save_path='dataset/cached_tensors/X_bafava_a+v+av.pt')
)
