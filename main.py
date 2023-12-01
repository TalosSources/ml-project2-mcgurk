import mc_gurk_classification

model = mc_gurk_classification.training_pipeline(X_cache_path='dataset/cached_tensors/X_badaga.pt',epochs=50000, learning_rate=0.0001, model_save_path='saved_models/lr_badaga.pt', labels_path='dataset/train_labels.txt')