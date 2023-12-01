import mc_gurk_classification

model = mc_gurk_classification.training_pipeline(X_cache_path='dataset/cached_tensors/X_3x20da20ga.pt',epochs=100000, learning_rate=0.0001, model_save_path='saved_models/lr_3x20da20ga.pt', labels_path='dataset/train_labels.txt')