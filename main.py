import mc_gurk_classification

model = mc_gurk_classification.training_pipeline(X_cache_path='dataset/cached_tensors/X_babgaga_a+v+av.pt',epochs=100000, learning_rate=0.00003, model_save_path='saved_models/lr_babgaga_a+v+av.pt', labels_path='dataset/train_labels_a+v+av.txt')