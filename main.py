import mc_gurk_classification

model = mc_gurk_classification.training_pipeline(X_cache_path='torch_tensors/X_cached_2',epochs=10000, learning_rate=0.00001, model_save_path='model.pt')