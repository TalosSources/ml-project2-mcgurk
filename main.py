import mc_gurk_classification

model, X, Y, accuracy = mc_gurk_classification.training_pipeline(labels_path='dataset/labels/train_labels_a+v.txt',
                                                                 #videos_paths=mc_gurk_classification.sorted_videos_paths('dataset/train_sets/train_bafava'), 
                                                                 X_cache_path='dataset/cached_tensors/X_bafava_a+av.pt',
                                                                 epochs=100000, 
                                                                 learning_rate=0.00002, 
                                                                 model_save_path='stuff/saved_models/lr_bafava_a+av.pt', )
                                                                 #X_save_path='dataset/cached_tensors/X_bafava_a+v+av.pt')
