import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 224
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4]
    config.base_channel = 32
    config.n_classes = 1

    # ********** useless **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config
