from aws_utils.s3_dual_paths import DualPath


with DualPath("s3://cvdb-data/prod/training_data/semantic_segmentation/point-corn/SET-corn-stand-spring23-test/20230420-112042-512_512-r4/images/", 'r', local_path='data_test') as dp:
    pass