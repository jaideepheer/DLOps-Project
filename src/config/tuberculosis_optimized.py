from datetime import datetime
import os

abs_path = os.path.dirname(__file__)
now = datetime.now()
now = now.strftime("%d_%m_%Y")


def build_args(*_, task_name = None, model_name = None, **overrides):
    assert  len(_) == 0
    model_name = model_name or 'resnet18'
    task_name = task_name or 'tuberculosis'
    return {
        'data_dir'                 : f'/workspace/Assignment_1/data/{task_name}',
        'image_dir'                : f'/workspace/Assignment_1/data/{task_name}',
        'df_train'                 : f'/workspace/Assignment_1/data/{task_name}/train.csv',
        'df_val'                   : f'/workspace/Assignment_1/data/{task_name}/val_split.csv',
        'df_test'                  : f'/workspace/Assignment_1/data/{task_name}/test.csv',
        'vocab'                    : f'/workspace/Assignment_1/data/{task_name}/labels.json',
        'triton_dir'               : f"/workspace/Assignment_1/triton_model_repository/{task_name}/",

        'GPU_ID'                   : '0',
        'batch_size'               : 32,
        'num_workers'              : 8,
        'pin_memory'               : True,
        'channels_last'            : False,
        'epochs'                   : 5,
        'test_size'                : 0.2,
        'learning_rate'            : 0.0001,
        

        ######### MODEL parameters ########
        'model_name'               : model_name,
        'out_weight_dir'           : f'/workspace/Assignment_1/saved_models/{task_name}',
        'out_weight'               : f'/workspace/Assignment_1/saved_models/{task_name}/{model_name}.pt',
        'image_size'               : 512,
        'n_classes'                : 2,
        ##################################

        'distributed'              : False,
        'local_rank'               : 0,
        'amp'                      : True,
        'opt_level'                : "O2",
        'wandb'                    : True,
        'project_name'             : f'task_{task_name}',
        'device'                   : 'gpu',
        'seed'                     : 42,
        'world_size'               : 4,
        'TF32'                     : True, 
        'benchmark'                : True,
        **overrides,
    }
