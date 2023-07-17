parent_work_dir="./data/.workdir"
parent_log_dir="./data/outputs/test_00_debug"


HYDRA_FULL_ERROR=1   OC_CAUSE=1 python main.py  \
                seed=1\
				dataset.name='CIFAR10'\
				dataset.path='/gpfswork/rech/tza/uki35ex/dataset'\
                +mlxpy.interactive_mode=True\
                +mlxpy.version_manager.parent_work_dir=$parent_work_dir\
                +mlxpy.logger.parent_log_dir=$parent_log_dir\
                +mlxpy.use_scheduler=True\
                +mlxpy.use_version_manager=False\
                +mlxpy.use_logger=True\
