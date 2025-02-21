parent_work_dir="./data/.workdir"
parent_log_dir="./data/outputs/test_00_debug"


HYDRA_FULL_ERROR=1   OC_CAUSE=1 python main.py  \
                seed=1\
				dataset.type='test'\
				dataset.name='FlickrFace'\
				dataset.path='/gpfswork/rech/tza/uki35ex/dataset/'\
				dataset.batch_size=8\
				dataset.num_workers=2\
				dataset.pin_memory=True\
				dataset.max_size=50000\
				optimizer.lr=.00008\
				optimizer.num_steps=500000\
				optimizer.grad_acc=2\
                checkpoints.mode='test'\
                checkpoints.save_path='/gpfsscratch/rech/tza/uki35ex/_Experiments/denoising-diffusion-pytorch/models/test'\
                checkpoints.load_path='/gpfsscratch/rech/tza/uki35ex/_Experiments/denoising-diffusion-pytorch/models/test/63/model-58.pt'\
				checkpoints.image_test='./10000.png'\
				sampling.calculate_fid=False\
				sampling.num_fid_samples=1000\
				sampling.objective='pred_x0'\
                +mlxpy.interactive_mode=True\
                +mlxpy.version_manager.parent_work_dir=$parent_work_dir\
                +mlxpy.logger.parent_log_dir=$parent_log_dir\
                +mlxpy.use_scheduler=True\
                +mlxpy.use_version_manager=False\
                +mlxpy.use_logger=True\
