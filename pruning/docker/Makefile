run:
	docker build -t pruning .
run2:
	docker build -t unet_noaug_train .

in:
	NV_GPU=3 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it pruning:latest bash

in2:
	NV_GPU=0 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it unet_noaug_train:latest bash

