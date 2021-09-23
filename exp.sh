python run.py --config ./configs/config_async_100c.json
mkdir allasync_res
mv *.png allasync_res/
mv *.csv allasync_res/

python run.py --config ./configs/config_async_10c.json
mkdir async_random_normal_res
mv *.png async_random_normal_res/
mv *.csv async_random_normal_res/

python run.py --config ./configs/config.json
mkdir async_latencyloss_normal_res
mv *.png async_latencyloss_normal_res/
mv *.csv async_latencyloss_normal_res/

python run.py --config ./configs/config_sync_10c.json
mkdir sync_res
mv *.png sync_res/
mv *.csv sync_res/
