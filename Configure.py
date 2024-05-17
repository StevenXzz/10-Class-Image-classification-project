model_configs = {
	"name": 'MyModel',
	"depth": 2,
    "lr": 0.1,
    'weight_decay':1e-4,
	'momentum': 0.9,
    'device': 'cuda',
    'batch_size': 32,
    'pretrained':False,
    'start_epoch': 300
	# ...
}

training_configs = {
	"lr": 0.1,
    "n_epochs": 300,
    'image_size' : 32,
    'device': 'cuda',
    'batch_size': 64,
    'result_dir': './models'
	# ...
}




### END CODE HERE
