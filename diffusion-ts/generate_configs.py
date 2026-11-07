from string import Template

datasets = ['geolife-100', 'porto-100', 'cabspotting-100']
seq_lens = [128]
ds_data = [2, 2, 2, 2, 2, 2]

with open('max_hyperparameter.template', 'r') as f:
    template = Template(f.read())

all_cmds = []

for seq_len in seq_lens:
    for dataset_str, d_data in zip(datasets, ds_data):
        config = template.substitute({
            'dataset_str': dataset_str,
            'd_data': d_data,
            'seq_len': seq_len
            })
        config_file_path = f'Config/cmp_{dataset_str}_{seq_len}.yaml'
        with open(config_file_path, 'w') as f:
            f.write(config)

        cmd = f'python main.py --name {dataset_str}{seq_len} --config_file {config_file_path} --gpu 0 --benchmark'
        single_experiment_path = f'train-{dataset_str}-{seq_len}.sh'
        with open(single_experiment_path, 'w') as f:
            f.write('#!/bin/bash\n\n' + cmd + '\n')


        all_cmds.append(cmd)

with open('train-all.sh', 'w') as f:
    f.write('#!/bin/bash\n' + '\n'.join(all_cmds))

