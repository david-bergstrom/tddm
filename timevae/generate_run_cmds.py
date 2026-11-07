def generate_bash_script():
    dataset_options = ['geolife-100', 'porto-100', 'cabspotting-100']
    ds_data = [2, 2, 2]
    seq_lens = [128]

    with open('run_all.sh', 'w') as bash_script:
        bash_script.write("#!/bin/bash\n")
        for seq_len in seq_lens:
            for dataset, d in zip(dataset_options, ds_data):
                bash_script.write(f"python test_vae.py --dataset {dataset} --seq_len {seq_len} 2>&1 | tee -a timevae-{dataset}-{seq_len}.log\n")

    with open('run_all_short.sh', 'w') as bash_script:
        bash_script.write("#!/bin/bash\n")
        for seq_len in seq_lens:
            for dataset, d in zip(dataset_options, ds_data):
                bash_script.write(f"python test_vae.py --dataset {dataset} --seq_len {seq_len} --epochs 5 2>&1 | tee -a timevae-short-{dataset}-{seq_len}.log\n")

if __name__ == "__main__":
    generate_bash_script()
