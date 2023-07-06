import os
from threading import Thread
import subprocess
import sys

tasks = []

def parseTrain(train_args:dict):
    
    if os.path.exists(train_args["pretrained_model_name_or_path"]) is False:
        # raise FileNotFoundError
        pass
    extra_args=[
        '--enable_bucket',
        '--output_dir="/kaggle/working/output"',
        '--logging_dir="/kaggle/working/logs"',
        '--mixed_precision="fp16"',
        '--save_precision="fp16"',
        '--seed="1337"',
        '--prior_loss_weight=1',
        '--max_token_length=225',
        '--caption_extension=".txt"',
        '--training_comment=lora_webui_script',
        '--xformers',
        '--lowram',
        f'--pretrained_model_name_or_path={train_args["pretrained_model_name_or_path"]}',
        f'--train_data_dir={train_args["train_data_dir"]}',
        f'--log_prefix={train_args["output_name"]}',
        f'--resolution={train_args["resolution"]}',
        f'--max_train_epochs={train_args["max_train_epochs"]}',
        f'--learning_rate={train_args["learning_rate"]}',
        f'--unet_lr={train_args["unet_lr"]}',
        f'--text_encoder_lr={train_args["text_encoder_lr"]}',
        f'--lr_scheduler={train_args["lr_scheduler"]}',
        f'--lr_warmup_steps={train_args["lr_warmup_steps"]}',
        f'--network_dim={train_args["network_dim"]}',
        f'--network_alpha={train_args["network_alpha"]}',
        f'--output_name={train_args["output_name"]}',
        f'--train_batch_size={train_args["train_batch_size"]}',
        f'--save_every_n_epochs={train_args["save_every_n_epochs"]}',
        f'--save_model_as={train_args["save_model_as"]}',
        f'--min_bucket_reso={train_args["min_bucket_reso"]}',
        f'--max_bucket_reso={train_args["max_bucket_reso"]}',
        '--v2' if "v2" in train_args else f'--clip_skip={train_args["clip_skip"]}',
        '--v_parameterization' if "v_parameterization" in train_args and "v_parameterization" in train_args else '',
        '--network_train_unet_only' if "network_train_unet_only" in train_args else '',
        '--network_train_text_encoder_only' if "network_train_text_encoder_only" in train_args else '',
        f'--stop_text_encoder_training={train_args["stop_text_encoder_training"]}' if train_args["stop_text_encoder_training"] != '0' else '',
        '--shuffle_caption' if "shuffle_caption" in train_args else '',
        '--weighted_captions' if "weighted_captions" in train_args else '',
        '--gradient_checkpointing' if "gradient_checkpointing" in train_args else '',
        f'--gradient_accumulation_steps={train_args["gradient_accumulation_steps"]}' if str(train_args["gradient_accumulation_steps"]) != str(0) else '',
        f'--noise_offset={train_args["noise_offset"]}' if train_args["noise_offset"] != '0' else (f'--multires_noise_iterations={train_args["multires_noise_iterations"]}' if train_args["multires_noise_iterations"] !='0' else ''),
        f'--multires_noise_discount={train_args["multires_noise_discount"]}' if train_args["noise_offset"] == '0' and train_args["multires_noise_iterations"] !='0' else '',
                       ]
    
    if "enable_block_weights" in train_args:
        if "enable_dylora" in train_args:
            train_args.pop("enable_dylora")
        if "enable_lycoris" in train_args:
            train_args.pop("enable_lycoris")

        extra_args.append("--network_args")
        extra_args.append(f"down_lr_weight={train_args['down_lr_weight']}")
        extra_args.append(f"mid_lr_weight={train_args['mid_lr_weight']}")
        extra_args.append(f"up_lr_weight={train_args['up_lr_weight']}")
        extra_args.append(f"block_lr_zero_threshold={train_args['block_lr_zero_threshold']}")
        if train_args("conv_dim") != '0':
            extra_args.append(f"conv_dim={train_args('conv_dim')}")
            if extra_args["conv_alpha"] != '0':
                extra_args.append(f"conv_alpha={train_args['conv_alpha']}")
            
    if "enable_lycoris" in train_args:
        if "enable_dylora" in train_args:
            train_args.pop("enable_dylora")
        extra_args.append("--network_module=lycoris.kohya")
        extra_args.append("--network_args")
        if train_args("conv_dim") != '0':
            extra_args.append(f"conv_dim={train_args('conv_dim')}")
            if extra_args["conv_alpha"] != '0':
                extra_args.append(f"conv_alpha={train_args['conv_alpha']}")
        extra_args.append(f"algo={train_args['algo']}")
    else:
        extra_args.append("--network_module=networks.lora")
    if "enable_dylora" in train_args:
        extra_args.append("--network_module=networks.dylora")
        extra_args.append("--network_args")
        extra_args.append(f"unit={train_args['unit']}")
        if train_args("conv_dim") != '0':
            extra_args.append(f"conv_dim={train_args('conv_dim')}")
            if extra_args["conv_alpha"] != '0':
               extra_args.append(f"conv_alpha={train_args['conv_alpha']}")
    if train_args["optimizer_type"] == "adafactor":
        extra_args.append("--optimizer_type=optimizer_type")
        extra_args.append("--optimizer_args")
        extra_args.append("scale_parameter=True")
        extra_args.append("warmup_init=True")
    if train_args["optimizer_type"] in ["DAdaptation", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptSGD"]:
        extra_args.append(f"--optimizer_type={train_args['optimizer_type']}")
        extra_args.append("--optimizer_args")
        if train_args["optimizer_type"] in ["DAdaptation", "DAdaptAdam"]:
             extra_args.append("decouple=True")
             extra_args.append("weight_decay=0.01")
        extra_args["lr"] = '1'
        extra_args["unet_lr"] = '1'
        extra_args["text_encoder_lr"] = '1'
    if train_args["optimizer_type"] in ["Lion", "Lion8bit"]:
        extra_args.append(f"--optimizer_type={train_args['optimizer_type']}")
        extra_args.append("--optimizer_args")
        extra_args.append("betas=.95,.98")
    if train_args["optimizer_type"] == "AdamW8bit":
        extra_args.append("--use_8bit_adam")
    # if train_args["network_weights"] 在已有的lora上训练
    if os.path.exists(train_args["reg_data_dir"]):
        extra_args.append(f'--reg_data_dir={train_args["reg_data_dir"]}')
        if extra_args["prior_loss_weight"] !='0':
            extra_args.append(f"--prior_loss_weight={train_args['prior_loss_weight']}")
    if train_args["keep_tokens"] != '0':
        extra_args.append(f"--keep_tokens={train_args['keep_tokens']}")
    if train_args["min_snr_gamma"] != '0':
        extra_args.append(f"--min_snr_gamma={train_args['min_snr_gamma']}")
    if "enable_sample" in train_args:
        extra_args.append(f"--sample_every_n_epochs={train_args['sample_every_n_epochs']}")
        extra_args.append(f"--sample_prompts={train_args['sample_prompts']}")
        extra_args.append(f"--sample_sampler={train_args['sample_sampler']}")
    launch_args = [
        '--multi_gpu',
        '--num_cpu_threads_per_process=2',
    ]
    
    return f"conda run -n venv --no-capture-output accelerate launch {' '.join(launch_args)} 'train_network.py' {' '.join((i for i in extra_args if i))}"

class Tasks:
    def __init__(self,pid) -> None:
        self.tasks = []
        self.key = None
        self.isRunning = False
        self.finished_close = False
        self.output_info = ''
        self.temp_info = []

    def addTask(self,cmd, train_args):  
        self.tasks.append((cmd,train_args))
        if self.isRunning is False:
            Thread(target=self.runTasks).start()
        return str(len(self.tasks))
    def runTasks(self):
        try:
            self.isRunning = True
            while len(self.tasks) != 0:
                p = subprocess.Popen(self.tasks[0][0], shell = True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.getOutput(p)
                self.uploadResults(self.tasks[0][1]["hugface_key"],self.tasks[0][1]["output_name"],self.tasks[0][1]["repo_id"],self.tasks[0][1]["repo_type"])
                self.tasks.remove(self.tasks[0])
            self.finishedProcess()
            self.isRunning = False
        except Exception as e:
            self.addTemp(str(e)+"服务器错误")
            self.isRunning = False
            self.tasks.remove(self.tasks[0])

    def getOutput(self,p):
        for info in iter(p.stdout.readline, b''):
            self.addTemp(info.decode('utf-8').strip())

    def uploadResults(self,key, name, repo_id, repo_type):
        if key:
            def upload_file(file):
                file_name = os.path.basename(file)
                response = api.upload_file(path_or_fileobj=file,path_in_repo=f"{name}/{file_name}", repo_id = repo_id,repo_type = repo_type)
                self.addTemp(response)
                os.remove(file)
            from huggingface_hub import login
            login(token=key)
            from huggingface_hub import HfApi
            api = HfApi()
            [upload_file(file) for file in [os.path.join(root, file) for root, folders, files in os.walk("/kaggle/working/output") for file in files]]

    def getTempInfo(self):
        info = '\n'.join(self.temp_info)
        self.temp_info.clear()
        return info

    def addTemp(self,info):
        self.output_info +=info + "\n"
        self.temp_info.append(info)

    def finishedProcess(self):
        if self.finished_close and len(self.tasks) == 0:
            if os.name == 'nt':
                os.system(f"taskkill /pid {os.getpid()} /f /t")
            elif os.name == 'posix':
                os.system(f"kill -KILL {os.getpid()}")
