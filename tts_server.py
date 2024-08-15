import json
from multiprocessing import Manager
import time
import os
from pathlib import Path
import wave
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import io
import asyncio
import aiofiles
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Form, Depends, Query
import nltk
import uvicorn


###########################
#### ENVIRONMENT SETUP ####
###########################

os.environ['TORCH_CUDA_ARCH_LIST'] = "7.0"

#######################
#### LOGGING SETUP ####
#######################

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

##############################
#### MODEL GPU RETRIEVAL #####
##############################

class GPUModelManager:
    def __init__(self, modeldownload_base_path, modeldownload_model_path, params):
        self.models: Dict[int, Optional[Xtts]] = {}
        self.locks: Dict[int, asyncio.Lock] = {}
        self.modeldownload_base_path = modeldownload_base_path
        self.modeldownload_model_path = modeldownload_model_path
        self.params = params
        
        # Initialize locks for each GPU
        for gpu_id in range(torch.cuda.device_count()):
            self.locks[gpu_id] = asyncio.Lock()

    async def load_model(self, gpu_id: int) -> None:
        device = torch.device(f'cuda:{gpu_id}')
        
        async with self.locks[gpu_id]:
            # Ensure the model entry exists in the dictionary
            if gpu_id not in self.models:
                self.models[gpu_id] = None
            
            if self.models[gpu_id] is None:
                try:
                    base_path = (
                        self.modeldownload_base_path / self.modeldownload_model_path
                        if str(self.modeldownload_base_path) == "models"
                        else self.modeldownload_base_path / self.modeldownload_model_path
                    )
                    config_path = base_path / "config.json"
                    vocab_path_dir = base_path / "vocab.json"
                    checkpoint_dir = base_path

                    if self.params["tts_method_api_tts"]:
                        model = TTS(self.params["tts_model_name"]).to(device)
                    else:
                        if self.params["tts_method_xtts_local"] or self.params.get("tts_method_xtts_ft", False):
                            config = XttsConfig()
                            config.load_json(str(config_path))
                            model = Xtts.init_from_config(config)
                            model.load_checkpoint(
                                config,
                                checkpoint_dir=str(checkpoint_dir),
                                vocab_path=str(vocab_path_dir),
                                use_deepspeed=self.params["deepspeed_activate"],
                            )
                        else:  # Fallback to API Local
                            model = TTS(
                                model_path=base_path,
                                config_path=config_path,
                            ).to(device)

                    logging.info(f"Coqui Public Model License: https://coqui.ai/cpml.txt")

                    model.to(device)
                    torch.cuda.synchronize(device)
                    self.models[gpu_id] = model
                    logging.info(f"Model loaded and synchronized on GPU {gpu_id}")
                    logging.debug(f"Model for GPU {gpu_id} added to model_instances")

                except torch.cuda.CudaError as cuda_error:
                    logging.error(f"CUDA error encountered on GPU {gpu_id}: {cuda_error}", exc_info=True)
                    raise
                except Exception as e:
                    logging.error(f"Error during model loading on GPU {gpu_id}: {e}", exc_info=True)
                    raise

    async def unload_model(self, gpu_id: int) -> None:
        async with self.locks[gpu_id]:
            if gpu_id in self.models and self.models[gpu_id] is not None:
                try:
                    model = self.models[gpu_id]
                    model.cpu()  # Move the model to CPU first
                    del model  # Delete the model object
                    self.models[gpu_id] = None  # Remove the reference from our dictionary
                    torch.cuda.empty_cache()  # Clear CUDA cache to free up memory
                    logging.info(f"Model unloaded from GPU {gpu_id}")
                except Exception as e:
                    logging.error(f"Error during model unloading on GPU {gpu_id}: {e}", exc_info=True)
                    raise
            else:
                logging.warning(f"No model loaded on GPU {gpu_id} to unload")

    async def get_model(self, gpu_id: int) -> Xtts:
        if gpu_id not in self.models or self.models[gpu_id] is None:
            await self.load_model(gpu_id)
        return self.models[gpu_id]

    def is_model_loaded(self, gpu_id: int) -> bool:
        return gpu_id in self.models and self.models[gpu_id] is not None


###########################
#### GLOBAL VARIABLES ####
###########################

device_count = torch.cuda.device_count()

this_dir = Path(__file__).parent.resolve()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load language configurations
with open(this_dir / "system" / "config" / "languages.json", encoding="utf8") as f:
    languages = json.load(f)

tts_method_xtts_ft = False

#################################################################
#### LOAD PARAMS FROM confignew.json - REQUIRED FOR BRANDING ####
#################################################################

def load_config(file_path):
    with open(file_path, "r") as configfile_path:
        configfile_data = json.load(configfile_path)
    return configfile_data

configfile_path = this_dir / "confignew.json"
params = load_config(configfile_path)
params["low_vram"] = "false" if not torch.cuda.is_available() else params["low_vram"]

temperature = params["local_temperature"]
repetition_penalty = params["local_repetition_penalty"]

modeldownload_config_file_path = this_dir / "modeldownload.json"

if modeldownload_config_file_path.exists():
    with open(modeldownload_config_file_path, "r") as modeldownload_config_file:
        modeldownload_settings = json.load(modeldownload_config_file)
    modeldownload_base_path = Path(modeldownload_settings.get("base_path", ""))
    modeldownload_model_path = Path(modeldownload_settings.get("model_path", ""))
else:
    logging.warning(f"modeldownload.config is missing. Please re-download it and save it in the alltalk_tts main folder.")

trained_model_directory = this_dir / "models" / "trainedmodel"
finetuned_model = trained_model_directory.exists()

if finetuned_model:
    required_files = ["model.pth", "config.json", "vocab.json"]
    finetuned_model = all((trained_model_directory / file).exists() for file in required_files)

try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    logging.error("Could not find the TTS module. Make sure to install the requirements for the alltalk_tts extension.")
    raise

deepspeed_available = False
try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    pass

gpu_model_manager = GPUModelManager(modeldownload_base_path=modeldownload_base_path, modeldownload_model_path=modeldownload_model_path, params=params)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#####################################
#### MODEL LOADING AND UNLOADING ####
#####################################

import subprocess

def setup_logging(rank):
    logging.basicConfig(
        level=logging.DEBUG,
        format=f'%(asctime)s - Rank {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

async def setup(rank, world_size):
    setup_logging(rank)
    logging.debug(f"Starting setup for rank {rank}")

    if dist.is_initialized():
        dist.barrier()

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    logging.debug(f"Device set to {device} for rank {rank}")

    generate_start_time = time.time()

    try:
        await gpu_model_manager.load_model(rank)
        logging.debug(f"Model loaded and added to model_instances for rank {rank}")
    except Exception as e:
        logging.error(f"Error loading model for rank {rank}: {str(e)}")
        raise

    if dist.is_initialized():
        dist.barrier()

    generate_end_time = time.time()
    generate_elapsed_time = generate_end_time - generate_start_time
    logging.info(f"Model Loaded in {generate_elapsed_time:.2f} seconds for rank {rank}")
    logging.info(f"Model Ready")

    params["tts_model_loaded"] = True

    output_directory = this_dir / params["output_folder_wav_standalone"]
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Output directory created: {output_directory}")

    if dist.is_initialized():
        # Synchronize all processes and gather model instance information
        dist.all_reduce(all_ranks, op=dist.ReduceOp.MAX)
        all_ranks = all_ranks.tolist()
        logging.debug(f"All ranks with loaded models: {all_ranks}")

    logging.debug(f"All processes have completed setup for rank {rank}")


async def handle_tts_method_change(tts_method):
    global model
    global tts_method_xtts_ft
    logging.info(f"Changing model (Please wait 15 seconds)")

    params.update({
        "tts_method_api_tts": tts_method == "API TTS",
        "tts_method_api_local": tts_method == "API Local",
        "tts_method_xtts_local": tts_method == "XTTSv2 Local",
        "deepspeed_activate": tts_method != "API TTS",
    })
    tts_method_xtts_ft = tts_method == "XTTSv2 FT"

    for rank in gpu_model_manager.models:
        if gpu_model_manager.models[rank]:
            gpu_model_manager.unl

    await setup(0, 1)

###########################
#### DDP Initialization ###
###########################


async def ddp_main(rank, world_size):
    await setup(rank=rank, world_size=world_size)

    if rank == 0:
        logging.info(f"Rank {rank} starting server...")
        await start_server()  # Await the asynchronous server start
    else:
        logging.info(f"Rank {rank} is idle...")

    # Keep the loop running
    while True:
        await asyncio.sleep(3600)

def run_ddp_main(rank, world_size):
    asyncio.run(ddp_main(rank, world_size))
    
def initialize_ddp():
    world_size = torch.cuda.device_count()
    logging.debug(f"World size (number of GPUs): {world_size}")
    
    if world_size > 1:
        logging.info("Multiple GPUs detected, initializing DDP...")
        mp.spawn(run_ddp_main, args=(world_size,), nprocs=world_size, join=True)
    else:
        logging.info("Single GPU or CPU detected, running setup without DDP...")
        asyncio.run(ddp_main(0, 1))

###########################
### DDP Setup Functions ###
###########################

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_IB_DISABLE'] = '1'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(f"DDP initialized successfully on rank {rank}")

def cleanup_ddp():
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            logging.info("DDP process group destroyed successfully.")
    except Exception as e:
        logging.error("Error during DDP cleanup:", exc_info=True)

async def load_ddp_model(rank, world_size, model_loader):
    logging.debug(f"Starting load_ddp_model for rank {rank}")
    try:
        # Initialize DDP only if it's not already initialized
        if not dist.is_initialized():
            setup_ddp(rank, world_size)
            logging.debug(f"DDP setup complete for rank {rank}")
        
        # Call the model_loader function to load the model
        logging.debug(f"About to call model_loader for rank {rank}")
        model = await model_loader()  # Properly await the asynchronous function
        
        # Move the model to the correct device
        model.to(rank)
        logging.debug(f"Model loaded and moved to device for rank {rank}")
        
        # Wrap the model in DDP
        ddp_model = DDP(model, device_ids=[rank])
        logging.debug(f"DDP model created for rank {rank}")
        
        return ddp_model
    except Exception as e:
        logging.error(f"Error in load_ddp_model for rank {rank}: {str(e)}", exc_info=True)
        raise

##################
#### LOW VRAM ####
##################
# LOW VRAM - MODEL MOVER VRAM(cuda)<>System RAM(cpu) for Low VRAM setting
async def switch_device():
    global model, device
    # Check if CUDA is available before performing GPU-related operations
    if torch.cuda.is_available():
        if device == "cuda":
            device = "cpu"
            model.to(device)
            torch.cuda.empty_cache()
        else:
            device = "cuda"
            model.to(device)

@app.post("/api/lowvramsetting")
async def set_low_vram(request: Request, new_low_vram_value: bool):
    global device
    try:
        if new_low_vram_value is None:
            raise ValueError("Missing 'low_vram' parameter")

        if params["low_vram"] == new_low_vram_value:
            return Response(
                content=json.dumps(
                    {
                        "status": "success",
                        "message": f"[{params['branding']}Model] LowVRAM is already {'enabled' if new_low_vram_value else 'disabled'}.",
                    }
                )
            )
        params["low_vram"] = new_low_vram_value
        if params["low_vram"]:
            await gpu_model_manager.unload_model(model)
            if torch.cuda.is_available():
                device = "cpu"
                print(
                    f"[{params['branding']}Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m"
                )
                print(
                    f"[{params['branding']}Model] \033[94mLowVRAM Enabled.\033[0m Model will move between \033[93mVRAM(cuda) <> System RAM(cpu)\033[0m"
                )
                await setup()
            else:
                # Handle the case where CUDA is not available
                print(
                    f"[{params['branding']}Model] \033[91mError:\033[0m Nvidia CUDA is not available on this system. Unable to use LowVRAM mode."
                )
                params["low_vram"] = False
        else:
            await gpu_model_manager.unload_model(model)
            if torch.cuda.is_available():
                device = "cuda"
                print(
                    f"[{params['branding']}Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m"
                )
                print(
                    f"[{params['branding']}Model] \033[94mLowVRAM Disabled.\033[0m Model will stay in \033[93mVRAM(cuda)\033[0m"
                )
                await setup()
            else:
                # Handle the case where CUDA is not available
                print(
                    f"[{params['branding']}Model] \033[91mError:\033[0m Nvidia CUDA is not available on this system. Unable to use LowVRAM mode."
                )
                params["low_vram"] = False
        return Response(content=json.dumps({"status": "lowvram-success"}))
    except Exception as e:
        return Response(content=json.dumps({"status": "error", "message": str(e)}))

###################
#### DeepSpeed ####
###################
# DEEPSPEED - Reload the model when DeepSpeed checkbox is enabled/disabled
async def handle_deepspeed_change(value):
    global model
    if value:
        # DeepSpeed enabled
        print(f"[{params['branding']}Model] \033[93mDeepSpeed Activating\033[0m")

        print(
            f"[{params['branding']}Model] \033[94mChanging model \033[92m(DeepSpeed can take 30 seconds to activate)\033[0m"
        )
        print(
            f"[{params['branding']}Model] \033[91mInformation\033[0m If you have not set CUDA_HOME path, DeepSpeed may fail to load/activate"
        )
        print(
            f"[{params['branding']}Model] \033[91mInformation\033[0m DeepSpeed needs to find nvcc from the CUDA Toolkit. Please check your CUDA_HOME path is"
        )
        print(
            f"[{params['branding']}Model] \033[91mInformation\033[0m pointing to the correct location and use 'set CUDA_HOME=putyoutpathhere' (Windows) or"
        )
        print(
            f"[{params['branding']}Model] \033[91mInformation\033[0m 'export CUDA_HOME=putyoutpathhere' (Linux) within your Python Environment"
        )
        model = await gpu_model_manager.unload_model(model)
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = True
        params["deepspeed_activate"] = True
        await setup()
    else:
        # DeepSpeed disabled
        print(f"[{params['branding']}Model] \033[93mDeepSpeed De-Activating\033[0m")
        print(
            f"[{params['branding']}Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m"
        )
        params["deepspeed_activate"] = False
        model = await gpu_model_manager.unload_model(model)
        await setup()

    return value  # Return new checkbox value

# DEEPSPEED WEBSERVER- API Enable/Disable DeepSpeed
@app.post("/api/deepspeed")
async def deepspeed(request: Request, new_deepspeed_value: bool):
    try:
        if new_deepspeed_value is None:
            raise ValueError("Missing 'deepspeed' parameter")
        if params["deepspeed_activate"] == new_deepspeed_value:
            return Response(
                content=json.dumps(
                    {
                        "status": "success",
                        "message": f"DeepSpeed is already {'enabled' if new_deepspeed_value else 'disabled'}.",
                    }
                )
            )
        params["deepspeed_activate"] = new_deepspeed_value
        await handle_deepspeed_change(params["deepspeed_activate"])
        return Response(content=json.dumps({"status": "deepspeed-success"}))
    except Exception as e:
        return Response(content=json.dumps({"status": "error", "message": str(e)}))

########################
#### TTS GENERATION ####
########################

debug_generate_audio = False
tts_stop_generation = False
tts_generation_lock = False
tts_narrator_generatingtts = False

async def generate_audio(text, voice, language, temperature, repetition_penalty, streaming=False):
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    text_batches = split_text_into_batches(text, num_gpus)

    tasks = []
    for i, batch in enumerate(text_batches):
        gpu_id = i % num_gpus  # Cycle through available GPUs
        tasks.append(generate_audio_on_gpu(batch, voice, language, temperature, repetition_penalty, gpu_id, streaming))

    # Run all tasks concurrently across GPUs
    if streaming:
        async for chunk in combine_outputs_streaming(tasks):
            yield chunk
    else:
        combined_output = await combine_outputs_non_streaming(tasks)
        yield combined_output

async def generate_audio_on_gpu(text, voice, language, temperature, repetition_penalty, rank, streaming):
    # Add debug log to track the status of model_instances
    logging.debug(f"Attempting to access model on GPU {rank} from model_instances.")
    
    # Await the get_model call to get the model
    model_on_device = await gpu_model_manager.get_model(rank)
    
    # Log the device information after ensuring the model is loaded
    logging.debug(f"Model on GPU {rank} is on device {next(model_on_device.parameters()).device}.")
    
    if model_on_device is None:
        raise ValueError(f"No model found for GPU {rank}")
    
    # Now that the model is available, proceed to generate the audio
    return await generate_audio_internal(model_on_device, text, voice, language, temperature, repetition_penalty, streaming, torch.device(f'cuda:{rank}'))


async def generate_audio_internal(model_on_device, text, voice, language, temperature, repetition_penalty, streaming, device):
    # Ensure all tensors are on the correct device
    gpt_cond_latent, speaker_embedding = model_on_device.get_conditioning_latents(
        audio_path=[f"{this_dir}/voices/{voice}"],
        gpt_cond_len=model_on_device.config.gpt_cond_len,
        max_ref_length=model_on_device.config.max_ref_len,
        sound_norm_refs=model_on_device.config.sound_norm_refs,
    )

    # Move tensors and model to the specific device
    model_on_device.to(device)
    gpt_cond_latent = gpt_cond_latent.to(device)
    speaker_embedding = speaker_embedding.to(device)

    logging.debug(f"Model is on device: {next(model_on_device.parameters()).device}")
    logging.debug(f"gpt_cond_latent is on device: {gpt_cond_latent.device}")
    logging.debug(f"speaker_embedding is on device: {speaker_embedding.device}")

    common_args = {
        "text": text,
        "language": language,
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding,
        "temperature": float(temperature),
        "length_penalty": float(model_on_device.config.length_penalty),
        "repetition_penalty": float(repetition_penalty),
        "top_k": int(model_on_device.config.top_k),
        "top_p": float(model_on_device.config.top_p),
        "enable_text_splitting": True,
        "stream_chunk_size": 20 if streaming else None
    }

    common_args = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in common_args.items()}

    if streaming:
        return await stream_audio(model_on_device.inference_stream, common_args, device)
    else:
        return await generate_full_audio(model_on_device.inference, common_args, device)

async def stream_audio(inference_func, args, device):
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(1)
        vfout.setsampwidth(2)
        vfout.setframerate(24000)
        vfout.writeframes(b"")
    wav_buf.seek(0)
    yield wav_buf.read()

    for i, chunk in enumerate(inference_func(**args)):
        if tts_stop_generation:
            print(f"[TTSGen] Stopping audio generation.")
            tts_stop_generation = False
            tts_generation_lock = False
            break

        chunk = process_chunk(chunk, device)
        wav_buf.write(chunk.tobytes())

        if debug_generate_audio:
            print(f"[Debug] Stream audio generation: Yielded audio chunk {i}.")
    
    wav_buf.seek(0)
    yield wav_buf.read()

async def generate_full_audio(inference_func, args, device):
    output = inference_func(**args)
    wav_buf = io.BytesIO()
    
    # Ensure the tensor is on CPU before saving
    output_tensor = torch.tensor(output["wav"]).unsqueeze(0).to('cpu')
    
    torchaudio.save(wav_buf, output_tensor, 24000, format="wav")
    wav_buf.seek(0)
    return wav_buf.read()

def process_chunk(chunk, device):
    if isinstance(chunk, list):
        chunk = torch.cat(chunk, dim=0)
    
    # Move the tensor to CPU before converting to NumPy
    chunk = chunk.to('cpu').clone().detach().numpy()
    
    chunk = chunk[None, : int(chunk.shape[0])]
    chunk = np.clip(chunk, -1, 1)
    return (chunk * 32767).astype(np.int16)

async def combine_outputs_non_streaming(tasks):
    # Run tasks in parallel using asyncio.gather
    outputs = await asyncio.gather(*tasks)
    combined_wav = b"".join(outputs)
    return combined_wav

async def combine_outputs_streaming(tasks):
    # Process tasks concurrently and yield chunks as they are ready
    for task in asyncio.as_completed(tasks):
        async for chunk in await task:
            yield chunk

def split_text_into_batches(text, num_batches):
    sentences = nltk.sent_tokenize(text)
    num_sentences = len(sentences)
    batch_size = max(1, num_sentences // num_batches)

    batches = []
    for i in range(0, num_sentences, batch_size):
        batch = " ".join(sentences[i:i+batch_size])
        batches.append(batch)
    
    return batches

@app.post("/api/generate")
async def generate(request: Request):
    try:
        # Parse the incoming JSON request
        data = await request.json()
        text = data["text"]
        voice = data["voice"]
        language = data["language"]
        temperature = data["temperature"]
        repetition_penalty = data["repetition_penalty"]
        streaming = data.get("streaming", False)  # Default to non-streaming if not specified

        logging.info("Starting TTS generation")

        # Start the TTS generation process
        audio_generator = generate_audio(text, voice, language, temperature, repetition_penalty, streaming)

        # Collect the generated audio data
        audio_data = b""
        async for chunk in audio_generator:
            audio_data += chunk
        
        logging.info("TTS generation completed")

        # Return the generated audio as a response
        return Response(audio_data, media_type="audio/wav")

    except Exception as e:
        # Log the error and return an error response
        logging.error(f"Error in API call: {str(e)}", exc_info=True)
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)



###################################################
#### POPULATE FILES LIST FROM VOICES DIRECTORY ####
###################################################
# List files in the "voices" directory
def list_files(directory):
    files = [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(".wav")
    ]
    return files

#############################
#### JSON CONFIG UPDATER ####
#############################

# Create an instance of Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory=this_dir / "system")

# Create a dependency to get the current JSON data
def get_json_data():
    with open(this_dir / "confignew.json", "r") as json_file:
        data = json.load(json_file)
    return data


# Define an endpoint function
@app.get("/settings")
async def get_settings(request: Request):
    wav_files = list_files(this_dir / "voices")
    # Render the template with the current JSON data and list of WAV files
    return templates.TemplateResponse(
        "/at_admin/at_settings.html",
        {
            "request": request,
            "data": get_json_data(),
            "modeldownload_model_path": modeldownload_model_path,
            "wav_files": wav_files,
        },
    )

@app.post("/update-settings")
async def update_settings(
    request: Request,
    activate: bool = Form(...),
    autoplay: bool = Form(...),
    deepspeed_activate: bool = Form(...),
    delete_output_wavs: str = Form(...),
    ip_address: str = Form(...),
    language: str = Form(...),
    local_temperature: str = Form(...),
    local_repetition_penalty: str = Form(...),
    low_vram: bool = Form(...),
    tts_model_loaded: bool = Form(...),
    tts_model_name: str = Form(...),
    narrator_enabled: bool = Form(...),
    narrator_voice: str = Form(...),
    output_folder_wav: str = Form(...),
    port_number: str = Form(...),
    remove_trailing_dots: bool = Form(...),
    show_text: bool = Form(...),
    tts_method: str = Form(...),
    voice: str = Form(...),
    data: dict = Depends(get_json_data),
):
    # Update the settings based on the form values
    data["activate"] = activate
    data["autoplay"] = autoplay
    data["deepspeed_activate"] = deepspeed_activate
    data["delete_output_wavs"] = delete_output_wavs
    data["ip_address"] = ip_address
    data["language"] = language
    data["local_temperature"] = local_temperature
    data["local_repetition_penalty"] = local_repetition_penalty
    data["low_vram"] = low_vram
    data["tts_model_loaded"] = tts_model_loaded
    data["tts_model_name"] = tts_model_name
    data["narrator_enabled"] = narrator_enabled
    data["narrator_voice"] = narrator_voice
    data["output_folder_wav"] = output_folder_wav
    data["port_number"] = port_number
    data["remove_trailing_dots"] = remove_trailing_dots
    data["show_text"] = show_text
    data["tts_method_api_local"] = tts_method == "api_local"
    data["tts_method_api_tts"] = tts_method == "api_tts"
    data["tts_method_xtts_local"] = tts_method == "xtts_local"
    data["voice"] = voice

    # Save the updated settings back to the JSON file
    with open(this_dir / "confignew.json", "w") as json_file:
        json.dump(data, json_file)

    # Redirect to the settings page to display the updated settings
    return RedirectResponse(url="/settings", status_code=303)


#########################
#### VOICES LIST API ####
#########################
# Define the new endpoint
@app.get("/api/voices")
async def get_voices():
    wav_files = list_files(this_dir / "voices")
    return {"voices": wav_files}

########################
#### GENERATION API ####
########################
import html
import re
import numpy as np
import soundfile as sf
import sys

##############################
#### Streaming Generation ####
##############################

@app.get("/api/tts-generate-streaming", response_class=StreamingResponse)
async def tts_generate_streaming(text: str, voice: str, language: str, output_file: str):
    try:
        output_file_path = this_dir / "outputs" / output_file
        stream = await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=True)
        return StreamingResponse(stream, media_type="audio/wav")
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

@app.post("/api/tts-generate-streaming", response_class=JSONResponse)
async def tts_generate_streaming(request: Request, text: str = Form(...), voice: str = Form(...), language: str = Form(...), output_file: str = Form(...)):
    try:
        output_file_path = this_dir / "outputs" / output_file
        await generate_audio(text, voice, language, temperature, repetition_penalty, output_file_path, streaming=False)
        return JSONResponse(content={"output_file_path": str(output_file)}, status_code=200)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": "An error occurred"}, status_code=500)

@app.put("/api/stop-generation")
async def stop_generation_endpoint():
    global tts_stop_generation, tts_generation_lock
    if tts_generation_lock and not tts_stop_generation:
        tts_stop_generation = True
    return {"message": "Generation stopped"}

##############################
#### Standard Generation ####
##############################

# Check for PortAudio library on Linux
try:
    import sounddevice as sd
    sounddevice_installed=True
except OSError:
    print(f"[{params['branding']}Startup] \033[91mInfo\033[0m PortAudio library not found. If you wish to play TTS in standalone mode through the API suite")
    print(f"[{params['branding']}Startup] \033[91mInfo\033[0m please install PortAudio. This will not affect any other features or use of Alltalk.")
    print(f"[{params['branding']}Startup] \033[91mInfo\033[0m If you don't know what the API suite is, then this message is nothing to worry about.")
    sounddevice_installed=False
    if sys.platform.startswith('linux'):
        print(f"[{params['branding']}Startup] \033[91mInfo\033[0m On Linux, you can use the following command to install PortAudio:")
        print(f"[{params['branding']}Startup] \033[91mInfo\033[0m sudo apt-get install portaudio19-dev")

from typing import Optional, Union, Dict, List
from pydantic import BaseModel, ValidationError, Field

def play_audio(file_path, volume):
    data, fs = sf.read(file_path)
    sd.play(volume * data, fs)
    sd.wait()

class Request(BaseModel):
    # Define the structure of the 'Request' class if needed
    pass

class JSONInput(BaseModel):
    text_input: str = Field(..., max_length=2000, description="text_input needs to be 2000 characters or less.")
    text_filtering: str = Field(..., pattern="^(none|standard|html)$", description="text_filtering needs to be 'none', 'standard' or 'html'.")
    character_voice_gen: str = Field(..., pattern="^.*\.wav$", description="character_voice_gen needs to be the name of a wav file e.g. mysample.wav.")
    narrator_enabled: bool = Field(..., description="narrator_enabled needs to be true or false.")
    narrator_voice_gen: str = Field(..., pattern="^.*\.wav$", description="narrator_voice_gen needs to be the name of a wav file e.g. mysample.wav.")
    text_not_inside: str = Field(..., pattern="^(character|narrator)$", description="text_not_inside needs to be 'character' or 'narrator'.")
    language: str = Field(..., pattern="^(ar|zh-cn|cs|nl|en|fr|de|hu|hi|it|ja|ko|pl|pt|ru|es|tr)$", description="language needs to be one of the following ar|zh-cn|cs|nl|en|fr|de|hu|hi|it|ja|ko|pl|pt|ru|es|tr.")
    output_file_name: str = Field(..., pattern="^[a-zA-Z0-9_]+$", description="output_file_name needs to be the name without any special characters or file extension e.g. 'filename'")
    output_file_timestamp: bool = Field(..., description="output_file_timestamp needs to be true or false.")
    autoplay: bool = Field(..., description="autoplay needs to be a true or false value.")
    autoplay_volume: float = Field(..., ge=0.1, le=1.0, description="autoplay_volume needs to be from 0.1 to 1.0")

    @classmethod
    def validate_autoplay_volume(cls, value):
        if not (0.1 <= value <= 1.0):
            raise ValueError("Autoplay volume must be between 0.1 and 1.0")
        return value


class TTSGenerator:
    @staticmethod
    def validate_json_input(json_data: Union[Dict, str]) -> Union[None, str]:
        try:
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            JSONInput(**json_data)
            return None  # JSON is valid
        except ValidationError as e:
            return str(e)

def process_text(text):
    # Normalize HTML encoded quotes
    text = html.unescape(text)
    # Replace ellipsis with a single dot
    text = re.sub(r'\.{3,}', '.', text)
    # Pattern to identify combined narrator and character speech
    combined_pattern = r'(\*[^*"]+\*|"[^"*]+")'
    # List to hold parts of speech along with their type
    ordered_parts = []
    # Track the start of the next segment
    start = 0
    # Find all matches
    for match in re.finditer(combined_pattern, text):
        # Add the text before the match, if any, as ambiguous
        if start < match.start():
            ambiguous_text = text[start:match.start()].strip()
            if ambiguous_text:
                ordered_parts.append(('ambiguous', ambiguous_text))
        # Add the matched part as either narrator or character
        matched_text = match.group(0)
        if matched_text.startswith('*') and matched_text.endswith('*'):
            ordered_parts.append(('narrator', matched_text.strip('*').strip()))
        elif matched_text.startswith('"') and matched_text.endswith('"'):
            ordered_parts.append(('character', matched_text.strip('"').strip()))
        else:
            # In case of mixed or improperly formatted parts
            if '*' in matched_text:
                ordered_parts.append(('narrator', matched_text.strip('*').strip('"')))
            else:
                ordered_parts.append(('character', matched_text.strip('"').strip('*')))
        # Update the start of the next segment
        start = match.end()
    # Add any remaining text after the last match as ambiguous
    if start < len(text):
        ambiguous_text = text[start:].strip()
        if ambiguous_text:
            ordered_parts.append(('ambiguous', ambiguous_text))
    return ordered_parts

def standard_filtering(text_input):
    text_output = (text_input
                        .replace("***", "")
                        .replace("**", "")
                        .replace("*", "")
                        .replace("\n\n", "\n")
                        .replace("&#x27;", "'")
                        )
    return text_output

def combine(output_file_timestamp, output_file_name, audio_files):
    audio = np.array([])
    sample_rate = None
    try:
        for audio_file in audio_files:
            audio_data, current_sample_rate = sf.read(audio_file)
            if audio.size == 0:
                audio = audio_data
                sample_rate = current_sample_rate
            elif sample_rate == current_sample_rate:
                audio = np.concatenate((audio, audio_data))
            else:
                raise ValueError("Sample rates of input files are not consistent.")
    except Exception as e:
        # Handle exceptions (e.g., file not found, invalid audio format)
        return None, None
    if output_file_timestamp:
        timestamp = int(time.time())
        output_file_path = os.path.join(this_dir / "outputs" / f'{output_file_name}_{timestamp}_combined.wav')
        output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}_{timestamp}_combined.wav'
        output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}_{timestamp}_combined.wav'
    else:
        output_file_path = os.path.join(this_dir / "outputs" / f'{output_file_name}_combined.wav')
        output_file_url = f'http://{params["ip_address"]}:{params["port_number"]}/audio/{output_file_name}_combined.wav'
        output_cache_url = f'http://{params["ip_address"]}:{params["port_number"]}/audiocache/{output_file_name}_combined.wav'
    try:
        sf.write(output_file_path, audio, samplerate=sample_rate)
        # Clean up unnecessary files
        for audio_file in audio_files:
            os.remove(audio_file)
    except Exception as e:
        # Handle exceptions (e.g., failed to write output file)
        return None, None
    return output_file_path, output_file_url, output_cache_url

##########################
#### Current Settings ####
##########################
# Define the available models
models_available = [
    {"name": "Coqui", "model_name": "API TTS"},
    {"name": "Coqui", "model_name": "API Local"},
    {"name": "Coqui", "model_name": "XTTSv2 Local"}
]

@app.get('/api/currentsettings')
def get_current_settings():
    # Determine the current model loaded
    if params["tts_method_api_tts"]:
        current_model_loaded = "API TTS"
    elif params["tts_method_api_local"]:
        current_model_loaded = "API Local"
    elif params["tts_method_xtts_local"]:
        current_model_loaded = "XTTSv2 Local"
    else:
        current_model_loaded = None  # or a default value if no method is active

    settings = {
        "models_available": models_available,
        "current_model_loaded": current_model_loaded,
        "deepspeed_available": deepspeed_available,
        "deepspeed_status": params["deepspeed_activate"],
        "low_vram_status": params["low_vram"],
        "finetuned_model": finetuned_model
    }
    return settings  # Automatically converted to JSON by Fas

#############################
#### Word Add-in Sharing ####
#############################
# Mount the static files from the 'word_addin' directory
app.mount("/api/word_addin", StaticFiles(directory=os.path.join(this_dir / 'system' / 'word_addin')), name="word_addin")

#############################################
#### TTS Generator Comparision Endpoints ####
#############################################
import subprocess
import aiofiles

class TTSItem(BaseModel):
    id: int
    fileUrl: str
    text: str
    characterVoice: str
    language: str

class TTSData(BaseModel):
    ttsList: List[TTSItem]

@app.post("/api/save-tts-data")
async def save_tts_data(tts_data: List[TTSItem]):
    # Convert the list of Pydantic models to a list of dictionaries
    tts_data_list = [item.dict() for item in tts_data]
    # Serialize the list of dictionaries to a JSON string
    tts_data_json = json.dumps(tts_data_list, indent=4)
    async with aiofiles.open(this_dir / "outputs" / "ttsList.json", 'w') as f:
        await f.write(tts_data_json)
    return {"message": "Data saved successfully"}

import sys

@app.get("/api/trigger-analysis")
async def trigger_analysis(threshold: int = Query(default=98)):
    venv_path = sys.prefix
    env = os.environ.copy()
    env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
    ttslist_path = this_dir / "outputs" / "ttsList.json"
    wavfile_path = this_dir / "outputs"
    subprocess.run(["python", "tts_diff.py", f"--threshold={threshold}", f"--ttslistpath={ttslist_path}", f"--wavfilespath={wavfile_path}"], cwd=this_dir / "system" / "tts_diff", env=env)
    # Read the analysis summary
    try:
        with open(this_dir / "outputs" / "analysis_summary.json", "r") as summary_file:
            summary_data = json.load(summary_file)
    except FileNotFoundError:
        summary_data = {"error": "Analysis summary file not found."}
    return {"message": "Analysis Completed", "summary": summary_data}


###############################
#### Internal script ready ####
###############################
@app.get("/ready")
async def ready():
    return Response("Ready endpoint")

############################
#### External API ready ####
############################
@app.get("/api/ready")
async def ready():
    return Response("Ready")

#################################
#### Start Uvicorn Webserver ####
#################################

host_parameter = params["ip_address"]
port_parameter = int(params["port_number"])

async def start_server():
    config = uvicorn.Config(app, host=host_parameter, port=port_parameter, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    initialize_ddp()
    
