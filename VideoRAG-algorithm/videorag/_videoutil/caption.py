import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames
    
import asyncio
from functools import partial

def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue, global_config=None):
    try:
        # The caption model is now configured via LLMConfig in the main script
        llm_config = global_config.get("llm", {})
        
        # Fallback to default if not provided, but the goal is to configure this from outside
        caption_model_func = llm_config.get("caption_model_func_raw", None)
        caption_model_name = llm_config.get("caption_model_name", "minicpm-v")

        if caption_model_func is None:
            raise ValueError("Caption model function not provided in LLMConfig.")

        caption_func = partial(caption_model_func, caption_model_name, global_config=global_config)

        async def run_captioning():
            with VideoFileClip(video_path) as video:
                for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                    frame_times = segment_times_info[index]["frame_times"]
                    video_frames = encode_video(video, frame_times)
                    segment_transcript = transcripts[index]

                    # Prepare content for minicpm_v_caption_complete
                    content_list = []
                    for frame in video_frames:
                        content_list.append({"type": "image_url", "image_url": {"url": frame}}) # This assumes the function can handle PIL images; might need adjustment
                    content_list.append({"type": "text", "text": f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in Chinese."})

                    # Call the async caption function
                    caption = await caption_func(content_list=content_list)
                    caption_result[index] = caption.replace("\n", "").replace("<|endoftext|>", "")

        asyncio.run(run_captioning())

    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        # raise RuntimeError # Commented out to avoid stopping the whole process on a single error

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments
        
def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    # model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # model.eval()
    
    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        # query = f"The transcript of the current video:\n{segment_transcript}.\nGiven a question: {query}, you have to extract relevant information from the video and transcript for answering the question."
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"
        msgs = [{'role': 'user', 'content': video_frames + [query]}]
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2
        segment_caption = caption_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=caption_tokenizer,
            **params
        )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        torch.cuda.empty_cache()
    
    return caption_result