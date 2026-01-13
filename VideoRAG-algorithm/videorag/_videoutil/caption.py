import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

import math
from scipy.spatial import cKDTree

MAX_NUM_FRAMES=180
MAX_NUM_PACKING=3
TIME_SCALE = 0.1

def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]

def encode_video_for_minicpm(video_path, choose_fps=3, force_packing=None):
    from decord import VideoReader, cpu
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps

    if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
    else:
        packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    frame_idx = [i for i in range(0, len(vr))]
    frame_idx =  np.array(uniform_sample(frame_idx, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)

    frames_numpy = vr.get_batch(frame_idx).asnumpy()

    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, TIME_SCALE)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames_numpy) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames_numpy]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)

    return frames, frame_ts_id_group
    
import asyncio
from functools import partial

def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue, global_config=None):
    try:
        llm_config = global_config.get("llm", {})
        
        caption_model_func = llm_config.get("caption_model_func_raw", None)
        caption_model_name = llm_config.get("caption_model_name", "minicpm-v")
        video_caption_fps = global_config.get("video_caption_fps", 3)

        if caption_model_func is None:
            raise ValueError("Caption model function not provided in LLMConfig.")

        caption_func = partial(caption_model_func, caption_model_name, global_config=global_config)

        async def run_captioning():
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):

                # Each segment is processed individually
                segment_info = segment_times_info[index]
                start_time = segment_info["start"]
                end_time = segment_info["end"]

                with VideoFileClip(video_path).subclip(start_time, end_time) as video_segment:
                    segment_video_path = f"/tmp/{video_name}_segment_{index}.mp4"
                    video_segment.write_videofile(segment_video_path, codec="libx264", audio_codec="aac", logger=None)

                    video_frames, temporal_ids = encode_video_for_minicpm(segment_video_path, choose_fps=video_caption_fps)

                    os.remove(segment_video_path)

                segment_transcript = transcripts[index]

                content_list = []
                for frame in video_frames:
                    content_list.append({"type": "image_url", "image_url": {"url": frame}})
                content_list.append({"type": "text", "text": f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in Chinese."})

                caption = await caption_func(content_list=content_list, temporal_ids=temporal_ids)
                caption_result[index] = caption.replace("\n", "").replace("<|endoftext|>", "")

        asyncio.run(run_captioning())

    except Exception as e:
        error_queue.put(f"Error in segment_caption: {str(e)}")

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