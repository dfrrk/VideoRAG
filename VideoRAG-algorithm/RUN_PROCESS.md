# VideoRAG-algorithm 运行流程分析

本文档旨在详细解析 `videorag_longervideos.py` 脚本在 Linux (或 WSL) 环境下的完整执行流程。

## 脚本概述

该脚本的核心目标分为两个阶段：

1.  **学习/索引 (Learn Phase)**: 首先，它会处理指定文件夹下的所有视频文件，通过一系列复杂的 AI 模型提取信息，并将这些信息构建成一个知识库（以多种文件形式存储在 `working_dir` 中）。
2.  **推理/问答 (Inference Phase)**: 然后，它会加载一个包含问题的数据集（`dataset.json`），并利用第一阶段构建的知识库，对这些问题进行回答，并将答案保存为单独的 Markdown 文件。

以下是脚本执行的详细步骤分解。

---

## 第一阶段：学习/索引 (Learn Phase)

此阶段由 `videorag.insert_video(video_path_list=video_paths)` 这一行代码触发。`VideoRAG` 类的 `insert_video` 方法会遍历所有输入的视频文件，并对每一个视频执行以下一系列操作。这些操作定义在 `videorag/_videoutil/` 目录下的各个模块中。

1.  **视频分割 (`split.py`)**:
    *   **目标**: 将长视频分割成一系列固定长度的、更易于处理的小视频片段。
    *   **过程**:
        *   脚本调用 `split_video` 函数。
        *   该函数使用 `moviepy` 库来读取视频，并根据 `video_segment_length` 参数（默认为 30 秒）将视频切割成多个片段。
        *   每个片段都会被保存为一个独立的视频文件，并同时提取出其对应的音频，保存为音频文件（默认为 mp3）。
        *   此外，它还会为每个片段采样一定数量的帧（由 `rough_num_frames_per_segment` 定义），这些帧的路径或时间戳信息会被记录下来，用于后续的视觉分析。

2.  **语音识别 (ASR - `asr.py`)**:
    *   **目标**: 将每个视频片段的音频转换成文字稿。
    *   **过程**:
        *   脚本调用 `speech_to_text` 函数。
        *   在 `VideoRAG-algorithm` 项目中，这通常会调用一个本地模型，例如 `faster_whisper`。它会加载一个预训练的 Whisper 模型。
        *   对于每个音频文件，模型会进行推理，生成对应的文字记录（transcript）。
        *   这些文字记录与它们所属的视频片段一一对应。

3.  **视觉分析/视频字幕 (`caption.py`)**:
    *   **目标**: 分析每个视频片段的视觉内容，生成一段描述性的文字（即字幕或 caption）。
    *   **过程**:
        *   脚本调用 `segment_caption` 函数。
        *   该函数会利用一个多模态的视觉语言模型 (VLM)。
        *   它将之前采样好的视频帧和对应的 ASR 文字稿一起作为输入，提供给 VLM。
        *   VLM 会“观看”这些帧并“阅读”文字稿，然后生成一段总结性的、描述该视频片段内容的英文文字。

4.  **信息整合**:
    *   **目标**: 将来自不同模型的信息（音频文字、视频字幕、时间戳）合并在一起。
    *   **过程**:
        *   `merge_segment_information` 函数被调用。
        *   它将每个视频片段的时间戳、ASR 文字稿和视觉字幕组合成一个结构化的数据单元。这创建了对每个视频片段的多模态描述。

5.  **特征编码 (`feature.py`)**:
    *   **目标**: 将视频片段的视觉内容转换成一个高维的数学向量（embedding），以便进行快速的相似性搜索。
    *   **过程**:
        *   脚本调用 `video_segment_feature_vdb.upsert` 方法，内部会使用 `ImageBind` 或类似的模型。
        *   `ImageBind` 是一个能将多种模态（视频、图像、文本、音频等）编码到同一个向量空间的多模态编码器。
        *   对于每个视频片段，模型会处理其视觉内容，并输出一个固定维度的特征向量。
        *   这些向量被存储在一个向量数据库中（例如 `HNSWlib` 或 `NanoVectorDB`），并与它们所代表的视频片段的 ID 关联起来。

6.  **知识图谱构建 (`_op.py`)**:
    *   **目标**: 从整合后的文本信息（ASR + Caption）中提取关键实体（如人名、地名、概念），并构建它们之间的关系，形成一个知识图谱。
    *   **过程**:
        *   在 `insert_video` 的最后，脚本会调用 `self.ainsert` 方法，该方法内部会触发 `extract_entities` 函数。
        *   这个函数使用一个大语言模型（LLM，例如 GPT-4o-mini）来读取每个片段的文本。
        *   LLM 被指示去识别文本中的命名实体，并将它们的关系以图（Graph）的形式进行组织。
        *   这个图被存储下来（例如，使用 `NetworkX` 库），它捕捉了视频内容的核心语义和知识结构。

完成以上所有步骤后，`working_dir` 目录中会包含视频的文本信息、特征向量、知识图谱等一系列文件。这些文件共同构成了对原始视频集合的完整知识索引，为第二阶段的问答做好了准备。


---

## 第二阶段：推理/问答 (Inference Phase)

在完成视频索引后，脚本会立即进入问答阶段。

1.  **加载问题集**:
    *   脚本首先会读取 `longervideos/dataset.json` 文件。这个 JSON 文件包含了一系列的问题，每个问题都有一个唯一的 ID 和具体的问题文本。

2.  **重新初始化 `VideoRAG` 实例**:
    *   脚本会再次创建一个 `VideoRAG` 类的实例。重要的是，它传入了与第一阶段**完全相同**的 `working_dir`。这使得新实例能够自动加载之前已经构建好的所有索引文件（知识图谱、向量数据库等）。

3.  **加载字幕模型**:
    *   脚本显式调用 `videorag.load_caption_model()`。这是一个在 `Vimo-desktop` 版本中不存在的方法，它可能是用来预加载在问答阶段需要用到的多模态大模型，以提高后续处理的效率。

4.  **循环回答问题**:
    *   脚本会遍历从 `dataset.json` 中加载的所有问题。对于每一个问题，它会执行以下操作：
        *   **调用查询方法**: `videorag.query(query=query, param=param)` 被调用。这是执行 RAG (Retrieval-Augmented Generation，检索增强生成) 的核心。
        *   **检索 (Retrieval)**:
            *   `query` 方法首先会使用文本编码模型（如 `text-embedding-3-small`）将用户的问题转换成一个向量。
            *   然后，它拿着这个查询向量，到第一阶段构建的向量数据库中进行相似性搜索，找出与问题最相关的视频片段的特征向量。
            *   同时，它可能还会利用知识图谱来查找与问题中的实体相关的其他信息。
        *   **增强 (Augmentation)**:
            *   脚本将检索到的最相关的视频片段的**原始信息**（包括 ASR 文字稿、视频字幕、时间戳等）提取出来。
            *   这些信息被组合成一个丰富的上下文（context）。
        *   **生成 (Generation)**:
            *   最后，脚本将原始的用户问题和刚刚构建的上下文信息，一起发送给一个强大的大语言模型（LLM，如 GPT-4o-mini）。
            *   它会要求 LLM 在提供的上下文信息的基础上，来回答用户的问题。这种“先检索、后生成”的方式，使得 LLM 能够回答关于视频内容的非常具体和深入的问题，而不是仅仅依赖其内部知识。
        *   **保存答案**: LLM 生成的最终答案会被保存到一个位于 `longervideos/videorag-answers/` 目录下的 Markdown 文件中，文件名与问题的 ID 对应。

这个流程完整地展示了 RAG 框架如何被应用于视频理解任务中：先将视频内容“知识化”（索引阶段），然后在回答问题时，精确地检索出相关的知识片段，并让大语言模型基于这些具体的知识来生成答案。


---

## 第三部分：中文模型替换指南

本指南旨在为您提供在 `VideoRAG-algorithm` 项目中替换和集成更适合中文视频处理的本地化模型的具体步骤和建议。

### 总体思路

代码中与模型相关的核心配置位于 `videorag_longervideos.py` 脚本顶部的 `longervideos_llm_config` 对象中。然而，这个配置对象所引用的具体模型调用函数则定义在 `videorag/_llm.py` 文件里。

因此，替换模型通常需要两步：

1.  **修改 `videorag/_llm.py`**: 在此文件中添加新的函数，用于调用您在本地部署的新模型（例如，通过 API 请求本地的推理服务器）。
2.  **修改 `videorag_longervideos.py`**: 更新 `longervideos_llm_config` 对象，使其引用您在 `_llm.py` 中新创建的函数，并传入新模型的名称。

以下是针对语音识别（ASR）和视频字幕（Caption）模型的具体替换建议。


### 1. 语音识别 (ASR) 模型替换

**推荐模型**: **FunASR Paraformer-large**

*   **优点**: 这是由阿里巴巴达摩院开发的业界领先的中文语音识别模型。它在准确率上表现非常出色，尤其擅长处理带有口音、语速变化和背景噪音的真实场景语音。它完全开源，并且有成熟的本地部署方案。
*   **部署**: 您需要根据 [FunASR 的官方文档](https://github.com/alibaba-damo-academy/FunASR) 在您的本地（或 WSL）环境中部署其推理服务。通常，这会涉及到运行一个 Docker 容器或一个 Python 服务器，它会在本地暴露一个 API 端点（例如 `http://localhost:8000/asr`）。

**修改步骤**:

**a) 修改 `videorag/_videoutil/asr.py`**

当前，`asr.py` 中的 `speech_to_text_online` 函数是为调用阿里云 DashScope 的在线 API 设计的。我们需要创建一个新的函数，或者修改现有的函数，来调用您本地的 FunASR 服务。

**示例 - 添加一个新的本地 ASR 函数**:

在 `asr.py` 文件中，您可以添加如下函数：

```python
# a_sr.py

import requests
import json

# ... (保留文件中的其他 import)

# 新增函数，用于调用本地 FunASR 服务
def call_funasr_local(audio_file_path: str) -> str:
    """
    Calls the local FunASR server to transcribe an audio file.
    """
    # FunASR 通常需要您将文件以二进制形式上传
    try:
        with open(audio_file_path, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_file_path), f, "audio/mpeg")}
            # 这里的 URL "http://localhost:8000/asr" 需要根据您自己的 FunASR 部署地址进行修改
            response = requests.post("http://localhost:8000/asr", files=files)
            response.raise_for_status()  # 如果请求失败则抛出异常

            # 解析 FunASR 的返回结果，这同样需要根据 FunASR 的 API 文档来确定
            # 假设它返回一个 JSON，其中包含一个 "text" 字段
            result_json = response.json()
            return result_json.get("text", "")

    except requests.exceptions.RequestException as e:
        logger.error(f"FunASR request failed for {audio_file_path}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Failed to process audio file {audio_file_path} with FunASR: {e}")
        return ""

# 您可以创建一个新的主调用函数，或者修改现有的 `speech_to_text_online`
async def speech_to_text_local_funasr(video_name, working_dir, segment_index2name, audio_output_format, global_config, max_concurrent=5):
    cache_path = os.path.join(working_dir, '_cache', video_name)
    transcripts = {}

    logger.info(f"🎤 Starting LOCAL ASR for {len(segment_index2name)} audio segments...")

    # 这里可以使用多线程或异步来并行处理
    for index, segment_name in tqdm(segment_index2name.items(), desc="Transcribing Audio"):
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")
        transcripts[index] = call_funasr_local(audio_file)

    logger.info("🎉 Local ASR processing completed!")
    return transcripts

# 最后，修改 `speech_to_text` 这个主入口函数
def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format, global_config):
    """
    Synchronous wrapper for speech-to-text function.
    Chooses between online and local based on config.
    """
    # 我们可以通过模型名称来判断是使用在线服务还是本地服务
    asr_model_name = global_config.get("asr_model", "")

    if "funasr" in asr_model_name.lower():
        # 如果模型名称包含 "funasr"，则调用本地服务
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            speech_to_text_local_funasr(video_name, working_dir, segment_index2name, audio_output_format, global_config)
        )
    else:
        # 否则，保持原有的在线服务逻辑
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            speech_to_text_async(video_name, working_dir, segment_index2name, audio_output_format, global_config)
        )

```

**b) 修改 `videorag_longervideos.py`**

现在，您只需要在主脚本中更新 `asr_model` 的名称即可。

```python
# videorag_longervideos.py

# ... (其他代码)

videorag = VideoRAG(
    llm=longervideos_llm_config,
    working_dir=f"./longervideos/videorag-workdir/{sub_category}",
    # 在这里或者在 LLMConfig 中，确保 asr_model 被设置
    asr_model="funasr-paraformer-large"  # 使用一个包含 "funasr" 的新名称
)
videorag.insert_video(video_path_list=video_paths)

# ... (其他代码)
```

通过以上修改，当您运行 `videorag_longervideos.py` 时，`speech_to_text` 函数会检测到模型名称中含有 "funasr"，并自动切换到调用您本地部署的 FunASR 服务。


### 2. 视频字幕 (Caption) 模型替换

**集成模型**: **MiniCPM-V**

*   **优点**: 这是一个强大的开源多模态大模型，能够处理视觉和语言任务。它非常适合为视频片段生成描述性字幕。
*   **部署**: 您需要根据 MiniCPM-V 的官方文档，在本地部署其推理服务。通常，这会通过 `vLLM` 或类似的框架来完成，最终会在本地暴露一个与 OpenAI API 兼容的 API 端点（例如 `http://localhost:8001/v1`）。

**修改步骤**:

**a) 修改 `videorag/_llm.py`**

我们需要在此文件中添加一个新函数，用于与本地部署的 MiniCPM-V 模型进行交互。我们还需要确保 `LLMConfig` 数据类可以保存字幕模型的信息。

**示例 - 添加一个新的本地 Caption 函数**:

在 `_llm.py` 文件中，添加以下函数。此函数将连接到本地模型服务器，发送视频帧和文本提示，并返回生成的字幕。

```python
# _llm.py

import base64
from io import BytesIO
from PIL import Image
from openai import AsyncOpenAI
from logging import getLogger

logger = getLogger(__name__)

# ... (保留文件中的其他 import 和函数)

async def minicpm_v_caption_complete(
    model_name: str, content_list: list, **kwargs
) -> str:
    """
    调用本地的、与 OpenAI API 兼容的 MiniCPM-V 模型端点。
    """
    global_config = kwargs.get("global_config", {})

    local_api_base = global_config.get("local_vlm_base_url", "http://localhost:8001/v1")

    local_client = AsyncOpenAI(
        api_key="your-dummy-api-key",
        base_url=local_api_base,
    )

    processed_content = []
    for item in content_list:
        if item["type"] == "image_url":
            pil_image = item["image_url"]["url"]
            if isinstance(pil_image, Image.Image):
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                processed_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                })
            else:
                 processed_content.append(item)
        else:
            processed_content.append(item)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that describes video content in Chinese."},
        {"role": "user", "content": processed_content}
    ]

    try:
        response = await local_client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Local MiniCPM-V request failed: {e}")
        return ""

```

接下来，更新 `LLMConfig` 数据类以包含字幕模型配置：

```python
# _llm.py

@dataclass
class LLMConfig:
    # ... (保留所有现有字段)

    cheap_model_max_token_size: int
    cheap_model_max_async: int

    caption_model_func_raw: callable = None
    caption_model_name: str = None

    # Assigned in post init
    embedding_func: EmbeddingFunc  = None
    best_model_func: callable = None
    cheap_model_func: callable = None
```

**b) 修改 `videorag/_videoutil/caption.py`**

现在，我们需要更新 `segment_caption` 函数，使其不再硬编码模型，而是使用我们通过 `LLMConfig` 传入的函数。

```python
# caption.py

import asyncio
from functools import partial
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

# ... (保留 encode_video 函数)

def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue, global_config=None):
    try:
        llm_config = global_config.get("llm", {})

        caption_model_func = llm_config.get("caption_model_func_raw")
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

                    content_list = []
                    for frame in video_frames:
                        content_list.append({"type": "image_url", "image_url": {"url": frame}})
                    content_list.append({"type": "text", "text": f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in Chinese."})

                    caption = await caption_func(content_list=content_list)
                    caption_result[index] = caption.replace("\n", "").replace("<|endoftext|>", "")

        asyncio.run(run_captioning())

    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")

```

**c) 修改 `videorag/videorag.py`**

我们需要更新 `insert_video` 方法，以将全局配置传递给 `segment_caption` 进程。

```python
# videorag.py

# ... (在 insert_video 方法中)
            process_segment_caption = multiprocessing.Process(
                target=segment_caption,
                args=(
                    video_name,
                    video_path,
                    segment_index2name,
                    transcripts,
                    segment_times_info,
                    captions,
                    error_queue,
                    asdict(self), # 传入全局配置
                )
            )
# ...
```

**d) 修改 `videorag_longervideos.py`**

最后，在主脚本中，更新 `longervideos_llm_config` 对象，以使用我们新创建的函数和模型。

```python
# videorag_longervideos.py

from videorag._llm import * # 确保新函数被导入

# ...

longervideos_llm_config = LLMConfig(
    # ... (保留 embedding 和其他模型的配置)

    # ↓↓↓ 添加以下部分 ↓↓↓
    # Caption model configuration
    caption_model_func_raw=minicpm_v_caption_complete,
    caption_model_name="minicpm-v" # 或您本地服务器特定的模型标识符
)

if __name__ == '__main__':
    # ... (后续代码不变)
```

通过以上修改，`VideoRAG` 实例在进行视频字幕生成时，将调用 `minicpm_v_caption_complete` 函数，该函数会将请求发送到您本地部署的 MiniCPM-V 模型服务，从而实现了完全本地化的、高质量的中文视频内容分析。
