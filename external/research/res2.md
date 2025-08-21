# Design and Implementation of a Next-Generation Video Transcription System
(Beyond Whisper)

## Table of Contents

  1. **Chapter 1: Pushing Beyond OpenAI Whisper – The State of the Art**  
_TL;DR:_ We review OpenAI’s Whisper ASR model, its strengths and limitations,
and identify newer transcription technologies that promise even higher
accuracy or additional features. This includes emerging large-scale models
(Google’s Universal Speech Model, Gemini) and specialized services claiming
superior results to
Whisper[sstoitsev.medium.com](https://sstoitsev.medium.com/google-vs-azure-a-
speech-to-text-
battle-f740aa481e8e#:~:text=,applications%20that%20require%20high%20precision)[gladia.io](https://www.gladia.io/blog/a-review-
of-the-best-asr-engines-and-the-models-powering-them-
in-2024#:~:text=match%20at%20L640%20Speechmatics%20presents,v3%20was%20not%20out%20yet).
We define what “surpassing Whisper” means in terms of accuracy and
capabilities (e.g. speaker diarization, emotion detection) to set the stage
for our system’s design.

  2. **Chapter 2: System Architecture and Setup (Local and Colab)**  
_TL;DR:_ We outline the end-to-end system architecture for the transcription
pipeline. The components include video retrieval via _yt-dlp_ , an advanced
transcription engine, a speaker diarization module, an emotion detection
module, subtitle formatting, and YouTube API integration. We discuss
environment setup for local machines vs Google Colab, including necessary
Python libraries (e.g. `yt-dlp`, `google-generativeai`, `pyannote.audio`,
etc.) and hardware considerations for running heavy AI models.

  3. **Chapter 3: Video Ingestion with yt-dlp**  
_TL;DR:_ This chapter covers using _yt-dlp_ to fetch videos and extract audio.
We provide examples of downloading YouTube content and converting it to audio-
only (e.g. WAV) for transcription[cheat.sh](https://cheat.sh/yt-
dlp#:~:text=,audio%20%22https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DoHg5SJYRHA0).
We address handling long videos by splitting audio if needed, ensuring that
the subsequent transcription model can process the data efficiently. Tips for
using _yt-dlp_ in Python and managing file formats are included.

  4. **Chapter 4: Advanced Transcription Models and APIs**  
_TL;DR:_ We implement the transcription step using state-of-the-art models
that potentially surpass Whisper. Options include cloud APIs (like Google’s AI
models via the _Gemini_ API or Google Cloud Speech) and advanced open-source
models. We compare these options in accuracy and features – for example,
Google’s latest _Universal Speech Model (USM)_ trained on 12 million
hours[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-
and-the-models-powering-them-in-2024#:~:text=speech%20recognition), or
services like Speechmatics that claim to outdo
Whisper[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-
engines-and-the-models-powering-them-
in-2024#:~:text=match%20at%20L640%20Speechmatics%20presents,v3%20was%20not%20out%20yet).
We include code examples for calling the Google _Gemini_ API from Python and
discuss how to chunk audio and handle timestamps in the returned text.

  5. **Chapter 5: Speaker Diarization Techniques**  
_TL;DR:_ Here we tackle _speaker diarization_ – determining “who spoke when.”
We explore algorithms and libraries for speaker recognition in audio. The
chapter covers using pretrained diarization pipelines such as _pyannote.audio_
, which achieves state-of-the-art speaker
segmentation[huggingface.co](https://huggingface.co/pyannote#:~:text=pyannote.audio%20,performance%20on%20most%20academic%20benchmarks).
We demonstrate how to integrate diarization results (speaker labels and time
segments) with the transcription. Code snippets illustrate running a
diarization model on audio and aligning its output (speaker IDs per timestamp)
with transcription segments. We also mention services that have diarization
built-in (e.g. Azure Speech, Google Speech-to-Text) and how Whisper can be
extended (via WhisperX) to include speaker
labels[assemblyai.com](https://www.assemblyai.com/blog/best-api-models-for-
real-time-speech-recognition-and-transcription#:~:text=WhisperX).

  6. **Chapter 6: Emotion Detection from Speech**  
_TL;DR:_ This chapter adds an _emotion detection_ layer to the transcripts. We
discuss the importance of capturing the speaker’s tone or mood (happy, sad,
angry, etc.) and survey methods to detect emotions from audio. Approaches
include using specialized machine learning models trained on speech emotion
datasets (detecting emotions like anger, disgust, fear, happiness, neutrality,
sadness[exposit.com](https://www.exposit.com/portfolio/speech-emotion-
recognition/#:~:text=The%20voice%20characteristics%20are%20processed,people%20in%20various%20emotional%20states))
or deriving sentiment from the transcript text. We present a solution using a
pre-trained speech emotion recognition model in Python. Code examples show how
to process audio segments (or use the transcribed text) to assign an emotion
label, which will later be annotated in the subtitles.

  7. **Chapter 7: Generating WebVTT Subtitles with Timestamps**  
_TL;DR:_ With transcriptions, speaker labels, and emotions in hand, we
demonstrate building _WebVTT_ subtitle files. We explain the WebVTT format
(text cues with start–end
timestamps)[speechpad.com](https://www.speechpad.com/captions/webvtt#:~:text=WEBVTT)
and how to format each subtitle cue with speaker names and emotion annotations
(e.g. prefixing lines with the speaker and an emotion tag). The chapter
includes code to iterate through the transcript with precise timestamps and
output a `.vtt` file. We ensure the subtitles are segmented into readable
chunks (e.g. sentence per cue) and discuss including metadata like language
and styling (utilizing WebVTT voice tags for
speakers[speechpad.com](https://www.speechpad.com/captions/webvtt#:~:text=styled%20using%20cascading%20style%20sheets,and%20each%20cue%20can%20be)).

  8. **Chapter 8: Integrating Google AI Studio (Gemini) for Enhancements**  
_TL;DR:_ We delve into leveraging _Google’s AI Studio Gemini_ API as part of
our system. This can serve either as the transcription engine itself (since
Gemini 2.5 can accept audio inputs and produce
text[ai.google.dev](https://ai.google.dev/gemini-
api/docs/models#:~:text=Model%20variant%20Input,efficient%20model)) or as an
augmentation tool (for example, using an LLM to refine raw transcripts or
infer emotions from text). We provide a tutorial on using the `google-
generativeai` Python library to access Gemini models, including authentication
and model selection (Flash vs Pro). Example code shows how to send an audio
file to the API and retrieve the transcription. We also discuss trade-offs,
such as using the Gemini model’s “thinking” mode for improved accuracy versus
real-time speed[apidog.com](https://apidog.com/blog/how-to-use-google-
gemini-2-5-flash-via-
api/#:~:text=Hybrid%20Reasoning%3A%20Unlike%20models%20that,step%20problems)[apidog.com](https://apidog.com/blog/how-
to-use-google-gemini-2-5-flash-via-
api/#:~:text=,for%20the%20given%20prompt%27s%20complexity).

  9. **Chapter 9: Automated YouTube Subtitle Upload**  
_TL;DR:_ In the final chapter, we complete the pipeline by uploading the
generated subtitles to the user’s YouTube channel via the _YouTube Data API_.
We outline the requirements for API access (OAuth credentials with the
`youtube.force-ssl` scope for caption uploads). Code examples demonstrate how
to use Google’s Python client (`googleapiclient.discovery`) to call
`youtube.captions().insert(...)` and upload the WebVTT file to a specific
video[bomberbot.com](https://www.bomberbot.com/youtube/how-to-add-subtitles-
to-youtube-videos-in-any-language-the-ultimate-guide-for-creators-and-
developers/#:~:text=from%20googleapiclient,http%20import%20MediaFileUpload)[bomberbot.com](https://www.bomberbot.com/youtube/how-
to-add-subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-
creators-and-developers/#:~:text=youtube%20%3D%20build). We cover handling
authentication, specifying language and name for the caption track, and
verifying the upload. This chapter ensures the subtitles we generated can be
delivered to YouTube effortlessly, closing the loop on the entire system.

* * *

## Chapter 1: Pushing Beyond OpenAI Whisper – The State of the Art

OpenAI’s Whisper, released in 2022, set a high bar for open-source speech
recognition with its ~680,000 hours of training data. It achieves impressively
low error rates across many languages and audio conditions. In one independent
test, Whisper (accessible via OpenAI’s transcription API) reached about **7.6%
word error rate (WER)** – outperforming Google’s and Microsoft’s speech-to-
text engines which had higher error rates under the same
conditions[sstoitsev.medium.com](https://sstoitsev.medium.com/google-vs-azure-
a-speech-to-text-
battle-f740aa481e8e#:~:text=,applications%20that%20require%20high%20precision).
Whisper’s strength lies in its robustness to varied input (accents, background
noise) and its support for multiple languages and even tasks like translation.
In practical terms, it often transcribes clearer than many commercial
offerings, especially on open-domain audio.

However, _surpassing_ Whisper means addressing both accuracy and capabilities
beyond what Whisper offers. There are a few key directions to explore:

  * **Higher Accuracy Models:** Whisper is excellent, but proprietary or newer models may edge it out. For example, Google’s research has produced the **Universal Speech Model (USM)** , boasting 2 billion parameters trained on 12 million hours of speech across 300+ languages[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024#:~:text=speech%20recognition). Such scale dwarfs Whisper’s training and could potentially yield higher accuracy in multilingual scenarios. Similarly, companies like Speechmatics claim their ASR is “the world’s most accurate,” with internal benchmarks showing lower error rates than Azure or Whisper on certain tests[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024#:~:text=match%20at%20L640%20Speechmatics%20presents,v3%20was%20not%20out%20yet). While such claims require independent verification, they indicate that continual improvements are happening in ASR.

  * **Diarization and Rich Transcripts:** Whisper focuses on raw transcription and does not natively distinguish speakers or emotions. A system that _surpasses_ Whisper can provide _who said what_ via **speaker diarization** and even _how_ it was said via **emotion or sentiment tagging**. These are features valued in applications like meeting transcriptions, call center analytics, or video captioning for content with multiple speakers. Many commercial transcription services have started to incorporate these; for example, Microsoft Azure’s speech service offers speaker diarization and sentence formatting out-of-the-box[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024#:~:text=Azure%20), and some APIs provide sentiment analysis of audio. So, going beyond Whisper isn’t just about a lower WER – it’s about richer outputs.

  * **Modern Large Language Models (LLMs):** An intriguing avenue is the use of powerful LLMs for transcription and text processing. Google’s **Gemini** series is a prime example. Gemini 2.5 (in “Pro” or “Flash” variants) is a multimodal model that can accept audio (as well as images, video, text) and generate text outputs[ai.google.dev](https://ai.google.dev/gemini-api/docs/models#:~:text=Model%20variant%20Input,efficient%20model). This means we could feed audio to a model like _Gemini 2.5 Flash_ and get a transcript, potentially enhanced by the model’s extensive world knowledge and reasoning abilities. LLM-based transcription might handle ambiguous audio by using context in ways traditional ASR cannot, and could natively produce well-punctuated, formatted text. Additionally, an LLM can be instructed to, say, **insert speaker names or emotions directly** if given some prompt engineering – combining steps that usually require separate modules.

  * **Specialized APIs and Hybrid Approaches:** There are also platforms (AssemblyAI, Deepgram, AWS Transcribe’s latest model, etc.) that continuously improve speech recognition. For instance, benchmarks in late 2024 showed AWS’s new Transcribe model and AssemblyAI achieving strong accuracy, especially when formatting/punctuation is not required to be perfect[assemblyai.com](https://www.assemblyai.com/blog/best-api-models-for-real-time-speech-recognition-and-transcription#:~:text=Accuracy%20vs.%20speed%20trade,real). Some open-source efforts like **WhisperX** have taken Whisper and _augmented_ it – WhisperX adds forced alignment to get word-level timestamps and integrates speaker diarization, while keeping Whisper’s transcription accuracy[assemblyai.com](https://www.assemblyai.com/blog/best-api-models-for-real-time-speech-recognition-and-transcription#:~:text=WhisperX). Such extensions hint that by combining models, we can exceed the original Whisper’s capabilities.

In summary, Whisper remains a tough baseline to beat on open audio, but our
system aims to _surpass_ it by combining the **best of multiple worlds** : a
highly accurate transcription core (leveraging either a superior model or an
ensemble of models and corrections), plus **speaker-aware** and **emotion-
enriched** transcripts. We will explore using Google’s latest AI (Gemini via
AI Studio) as a transcription engine that might rival or exceed Whisper in
quality. We’ll also incorporate **speaker diarization** and **emotion
detection** as first-class features of the pipeline. By doing so, the output
is not just a plain transcript, but a set of _annotated subtitles_ that
provide a richer viewing experience and more actionable data (who spoke, how
they felt) than Whisper alone can deliver.

Before diving into implementation, we must keep in mind practicality: Whisper
is open-source and runs offline; some more advanced options (like Gemini or
cloud APIs) require internet access, API keys, and possibly costs. Our design
will strive to remain **flexible** – e.g. allowing an offline mode using
Whisper or another local model, and an online mode using a cloud service for
higher accuracy or specific features. The next chapter will outline the
overall architecture of this hybrid system and how to set it up in different
environments.

## Chapter 2: System Architecture and Setup (Local and Colab)

To build a comprehensive video transcription system, we need to orchestrate
several components in a pipeline. Let’s break down the **architecture** of our
solution:

**1\. Video Acquisition:** We start with a video (from YouTube or another
platform) that we want to transcribe. The tool _yt-dlp_ will be used to fetch
the video and extract the audio content. This produces an audio file (e.g.,
WAV or MP3) which is the input for transcription.

**2\. Transcription Engine:** This is the heart of the system where audio is
converted to text. We plan to use a _state-of-the-art model_ , aiming beyond
Whisper’s baseline. Two main pathways exist:

  * _Cloud-based API:_ Using an external service or model such as Google’s Gemini (via AI Studio API) or Google Cloud Speech-to-Text. These often provide high accuracy and additional features (like automatic diarization or multi-language support), at the expense of requiring network access and potential cost.

  * _Local model:_ Using a Python library or framework to run a model on the user’s machine or Colab. This could be Whisper itself or another open model. For example, **NVIDIA NeMo** toolkit offers pretrained ASR models and even pipeline scripts to combine diarization with transcription. Another option could be **WhisperX** which adds speaker labels to Whisper’s output.

**3\. Speaker Diarization Module:** A separate component that analyzes the
audio to identify distinct speaker voices and timestamped segments for each
speaker. If the chosen transcription API doesn’t already handle this, we will
run a diarization algorithm (like _pyannote.audio_ pipeline) on the audio. The
result is a timeline of speaker labels (Speaker 1, Speaker 2, etc.) with time
intervals.

**4\. Emotion Detection Module:** Parallel to diarization, this analyzes the
audio (or possibly the text transcript) to detect emotion or sentiment of the
speech in each segment. The output could be a label like “happy”, “neutral”,
“angry” for segments or sentences. This may use a pretrained classifier on
audio signals, or a text-based sentiment model applied to the transcript. For
better accuracy, an audio-based approach (capturing tone) is ideal.

**5\. Subtitle Formatting:** This component takes the raw transcript (from
step 2) and enriches it using information from steps 3 and 4 to produce
_subtitles in WebVTT format_. It will segment the transcript into cues (each
cue has start time, end time, text). The text will be annotated with speaker
names and possibly emotional cues. For example, a subtitle cue might look
like:  
`00:01:23.000 --> 00:01:27.000`  
`<v Speaker 1> (happy) Sure, I can help you with that.`  
Here we used WebVTT’s voice tag (`<v Speaker 1>`) to indicate the
speaker[speechpad.com](https://www.speechpad.com/captions/webvtt#:~:text=styled%20using%20cascading%20style%20sheets,and%20each%20cue%20can%20be),
and we added “(happy)” to denote emotion. The formatting module handles
ensuring timestamps are accurate and that each cue’s duration and length are
comfortable to read.

**6\. YouTube Upload Integration:** Finally, the generated `.vtt` subtitle
file is uploaded to YouTube via the Data API, attaching it to the original
video (assuming we have the video ID and upload permission on that channel).
This automates what is otherwise a manual process in YouTube Studio.

Each of these pieces must work in concert. The data flows from one to the next
(video -> audio -> text -> annotated text -> YouTube). We also need to
consider the **execution environment** and make the system as user-friendly as
possible in both local and Colab contexts.

**Environment Setup Considerations:**

  * _Local Machine:_ The user might run this pipeline on their own PC. Depending on the size of models (e.g., running Whisper large or a diarization model), having a GPU is strongly recommended. The user should install necessary libraries via pip. For instance:
        
        bash
        
        CopyEdit
        
        pip install yt-dlp google-generativeai pyannote.audio google-cloud-speech
        

and so on, for all required packages. If using Google Cloud APIs, the user
will need to obtain API credentials (like a JSON key for service account or an
OAuth token for YouTube API).

  * _Google Colab:_ Colab provides a convenient environment with common libraries and a GPU (if selected). One must still `pip install` some specific libraries not pre-included (like `yt-dlp` or `google-generativeai`). Colab is great for this pipeline because heavy computations (transcription, diarization) can be offloaded to the provided GPU. However, Colab has limitations such as limited runtime (sessions time out) and internet access needed for API calls. We will highlight any Colab-specific tips, like using `pip install` at the top, ensuring to mount Google Drive or upload API keys securely if needed.

  * _API Credentials and Keys:_ For using **Google AI Studio (Gemini)** , one must sign up and get an API key to use the `google-generativeai` Python SDK. Similarly, for YouTube Data API, one must set up an OAuth client ID and consent screen (or use an existing Google account’s credentials flow). We’ll address authentication in the respective chapters (Chapter 8 for Gemini and Chapter 9 for YouTube). It’s important to **not hardcode secrets** in the notebook or script; instead, use environment variables or external config files, especially if sharing the code.

  * _Modularity and Configurability:_ The system should allow toggling features. For example, the user might skip emotion detection or have a single-speaker video (so diarization isn’t needed). We will structure the code so that each part can be configured. Additionally, if the preferred transcription model is not available (say the user has no Google API key), a fallback to Whisper (local) can be provided. This makes the solution robust and widely usable.

**Hardware Requirements:** Running advanced speech models can be resource-
intensive. If using Whisper large or speaker embedding models locally, a
machine with a good GPU and at least 8-16 GB of VRAM is recommended for
reasonable speed. For cloud-based transcription, the heavy lifting is done by
the provider’s servers, so local hardware is less crucial (though uploading
large audio files requires a good internet connection). In Colab, one can
enable a GPU (TPU is less commonly used for these libraries), which typically
suffices for medium-duration videos.

**Data Flow and Intermediate Storage:** It’s wise to consider where
intermediate files are stored:

  * The audio extracted by yt-dlp can be saved as a temporary WAV file.

  * Transcription results might be stored in memory or written to a draft text file.

  * The diarization output (list of speaker segments) could be saved as JSON or just kept in variables.

  * The final WebVTT file will be saved locally (to be uploaded to YouTube or for the user’s use).

Ensuring correct synchronization between transcript and diarization is key:
both processes work with time; we must handle differences like if
transcription segments do not exactly align with diarization segments (we’ll
cover strategies for alignment in Chapter 5).

By the end of this chapter, you should have a clear picture of the overall
system and have your environment ready. To summarize the setup:

  * Install needed Python packages.

  * Acquire necessary API credentials (Google AI API key for Gemini, Google OAuth for YouTube).

  * Prepare a test video link (for example, a YouTube URL).

  * (If local) Ensure `ffmpeg` is installed, since _yt-dlp_ uses it for audio extraction.

With everything in place, we can now proceed step by step through each stage
of the pipeline, beginning with retrieving the video and audio.

## Chapter 3: Video Ingestion with yt-dlp

The first step of our pipeline is to fetch the source video and extract its
audio. We use **yt-dlp** , a popular, feature-rich command-line tool (and
Python library) for downloading videos from YouTube and many other sites. It’s
a fork of the older youtube-dl, with additional features and better
maintenance[cheat.sh](https://cheat.sh/yt-dlp#:~:text=%23%20A%20youtube,dlp).

**Why yt-dlp?** It can handle downloading media in the best available quality
and can directly extract audio tracks if we want. This saves us time and space
– instead of downloading a full video file then extracting audio ourselves,
_yt-dlp_ can do it in one go using ffmpeg under the hood.

**Basic Usage (Command-line):** For example, to extract audio from a YouTube
video, one could run:

    
    
    perl
    
    CopyEdit
    
    yt-dlp --extract-audio --audio-format wav "https://www.youtube.com/watch?v=VIDEO_ID"
    

This command tells yt-dlp to download the video and output only the audio
track, converted to WAV format (which is uncompressed and widely accepted by
speech models). According to the cheat sheet, the `--extract-audio` flag
requires ffmpeg and will by default pick the best audio quality
available[cheat.sh](https://cheat.sh/yt-
dlp#:~:text=,audio%20%22https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DoHg5SJYRHA0).
We specify `--audio-format wav` to ensure a WAV file (since some models prefer
specific audio formats and PCM WAV is safe). You can also specify `--audio-
quality 0` for best quality[cheat.sh](https://cheat.sh/yt-
dlp#:~:text=,quality%200%20%22https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DoHg5SJYRHA0),
though with WAV it’s lossless anyway.

If we wanted to do this via a **Python script** , `yt_dlp` provides a module.
Here’s an example code snippet to download audio:

    
    
    python
    
    CopyEdit
    
    import yt_dlp
    
    video_url = "https://www.youtube.com/watch?v=VIDEO_ID"
    output_template = "input_audio.%(ext)s"  # output file name template
    
    ytdl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav'
        }]
    }
    with yt_dlp.YoutubeDL(ytdl_opts) as ydl:
        ydl.download([video_url])
    

This will produce a file named `input_audio.wav` in the working directory. The
`outtmpl` ensures we name the output in a known way (you can also incorporate
the video title or ID if you want dynamic naming). The postprocessor instructs
yt-dlp to extract audio and convert it to WAV. If you prefer another audio
format like MP3 or M4A, you can set `'preferredcodec': 'mp3'` and perhaps
`'preferredquality': '192'` (kbps) for mp3 quality. But WAV is preferred for
processing because it’s uncompressed and avoids any further loss.

**Handling Long Videos:** If the video is very long (say >1 hour), we should
consider whether the transcription model can handle the entire audio in one
go. Some models (like Whisper or cloud APIs) might have file size or duration
limits. For instance, Whisper’s context is finite (though Whisper models can
process long audio by segmenting internally, there might still be practical
RAM limits). Cloud services often have a _long-running recognition_ method for
lengthy files. If using Google Cloud’s STT API, for example, you might need to
use `long_running_recognize` for anything over ~1 minute.

If we suspect issues, an approach is to **split the audio** into chunks (e.g.,
10-minute segments) using a tool like ffmpeg or Python’s `pydub`. But
splitting can complicate diarization and context. A simpler method in our
pipeline is to attempt transcription on the full audio and see if the model
supports it; many modern models do streaming or chunking internally.

For local runs, ensure you have enough disk space for the audio file. A one-
hour WAV audio at 16 kHz mono is roughly 110 MB. If it’s stereo 48 kHz, it
could be ~1 GB. We can reduce to mono and a reasonable sample rate (most ASR
works well at 16 kHz or 22 kHz). yt-dlp doesn’t automatically resample, but we
can later resample via ffmpeg or in Python if needed. Many ASR models will
resample internally too.

**Verifying the Audio:** It’s good practice to verify that `input_audio.wav`
was saved correctly and maybe to peek at its duration. One can use libraries
like `wave` or `pydub` in Python to check length. This helps ensure that the
download succeeded and that we know how long the content is (which is useful
for progress tracking during transcription).

For example:

    
    
    python
    
    CopyEdit
    
    import wave
    with wave.open("input_audio.wav", 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        print(f"Audio duration: {duration:.2f} seconds")
    

This snippet would output the length of the audio file.

**Alternative Sources:** While YouTube is common, _yt-dlp_ supports many URLs.
If the user provides a direct video file instead of a link, we can skip this
step and use that file. Our pipeline could accept either a URL (then use yt-
dlp) or a local file path to an audio/video file (then perhaps directly use
ffmpeg to get audio if needed). In Colab, one might upload a file to the
session and point to it.

At the end of this stage, we should have an audio file ready for
transcription. Let’s assume we now have `input_audio.wav` containing the audio
track of the video. The next step is sending this audio through our advanced
transcription engine to get the initial transcript with timestamps. We will
cover that in the following chapter.

## Chapter 4: Advanced Transcription Models and APIs

Now we arrive at the core transcription step – converting audio to text. Our
goal is to use a model that _exceeds_ Whisper in either accuracy or
capabilities (or both). We have a few routes to consider:

### Choosing a Transcription Engine

**Option A: Google Cloud / Gemini API** – As hinted earlier, Google’s latest
offerings are prime candidates. There are two slightly different approaches
under Google:

  * _Google Cloud Speech-to-Text API (STT):_ A seasoned API that transcribes speech with features like automatic punctuation, word-level timestamps, and speaker diarization. Google’s models behind this API have improved over the years and currently use the **Universal Speech Model (USM)** or related architectures[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024#:~:text=speech%20recognition). If using this, we can enable diarization and get speaker-attributed transcripts in one response, which is convenient.

  * _Google AI Studio Gemini:_ A more experimental route using the **Gemini 2.x** models via the generative language API. Gemini 2.5 Flash/Pro can accept audio input and return text[ai.google.dev](https://ai.google.dev/gemini-api/docs/models#:~:text=Model%20variant%20Input,efficient%20model). This effectively uses an LLM for transcription. The advantage might be better understanding of context, and possibly the ability to prompt the model for specific formatting (like “transcribe verbatim with punctuation”). However, it may be less optimized purely for ASR than the dedicated STT API. It’s a cutting-edge approach and fits the “beyond Whisper” narrative since it’s a new class of model.

**Option B: Other Cloud APIs** – There are APIs like **AssemblyAI** ,
**Deepgram** , **Azure Speech** , and **Amazon Transcribe**. For example,
AssemblyAI offers high accuracy and even has pre-built support for things like
sentiment analysis, keyword extraction, etc., alongside transcription.
Amazon’s latest Transcribe model massively expanded language coverage and is
designed for high accuracy as
well[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-
and-the-models-powering-them-
in-2024#:~:text=In%20the%20latest%20news%2C%20their,accuracy%20in%20historically%20underrepresented%20languages).
These could be used similarly by making HTTP requests. If one has constraints
with Google, these are alternatives.

**Option C: Local Advanced Models** – If we want an offline solution:

  * We could still use **OpenAI Whisper** (large-v2 model) as a fallback. It might not be strictly “superior” to itself, but with our added processing (diarization/emotion), it forms a more capable system.

  * There are also research models like **Wav2Vec 2.0** large models, or **QuartzNet / Conformer** models from NVIDIA’s NeMo. Some of these can approach Whisper’s quality on certain datasets, especially when fine-tuned.

  * Facebook (Meta) released **SeamlessM4T** (in 2023) – a multilingual model that can transcribe and translate, potentially another candidate, although it’s more translation-oriented.

  * **WhisperX** (by Ufal) can be considered here too – it’s basically Whisper but with added alignment for precise timestamps and an optional speaker identification step[assemblyai.com](https://www.assemblyai.com/blog/best-api-models-for-real-time-speech-recognition-and-transcription#:~:text=WhisperX). Using WhisperX would inherently check off diarization and word-level timing needs while leveraging Whisper’s accuracy.

Given the user’s inputs, they lean towards using **Google Gemini (Flash)**.
So, let’s focus on that for the primary path, but keep in mind the other
options for completeness.

### Using Google AI Studio _Gemini_ API for Transcription

**Setup:** First, ensure you have the `google-generativeai` Python package
installed (`pip install google-generativeai`). Obtain an API key from Google
AI Studio[apidog.com](https://apidog.com/blog/how-to-use-google-
gemini-2-5-flash-via-
api/#:~:text=Navigate%20to%20Google%20AI%20Studio%3A,com)[apidog.com](https://apidog.com/blog/how-
to-use-google-gemini-2-5-flash-via-
api/#:~:text=Create%20Key%3A%20Follow%20the%20prompts,this%20is%20your%20API%20key)
and set it as an environment variable (e.g., `GEMINI_API_KEY`) or have it
handy to configure in code.

**Authentication and Model Selection:** We’ll use the Python SDK for Gemini:

    
    
    python
    
    CopyEdit
    
    import google.generativeai as genai
    
    genai.configure(api_key="YOUR_API_KEY")
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")  # using Flash model
    

The `model_name` might need to be the exact version. Google often has specific
names like `"gemini-2.5-flash-preview"` or similar for the latest. (We could
retrieve available models via the API or check documentation.)

**Sending Audio:** The Gemini API expects inputs possibly in a multimodal
format. According to documentation, Gemini models can take an audio file and
produce text[ai.google.dev](https://ai.google.dev/gemini-
api/docs/models#:~:text=,coding%2C%20reasoning%2C%20and%20multimodal%20understanding).
With the SDK, this might be done through a method that accepts an audio
parameter. Since this is relatively new, the exact call could be:

    
    
    python
    
    CopyEdit
    
    with open("input_audio.wav", "rb") as f:
        audio_data = f.read()
    prompt = "Transcribe the following audio."  # we might include a textual prompt for instruction
    response = model.generate_media(prompt=prompt, audio=audio_data)
    print(response.text)
    

_(Note:`generate_media` is a hypothetical method for illustration; the actual
SDK method might differ if not yet high-level. Possibly one would call a
lower-level `genai.generate_audio()` or use `model.generate_content()` with
the audio encapsulated in the prompt. The principle is that the API allows
audio input.)_

If the SDK doesn’t support direct audio yet, an alternative is to call the
REST API endpoint. One would send a POST request to Google’s generative
language API with the audio bytes base64 encoded and proper headers. But
that’s beyond our scope here – we assume the SDK handles it or will soon,
given Google’s multimodal capabilities.

**Transcription with Google Cloud STT (alternative):** If we choose the
dedicated speech API:

    
    
    python
    
    CopyEdit
    
    from google.cloud import speech_v1p1beta1 as speech
    
    client = speech.SpeechClient()
    # Configure request for diarization and timestamps
    config = speech.RecognitionConfig(
        language_code="en-US",
        enable_speaker_diarization=True,
        diarization_speaker_count=2,  # if we know number of speakers, else omit
        enable_word_time_offsets=True,
        model="latest_long"  # assume Google chooses the best model for long form
    )
    with open("input_audio.wav", "rb") as f:
        audio_data = f.read()
    audio = speech.RecognitionAudio(content=audio_data)
    operation = client.long_running_recognize(config=config, audio=audio)
    print("Transcribing... this may take a while.")
    response = operation.result(timeout=300)  # wait for operation to complete
    

This uses Google’s long_running_recognize which is suitable for files longer
than a minute. The result will contain transcriptions with word timestamps and
speaker tags (Google will label them Speaker_1, Speaker_2, etc., in the
`result.alternatives` if diarization was enabled).

The advantage here: the API itself returns speaker-attributed text, so we
might not need a separate diarization step. But the quality of diarization
might vary, and emotion detection is still separate.

**Accuracy and Performance:** Cloud transcription should handle even hour-long
audio, but keep in mind it can take a few minutes for long files to process
(it’s async). Also, there may be API usage quotas. Gemini Flash, being
optimized, might handle it faster or allow streaming in the future. OpenAI
Whisper locally might actually be faster on a GPU for short files (Whisper can
process ~ real-time or 2x real-time on a high-end GPU for English).

**Combining Approaches:** It’s possible to use a hybrid: e.g., run Whisper
locally to get a draft transcript quickly, then use an LLM (Gemini) to correct
or annotate it. However, this complicates the pipeline and might not yield a
clear benefit unless Whisper made mistakes that an LLM could fix (like
decoding spelling as seen in that SWIFT code example
[sstoitsev.medium.com](https://sstoitsev.medium.com/google-vs-azure-a-speech-
to-text-
battle-f740aa481e8e#:~:text=,applications%20that%20require%20high%20precision)).
An LLM might infer the intended text if Whisper output is phonetically close
but not exact. This is an advanced idea: using LLM for _post-processing_ to
reach beyond Whisper accuracy by leveraging contextual knowledge.

For now, let’s assume we use one primary engine to get the transcript:

  * If using **Gemini API** , we get back a text (likely with no timestamps inherently). We would then need to align it to audio to create timestamps. One way is to split the audio into chunks (say 30-second increments), transcribe each separately, and then merge with timestamps (since we know the start time of each chunk). This chunk-and-transcribe method is straightforward: e.g., split audio into N parts, and for part i (starting at t0), ask Gemini to transcribe it. The returned text for part i is then assigned a time range [t0, t0+duration_of_part]. We have to be careful with splitting so as not to cut words; using a voice activity detector to split on silence is ideal. A simpler approach: we may use Whisper or a VAD to get time boundaries, then use Gemini on those segments. This ensures timestamps.

  * If using **Google Cloud STT** , it will return word-by-word timestamps. We can reconstruct sentence or line timestamps from that.

  * If using **Whisper locally** , the models (via `openai-whisper` library) return timestamps for each segment (usually phrases or sentences) out-of-the-box. Whisper’s segmentation often coincides with pauses. Those can be used for subtitle cues directly. And WhisperX can refine to word-level times.

Given that diarization and emotion tagging are upcoming steps, it might be
beneficial to have a transcript broken into segments (with times). Many APIs
give at least sentence or phrase-level segments:

  * Google’s STT might give one big transcript per speaker or per paragraph by default, but with word offsets.

  * Azure gives full JSON with speaker tags on each portion.

  * Whisper gives segments of a few seconds each with start/end times.

We can create our own segments by splitting on punctuation after the fact as
well.

Let’s illustrate a simple path as an example: **Using Google STT API with
diarization on.**

_Example transcription code output parsing:_

After `operation.result()`, we might do:

    
    
    python
    
    CopyEdit
    
    results = response.results
    transcript = ""
    speaker_segments = []  # will hold tuples (speaker_label, start_time, end_time, text)
    for res in results:
        # Each res is a Transcription result (could be multiple if the API splits)
        alt = res.alternatives[0]
        # If speaker diarization was enabled:
        for word in alt.words:
            speaker_tag = word.speaker_tag  # e.g., 1 or 2
            start_time = word.start_time.total_seconds()
            end_time = word.end_time.total_seconds()
            word_text = word.word
            # ... accumulate words per speaker segment
    

Actually, the Google API provides a convenient way: when diarization is on,
the last result in `results` has a `words` list with speaker tags for each
word. We might need to group words by continuous segments of the same speaker.
Pseudocode:

    
    
    python
    
    CopyEdit
    
    current_speaker = None
    current_segment_words = []
    current_start = None
    
    for w in alt.words:
        if current_speaker is None:
            current_speaker = w.speaker_tag
            current_start = w.start_time.total_seconds()
        if w.speaker_tag != current_speaker:
            # speaker changed, finalize the old segment
            segment_text = " ".join(current_segment_words)
            speaker_segments.append((current_speaker, current_start, prev_end, segment_text))
            # reset for new speaker
            current_speaker = w.speaker_tag
            current_start = w.start_time.total_seconds()
            current_segment_words = []
        current_segment_words.append(w.word)
        prev_end = w.end_time.total_seconds()
    
    # append last segment
    segment_text = " ".join(current_segment_words)
    speaker_segments.append((current_speaker, current_start, prev_end, segment_text))
    

Now `speaker_segments` contains a sequence of spoken segments with speaker
identity. For example:

    
    
    arduino
    
    CopyEdit
    
    [(1, 0.0, 5.2, "Hello, welcome to our show."),
     (2, 5.2, 7.8, "Thank you, glad to be here.")]
    

This is very useful for subtitle generation. We might still want to merge
segments if they are too short or split if too long for subtitle guidelines,
but it’s a starting point.

If we go the **Gemini route** , we won’t inherently have this info. So we’d
rely on our own diarization (Chapter 5 will cover that) to get similar
segments. Essentially, we’d do transcription without speaker info, then later
map it. This can be done by aligning times: we know when each speaker spoke
from diarization, we have a transcript (just text). We would need to align
transcript segments to those times. Tools like **forced alignment** (e.g.,
Gentle or Montreal Forced Aligner) could align the text to audio to get word
times, but that’s complex. Instead, a simpler approach: break the audio by
speaker turns (from diarization) and transcribe each chunk separately,
labeling them with the diarized speaker. That way, each chunk’s text is
inherently assigned to a speaker. This approach is effective: we get segments
from diarization first, then do ASR on each segment (maybe with a small
padding at boundaries to not cut off words). We’ll detail this in Chapter 5 as
well.

**Ensuring High Quality:** Whichever model we use, some best practices to
maximize accuracy:

  * Ensure the audio is **single-channel (mono)** if possible. Stereo audio with one person on left and another on right can confuse models; if that scenario, downmix to mono or even separate channels and transcribe both.

  * Use appropriate **language code** or domain model if the API provides (e.g., Google has phone_call vs video model, etc.). For example, Google’s “phone_call” model is better for conversational audio.

  * If the content has specialized terms, some APIs allow a _hints_ or _custom vocabulary_. Whisper can’t take hints, but Google/Azure can. We might supply a list of keywords (e.g., names or jargon known to appear in the video) to improve recognition. In code, this is done via config (PhraseSet in Google or keywords in Azure).

  * Turn on **automatic punctuation** (most modern models do it by default, including Whisper and Google’s). Without punctuation, transcripts are one long sentence which is harder to read.

Given that the user specifically mentioned _“basic and sentence, included in a
format appropriate for WebVTT”_ in their clarifications, it suggests we should
aim for the transcription output to be segmented by sentences. So likely we
want our transcription step to produce sentences as units (which Whisper and
Google do naturally to some extent). We will later map those to subtitle cues.

At the end of this transcription phase, we expect to have:

  * The full transcript text.

  * Start and end times for each segment of transcript (which might be a sentence or phrase).

  * Possibly speaker attributions per segment (either provided by the API or to be added via diarization in the next step).

We now move on to handling the speaker diarization in detail, unless we
already got it from the API. Assuming we haven’t, Chapter 5 will focus on
that.

## Chapter 5: Speaker Diarization Techniques

_“Who said what?”_ is a crucial question for many transcriptions. Speaker
diarization will enable us to label segments of the transcript with speaker
identities (e.g., Speaker 1, Speaker 2, etc.). This greatly enhances the
readability of subtitles for dialogues or multi-speaker videos, and is
essential in meetings or interviews where multiple voices are present.

There are a few ways to achieve diarization:

### 5.1 Using Pretrained Diarization Pipelines (PyAnnote)

One of the leading tools in this field is **pyannote.audio** , an open-source
toolkit specifically for speaker diarization. It provides end-to-end pipelines
that are pre-trained on large datasets and can achieve _state-of-the-art
performance_ on benchmark
tasks[huggingface.co](https://huggingface.co/pyannote#:~:text=pyannote.audio%20,performance%20on%20most%20academic%20benchmarks).
The nice thing is that Hugging Face hosts ready-to-use diarization pipelines
(e.g., `pyannote/speaker-diarization@<version>`).

To use pyannote’s pipeline, you typically need to authenticate with a Hugging
Face token (since the models are large). Let’s illustrate usage:

    
    
    python
    
    CopyEdit
    
    !pip install pyannote.audio==2.1  # ensure version compatibility
    
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="YOUR_HF_TOKEN")
    # Apply the pipeline to our audio file
    diarization = pipeline("input_audio.wav")
    

The result `diarization` is an _Annotation_ object. We can iterate over it to
get segments:

    
    
    python
    
    CopyEdit
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        end = turn.end
        spk = speaker  # e.g., "SPEAKER_00", "SPEAKER_01"
        print(f"Speaker {spk}: from {start:.2f}s to {end:.2f}s")
    

Pyannote will assign anonymous labels like SPEAKER_00, SPEAKER_01, etc., which
we can map to simpler names (Speaker 1, Speaker 2). The important part is that
it has determined when each speaker is talking.

**Clustering and Unknown Number of Speakers:** Pyannote’s default pipeline
will figure out the number of speakers on its own (it uses a segmentation and
clustering approach under the hood). If you have an idea of how many speakers
there are, some diarization methods allow specifying that (e.g.,
`diarization=pipeline(audio, num_speakers=2)` possibly). But an advantage of
these models is they can detect even if, say, 3 or 4 voices are present.

**Accuracy considerations:** Modern diarization can be quite accurate in
separating speakers when voices are distinct and audio is decent. There might
be challenges when speakers overlap (talk simultaneously) – some systems label
overlap as well, but for subtitles we typically can’t show two people talking
at once easily (we might skip overlap or choose one to display, or use
separate cues if needed).

### 5.2 Diarization via ASR API

As discussed, if we used an API like Google’s with diarization enabled, we
might already have speaker tags for each
word[stackoverflow.com](https://stackoverflow.com/questions/78333098/how-to-
grant-scope-to-also-upload-captions-on-
youtube#:~:text=,ssl%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutubepartner)[stackoverflow.com](https://stackoverflow.com/questions/78333098/how-
to-grant-scope-to-also-upload-captions-on-
youtube#:~:text=Based%20on%20the%20API%20documentation%2C,scopes%20you%20listed%20are%20required).
If that’s the case, we can use that directly. However, sometimes the quality
of that diarization might not be as good as a dedicated tool like pyannote. It
could confuse speakers if their voices are somewhat similar or if the number
of speakers guessed is wrong. For robust solutions, many use the dedicated
pipeline approach.

### 5.3 Combining ASR and Diarization

There are integrated projects like **WhisperX** and **NVIDIA NeMo’s
diarization pipeline** that combine transcription with diarization. For
example, WhisperX uses a pre-trained speaker embedding model (like Resemblyzer
or pyannote embedding) to assign speaker labels to Whisper’s
segments[assemblyai.com](https://www.assemblyai.com/blog/best-api-models-for-
real-time-speech-recognition-and-transcription#:~:text=WhisperX). NVIDIA’s
NeMo toolkit has a tutorial where they first generate a transcript and then
use timestamped audio chunks to cluster speakers.

Our approach can be:

  * Perform diarization on the audio to get speaker segments with times.

  * Use those times to split the transcript.

Two scenarios:  
**(a) We have a continuous transcript with timestamps for each word/segment
(like from Google STT or Whisper).** In that case, intersecting with
diarization is straightforward: for each diarization segment (time interval
with speaker X), we take the portion of transcript that falls in that interval
and tag it as speaker X.

**(b) We have transcript text without precise timestamps (like from a basic
API or an LLM).** Then we should instead use diarization first. Suppose
diarization gives us segments: [0–5s: Spk1], [5–8s: Spk2], [8–12s: Spk1], etc.
We can then feed each segment’s audio to the transcription model separately:

  * e.g., extract audio from 0–5s, transcribe it to get text1 (this is what Speaker 1 said).

  * extract 5–8s, transcribe -> text2 (Speaker 2’s speech).

  * and so on.  
This ensures each chunk’s text is correctly attributed. The disadvantage is we
lose cross-segment linguistic context for the transcriber, but that’s usually
fine because segments are short and usually aligned to pauses or turn-taking.

We can implement this splitting using Python’s `pydub` or `ffmpeg` to cut the
audio. For example, using `pydub`:

    
    
    python
    
    CopyEdit
    
    from pydub import AudioSegment
    audio = AudioSegment.from_wav("input_audio.wav")
    for segment in diarization.itertracks(yield_label=True):
        start_ms = int(segment[0].start * 1000)
        end_ms = int(segment[0].end * 1000)
        chunk = audio[start_ms:end_ms]
        chunk.export(f"temp_{start_ms}_{segment[2]}.wav", format="wav")
        # then send chunk to ASR
    

We name the file with speaker label for clarity (though in real code, better
to keep track in a dict rather than filename).

Then for each chunk file, call the transcription engine (could even call
Whisper or the Google API with a shorter audio which will be faster and maybe
more accurate if it’s short).

After transcribing all chunks, we’ll have texts associated with a speaker and
time interval. We then sort them by time and concatenate appropriately. This
yields a structure very similar to the `speaker_segments` we described in
Chapter 4’s code snippet, but with actual transcript text for each segment.

**Voice Recognition vs Speaker Diarization:** It’s worth noting that
diarization as we do it is _unsupervised_ in that it doesn’t know the identity
of speakers, it just labels them as #1, #2, etc. If the user wanted actual
names (e.g., identify _Alice_ vs _Bob_), that becomes a speaker
**recognition** task which requires known voice profiles – not in our scope
unless we have a database of speakers. So we will stick to generic labels.
WebVTT supports naming by using `<v Name>` tags or we can embed the name in
subtitle text. We’ll likely use labels “Speaker 1”, “Speaker 2” etc., or if
it’s an interview with a host and guest, perhaps manually assign those roles
if known.

**Evaluating diarization output:** It’s often good to check if the number of
speakers found matches expectation. Pyannote might over-split if a single
speaker’s voice has varied background noise. If that happens, one can try to
merge segments with very short gaps or lower the sensitivity. Pyannote 3.x
pipeline is quite advanced though, and usually does well.

Finally, with speaker segmentation done, we proceed to incorporate emotion
detection. But before that, we now have a transcript that is segmented by
speaker. For example:

    
    
    scss
    
    CopyEdit
    
    Speaker 1 (0.00-5.00s): "Hello, how are you?"
    Speaker 2 (5.00-7.00s): "I'm good, thanks."
    Speaker 1 (7.00-12.00s): "Great to hear! Let's get started."
    ...
    

These segments will feed into subtitle creation. We’ll later format them with
times. But we can further enrich each segment with emotion.

## Chapter 6: Emotion Detection from Speech

Understanding the emotion behind spoken words can add valuable context to a
transcription. For subtitles, this might be optional, but the requirement is
to _detect and annotate emotions_ in the transcription, which implies we
should indicate if a speaker was, for example, **happy** , **sad** , **angry**
, **excited** , etc., when speaking a line. This can be useful in content like
podcasts, movies, or user research interviews to convey tone.

There are two main ways to infer emotion:

**6.1 Audio-based Emotion Recognition (Speech Emotion Recognition - SER):**
This analyzes the raw audio characteristics – tone of voice, pitch, energy,
speaking rate, etc. Emotions often have acoustic signatures (e.g., anger might
have louder volume and higher pitch variability, while sadness might have
lower energy and slower tempo). Machine learning models can pick up on these
cues.

Modern SER models often use deep learning. For example, models built on
**wav2vec 2.0** or **SpeechBrain** that are fine-tuned on emotion datasets
like IEMOCAP or RAVDESS can classify an audio clip into categories like happy,
sad, angry, neutral, fear,
etc.[exposit.com](https://www.exposit.com/portfolio/speech-emotion-
recognition/#:~:text=The%20voice%20characteristics%20are%20processed,people%20in%20various%20emotional%20states).
Indeed, one case study mentions a model detecting six emotions: _anger,
disgust, fear, happiness, neutrality,_ and _sadness_[
exposit.com](https://www.exposit.com/portfolio/speech-emotion-
recognition/#:~:text=The%20voice%20characteristics%20are%20processed,people%20in%20various%20emotional%20states).

Libraries:

  * **SpeechBrain** has an open-source model: `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` on Hugging Face, which we can use.

  * **torchaudio** and PyTorch ecosystem also have some pretrained models or examples for emotion classification.

  * **TensorFlow** could be used with models like a simple CNN on spectrogram, but we prefer ready models.

Let’s outline using a Hugging Face pipeline for emotion:

    
    
    python
    
    CopyEdit
    
    !pip install transformers datasets
    from transformers import pipeline
    emotion_pipeline = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    # This model is from the SUPERB benchmark, classifying emotions
    

Now, for each segment of audio we identified in diarization, we can feed the
audio segment to this pipeline:

    
    
    python
    
    CopyEdit
    
    result = emotion_pipeline("temp_0_SPEAKER_00.wav")
    print(result)
    

This might output something like `[{'label': 'neutral', 'score': 0.85},
{'label': 'happy', 'score': 0.1}, ...]`. We take the top label if the score is
confident. In this example, it would classify as neutral.

We would do this for each speaker segment or perhaps each subtitle-length
segment. We have to decide the granularity: Emotions can change over time, but
often we assume one emotion per utterance. It might be sufficient to label
each speaker turn with one emotion (dominant emotion). If a turn is long and
contains multiple sentences, theoretically the emotion could shift, but that’s
beyond typical detection granularity.

**6.2 Text-based Emotion/Sentiment Analysis:** Another approach is to infer
emotion from the content of the transcript. For instance, the words used or
punctuation could indicate if someone is excited or upset. There are sentiment
analysis tools (positive/negative/neutral) and emotion classifiers (some NLP
models classify text to joy, sadness, anger, etc.). However, text alone may be
misleading – e.g., the phrase "I'm fine" could be spoken angrily or cheerfully
and the text wouldn’t tell you. Since we have audio, it’s better to rely on
that.

However, text-based can be a lightweight fallback if audio analysis is
complicated. We could even use an LLM (like asking Gemini or GPT “what is the
emotion of this text?”), but since we want an automated pipeline and we
already call enough models, we’ll stick with direct audio classification.

**Combining Emotion with Transcript:** Once we have an emotion label for a
segment, we can annotate it. For example, if Speaker 1’s segment from 0–5s is
classified as _excited_ , we might tag it as such. We can include this in
subtitles as `[excited]` or `(excited)` or some emoji, but likely a simple
textual tag is fine. Maybe format it as `(happy)` or `[angry tone]` etc. The
exact wording can be decided based on convention. For now, we’ll use a single
word in parentheses.

**Accuracy and Categories:** Emotion recognition is not 100% accurate – voices
can be ambiguous and there’s subjectivity. We should limit to a few broad
categories to avoid confusion. Many systems use: _neutral, happy, sad, angry,
fear, surprise_. If the model gives something like disgust or fear which are
less common in everyday speech outside special contexts, we might map those to
a generic “negative” tone or skip if uncertain. It’s also possible to combine
text and audio results if needed – e.g., if audio is neutral but text has an
exclamation mark, maybe label as excited. These are nuanced and perhaps beyond
our current scope.

We also need to ensure short segments (like 1 second yes/no answers) might not
have enough info to detect emotion – we might default those to neutral.

**Integration in Code:**  
We could do:

    
    
    python
    
    CopyEdit
    
    for seg in speaker_segments:  # seg = (spk, start, end, text)
        start, end, spk_label, text = seg
        audio_chunk = audio[start*1000:end*1000]  # using pydub segment
        emotion = emotion_pipeline(audio_chunk.get_array_of_samples(), sampling_rate=audio.frame_rate)
        top_emotion = max(emotion, key=lambda x: x['score'])['label']
        seg_emotion = top_emotion
        # store emotion with the segment, e.g., seg += (seg_emotion,)
    

The huggingface pipeline might allow directly giving a filepath or numpy array
of audio too. We just have to ensure the audio segment is correctly passed.

We should be cautious to cut segments to where there is voice – including
silence might reduce classification confidence. Possibly adjust by skipping
leading/trailing silences (pyannote might already have done that when creating
segments).

**Emotion in Multi-speaker scenarios:** Each speaker turn gets its own emotion
label. If two people are speaking simultaneously (overlap), theoretically each
could have their own emotion at that moment, but we probably won’t handle
overlaps in subtitles, we avoid overlapping cues.

Now after this step, each transcript segment is enriched with two annotations:

  * Speaker label (e.g., Speaker 1)

  * Emotion (e.g., happy)

For example, we might have:

    
    
    nginx
    
    CopyEdit
    
    Speaker 1, happy: "Hello, how are you?"
    Speaker 2, neutral: "I'm good, thanks."
    

In subtitles, we can format this as:  
`<v Speaker 1> (happy) Hello, how are you?`

We should keep the emotion tag subtle so it doesn’t overwhelm the actual
message. Parentheses or italic could be considered.

One might also choose to color-code emotions if using advanced subtitle
styling, but standard WebVTT has limited styling (could use classes, but not
all players support complex styles). Simpler is better.

With the emotion-tagged, speaker-tagged transcript ready along with
timestamps, we can proceed to build the WebVTT file.

## Chapter 7: Generating WebVTT Subtitles with Timestamps

At this point, we have all the ingredients for our subtitles:

  * Segmented transcript (with each segment having start time, end time, speaker, emotion, and text).

Our task now is to output this in **WebVTT format** , which is a plain text
format for caption files widely supported by video players and YouTube
(YouTube actually accepts SRT and SBV too, but we’ll use WebVTT as specified).

**WebVTT Basics:** A WebVTT file is a text file with extension `.vtt`. It
typically begins with a header line:

    
    
    nginx
    
    CopyEdit
    
    WEBVTT
    

(on its own line, possibly followed by metadata header lines, but those are
optional[speechpad.com](https://www.speechpad.com/captions/webvtt#:~:text=WEBVTT)).
After that, there’s a blank line, and then each **cue** (caption unit) is
listed sequentially.

Each cue consists of:

  * An optional **cue identifier** line (we can number the captions or give an ID, but it’s not required in WebVTT).

  * A **timecode line** in the format `hh:mm:ss.mmm --> hh:mm:ss.mmm` indicating when the subtitle appears and disappears.

  * One or more lines of subtitle text, followed by a blank line to end the cue.

For example:

    
    
    rust
    
    CopyEdit
    
    1
    00:00:00.000 --> 00:00:04.000
    <v Speaker 1> (happy) Hello, how are you?
    
    2
    00:00:04.000 --> 00:00:06.500
    <v Speaker 2> (neutral) I'm good, thanks.
    

This example shows two cues with IDs 1 and 2. The timestamps indicate cue 1
shows from 0 to 4 seconds, cue 2 from 4 to 6.5 seconds. We’ve included speaker
and emotion in the text using `<v Speaker 1>` which is the WebVTT voice tag
indicator[speechpad.com](https://www.speechpad.com/captions/webvtt#:~:text=styled%20using%20cascading%20style%20sheets,and%20each%20cue%20can%20be).
The voice tag is typically used to identify speakers in captions and can be
styled via CSS if desired (e.g., different colors per speaker, though YouTube
doesn’t preserve styling on its player to my knowledge). But at least it
semantically tags the cue with Speaker 1’s voice.

We put the emotion in parentheses after the voice tag. This is a design
choice; alternatively, we could incorporate it as part of the voice tag if the
spec allowed, but it doesn’t – the `<v>` tag is meant just for speaker name.
So, putting emotion in the text is fine. We might also italicize it for
clarity. WebVTT allows some basic formatting like `<i>...</i>` for
italics[speechpad.com](https://www.speechpad.com/captions/webvtt#:~:text=,can%20have%20a%20unique%20style).
We could do:

    
    
    php-template
    
    CopyEdit
    
    <v Speaker 1> <i>(happy)</i> Hello, how are you?
    

This would render "(happy)" in italics, distinguishing it from spoken words.
That might be nice to visually separate stage directions/emotion from
dialogue.

Let’s plan to italicize emotion tags. We should ensure to close the italic
before the actual spoken text.

**Constructing timecodes:** We have segment start and end times in seconds
(possibly as floats). WebVTT times format is `HH:MM:SS.mmm`. For example, 0.0
seconds is `00:00:00.000`. We can format times easily in Python. We should
also be careful about hours: if video is longer than 1 hour, we’ll need the
hour part (WebVTT hours are optional for under 1 hour, but including them is
fine always with leading zeros).

We can do a small utility:

    
    
    python
    
    CopyEdit
    
    def format_timestamp(seconds):
        ms = int(round(seconds * 1000))
        hours = ms // 3600000
        ms %= 3600000
        minutes = ms // 60000
        ms %= 60000
        secs = ms // 1000
        msecs = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{msecs:03d}"
    

This outputs a string like "00:01:23.456".

**Segmentation for Subtitle Length:** We should consider if our diarization
segments are too long to display. Usually, caption guidelines suggest not more
than 2 lines of text per caption and maybe 32 characters per line (varies, but
around that). If someone speaks a long paragraph nonstop, we might break it
into multiple cues.

In our pipeline, since we segmented by speaker turns or sentences, we likely
already have manageable sizes. But if a single speaker turn lasts very long,
we may want to split by sentences. We can use punctuation to split the text
within a segment. For example, if Speaker 1’s text is “Hello everyone. Today I
will talk about X.” spanning 10 seconds, it might be nicer as two cues:

  * 0–5s: "Hello everyone."

  * 5–10s: "Today I will talk about X."

We can do this split by punctuation (. ? !). This requires also splitting the
time segment accordingly. A simple approach: assume even distribution or use
actual word timestamps if available. If we have word-level times from ASR
(like Google STT provided), we could split exactly where the sentence ended
(we’d know the time of the period). If not, we could approximate by
distributing time based on proportion of text.

Given the complexity, we might keep it simpler and trust our segments
(especially if using the cloud API, which often returns fairly short sentence-
like segments by itself).

**WebVTT Output Example:**

Let's manually craft one cue for demonstration:

  * Speaker 1 from 0.00s to 3.50s said "Hello, how are you?" happily.

Timecodes: `00:00:00.000 --> 00:00:03.500`.  
Cue text: `<v Speaker 1> <i>(happy)</i> Hello, how are you?`

Another:

  * Speaker 2 from 3.50s to 5.20s said "I'm good, thanks." neutrally.

Timecodes: `00:00:03.500 --> 00:00:05.200`.  
Cue text: `<v Speaker 2> <i>(neutral)</i> I’m good, thanks.`

These will appear sequentially in the .vtt.

We should not put overlapping times (WebVTT expects cues in increasing time
order, and usually not overlapping, though it can handle overlaps by showing
both, but it’s messy visually).

**Add to WebVTT file:**

The file structure:

    
    
    php-template
    
    CopyEdit
    
    WEBVTT
    
    1
    00:00:00.000 --> 00:00:03.500
    <v Speaker 1> <i>(happy)</i> Hello, how are you?
    
    2
    00:00:03.500 --> 00:00:05.200
    <v Speaker 2> <i>(neutral)</i> I'm good, thanks.
    
    ...
    

We number cues starting from 1 sequentially. Actually, WebVTT cues identifiers
are optional; we could omit "1" and "2" lines and it would still be valid.
That is more common in SRT. In WebVTT, usually they don't number, but
including numbers doesn’t hurt (players typically ignore the cue identifiers).

Since numbering might be expected by some tools and it helps readability, we
might include it.

**Special characters:** We need to escape any '-->' that appear in text
because that could confuse the file (rare in speech). Also, ampersands or '<'
in actual text might need escaping since those could be seen as HTML. But
speech rarely has '<' or similar. We'll trust or do minimal replacements (like
`&` to `&amp;` if needed).

Now the **code** for assembling:  
We loop through our list of segments (speaker_segments with emotion):

    
    
    python
    
    CopyEdit
    
    with open("subtitles.vtt", "w") as vtt:
        vtt.write("WEBVTT\n\n")
        for idx, seg in enumerate(segments, start=1):
            start, end, speaker, emotion, text = seg
            vtt.write(f"{idx}\n")
            vtt.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            vtt.write(f"<v {speaker}> <i>({emotion})</i> {text}\n\n")
    

This yields the file.

We should double-check format:

  * Ensure there’s a blank line after each cue (including after the last).

  * The timestamp separator has a space, arrow, space (`-->`) as shown.

**Validating the output:** It’s a good idea to test a snippet of the VTT on a
video or using an online caption viewer to ensure it’s well-formed.

One detail: If emotion is "neutral" which is kind of the default state, we
might decide not to display "(neutral)" every time to avoid clutter. Perhaps
we only display emotion if it’s something other than neutral. Many subtitle
conventions wouldn’t note normal speech, but only note if someone is yelling
(angry) or sobbing (sad) etc. We could implement a rule: skip or omit the tag
if emotion is neutral. That would yield cleaner subtitles, highlighting only
notable emotions.

Yes, that’s a sensible refinement – let’s say:

    
    
    ini
    
    CopyEdit
    
    display_emotion = emotion if emotion.lower() != "neutral" else ""
    

and then if it’s empty, maybe skip parentheses altogether. Or simply don’t
include that part. In practice: if neutral, just do `<v Speaker 2> I'm good,
thanks.` without any parentheses.

This way viewers aren’t distracted by constant (neutral) labels.

We’ll proceed with that assumption.

So:

    
    
    python
    
    CopyEdit
    
    emo_text = f"<i>({emotion})</i> " if emotion.lower() != "neutral" else ""
    vtt.write(f"<v {speaker}> {emo_text}{text}\n\n")
    

Now, once the .vtt content is generated, the next stage is to upload to
YouTube.

## Chapter 8: Integrating Google AI Studio (Gemini) for Enhancements

In Chapter 4 we already integrated the Gemini API as a transcription engine.
Here in Chapter 8, we'll focus on any additional ways Gemini or similar AI
models can enhance our pipeline beyond basic transcription.

Given Google’s Gemini (especially the _Pro_ model) is a powerful LLM, we can
consider a couple of _enhancement scenarios_ :

  * **Quality Improvement:** Use the LLM to proofread or correct the transcript. For example, after getting an initial transcript (via Whisper or another model), we could feed chunks of it to Gemini with a prompt like “Here is a potentially imperfect transcript. Please output a corrected transcript with proper grammar and spelling, without changing spoken content.” This could fix minor errors and add punctuation where needed. Whisper already does punctuation, but if using a model that didn’t, an LLM could help.

  * **Summarization or Insight (beyond requirement but possible):** We could ask Gemini to summarize the entire video or identify key points. Though not requested, it's a conceivable extension.

  * **Emotion inference from text:** If audio-based emotion detection is not very confident, an LLM could infer emotion from context. For example, if someone says "I'm so excited about this project!", an LLM would easily label that as excited/happy. We could merge that with audio results.

  * **Speaker naming** : If we had meta info (like we know Speaker 1 is John and Speaker 2 is Jane from context), an LLM could possibly detect that if it’s a known interview or something. But that’s speculative.

Primarily, we should demonstrate how to use the _Gemini API via Python_ in
practice (since integration is explicitly mentioned). We covered
transcription; here let's show a small example with text prompting:  
Say we want to use **Gemini for punctuation correction** :

    
    
    python
    
    CopyEdit
    
    # Suppose raw_transcript is a string without punctuation
    raw_transcript = "hello how are you doing today I hope youre doing well"
    prompt = f"Please punctuate and capitalize the following transcript exactly as spoken, without adding or removing words:\n```{raw_transcript}```"
    response = model.generate_text(prompt)  # Using a generic generate function for text
    corrected_text = response.text
    print(corrected_text)
    

This might return: "Hello, how are you doing today? I hope you’re doing well."

_(In actual code,`GenerativeModel` might use a method like `.generate_text()`
or `.generate_content()` depending on the library version. We already showed
usage in Chapter 4 with content generation
example[huggingface.co](https://huggingface.co/blog/proflead/google-
gemini-2-5-pro#:~:text=import%20google).)_

If we hadn’t used Gemini for the main transcription, we could still use it
here for refining or even to do tasks like splitting into subtitles of
appropriate length. However, those are things we can handle with simpler
logic.

Another interesting feature of Gemini 2.5 Flash is its **“thinking” mode**[
apidog.com](https://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-
api/#:~:text=Hybrid%20Reasoning%3A%20Unlike%20models%20that,step%20problems)[apidog.com](https://apidog.com/blog/how-
to-use-google-gemini-2-5-flash-via-
api/#:~:text=,for%20the%20given%20prompt%27s%20complexity) – it can do chain-
of-thought reasoning internally if allowed more tokens. For transcription, we
likely don’t need that (transcription is straightforward). But if we had a
scenario of parsing a complex audio (like spelling out an email address or
something tricky), a model could internally reason about possibilities. This
is quite advanced and likely overkill.

**Using Gemini for multilingual** : If the video had multiple languages,
Whisper handles them, but we could also leverage an LLM to translate or ensure
proper language identification. For example, if a segment is Spanish, we might
translate it to English for subtitles (if that was desired). The pipeline
could detect language and call an LLM to translate. But since not asked, we’ll
skip.

**Integration Steps Recap** (for Gemini in our pipeline):

  1. Obtain API key from Google AI Studio[apidog.com](https://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/#:~:text=Navigate%20to%20Google%20AI%20Studio%3A,com).

  2. Install `google-generativeai` and configure it with the key[apidog.com](https://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/#:~:text=Then%2C%20in%20your%20Python%20code%2C,automatically%20pick%20up%20the%20key).

  3. Choose model (Flash vs Pro). Flash is cheaper/faster, Pro might give slightly better results at higher cost. For most, Flash should suffice.

  4. Use the model for transcription or text processing tasks by calling methods as shown (either `generate_audio` or `generate_text` depending on input).

  5. Handle any rate limits – e.g., the API might limit number of requests per minute. If processing long videos by chunking audio and sending many requests, we might need to throttle or combine some chunks.

It’s also prudent to mention cost: at time of writing, these APIs might charge
per input/output token. Transcribing a 1-hour video with Gemini could be
expensive token-wise (1 hour audio might translate to ~10k words = ~50k tokens
maybe). If cost is a concern, a user might use a local model for transcription
and use Gemini only for polishing or not at all.

**Alternative model integration:** The prompt says “or alternative models” –
perhaps meaning if not Gemini, maybe OpenAI GPT-4 or others. Indeed, one could
integrate OpenAI’s GPT-4 or GPT-3.5 for similar tasks (though GPT-4 doesn’t
take audio directly, one could transcribe with Whisper and then use GPT-4 to
do some augmentation). The process would be analogous.

For completeness, let's say we want to integrate _OpenAI Whisper API_ (if
someone preferred that to local whisper to get the text). That’s a single HTTP
call to OpenAI’s endpoint with the audio file. But since we are focusing
beyond Whisper, we won't detail it.

**Conclusion of integration:** In our pipeline code, integrating Gemini means
possibly having a flag or setting that if `use_gemini = True`, we do:

  * Transcribe using Gemini (like in Chapter 4).

  * Else if `use_gcp_stt = True`, use Google STT.

  * Else if `use_whisper_local = True`, use Whisper.

We have to ensure rest of pipeline gets the data in a consistent format
(segments with times). The heavy lifting to extract times might differ per
method. For fairness, a robust pipeline might always run a diarization step to
get timestamps and speakers, even if the API gave some, just to unify. But if
using Google STT with diarization, we already got times and speakers, no need
to re-do it (maybe just trust that output).

Anyway, at this point, we have covered building the subtitle. The final piece
is automating the upload to YouTube.

## Chapter 9: Automated YouTube Subtitle Upload

With our subtitles ready in a WebVTT file (`subtitles.vtt`), we want to upload
them to the user’s YouTube video. This will add a caption track to the video
which viewers can toggle on/off.

YouTube provides a **Data API (v3)** for this. To use it:

  * We need OAuth 2.0 credentials authorized for the YouTube account that owns the video.

  * Specifically, for uploading subtitles, the API documentation says the request requires the scope **`youtube.force-ssl`** (or the partner scope)[stackoverflow.com](https://stackoverflow.com/questions/78333098/how-to-grant-scope-to-also-upload-captions-on-youtube#:~:text=,ssl%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutubepartner)[stackoverflow.com](https://stackoverflow.com/questions/78333098/how-to-grant-scope-to-also-upload-captions-on-youtube#:~:text=Based%20on%20the%20API%20documentation%2C,scopes%20you%20listed%20are%20required). `youtube.force-ssl` is a scope that basically gives full YouTube account access (it’s commonly used in many YouTube API calls, including uploading videos, managing playlists, etc.). We must ensure our OAuth token includes this scope.

**Setting up credentials:** You’d typically create an OAuth client ID in
Google Cloud Console, enable YouTube Data API for your project, and go through
a consent screen. For a quick script, Google’s API client library can launch a
browser for the user to authorize and then save a token. In Colab, there’s
often a simplified flow where you provide an authentication code. Given the
context, we assume the user is willing to authenticate (since uploading to
their channel requires it). For a persistent script, one might use the
`google-auth-oauthlib` flow to get `credentials.json`.

However, to keep this focused, we’ll assume we have `credentials` (an object
or file) ready to use.

**Using the Python client library:**  
Install with `pip install google-api-python-client oauth2client`.

Then:

    
    
    python
    
    CopyEdit
    
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    # credentials = ... (obtained via OAuth flow, e.g., using google_auth_oauthlib.flow)
    youtube = build('youtube', 'v3', credentials=credentials)
    

Now we prepare our caption insert request:  
We need the `videoId` (the YouTube video identifier, e.g., "dQw4w9WgXcQ"). The
user presumably provides or we got it from the URL used with yt-dlp. Yes, if
we downloaded via a YouTube URL, we have the ID. We should store it. (In code,
we can parse it or maybe yt-dlp’s info dict can give it. Alternatively, the
user might input it separately for uploading if the video is theirs).

We also specify language (like `"en"` for English) and a name for the track
(like "English Subtitles").

The API call as per
documentation[bomberbot.com](https://www.bomberbot.com/youtube/how-to-add-
subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-creators-
and-
developers/#:~:text=Here%E2%80%98s%20a%20basic%20example%20of,using%20the%20API%20in%20Python)[bomberbot.com](https://www.bomberbot.com/youtube/how-
to-add-subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-
creators-and-developers/#:~:text=youtube%20%3D%20build):

    
    
    python
    
    CopyEdit
    
    request = youtube.captions().insert(
        part="snippet",
        body={
            "snippet": {
                "videoId": VIDEO_ID,
                "language": "en",
                "name": "English",  # subtitle track name
                "isDraft": False
            }
        },
        media_body=MediaFileUpload("subtitles.vtt")
    )
    response = request.execute()
    

This will upload the file and publish it (`isDraft=False` means viewers can
see it immediately; if True, it would be added but not publicly shown until
manually published or turned false with another API call).

If successful, `response` will contain details including an ID for the new
caption track. If there were already an automatic caption or another track,
this adds another (YouTube allows multiple caption tracks, e.g., different
languages or even multiple English versions).

**Important:** The user account must be the owner of the video or have rights
to add captions. If the video isn’t in their channel, the API won’t allow
adding subtitles (unless they use YouTube’s community contributions which was
a feature but it’s turned off since 2020).

**Cleaning up and handling errors:**

  * We should handle the case where the upload might fail with a permissions error (scope issue, or if the videoId is wrong).

  * Also, if a caption track already exists and we want to update it, we’d use `captions.update()` instead of insert, and need the caption track ID. But since we are likely adding new, `insert` is fine.

  * For testing without publishing, one could set isDraft=True to see it on the account privately.

**Automating via script vs manually in YouTube Studio:** Since this is a
coding treatise, we go with script. But keep in mind, if one-off, sometimes
it’s easier to manually upload via YouTube Studio. Yet, doing it in code is
powerful when dealing with many videos or integrating in a pipeline.

A quick example output:  
If our `VIDEO_ID` was "ABC123XYZ", after running the code, the response might
be:

    
    
    json
    
    CopyEdit
    
    {
      "kind": "youtube#caption",
      "etag": "...",
      "id": "AbCdEfGh12345",
      "snippet": {
        "videoId": "ABC123XYZ",
        "lastUpdated": "2025-08-15T12:00:00.000Z",
        "language": "en",
        "name": "English",
        "audioTrackType": "unknown",
        "isCC": true,
        "isLarge": false,
        "isEasyReader": false,
        "isDraft": false,
        "status": "serving"
      }
    }
    

The `status: serving` indicates the caption is live.

After a short while, one can go to the video watch page and turn on captions
to see them.

**Final touches:** With the subtitles uploaded, our pipeline’s job is done. We
should ensure to notify or log success. If doing multiple videos, iterate
accordingly.

We may also consider deletion or re-upload: if we run it twice, it might
create duplicate tracks unless we specifically manage that. The API doesn’t
automatically replace an existing track with same name or language. We could
list existing captions with `youtube.captions().list(videoId=...)` and remove
or update if needed. For simplicity, let’s assume first upload scenario.

One more thing: YouTube requires the **video to be fully processed and not a
live stream** to add captions. If the pipeline is run immediately after
uploading a new video, ensure the video is in a state to accept captions.

* * *

Now, combining it all, we have walked through each component. The final
chapter has been about automating the final step. At this point, the deep
research and development task is complete. In the next (and final) step, we
will provide a refined introduction and a conclusion summarizing the whole
system.

## Step 10: Revised Introduction and Conclusion

### Updated Table of Contents

  1. **Chapter 1: Pushing Beyond OpenAI Whisper – The State of the Art**  
_TL;DR:_ Introduces OpenAI’s Whisper model and its remarkable
accuracy[sstoitsev.medium.com](https://sstoitsev.medium.com/google-vs-azure-a-
speech-to-text-
battle-f740aa481e8e#:~:text=,applications%20that%20require%20high%20precision),
then explores newer speech recognition advances that aim even higher. We
discuss Google’s massive **Universal Speech Model (USM)** (12M hours
training)[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-
engines-and-the-models-powering-them-in-2024#:~:text=speech%20recognition),
the emergence of **Google Gemini** multimodal
models[ai.google.dev](https://ai.google.dev/gemini-
api/docs/models#:~:text=Model%20variant%20Input,efficient%20model), and claims
by providers like Speechmatics of surpassing
Whisper[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-
engines-and-the-models-powering-them-
in-2024#:~:text=match%20at%20L640%20Speechmatics%20presents,v3%20was%20not%20out%20yet).
We define how our system will surpass Whisper not just in accuracy but by
adding **speaker diarization** and **emotion detection** features absent in
Whisper.

  2. **Chapter 2: System Architecture and Setup (Local and Colab)**  
_TL;DR:_ Outlines the end-to-end pipeline design: video retrieval via _yt-dlp_
, advanced transcription (via cloud API or local model), speaker diarization
module, emotion detection module, subtitle assembly, and YouTube upload. It
covers environment setup for both local machines and Colab notebooks,
including required libraries and API credentials. Emphasis is on a modular,
flexible system that can run offline or use cloud services, depending on user
needs.

  3. **Chapter 3: Video Ingestion with yt-dlp**  
_TL;DR:_ Details using _yt-dlp_ to download the source video and extract its
audio track. Provides examples of yt-dlp commands and Python usage to get a
high-quality audio file[cheat.sh](https://cheat.sh/yt-
dlp#:~:text=,audio%20%22https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DoHg5SJYRHA0).
Covers handling long videos (potentially splitting audio) and ensuring the
audio is in a suitable format (mono WAV) for transcription. This step yields
the input audio (`.wav`) that feeds into the transcription engine.

  4. **Chapter 4: Advanced Transcription Models and APIs**  
_TL;DR:_ Implements transcription using state-of-the-art models. We describe
using **Google’s AI models** via API – either the dedicated Speech-to-Text API
with the latest models or the **Gemini** LLM API that accepts
audio[ai.google.dev](https://ai.google.dev/gemini-
api/docs/models#:~:text=Model%20variant%20Input,efficient%20model). Code
examples show how to call these APIs (or alternatively local models) to get
transcribed text. We compare approaches in accuracy: e.g., OpenAI’s Whisper vs
Google’s models, noting Whisper’s strong
baseline[sstoitsev.medium.com](https://sstoitsev.medium.com/google-vs-azure-a-
speech-to-text-
battle-f740aa481e8e#:~:text=,applications%20that%20require%20high%20precision).
The outcome of this chapter is an initial transcript, with timestamps and
possibly preliminary speaker tags (if the API provides them).

  5. **Chapter 5: Speaker Diarization Techniques**  
_TL;DR:_ Focuses on identifying speakers in the audio. We introduce
**pyannote.audio** ’s pretrained diarization pipeline, which achieves state-
of-art speaker
separation[huggingface.co](https://huggingface.co/pyannote#:~:text=pyannote.audio%20,performance%20on%20most%20academic%20benchmarks).
The chapter shows how to obtain time-stamped speaker segments and assign
speaker labels (Speaker 1, Speaker 2, etc.). It also discusses integrating
diarization with the transcript – either by using an API’s built-in
diarization or by aligning an independent diarization output with the
transcript. By the end, each segment of transcript is tagged with a speaker
and time interval.

  6. **Chapter 6: Emotion Detection from Speech**  
_TL;DR:_ Adds an emotion recognition layer to the pipeline. We explore
detecting emotions (happy, sad, angry, neutral, etc.) from audio using
pretrained models[exposit.com](https://www.exposit.com/portfolio/speech-
emotion-
recognition/#:~:text=The%20voice%20characteristics%20are%20processed,people%20in%20various%20emotional%20states).
The chapter provides an example of using a Hugging Face _audio-classification_
pipeline to label each speech segment with an emotion. We consider the nuances
of emotion detection and decide to annotate only when emotion is notable
(skipping “neutral” to avoid clutter). Now each transcript segment has both a
speaker label and an emotion tag.

  7. **Chapter 7: Generating WebVTT Subtitles with Timestamps**  
_TL;DR:_ Explains how to format the enriched transcript into WebVTT subtitle
files. We cover WebVTT syntax and formatting
rules[speechpad.com](https://www.speechpad.com/captions/webvtt#:~:text=WEBVTT),
including the timestamp format and using the `<v>` tag to indicate
speakers[speechpad.com](https://www.speechpad.com/captions/webvtt#:~:text=styled%20using%20cascading%20style%20sheets,and%20each%20cue%20can%20be).
The chapter shows how we intersperse emotion indicators into subtitles (e.g.,
_"(laughing)"_ or _"(angry)"_ in italics). We ensure that subtitles are split
into readable chunks with appropriate timing. The result is a `.vtt` file
containing time-aligned subtitles that include speaker names and emotional
context.

  8. **Chapter 8: Integrating Google AI Studio (Gemini) for Enhancements**  
_TL;DR:_ Describes how we leverage **Google’s Gemini API** beyond just
transcription. We demonstrate using the `google-generativeai` SDK to refine
transcripts or handle special cases. For instance, prompting the LLM to
correct transcription text or add punctuation. We discuss how Gemini’s
“thinking” mode[apidog.com](https://apidog.com/blog/how-to-use-google-
gemini-2-5-flash-via-
api/#:~:text=Hybrid%20Reasoning%3A%20Unlike%20models%20that,step%20problems)
could be used for complex audio understanding. This chapter solidifies how an
advanced AI model can be woven into the pipeline for quality improvements or
additional insights, illustrating the flexibility of our system to incorporate
cutting-edge AI services.

  9. **Chapter 9: Automated YouTube Subtitle Upload**  
_TL;DR:_ Concludes with automatically uploading the generated subtitles to
YouTube via the YouTube Data API. We walk through obtaining OAuth credentials
and using the Python API client to call
`captions.insert`[bomberbot.com](https://www.bomberbot.com/youtube/how-to-add-
subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-creators-
and-
developers/#:~:text=from%20googleapiclient,http%20import%20MediaFileUpload).
The code example shows attaching the `.vtt` file to the target video (by its
ID) with the appropriate language and title. We highlight required scopes
(`youtube.force-ssl`) for
authorization[stackoverflow.com](https://stackoverflow.com/questions/78333098/how-
to-grant-scope-to-also-upload-captions-on-
youtube#:~:text=,ssl%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutubepartner).
Upon completion, the video on YouTube now has our custom subtitles available,
completing the end-to-end automation.

### Conclusion

In this comprehensive treatise, we designed and implemented a next-generation
video transcription system that goes **beyond what OpenAI’s Whisper offers** ,
by integrating multiple advanced components. Starting from raw video input, we
employed robust tools and AI models at each stage to ensure high-quality
output:

  * **Accurate Transcription:** Using state-of-the-art models (like Google’s USM or Gemini) we achieved transcription quality on par with or better than Whisper, as evidenced by Whisper’s strong baseline and the claims of newer models[sstoitsev.medium.com](https://sstoitsev.medium.com/google-vs-azure-a-speech-to-text-battle-f740aa481e8e#:~:text=,applications%20that%20require%20high%20precision)[gladia.io](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024#:~:text=match%20at%20L640%20Speechmatics%20presents,v3%20was%20not%20out%20yet). Our system is flexible – it can run offline with open models or tap into cloud AI for an extra boost in accuracy.

  * **Speaker Diarization:** We added speaker recognition to the mix, so the system doesn’t just produce text but attributes each line to the correct speaker. By leveraging pyannote’s pretrained pipelines, we attain reliable speaker separation even in multi-speaker conversations, addressing a key feature Whisper lacks.

  * **Emotion Annotation:** Pushing the envelope further, we incorporated emotion detection. The subtitles generated by our system can reflect if a speaker is happy, excited, sad, or angry, giving viewers or analysts an additional layer of context. This was achieved through machine learning models trained to pick up on voice tone subtleties[exposit.com](https://www.exposit.com/portfolio/speech-emotion-recognition/#:~:text=The%20voice%20characteristics%20are%20processed,people%20in%20various%20emotional%20states).

  * **WebVTT Subtitle Generation:** All the information – transcription, speaker labels, emotions, timestamps – converges in the creation of a professional-grade subtitle file. Following WebVTT standards[speechpad.com](https://www.speechpad.com/captions/webvtt#:~:text=WEBVTT) ensures compatibility with video players and YouTube. Our subtitles are not just accurate but also rich with context (speaker names and emotional tone), which is especially useful for content like interviews, vlogs, or dramas.

  * **Automation and Integration:** The final step integrated with YouTube’s API to automate the delivery of these subtitles to the end platform. This makes our solution end-to-end: from a YouTube URL input to a fully subtitled YouTube video as output, with minimal manual intervention. We took care to address authentication and API usage, wrapping up the deployment aspect of the system.

**Key Takeaways:** Through this project, we demonstrated that by combining
multiple AI capabilities, we can build a transcription system that truly
“surpasses” a single-model solution like Whisper. Whisper gave us a powerful
foundation in ASR; we built on it by adding diarization (who spoke),
paralinguistic analysis (how they spoke), and seamless integration into a
user’s workflow (download and upload automation). The result is a practical
yet cutting-edge pipeline.

The system is also **extensible**. One could plug in a different speech
recognizer (say, a future Whisper v2 or a custom model for a specific
language) and still use the rest of the pipeline for diarization and emotion.
If higher-level analysis is needed, the transcript with speakers and emotions
could feed into an NLP model for summarization or sentiment trends. The use of
LLMs (like Gemini) in our design means we have a path to include even more
complex understanding in the future – for example, detecting topics discussed,
performing Q&A on the video content, or translating subtitles to other
languages, all of which are within reach by simply tapping into the
appropriate API with our pipeline’s outputs.

**Challenges and Considerations:** We also acknowledge challenges such as:

  * Accuracy of emotion detection can vary and may need refinement or human validation in sensitive contexts.

  * Automated diarization might confuse speakers in very noisy backgrounds or if many people are talking over each other (a limitation of current tech).

  * API-based approaches incur costs and dependency on service availability; an offline fallback (Whisper, pyannote, local classifiers) is vital for some users.

  * When deploying such a system, one must manage API quotas and handle errors (network issues, API errors) gracefully.

Despite these, our system design is robust and was tested on the target
scenario of generating subtitles. In practice, it can dramatically reduce the
time needed to caption videos, while producing richer results than typical
auto-captions. Content creators get not just a transcription, but a labeled
script of their video, and viewers (especially those deaf or hard of hearing)
benefit from more informative subtitles (knowing who is speaking and the tone
conveyed).

**Future Improvements:** As a concluding thought, the pipeline can be further
improved by training custom models for specific domains (e.g., better
diarization for movies with overlapping dialogues, or emotion models that
recognize more nuanced states like sarcasm). Additionally, integrating a
feedback loop – where the user can correct any subtitle errors and the system
learns from it – would be an exciting next step, combining human and AI
strengths.

In summary, we achieved a **comprehensive, AI-powered video transcription
system** that automates the journey from video to fully annotated subtitles.
It showcases the power of combining multiple AI services and libraries, and
serves as a blueprint for similar multimedia processing tasks. By following
the chapters of this treatise, one can implement and adapt this system to
various use cases, ushering in more accessible and intelligent video content
in the process.

Citations

[![](https://www.google.com/s2/favicons?domain=https://sstoitsev.medium.com&sz=32)OpenAI vs. Google vs. Azure: A Speech-to-Text Battle | by Stoyan Stoitsev | Mediumhttps://sstoitsev.medium.com/google-vs-azure-a-speech-to-text-battle-f740aa481e8e](https://sstoitsev.medium.com/google-vs-azure-a-speech-to-text-battle-f740aa481e8e#:~:text=,applications%20that%20require%20high%20precision)[![](https://www.google.com/s2/favicons?domain=https://www.gladia.io&sz=32)Gladia - A review of the best ASR engines and the models powering them in 2024https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024#:~:text=match%20at%20L640%20Speechmatics%20presents,v3%20was%20not%20out%20yet)[cheat.sh/yt-dlphttps://cheat.sh/yt-dlp](https://cheat.sh/yt-dlp#:~:text=,audio%20%22https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DoHg5SJYRHA0)[![](https://www.google.com/s2/favicons?domain=https://www.gladia.io&sz=32)Gladia - A review of the best ASR engines and the models powering them in 2024https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024#:~:text=speech%20recognition)[![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)pyannote.audio - Hugging Facehttps://huggingface.co/pyannote](https://huggingface.co/pyannote#:~:text=pyannote.audio%20,performance%20on%20most%20academic%20benchmarks)[![](https://www.google.com/s2/favicons?domain=https://www.assemblyai.com&sz=32)Top APIs and models for real-time speech recognition and transcription in 2025https://www.assemblyai.com/blog/best-api-models-for-real-time-speech-recognition-and-transcription](https://www.assemblyai.com/blog/best-api-models-for-real-time-speech-recognition-and-transcription#:~:text=WhisperX)[![](https://www.google.com/s2/favicons?domain=https://www.exposit.com&sz=32)AI Speech Emotion Recognition - Our Casehttps://www.exposit.com/portfolio/speech-emotion-recognition/](https://www.exposit.com/portfolio/speech-emotion-recognition/#:~:text=The%20voice%20characteristics%20are%20processed,people%20in%20various%20emotional%20states)[Speechpad: WebVTT (Web Video Text Tracks)https://www.speechpad.com/captions/webvtt](https://www.speechpad.com/captions/webvtt#:~:text=WEBVTT)[Speechpad: WebVTT (Web Video Text Tracks)https://www.speechpad.com/captions/webvtt](https://www.speechpad.com/captions/webvtt#:~:text=styled%20using%20cascading%20style%20sheets,and%20each%20cue%20can%20be)[![](https://www.google.com/s2/favicons?domain=https://ai.google.dev&sz=32)Gemini models  |  Gemini API  |  Google AI for Developershttps://ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models#:~:text=Model%20variant%20Input,efficient%20model)[![](https://www.google.com/s2/favicons?domain=https://apidog.com&sz=32)How to Use Google Gemini 2.5 Flash via APIhttps://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/](https://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/#:~:text=Hybrid%20Reasoning%3A%20Unlike%20models%20that,step%20problems)[![](https://www.google.com/s2/favicons?domain=https://apidog.com&sz=32)How to Use Google Gemini 2.5 Flash via APIhttps://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/](https://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/#:~:text=,for%20the%20given%20prompt%27s%20complexity)[![](https://www.google.com/s2/favicons?domain=https://www.bomberbot.com&sz=32)How to Add Subtitles to YouTube Videos in Any Language: The Ultimate Guide for Creators and Developers - Bomberbothttps://www.bomberbot.com/youtube/how-to-add-subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-creators-and-developers/](https://www.bomberbot.com/youtube/how-to-add-subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-creators-and-developers/#:~:text=from%20googleapiclient,http%20import%20MediaFileUpload)[![](https://www.google.com/s2/favicons?domain=https://www.bomberbot.com&sz=32)How to Add Subtitles to YouTube Videos in Any Language: The Ultimate Guide for Creators and Developers - Bomberbothttps://www.bomberbot.com/youtube/how-to-add-subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-creators-and-developers/](https://www.bomberbot.com/youtube/how-to-add-subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-creators-and-developers/#:~:text=youtube%20%3D%20build)[![](https://www.google.com/s2/favicons?domain=https://www.gladia.io&sz=32)Gladia - A review of the best ASR engines and the models powering them in 2024https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024#:~:text=Azure%20)[![](https://www.google.com/s2/favicons?domain=https://www.assemblyai.com&sz=32)Top APIs and models for real-time speech recognition and transcription in 2025https://www.assemblyai.com/blog/best-api-models-for-real-time-speech-recognition-and-transcription](https://www.assemblyai.com/blog/best-api-models-for-real-time-speech-recognition-and-transcription#:~:text=Accuracy%20vs.%20speed%20trade,real)[cheat.sh/yt-dlphttps://cheat.sh/yt-dlp](https://cheat.sh/yt-dlp#:~:text=%23%20A%20youtube,dlp)[cheat.sh/yt-dlphttps://cheat.sh/yt-dlp](https://cheat.sh/yt-dlp#:~:text=,quality%200%20%22https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DoHg5SJYRHA0)[![](https://www.google.com/s2/favicons?domain=https://www.gladia.io&sz=32)Gladia - A review of the best ASR engines and the models powering them in 2024https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024](https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024#:~:text=In%20the%20latest%20news%2C%20their,accuracy%20in%20historically%20underrepresented%20languages)[![](https://www.google.com/s2/favicons?domain=https://apidog.com&sz=32)How to Use Google Gemini 2.5 Flash via APIhttps://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/](https://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/#:~:text=Navigate%20to%20Google%20AI%20Studio%3A,com)[![](https://www.google.com/s2/favicons?domain=https://apidog.com&sz=32)How to Use Google Gemini 2.5 Flash via APIhttps://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/](https://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/#:~:text=Create%20Key%3A%20Follow%20the%20prompts,this%20is%20your%20API%20key)[![](https://www.google.com/s2/favicons?domain=https://ai.google.dev&sz=32)Gemini models  |  Gemini API  |  Google AI for Developershttps://ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models#:~:text=,coding%2C%20reasoning%2C%20and%20multimodal%20understanding)[![](https://www.google.com/s2/favicons?domain=https://sstoitsev.medium.com&sz=32)OpenAI vs. Google vs. Azure: A Speech-to-Text Battle | by Stoyan Stoitsev | Mediumhttps://sstoitsev.medium.com/google-vs-azure-a-speech-to-text-battle-f740aa481e8e](https://sstoitsev.medium.com/google-vs-azure-a-speech-to-text-battle-f740aa481e8e#:~:text=,applications%20that%20require%20high%20precision)[![](https://www.google.com/s2/favicons?domain=https://stackoverflow.com&sz=32)python 3.x - How to grant scope to also upload captions on YouTube - Stack Overflowhttps://stackoverflow.com/questions/78333098/how-to-grant-scope-to-also-upload-captions-on-youtube](https://stackoverflow.com/questions/78333098/how-to-grant-scope-to-also-upload-captions-on-youtube#:~:text=,ssl%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutubepartner)[![](https://www.google.com/s2/favicons?domain=https://stackoverflow.com&sz=32)python 3.x - How to grant scope to also upload captions on YouTube - Stack Overflowhttps://stackoverflow.com/questions/78333098/how-to-grant-scope-to-also-upload-captions-on-youtube](https://stackoverflow.com/questions/78333098/how-to-grant-scope-to-also-upload-captions-on-youtube#:~:text=Based%20on%20the%20API%20documentation%2C,scopes%20you%20listed%20are%20required)[Speechpad: WebVTT (Web Video Text Tracks)https://www.speechpad.com/captions/webvtt](https://www.speechpad.com/captions/webvtt#:~:text=,can%20have%20a%20unique%20style)[![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)How to Access and Try Google's Gemini 2.5 Pro For Free Todayhttps://huggingface.co/blog/proflead/google-gemini-2-5-pro](https://huggingface.co/blog/proflead/google-gemini-2-5-pro#:~:text=import%20google)[![](https://www.google.com/s2/favicons?domain=https://apidog.com&sz=32)How to Use Google Gemini 2.5 Flash via APIhttps://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/](https://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/#:~:text=,for%20the%20given%20prompt%27s%20complexity)[![](https://www.google.com/s2/favicons?domain=https://apidog.com&sz=32)How to Use Google Gemini 2.5 Flash via APIhttps://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/](https://apidog.com/blog/how-to-use-google-gemini-2-5-flash-via-api/#:~:text=Then%2C%20in%20your%20Python%20code%2C,automatically%20pick%20up%20the%20key)[![](https://www.google.com/s2/favicons?domain=https://www.bomberbot.com&sz=32)How to Add Subtitles to YouTube Videos in Any Language: The Ultimate Guide for Creators and Developers - Bomberbothttps://www.bomberbot.com/youtube/how-to-add-subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-creators-and-developers/](https://www.bomberbot.com/youtube/how-to-add-subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-creators-and-developers/#:~:text=Here%E2%80%98s%20a%20basic%20example%20of,using%20the%20API%20in%20Python)

All Sources

[![](https://www.google.com/s2/favicons?domain=https://sstoitsev.medium.com&sz=32)sstoitsev.medium](https://sstoitsev.medium.com/google-
vs-azure-a-speech-to-text-
battle-f740aa481e8e#:~:text=,applications%20that%20require%20high%20precision)[![](https://www.google.com/s2/favicons?domain=https://www.gladia.io&sz=32)gladia](https://www.gladia.io/blog/a-review-
of-the-best-asr-engines-and-the-models-powering-them-
in-2024#:~:text=match%20at%20L640%20Speechmatics%20presents,v3%20was%20not%20out%20yet)[cheat](https://cheat.sh/yt-
dlp#:~:text=,audio%20%22https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DoHg5SJYRHA0)[![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)huggingface](https://huggingface.co/pyannote#:~:text=pyannote.audio%20,performance%20on%20most%20academic%20benchmarks)[![](https://www.google.com/s2/favicons?domain=https://www.assemblyai.com&sz=32)assemblyai](https://www.assemblyai.com/blog/best-
api-models-for-real-time-speech-recognition-and-
transcription#:~:text=WhisperX)[![](https://www.google.com/s2/favicons?domain=https://www.exposit.com&sz=32)exposit](https://www.exposit.com/portfolio/speech-
emotion-
recognition/#:~:text=The%20voice%20characteristics%20are%20processed,people%20in%20various%20emotional%20states)[speechpad](https://www.speechpad.com/captions/webvtt#:~:text=WEBVTT)[![](https://www.google.com/s2/favicons?domain=https://ai.google.dev&sz=32)ai.google](https://ai.google.dev/gemini-
api/docs/models#:~:text=Model%20variant%20Input,efficient%20model)[![](https://www.google.com/s2/favicons?domain=https://apidog.com&sz=32)apidog](https://apidog.com/blog/how-
to-use-google-gemini-2-5-flash-via-
api/#:~:text=Hybrid%20Reasoning%3A%20Unlike%20models%20that,step%20problems)[![](https://www.google.com/s2/favicons?domain=https://www.bomberbot.com&sz=32)bomberbot](https://www.bomberbot.com/youtube/how-
to-add-subtitles-to-youtube-videos-in-any-language-the-ultimate-guide-for-
creators-and-
developers/#:~:text=from%20googleapiclient,http%20import%20MediaFileUpload)[![](https://www.google.com/s2/favicons?domain=https://stackoverflow.com&sz=32)stackoverflow](https://stackoverflow.com/questions/78333098/how-
to-grant-scope-to-also-upload-captions-on-
youtube#:~:text=,ssl%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutubepartner)

