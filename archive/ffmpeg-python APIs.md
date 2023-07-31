# ffmpeg-python APIs

真不熟悉 ffmpeg 命令行操作，每次都要去翻看笔记，但是 ffmpeg 真的很强大！如果有一个好用的 api 来方便处理就太好了

[ffmpeg-python](https://github.com/kkroening/ffmpeg-python) 能够较好地解决这个需求🥳

## 功能整理

整理常用核心功能！在此之前先说一个概念 `filter`，这个概念在 ffmpeg 中不是过滤器的意思，而是指一个对视频的操作或者变换

### 格式转换

```python
import ffmpeg
stream = ffmpeg.input('input.mp4')
stream = ffmpeg.output(stream, 'output.mkv')
ffmpeg.run(stream)
```

也可以进行连续操作

```python
import ffmpeg
command = (
    ffmpeg.input('input.mp4').
    output('output.mkv').
    run()
)
```

### 音频与视频

ffmpeg 返回对象拥有`video & audio` 两个属性，代表视频流和音频流，可以通过导出视频流的方式直接进行操作

```python
import ffmpeg

input_file = 'input.mp4'
output_file = 'output.mp4'

ffmpeg_command = (
    ffmpeg
    .input(input_file)
    .video
    .output(output_file)
    .overwrite_output() # overwrite the existing file without asking
    .run()
)
```

### 截取与连接

使用 `trim` 和 `concat` 命令对视频进行截取和连接

```python
import ffmpeg

input_file = 'input.mp4'
output_file = 'output.mp4'
start_time = '00:00:00.90'	# start_frame
end_time = '00:00:05'
duration = 5

(
    ffmpeg
    .input(input_file, ss=start_time, t=duration, to=end_time)
    .output(output_file)
    .overwrite_output()
    .run()
)
```

连接视频

```python
import ffmpeg

input_files = ['input1.mp4', 'input2.mp4']
output_file = 'output.mp4'

input_args = []
for file in input_files:
    input_args.append(ffmpeg.input(file))
    
joined = ffmpeg.concat(*input_args)
(
    joined
    .output(output_file)
    .overwrite_output()
    .run()
)
```

### 画幅剪裁

使用 crop

```python
import ffmpeg

input_file = 'input.mp4'
output_file = 'output.mp4'

(ffmpeg.input(input_file)
       .crop(x=0, y=0, width=1280, height=720)
       .output(output_file)
       .overwrite_output()
       .run())
```

### 变速

使用 filter fps

```python
import ffmpeg

input_file = 'input.mp4'
output_file = 'output.mp4'

(
    ffmpeg
    .input('dummy.mp4')
    .filter('fps', fps=25, round='up')
    .output('dummy2.mp4')
    .run()
)

import ffmpeg
# Example usage: Change video speed to half (0.5x)
spped = 0.5
command = ffmpeg.input(input_file)
        .filter('setpts', f'{1/speed}*PTS')
        .output(output_file)
        .run(overwrite_output=True)
```

### 图片与视频转换

```python
import ffmpeg

input_file = 'input.mp4'
output_folder = 'output/'

# video to picture
(
    ffmpeg
    .input(input_file)
    .output(output_folder + 'image_%05d.png', start_number=0)
    .run()
)

# picture to video
image_folder = 'input/'
output_file = 'output.mp4'
framerate = 30

(
    ffmpeg
    .input(image_folder + 'image_%05d.png', framerate=framerate)
    .output(output_file)
    .run()
)
```

### 添加封面与水印

为音频添加封面

```python
import ffmpeg

input_image_path = "image.png"
input_audio_path = "output.mp3"
output_video_path = "video.mp4"

audio_duration = float(ffmpeg.probe(input_audio_path)['format']['duration'])

video_stream = ffmpeg.input(input_image_path, loop=1)	# loop is neccessary for picture stream
audio_stream = ffmpeg.input(input_audio_path)

(
    ffmpeg
    .output(video_stream, audio_stream, output_video_path, t=audio_duration)
    .overwrite_output()
    .run()
)
```

为视频添加水印

```python
import ffmpeg

input_video_path = "input.mp4"
input_logo_path = "logo.png"
output_video_path = "video.mp4"

input_video = ffmpeg.input(input_video_path)

logo = ffmpeg.input(input_logo_path).filter("scale", 200, -1)	# width 200, height adaptive

overlay_video = ffmpeg.overlay(input_video, logo, x=0, y=0)

(
    ffmpeg
    .output(overlay_video, output_video_path)
    .overwrite_output()
    .run()
)
```

### 音频视频信息

使用 probe 方法

```python
import ffmpeg
from pprint import pprint

image_file = 'image.png'
audio_file = 'output.mp3'

# information as a dict
pprint(ffmpeg.probe(audio_file))
```

输出

```txt
{'format': {'bit_rate': '128466',
            'duration': '7.053061',
            'filename': 'output.mp3',
            'format_long_name': 'MP2/3 (MPEG audio layer 2/3)',
            'format_name': 'mp3',
            'nb_programs': 0,
            'nb_streams': 1,
            'probe_score': 51,
            'size': '113260',
            'start_time': '0.025057',
            'tags': {'compatible_brands': 'isomiso2avc1mp41',
                     'description': 'Packed by Bilibili XCoder v2.0.2',
                     'encoder': 'Lavf60.4.100',
                     'major_brand': 'isom',
                     'minor_version': '512'}},
 'streams': [{'avg_frame_rate': '0/0',
              'bit_rate': '128000',
              'bits_per_sample': 0,
              'channel_layout': 'stereo',
              'channels': 2,
              'codec_long_name': 'MP3 (MPEG audio layer 3)',
              'codec_name': 'mp3',
              'codec_tag': '0x0000',
              'codec_tag_string': '[0][0][0][0]',
              'codec_type': 'audio',
              'disposition': {'attached_pic': 0,
                              'captions': 0,
                              'clean_effects': 0,
                              'comment': 0,
                              'default': 0,
                              'dependent': 0,
                              'descriptions': 0,
                              'dub': 0,
                              'forced': 0,
                              'hearing_impaired': 0,
                              'karaoke': 0,
                              'lyrics': 0,
                              'metadata': 0,
                              'original': 0,
                              'still_image': 0,
                              'timed_thumbnails': 0,
                              'visual_impaired': 0},
              'duration': '7.053061',
              'duration_ts': 99532800,
              'index': 0,
              'initial_padding': 0,
              'r_frame_rate': '0/0',
              'sample_fmt': 'fltp',
              'sample_rate': '44100',
              'start_pts': 353600,
              'start_time': '0.025057',
              'tags': {'encoder': 'Lavc60.6.'},
              'time_base': '1/14112000'}]}
```

