---
title: FFMPEG note
tags:
  - FFMPEG
categories:
  - 软件
  - FFMPEG
abbrlink: '97736837'
date: 2021-09-14 14:50:37
---

# FFMPEG

## 常用命令

### 01.下载，配置

用的系统是 Ubuntu 可以直接 apt-get

```shell
sudo apt-get install ffmpeg
```

windows 可以去官网下载 [windows build](https://www.gyan.dev/ffmpeg/builds/)

### 02.简介，上手(FFmpeg FFprobe FFplay)

(1) 查看 ffmpeg 的帮助说明，提供的指令。建议将其中的命令大致看看，在本笔记最后附有该帮助说明

```shell
ffmpeg -h
```

(2) 播放媒体的指令

```shell
ffplay video.mp4
ffplay music.mp3
```

(3) 常用快捷键

按键"Q"或"Esc"：退出媒体播放
键盘方向键：媒体播放的前进后退
点击鼠标右键：拖动到该播放位置
按键"F"：全屏
按键"P"或空格键：暂停
按键"W":切换显示模式

(4) 查看媒体参数信息

```shell
ffprobe video.mp4
```

输出形如下面的内容

```shell
Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709), 1440x720, 163 kb/s, 30 fps, 30 tbr, 16k tbn (default)
    Metadata:
      handler_name    : VideoHandler
      vendor_id       : [0][0][0][0]
Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 128 kb/s (default)
    Metadata:
      handler_name    : SoundHandler
      vendor_id       : [0][0][0][0]
```

可以看到视频为 h264 编码，分辨率 1440x720、比特率(码率) 163 kb/s、帧率 30 fps。音频为 acc 编码，音频采样率 44100 Hz，比特率 128 kb/s

### 03.转换格式(文件格式,封装格式)

(1) 文件名可以是中英文，但不能有空格

(2) **转换格式**

```shell
ffmpeg -i video.mp4 video_avi.avi
```

ffmpeg 使用 -i 参数来表示输入文件，也可以用 -f 参数指定输出格式，但为了省事儿好像也可以不用指定，ffmpeg 会根据输出文件名的后缀自动识别

### 04.提取音视频

(1) 单独提取视频（不含音频流）

```shell
ffmpeg -i video.mp4 -vcodec copy -an video_silent.mp4
```

(2) 单独提取音频（不含视频流）

```shell
ffmpeg -i video.mp4 -vn -acodec copy video_novideo.m4a
```

具备多个音频流的，如

```shell
Stream #0:2[0x81]:Audio:ac3,48000Hz,5.1,s16,384kb/s
Stream #0:3[0x82]:Audio:ac3,48000Hz,5.1,s16,384kb/s
Stream #0:4[0x80]:Audio:ac3,48000Hz,5.1,s16,448kb/s
```

针对性的单一的提取，例如提取第2条，用指令： -map 0:3

(3) 合并音视频

```shell
ffmpeg -i video_novideo.m4a -i video_silent.mp4 -c copy video_merge.mp4
```

-c 为 -codec 的缩写，传入参数 copy 即使用原视频/音频编码器，这样会大大加快处理时间而不用重新转码。一般在 codec 前/后有 a/v 即代表音频/视频

### 05.截取，连接音视频

(1) 截取

```shell
ffmpeg -i music.mp3 -ss 00:00:30 -to 00:02:00 -acodec copy music_cutout.mp3
```

输入时间可以是如上的 `时:分:秒`，也可以是 `分:秒`，也可以直接输入多少秒，秒可以是浮点数。还可以截取指定长度的音视频，如截取60秒

```shell
ffmpeg -i music.mp3 -ss 00:00:30 -t 60 -acodec copy music_cutout60s.mp3
```

-sseof time_offset: 开始时间从媒体末尾开始计算，传入参数为复数

```shell
ffmpeg -i in.mp4 -ss 00:01:00 -to 00:01:10 -c copy out.mp4
ffmpeg -ss 00:01:00 -i in.mp4 -to 00:01:10 -c copy out.mp4
ffmpeg -ss 00:01:00 -i in.mp4 -to 00:01:10 -c copy -copyts out.mp4
# 从末尾往前 10s 截取 5s
ffmpeg -sseof -10 -t 5 -i in.mp4 -c copy out.mp4
```

把-ss放到-i之前，启用了关键帧技术，加速操作。但截取的时间段不一定准确。可用最后一条指令，保留时间戳，保证时间准确。

(2) 连接音视频

我推荐使用下面的方法

1. 新建一个 list.txt 文件，里面包含了需要连接的音视频。格式如下

   ```txt
   file '/path/to/file1'
   file '/path/to/file2'
   file '/path/to/file3'
   ```

2. 可以使用 -f concat，来合并音视频

   ```shell
   ffmpeg -f concat -i mylist.txt -c copy output.mp4
   ```

对于精细的音视频连接并不推荐使用 ffmeg

### 06.图片视频转换，水印，gif

**(1) 图片视频转换**

截取第7秒第1帧的画面

```shell
ffmpeg -i video.mp4 -ss 7 -vframes 1 video_image.jpg
```

视频分离成图片

```shell
ffmpeg -i input_test.mp4 -r 1 -f image2 output_image-%03d.png
```

图片也能合成为视频

```shell
ffmpeg -f image2 -r 15 -i output_image-%03d.png output_test.mp4
```

这里的 `-r` 代表 rate 可以理解为帧率。`output_image-%03d.png` 是图片的命名格式，这种形式在 python 格式化字符串中经常看到

注意，这里选择 png 格式而不是 jpg 格式是因为 png 为无损压缩图，这样在视频和图片的转换当中，就不会有损失

(2) 水印

```shell
ffmpeg -i video.mp4 -i qt.png -filter_complex "overlay=20:80" video_watermark.mp4
```

(3) 截取动图

```shell
ffmpeg -i video.mp4 -ss 7.5 -to 8.5 -s 640x320 -r 15 video_gif.gif
```

## 补充：视频编码

### 01.改变编码 上(编码,音频转码)

(1)查看编解码器

```shell
ffmpeg -codecs
```

(2)**网站常用编码**

MP4封装：H264视频编码+ACC音频编码
WebM封装：VP8视频编码+Vorbis音频编码
OGG封装：Theora视频编码+Vorbis音频编码

(3)**无损编码格式.flac转换编码**

```shell
ffmpeg -i music_flac.flac -acodec libmp3lame -ar 44100 -ab 320k -ac 2 music_flac_mp3.mp3
```

**说明：**

* acodec:audio Coder Decoder 音频编码解码器
* libmp3lame:mp3解码器
* ar:audio rate：音频采样率
* **44100:设置音频的采样率44100。若不输入，默认用原音频的采样率**
* ab:audio bit rate 音频比特率
* **320k：设置音频的比特率。若不输入，默认128K**
* ac: aduio channels 音频声道
* 2:声道数。若不输入，默认采用源音频的声道数

概括：设置格式的基本套路-先是指名属性，然后跟着新的属性值

(4)查看结果属性

```shell
ffprobe music_flac_mp3.mp3
```

### 02.改变编码 中(视频压制)

(1)视频转码

```shell
ffmpeg -i video.mp4 -s 1920x1080 -pix_fmt yuv420p -vcodec libx264 -preset medium -profile:v high -level:v 4.1 -crf 23 -acodec aac -ar 44100 -ac 2 -b:a 128k video_avi.avi
```

**说明:**

* **-s 1920x1080：缩放视频新尺寸(size)**
* -pix_fmt yuv420p：pixel format,用来设置视频颜色空间。参数查询：ffmpeg -pix_fmts
* -vcodec libx264：video Coder Decoder，视频编码解码器
* -preset medium: 编码器预设。参数：ultrafast,superfast,veryfast,faster,fast,medium,slow,slower,veryslow,placebo
* -profile:v high :编码器配置，与压缩比有关。实时通讯-baseline,流媒体-main,超清视频-high
* -level:v 4.1 ：对编码器设置的具体规范和限制，权衡压缩比和画质。
* -crf 23 ：设置码率控制模式。constant rate factor-恒定速率因子模式。范围0\~51,默认23。数值越小，画质越高。一般在8~28做出选择。
* **-r 30 :设置视频帧率**
* -acodec aac :audio Coder Decoder-音频编码解码器
* -b:a 128k :音频比特率.大多数网站限制音频比特率128k,129k
  其他参考上一个教程

### 03.改变编码 下(码率控制模式)

ffmpeg支持的码率控制模式：-qp -crf -b

(1)  -qp :constant quantizer,恒定量化器模式 

无损压缩的例子（快速编码）

```shell
ffmpeg -i input -vcodec libx264 -preset ultrafast -qp 0 output.mkv
```

无损压缩的例子（高压缩比）

```shell
ffmpeg -i input -vcodec libx264 -preset veryslow -qp 0 output.mkv
```

(2) -crf :constant rate factor,恒定速率因子模式

(3) -b ：bitrate,固定目标码率模式。一般不建议使用

3种模式默认单遍编码

VBR(Variable Bit Rate/动态比特率) 例子

```shell
ffmpeg -i input -vcodec libx264 -preset veryslow output
```

ABR(Average Bit Rate/平均比特率) 例子

```shell
ffmpeg -i input -vcodec libx264 -preset veryslow -b:v 3000k output
```

CBR(Constant Bit Rate/恒定比特率) 例子

```shell
... -b:v 4000k -minrate 4000k -maxrate 4000k -bufsize 1835k ...
```

## 参考链接

笔记来源：https://www.bilibili.com/video/av40146374

官方教程： http://ffmpeg.org/ffmpeg-all.html

博客：https://www.jianshu.com/p/f07f0be088d0

## 帮助文档，用于查询

下面是 ffmpeg 的帮助文档，用于查询

```shell
ffmpeg version 2021-09-08-git-5e7e2e5031-full_build-www.gyan.dev Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 10.3.0 (Rev5, Built by MSYS2 project)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-fontconfig --enable-iconv --enable-gnutls --enable-libxml2 --enable-gmp --enable-lzma --enable-libsnappy --enable-zlib --enable-librist --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-libbluray --enable-libcaca --enable-sdl2 --enable-libdav1d --enable-libzvbi --enable-librav1e --enable-libsvtav1 --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxvid --enable-libaom --enable-libopenjpeg --enable-libvpx --enable-libass --enable-frei0r --enable-libfreetype --enable-libfribidi --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-ffnvcodec --enable-nvdec --enable-nvenc --enable-d3d11va --enable-dxva2 --enable-libmfx --enable-libglslang --enable-vulkan --enable-opencl --enable-libcdio --enable-libgme --enable-libmodplug --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libshine --enable-libtheora --enable-libtwolame --enable-libvo-amrwbenc --enable-libilbc --enable-libgsm --enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-ladspa --enable-libbs2b --enable-libflite --enable-libmysofa --enable-librubberband --enable-libsoxr --enable-chromaprint
  libavutil      57.  4.101 / 57.  4.101
  libavcodec     59.  7.102 / 59.  7.102
  libavformat    59.  5.100 / 59.  5.100
  libavdevice    59.  0.101 / 59.  0.101
  libavfilter     8.  7.101 /  8.  7.101
  libswscale      6.  1.100 /  6.  1.100
  libswresample   4.  0.100 /  4.  0.100
  libpostproc    56.  0.100 / 56.  0.100
Hyper fast Audio and Video encoder
usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...

Getting help:
    -h      -- print basic options
    -h long -- print more options
    -h full -- print all options (including all format and codec specific options, very long)
    -h type=name -- print all options for the named decoder/encoder/demuxer/muxer/filter/bsf/protocol
    See man ffmpeg for detailed description of the options.

Print help / information / capabilities:
-L                  show license
-h topic            show help
-? topic            show help
-help topic         show help
--help topic        show help
-version            show version
-buildconf          show build configuration
-formats            show available formats
-muxers             show available muxers
-demuxers           show available demuxers
-devices            show available devices
-codecs             show available codecs
-decoders           show available decoders
-encoders           show available encoders
-bsfs               show available bit stream filters
-protocols          show available protocols
-filters            show available filters
-pix_fmts           show available pixel formats
-layouts            show standard channel layouts
-sample_fmts        show available audio sample formats
-colors             show available color names
-sources device     list sources of the input device
-sinks device       list sinks of the output device
-hwaccels           show available HW acceleration methods

Global options (affect whole program instead of just one file):
-loglevel loglevel  set logging level
-v loglevel         set logging level
-report             generate a report
-max_alloc bytes    set maximum size of a single allocated block
-y                  overwrite output files
-n                  never overwrite output files
-ignore_unknown     Ignore unknown stream types
-filter_threads     number of non-complex filter threads
-filter_complex_threads  number of threads for -filter_complex
-stats              print progress report during encoding
-max_error_rate maximum error rate  ratio of decoding errors (0.0: no errors, 1.0: 100% errors) above which ffmpeg returns an error instead of success.
-bits_per_raw_sample number  set the number of bits per raw sample
-vol volume         change audio volume (256=normal)

Per-file main options:
-f fmt              force format
-c codec            codec name
-codec codec        codec name
-pre preset         preset name
-map_metadata outfile[,metadata]:infile[,metadata]  set metadata information of outfile from infile
-t duration         record or transcode "duration" seconds of audio/video
-to time_stop       record or transcode stop time
-fs limit_size      set the limit file size in bytes
-ss time_off        set the start time offset
-sseof time_off     set the start time offset relative to EOF
-seek_timestamp     enable/disable seeking by timestamp with -ss
-timestamp time     set the recording timestamp ('now' to set the current time)
-metadata string=string  add metadata
-program title=string:st=number...  add program with specified streams
-target type        specify target file type ("vcd", "svcd", "dvd", "dv" or "dv50" with optional prefixes "pal-", "ntsc-" or "film-")
-apad               audio pad
-frames number      set the number of frames to output
-filter filter_graph  set stream filtergraph
-filter_script filename  read stream filtergraph description from a file
-reinit_filter      reinit filtergraph on input parameter changes
-discard            discard
-disposition        disposition

Video options:
-vframes number     set the number of video frames to output
-r rate             set frame rate (Hz value, fraction or abbreviation)
-fpsmax rate        set max frame rate (Hz value, fraction or abbreviation)
-s size             set frame size (WxH or abbreviation)
-aspect aspect      set aspect ratio (4:3, 16:9 or 1.3333, 1.7777)
-bits_per_raw_sample number  set the number of bits per raw sample
-vn                 disable video
-vcodec codec       force video codec ('copy' to copy stream)
-timecode hh:mm:ss[:;.]ff  set initial TimeCode value.
-pass n             select the pass number (1 to 3)
-vf filter_graph    set video filters
-ab bitrate         audio bitrate (please use -b:a)
-b bitrate          video bitrate (please use -b:v)
-dn                 disable data

Audio options:
-aframes number     set the number of audio frames to output
-aq quality         set audio quality (codec-specific)
-ar rate            set audio sampling rate (in Hz)
-ac channels        set number of audio channels
-an                 disable audio
-acodec codec       force audio codec ('copy' to copy stream)
-vol volume         change audio volume (256=normal)
-af filter_graph    set audio filters

Subtitle options:
-s size             set frame size (WxH or abbreviation)
-sn                 disable subtitle
-scodec codec       force subtitle codec ('copy' to copy stream)
-stag fourcc/tag    force subtitle tag/fourcc
-fix_sub_duration   fix subtitles duration
-canvas_size size   set canvas size (WxH or abbreviation)
-spre preset        set the subtitle options to the indicated preset
```

