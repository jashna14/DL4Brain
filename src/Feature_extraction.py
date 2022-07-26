import os
import sys
import time
import argparse
import logging
import math
import gc
import json

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.data import VideoClsCustom
from gluoncv.utils.filesystem import try_import_decord
from moviepy.editor import *
import librosa
import librosa.display
import IPython.display as ipd




def read_data(video_name, transform,video_utils):
    new_width = 340
    new_height = 256
    new_length = 32
    new_step = 1
    video_loader = True
    use_decord = True
    slowfast = True
    slow_temporal_stride = 16
    fast_temporal_stride = 2
    input_size = 224
    num_segments = 1
    num_crop = 1
    
    decord = try_import_decord()
    decord_vr = decord.VideoReader(video_name, width=new_width, height=new_height)
    duration = len(decord_vr)

    skip_length = new_length * new_stepyTorch
    segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

    if video_loader:
        if slowfast:
            clip_input = video_utils._video_TSN_decord_slowfast_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = video_utils._video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
    else:
        raise RuntimeError('We only support video-based inference.')

    clip_input = transform(clip_input)

    if slowfast:
        sparse_sampels = len(clip_input) // (num_segments * num_crop)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, input_size, input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (new_length, 3, input_size, input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if new_length == 1:
        clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

    return nd.array(clip_input)



def extract_text_features(tr_time,transcript_path):
    tr_words = []
    f = open(transcript_path,'r')
    lines = f.readlines()
    cur_time = tr_time
    words = []

    for line in lines:
        line = line.strip()
        if(line == ""):
            continue
        else:
            line = line.split(':')
            time = float(line[0])
            word = re.sub(r'[^a-zA-Z ]+'," ",line[1])
            word = word.strip()
            if word == "":
                continue
            else:
                if time <= cur_time:
                    words.extend(word.split(' '))
                else:
                    tr_words.append(words)
                    cur_time += tr_time
                    words = []
                    while(time > cur_time):
                        tr_words.append([])
                        cur_time += tr_time
                    words.extend(word.split(' '))

    if(tr_words[-1] != words):
        tr_words.append(words)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    text_feats = []
    for words in tr_words:
        if len(words) == 0:
            text_feats.append(np.array([]))
            continue
        input_ids = torch.tensor(tokenizer.encode(words)).unsqueeze(0)
        outputs = model(input_ids)
        cls = outputs[0][0][0]
        text_feats.append(np.array(cls.tolist()))

    return text_feats

def extract_audio_features():
    y,sr = librosa.load("./audio.mp3")
    y = librosa.to_mono(y)
    mfcc = librosa.feature.mfcc(y,sr=sr)
    mfcc = np.mean(mfcc,axis = 1)
    return mfcc

def extract_video_features():

    gc.set_threshold(100, 5, 5)
    gpu_id = 0
    input_size = 224
    use_pretrained = True
    hashtag = ''
    num_classes = 400
    model  = 'i3d_resnet50_v1_kinetics400'
    num_segments = 1
    resume_params = ''
    video_path = "./clip.mp4"
    new_width = 340
    new_height = 256
    new_length = 32
    new_step = 1
    video_loader = True
    use_decord = True
    slowfast = True
    slow_temporal_stride = 16
    fast_temporal_stride = 2
    data_aug = 'v1'



  # set env
    if gpu_id == -1:
        context = mx.cpu()
    else:
        context = mx.gpu(gpu_id)

  # get data preprocess
    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    transform_test = video.VideoGroupValTransform(size= input_size, mean=image_norm_mean, std=image_norm_std)
    num_crop = 1

  # get model
    if use_pretrained and len(hashtag) > 0:
        use_pretrained = hashtag
    classes = num_classes
    model_name = model
    net = get_model(name=model_name, nclass=classes, pretrained= use_pretrained,feat_ext=True, num_segments=num_segments, num_crop= num_crop)
    net.cast('float32')
    net.collect_params().reset_ctx(context)
    if resume_params != '' and not use_pretrained:
        net.load_parameters(resume_params, ctx=context)


    video_utils = VideoClsCustom(
                                root='',
                                setting='',
                                num_segments=num_segments,
                                num_crop=num_crop,
                                new_length=new_length,
                                new_step=new_step,
                                new_width=new_width,
                                new_height=new_height,
                                video_loader=video_loader,
                                use_decord=use_decord,
                                slowfast=slowfast,
                                slow_temporal_stride=slow_temporal_stride,
                                fast_temporal_stride=fast_temporal_stride,
                                data_aug=data_aug,
                                lazy_init=True)


    video_data = read_data(video_path, transform_test,video_utils)
    video_input = video_data.as_in_context(context)
    video_feat = net(video_input.astype('float32', copy=False))

    return video_feat.asnumpy()


def extract_stimuli_features(video_path,tr_time,transcript_path):
    
    clip = VideoFileClip(video_path)
    clip.duration

    TRs = []
    video_feats = []
    audio_feats = []


    i=0
    while i < clip.duration:
        TRs.append(i)
        i += tr_time

    if(TRs[-1] != clip.duration):
        TRs.append(clip.duration)

    for i in range(0,len(TRs)-1):
        if(i%20 == 0):
            print(str(i) + "/" + str(len(TRs)-1))
        clip1 = clip.subclip(TRs[i], TRs[i+1])
        clip1.write_videofile("./clip.mp4",verbose=False, progress_bar=False)
        audio = clip1.audio
        audio.write_audiofile("./audio.mp3",verbose=False,progress_bar=False)
        video_features = extract_video_features()
        video_feats.append(video_features[0])
        audio_feats.append(extract_audio_features())
    
    text_feats = extract_text_features(tr_time,transcript_path)
    
    return np.array(video_feats),np.array(audio_feats),np.array(text_feats)