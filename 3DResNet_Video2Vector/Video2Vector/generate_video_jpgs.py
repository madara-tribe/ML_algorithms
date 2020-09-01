import subprocess
import argparse
from pathlib import Path
from subprocess import PIPE
from joblib import Parallel, delayed
import pathlib

def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    if ext != video_file_path.suffix:
        return

    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate,duration').split()
    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd, stdout=PIPE, stderr=PIPE)
    res = p.stdout.decode('utf-8').splitlines()
    if len(res) < 4:
        return
    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]
    duration = float(res[3])
    n_frames = int(frame_rate * duration)

    name = video_file_path.stem
    dst_dir_path = dst_root_path + '/' + name
    dst_dir_path = pathlib.Path(dst_dir_path)
    dst_dir_path.mkdir(exist_ok=True)
    n_exist_frames = len([
        x for x in dst_dir_path.iterdir()
        if x.suffix == '.jpg' and x.name[0] != '.'
    ])

    if n_exist_frames >= n_frames:
        return

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)
    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    print('\n')


def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path + '/' + class_dir_path.name
    pathlib.Path(dst_class_path).mkdir(exist_ok=True)

    for video_file_path in sorted(class_dir_path.iterdir()):
        video_process(video_file_path, dst_class_path, ext, fps, size)

if __name__ == '__main__':
    # config
    dir_path = 'UCF101'
    dst_path = 'jpg'
    dataset = 'ucf101'
    n_jobs = -1
    ext = '.avi'
    
    # main
    class_dir_paths = [x for x in sorted(pathlib.Path(dir_path).iterdir())]
    test_set_video_path = dir_path +'/'+ 'test'
    #if test_set_video_path.exists():
        #class_dir_paths.append(test_set_video_path)
    status_list = Parallel(
        n_jobs=n_jobs,
        backend='threading')(delayed(class_process)(
            class_dir_path, dst_path, ext)
                             for class_dir_path in class_dir_paths)