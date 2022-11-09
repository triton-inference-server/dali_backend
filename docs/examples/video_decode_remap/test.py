import multiprocessing as mp
import os
import glob
import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline.experimental import pipeline_def

FRAMES_PER_SEQUENCE = 5

filenames = glob.glob(f'/home/mszolucha/workspace/DALI_extra/db/video/[cv]fr/*.mp4')
# filter out HEVC because some GPUs do not support it
filenames = filter(lambda filename: 'hevc' not in filename, filenames)
# mpeg4 is not yet supported in the CPU operator itself
filenames = filter(lambda filename: 'mpeg4' not in filename, filenames)

files = [np.fromfile(
    filename, dtype=np.uint8) for filename in filenames]

print(files)


# load remap
npz = np.load('./remap.npz')
remap_u = npz['remap_u']
remap_v = npz['remap_v']
print(remap_u.shape, remap_v.shape)
print(remap_u)

# Set video size
VIDEO_WIDTH=remap_u.shape[0]
VIDEO_HEIGHT=remap_u.shape[1]


@pipeline_def
def video_decoder_pipeline(source, device='cpu'):
    # Decode video
    data = fn.external_source(source=source, dtype=dali.types.UINT8, ndim=1)
    vid = fn.experimental.decoders.video(data, device=device)

    # Resize if necessary
    vid = fn.resize(vid, resize_x=VIDEO_WIDTH, resize_y=VIDEO_HEIGHT)

    # Undistort
    mapx = fn.external_source(source=[remap_u], batch=False, cycle=True)
    mapy = fn.external_source(source=[remap_v], batch=False, cycle=True)
    vid = fn.experimental.remap(vid, mapx, mapy)
    return vid


def video_loader(batch_size, epochs):
    idx = 0
    while idx < epochs * len(files):
        batch = []
        for _ in range(batch_size):
            batch.append(files[idx % len(files)])
            idx = idx + 1
        yield batch


def main(device):
    batch_size = 3
    epochs = 3
    pipe = video_decoder_pipeline(batch_size=batch_size, device_id=0, num_threads=4, debug=True,
                                  source=video_loader(batch_size, epochs), device=device)
    pipe.build()
    oo = pipe.run()
    print(oo)


if __name__ == '__main__':
    main('mixed')
