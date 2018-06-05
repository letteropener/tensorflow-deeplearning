import glob, os

def rename(dir, pattern, titlePattern):
    i = 1;
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename,
                  os.path.join(dir, titlePattern % i + ext))
        i = i + 1

rename(r'hand/Fa/', r'*.jpg', r'Fa_%d')
