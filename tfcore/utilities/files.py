import glob
import os
import platform


def get_video_paths(directory, ignore_empty=False):
    """ Return a list of all the paths to images in a
    directory (not recursive).

    # Arguments
        directory: The directory in which to find the videos.
        ignore_empty: By default, `get_video_paths` throws a `FileNotFound`
            exception if no video is found to avoids surprises.
            If `ignore_empty` is set to `True`, the function
            can returns an empty list without throwing an error.

    """
    return get_paths(directory, ['mp4', 'avi', 'mov'], ignore_empty)


def get_img_paths(directory, ignore_empty=False):
    """ Return a list of all the paths to images in a
    directory (not recursive).

    # Arguments
        directory: The directory in which to find the videos.
        ignore_empty: By default, `get_video_paths` throws a `FileNotFound`
            exception if no video is found to avoids surprises.
            If `ignore_empty` is set to `True`, the function
            can returns an empty list without throwing an error.

    """
    return get_paths(directory, ["jpg", "png", "jpeg"], ignore_empty)


def get_paths(directory, exts, ignore_empty):
    list_files = []
    if platform.system() in ['Linux', 'Darwin']:
        exts += [x.upper() for x in exts]
    for ext in exts:
        list_files += _get_files_with_ext(directory, ext)

    if not list_files and not ignore_empty:
        error_msg = ""
        if not os.path.isdir(directory):
            abs_path = os.path.abspath(directory)
            error_msg += "The directory `" + abs_path
            error_msg += "` which is the absolute path of `" + directory + "` does not exist.\n"

        error_msg += "No files with one of the extension "
        error_msg += str(exts) + " was found in the directory " + directory
        raise FileNotFoundError(error_msg)
    return list_files


def _get_files_with_ext(directory, extension):
    reg = os.path.join(directory, "*." + extension)
    return list(glob.glob(reg, recursive=False))


def get_filename(idx, filename='', extension='.png', decimals=5):
    for n in range(decimals, -1, -1):
        if idx < pow(10, n):
            filename += '0'
        else:
            filename += str(idx)
            break
    return filename + extension